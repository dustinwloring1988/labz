"""Depth presets and what they cost, for consumer hardware.

nanochat has exactly one architectural dial: ``--depth``. Everything else is
derived (model_dim = depth * aspect_ratio rounded up to a multiple of head_dim,
heads = model_dim / head_dim). That makes depth a good thing to expose directly,
but it also means a user can pick a value that will not fit their GPU or will
take a week to train, and only find out after the dataset has downloaded.

So the UI needs, before starting:
  - what the model at depth N actually is (dims, heads, parameter count)
  - whether it fits the VRAM this machine has
  - roughly how long it will take on this machine

The parameter arithmetic mirrors ``nanochat/shapes.py``, which is the
single source of truth on the nanochat side and is tested against the real
model. It is duplicated here because the backend cannot import nanochat (separate
virtualenv), so ``tests/test_nanochat_presets.py`` pins both against the same
vectors and fails if they drift.

Numbers in this module are estimates for *choosing*, not promises. The exact
figure for a run is reported by nanochat itself in its ``config`` event, and the
UI shows that once the run starts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# nanochat's defaults. The speedrun uses depth 24 on 8xH100; these are the
# knobs a desktop user is more likely to be running.
DEFAULT_ASPECT_RATIO = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_VOCAB_SIZE = 32768  # 2**15, what scripts.tok_train produces
PAD_VOCAB_TO = 64
MLP_EXPANSION = 4
VE_GATE_CHANNELS = 12
SMEAR_GATE_CHANNELS = 24

# Rough sustained throughput in tokens/sec per unit of peak bf16 FLOPs. Derived
# from published nanochat runs (44% MFU on a d24 speedrun) and rounded down,
# because a user running on a consumer card with no Flash Attention and a
# sliding-window pattern gets closer to half that. Deliberately pessimistic: a
# time estimate that under-promises is a time estimate people trust.
EFFECTIVE_MFU = 0.25

# VRAM headroom required before a run is considered safe to start.
VRAM_HEADROOM_FRACTION = 0.85


# -----------------------------------------------------------------------------
# Shape arithmetic (mirrors nanochat/shapes.py)
# -----------------------------------------------------------------------------

def resolve_shape(depth: int, aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                  head_dim: int = DEFAULT_HEAD_DIM) -> tuple[int, int]:
    """(n_embd, n_head) for a depth.

    n_embd is rounded up to a whole multiple of head_dim, so it is not simply
    depth * aspect_ratio; getting that wrong would misreport every derived number.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")
    if head_dim < 1:
        raise ValueError(f"head_dim must be >= 1, got {head_dim}")
    base_dim = depth * aspect_ratio
    n_embd = ((base_dim + head_dim - 1) // head_dim) * head_dim
    return n_embd, n_embd // head_dim


def pad_vocab_size(vocab_size: int) -> int:
    return ((vocab_size + PAD_VOCAB_TO - 1) // PAD_VOCAB_TO) * PAD_VOCAB_TO


def param_breakdown(depth: int, vocab_size: int = DEFAULT_VOCAB_SIZE,
                    aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                    head_dim: int = DEFAULT_HEAD_DIM) -> dict[str, int]:
    """Parameter counts split the same way GPT.num_scaling_params() splits them."""
    n_embd, n_head = resolve_shape(depth, aspect_ratio, head_dim)
    padded_vocab = pad_vocab_size(vocab_size)
    n_ve = (depth + 1) // 2  # gpt.py's has_ve alternates layers

    attn = 4 * n_embd * n_embd                      # q, k, v, out
    mlp = 2 * n_embd * (MLP_EXPANSION * n_embd)     # up, down
    transformer_matrices = depth * (attn + mlp) + n_ve * VE_GATE_CHANNELS * n_head

    wte = padded_vocab * n_embd
    value_embeds = n_ve * padded_vocab * n_head * head_dim
    lm_head = n_embd * padded_vocab
    scalars = 2 * depth + SMEAR_GATE_CHANNELS + 2   # resid, x0, smear_gate, smear_lambda, backout_lambda

    return {
        "wte": wte,
        "value_embeds": value_embeds,
        "lm_head": lm_head,
        "transformer_matrices": transformer_matrices,
        "scalars": scalars,
        "total": wte + value_embeds + lm_head + transformer_matrices + scalars,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": depth,
        "padded_vocab_size": padded_vocab,
    }


def flops_per_token(depth: int, vocab_size: int = DEFAULT_VOCAB_SIZE,
                    aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                    head_dim: int = DEFAULT_HEAD_DIM) -> float:
    """Forward+backward FLOPs per token, the 6*N term plus the output projection."""
    counts = param_breakdown(depth, vocab_size, aspect_ratio, head_dim)
    n_embd = counts["n_embd"]
    matrix_flops = 6 * (counts["n_layer"] * 12 * n_embd * n_embd)
    embed_flops = 6 * counts["padded_vocab_size"] * n_embd
    return float(matrix_flops + embed_flops)


def _memory_terms(depth: int, vocab_size: int, total_batch_size: int,
                  max_seq_len: int, aspect_ratio: int, head_dim: int) -> dict[str, int]:
    """Split the training memory estimate into a fixed part and a per-token part.

    Splitting them is what makes a useful suggestion possible: model state
    (weights, master weights, Adam moments, gradients) does not depend on the
    batch, so it cannot be reduced away, while activations and the fp32 logits
    scale linearly with tokens in the batch.
    """
    counts = param_breakdown(depth, vocab_size, aspect_ratio, head_dim)
    n_embd = counts["n_embd"]
    n_layer = counts["n_layer"]
    total_params = counts["total"]
    padded_vocab = counts["padded_vocab_size"]

    fixed = total_params * (2 + 12 + 2)  # bf16 weights, fp32 master + Adam m/v, grads
    per_token = n_layer * n_embd * 2 * 12 + padded_vocab * 4 * 2
    return {
        "fixed": fixed,
        "per_token": per_token,
        "total": fixed + per_token * total_batch_size,
    }


def training_memory_bytes(depth: int, vocab_size: int = DEFAULT_VOCAB_SIZE,
                          total_batch_size: int = 524288, max_seq_len: int = 2048,
                          aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                          head_dim: int = DEFAULT_HEAD_DIM) -> int:
    """Approximate steady-state training memory in bytes.

    At a 32768 vocab the fp32 logits dominate: one token of context costs
    ``vocab * 4 * 2`` bytes in activations alone, which is why a consumer card
    runs nanochat with a batch in the tens of thousands rather than half a million.
    """
    terms = _memory_terms(depth, vocab_size, total_batch_size, max_seq_len,
                          aspect_ratio, head_dim)
    return terms["total"]


def kv_cache_bytes(depth: int, context_tokens: int, aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                   head_dim: int = DEFAULT_HEAD_DIM) -> int:
    _, n_head = resolve_shape(depth, aspect_ratio, head_dim)
    return depth * 2 * n_head * head_dim * 2 * context_tokens


# -----------------------------------------------------------------------------
# Presets
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DepthPreset:
    """A named depth the picker offers, with the caveats that go with it."""

    key: str
    depth: int
    label: str
    tier: str  # tiny | small | medium | large | xlarge
    summary: str
    # What the run is realistically for, so a preset is a recommendation and
    # not just a number.
    good_for: str
    # Depths at or below this are the ones nanochat's own run scripts exercise.
    caveat: str | None = None


# Parameter counts below are with the default 32768 vocab, and are the
# transformer *matrices* alongside the total, because at small depths the
# embedding tables dominate and quoting the total alone is misleading: d4 is 37M
# parameters but only 3.1M of them are transformer, the rest is a lookup table
# that does not get smarter with depth. The UI shows both.
DEPTH_PRESETS: tuple[DepthPreset, ...] = (
    DepthPreset(
        key="d4", depth=4, label="d4", tier="tiny",
        summary="37M params, but only 3M of them transformer",
        good_for="Seeing the whole pipeline end to end. Not a usable model.",
        caveat="Almost all of it is the token embedding table. Far too small to "
               "produce coherent text; use it to check the plumbing.",
    ),
    DepthPreset(
        key="d6", depth=6, label="d6", tier="tiny",
        summary="74M params, 11M transformer",
        good_for="A working toy model you can watch learn a small domain.",
        caveat="Output is mostly word salad.",
    ),
    DepthPreset(
        key="d8", depth=8, label="d8", tier="small",
        summary="126M params, 25M transformer, minutes on a modern GPU",
        good_for="A first model that completes short prompts recognisably.",
    ),
    DepthPreset(
        key="d10", depth=10, label="d10", tier="small",
        summary="196M params, 49M transformer",
        good_for="Follows simple instructions. A sensible first real run.",
    ),
    DepthPreset(
        key="d12", depth=12, label="d12", tier="medium",
        summary="286M params, 85M transformer, the smallest nanochat miniseries depth",
        good_for="Coherent short answers, roughly GPT-2-small quality.",
    ),
    DepthPreset(
        key="d16", depth=16, label="d16", tier="medium",
        summary="537M params, 201M transformer",
        good_for="Noticeably better reasoning. Wants 12GB+ of VRAM.",
    ),
    DepthPreset(
        key="d20", depth=20, label="d20", tier="large",
        summary="897M params, 393M transformer, nanochat's default depth",
        good_for="Chattier and more accurate. Expect a multi-hour run on one GPU.",
    ),
    DepthPreset(
        key="d24", depth=24, label="d24", tier="large",
        summary="1.38B params, 679M transformer, the depth of the published speedrun",
        good_for="GPT-2 class quality.",
        caveat="The speedrun reached this on 8xH100 in about 90 minutes. On one "
               "consumer GPU it is a multi-day commitment.",
    ),
    DepthPreset(
        key="d32", depth=32, label="d32", tier="xlarge",
        summary="2.8B params, 1.6B transformer",
        good_for="Scaling-law experiments.",
        caveat="Weeks on one consumer GPU. Included so the scaling curve is "
               "explorable, not because it is practical here.",
    ),
)

DEFAULT_DEPTH = 12


def preset_for_depth(depth: int) -> DepthPreset | None:
    for preset in DEPTH_PRESETS:
        if preset.depth == depth:
            return preset
    return None


# -----------------------------------------------------------------------------
# Hardware fit
# -----------------------------------------------------------------------------

@dataclass
class FitEstimate:
    """Whether a configuration fits this machine, and what it will cost."""

    depth: int
    num_params: int
    n_embd: int
    n_head: int
    model_memory_bytes: int
    training_memory_bytes: int
    tokens: int
    total_flops: float
    est_seconds: float | None = None
    fits: bool = True
    # Empty when it fits. A list is more useful than a bool: the smallest fix is
    # usually "reduce the batch size", not "buy a bigger GPU".
    reasons: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    device_name: str | None = None
    peak_flops_bf16: float | None = None
    device_type: str = "cpu"
    # Largest batch that would fit, when one could be worked out. None means
    # either "no limit needed" or "the batch size cannot fix this".
    max_batch_size: int | None = None


def _device_peak_flops() -> tuple[str | None, float | None, str]:
    """(name, peak bf16 FLOPs, device_type) for the local accelerator.

    The FLOPS numbers come from nanochat's own table in nanochat/common.py, so a
    card the table knows about is measured the same way nanochat will measure it.
    Cards it does not know about return None, which downgrades the estimate to
    "cannot predict" rather than inventing a number.
    """
    try:
        from utils.hardware import get_device, get_gpu_memory_info

        device = get_device()
        device_type = getattr(device, "value", "cpu") if device is not None else "cpu"
        if device_type in ("cpu", "mlx"):
            return None, None, device_type
        info = get_gpu_memory_info()
        if not isinstance(info, dict) or not info.get("available"):
            return None, None, device_type
        # get_gpu_memory_info reports sizes in GiB, under *_gb.
        name = info.get("device_name")
        flops = _table_peak_flops(_normalize_gpu_name(name or ""))
        return name, flops, device_type
    except Exception:
        return None, None, "cpu"


# Mirrors _PEAK_FLOPS_TABLE in nanochat/common.py for the consumer and data
# center cards people actually run nanochat on. Keeping the two in step matters:
# a different peak FLOPS here changes the time estimate, not just the label.
_TABLE_PEAK_FLOPS: dict[str, float] = {
    "gb200": 2.5e15, "grace blackwell": 2.5e15, "b200": 2.25e15, "b100": 1.8e15,
    "h200 nvl": 836e12, "h200 pcie": 836e12, "h200": 989e12,
    "h100 nvl": 835e12, "h100 pcie": 756e12, "h100": 989e12,
    "h800 nvl": 989e12, "h800 pcie": 756e12, "h800": 989e12,
    "a100": 312e12, "a800": 312e12, "a40": 149.7e12, "a30": 165e12,
    "l40s": 362e12, "l40 s": 362e12, "l4": 121e12,
    "mi355": 2.5e15, "mi325": 1.3074e15, "mi300x": 1.3074e15,
    "mi300a": 980.6e12, "mi250x": 383e12, "mi250": 362.1e12,
    "5090 laptop": 167.6e12, "5090": 209.5e12, "5080": 125e12,
    "5070 ti": 92e12, "5070": 100.5e12, "5060 ti": 63e12,
    "4090 laptop": 103.2e12, "4090": 165.2e12,
    "4080 super": 52.2e12, "4080": 61.3e12,
    "4070 ti super": 40.1e12, "4070 ti": 40.1e12, "4070": 29.1e12,
    "4060 ti": 22.2e12, "4060": 15.1e12,
    "3090 ti": 21.7e12, "3090": 71e12,
}


def _normalize_gpu_name(name: str) -> str:
    """Strip the vendor prefix so a table lookup can match.

    Most specific patterns must be tried first, which a dict keyed by the whole
    name does not give, so the longest keys are checked in order.
    """
    lowered = name.lower()
    for vendor in ("nvidia geforce ", "nvidia ", "amd radeon ", "amd "):
        if lowered.startswith(vendor):
            lowered = lowered[len(vendor):]
            break
    return lowered.strip()


def _table_peak_flops(normalized: str) -> float | None:
    for key in sorted(_TABLE_PEAK_FLOPS, key=len, reverse=True):
        if key in normalized:
            return _TABLE_PEAK_FLOPS[key]
    return None


def available_vram_bytes() -> tuple[int | None, str]:
    """Free VRAM in bytes on the local accelerator, or (None, reason).

    get_gpu_memory_info reports GiB, which is converted here so every size in this
    module is bytes. Mixing the two is how a 16GB card turns into a 16 *billion*
    byte budget and everything looks like it fits.
    """
    try:
        from utils.hardware import get_device, get_gpu_memory_info

        device = get_device()
        if device is None or getattr(device, "value", "cpu") in ("cpu", "mlx"):
            return None, "no CUDA device"
        info = get_gpu_memory_info()
        if isinstance(info, dict) and info.get("available"):
            free_gb = info.get("free_gb")
            if free_gb is not None:
                return int(float(free_gb) * (1024 ** 3)), "ok"
            total_gb = info.get("total_gb")
            if total_gb:
                # No free figure: fall back to total, which over-promises. The
                # headroom factor below absorbs most of that.
                return int(float(total_gb) * (1024 ** 3)), "total (free unavailable)"
        error = info.get("error") if isinstance(info, dict) else None
        return None, str(error or "could not read free VRAM")
    except Exception as exc:
        return None, str(exc)


def estimate_fit(depth: int, *, num_iterations: int, total_batch_size: int = 524288,
                 vocab_size: int = DEFAULT_VOCAB_SIZE, max_seq_len: int = 2048,
                 aspect_ratio: int = DEFAULT_ASPECT_RATIO,
                 head_dim: int = DEFAULT_HEAD_DIM,
                 param_data_ratio: float = 12.0) -> FitEstimate:
    """Model shape, memory footprint and a wall-clock estimate for a run."""
    counts = param_breakdown(depth, vocab_size, aspect_ratio, head_dim)
    params = counts["total"]
    n_layer = counts["n_layer"]
    n_embd = counts["n_embd"]

    # Weights only, for inference/chat sizing.
    model_bytes = params * 2
    train_bytes = training_memory_bytes(depth, vocab_size, total_batch_size,
                                        max_seq_len, aspect_ratio, head_dim)
    per_token_flops = flops_per_token(depth, vocab_size, aspect_ratio, head_dim)
    tokens = int(total_batch_size * num_iterations)
    total_flops = per_token_flops * tokens

    name, peak_flops, device_type = _device_peak_flops()
    estimate = FitEstimate(
        depth=depth,
        num_params=params,
        n_embd=n_embd,
        n_head=counts["n_head"],
        model_memory_bytes=model_bytes,
        training_memory_bytes=train_bytes,
        tokens=tokens,
        total_flops=total_flops,
        device_name=name,
        peak_flops_bf16=peak_flops,
        device_type=device_type,
    )

    # --- time ---
    if peak_flops and peak_flops > 0:
        effective = peak_flops * EFFECTIVE_MFU
        estimate.est_seconds = total_flops / effective if effective else None
    elif device_type in ("cpu", "mlx"):
        # A CPU number is dominated by memory bandwidth and is far too machine
        # specific to guess. Say so rather than implying it is knowable.
        estimate.reasons.append(
            "This looks like a CPU-only machine. nanochat will run, but a real "
            "run here is measured in days, not hours."
        )
        estimate.fits = False
        estimate.suggestions.append("A CUDA GPU changes this by two orders of magnitude.")

    # --- memory ---
    free, why = available_vram_bytes()
    if free is not None:
        budget = free * VRAM_HEADROOM_FRACTION
        if train_bytes > budget:
            estimate.fits = False
            estimate.reasons.append(
                f"Needs about {train_bytes / 1e9:.1f} GB of VRAM for training but only "
                f"about {free / 1e9:.1f} GB is free (allowing {VRAM_HEADROOM_FRACTION:.0%} "
                f"headroom)."
            )
            terms = _memory_terms(depth, vocab_size, total_batch_size, max_seq_len,
                                  aspect_ratio, head_dim)
            # Solve for the batch that fits: the activations and fp32 logits are
            # linear in tokens, so (budget - fixed) / per_token is exact for this
            # model of memory rather than a guess.
            if terms["per_token"] > 0:
                affordable = (budget - terms["fixed"]) / terms["per_token"]
                max_batch = int(max(0, affordable))
                if max_batch >= 1024:
                    # Round down to something a person would actually enter.
                    rounded = (max_batch // 1024) * 1024
                    estimate.suggestions.append(
                        f"Lower total batch size to about {rounded:,} tokens to fit "
                        f"(down from {total_batch_size:,})."
                    )
                    estimate.max_batch_size = rounded
                else:
                    # Not fixable by batch size: the model state alone is too big.
                    estimate.suggestions.append(
                        "Reducing the batch size will not be enough here: the model "
                        "state alone does not fit. Choose a shallower depth."
                    )
            estimate.suggestions.append(
                "Close other GPU apps to free VRAM, and prefer sliding-window "
                "attention off (--window-pattern L) if throughput is poor."
            )
    elif device_type in ("cpu", "mlx"):
        pass  # already reported above
    else:
        estimate.reasons.append(f"Could not check free VRAM ({why}).")

    return estimate


def describe_presets() -> list[dict]:
    """Preset list enriched with shape, for the depth picker.

    Not a FitEstimate per preset: fit depends on the rest of the configuration
    (batch size, iterations), and the picker changes that independently.
    """
    out = []
    for preset in DEPTH_PRESETS:
        counts = param_breakdown(preset.depth)
        row = asdict(preset)
        embedding_params = counts["wte"] + counts["lm_head"] + counts["value_embeds"]
        row.update({
            "num_params": counts["total"],
            # The number that actually reflects what depth buys. At small depths
            # the embedding table is most of the model.
            "transformer_params": counts["transformer_matrices"],
            "embedding_params": embedding_params,
            "n_embd": counts["n_embd"],
            "n_head": counts["n_head"],
            "kv_cache_bytes_per_token": kv_cache_bytes(preset.depth, 1),
            "model_tag": f"d{preset.depth}",
            "is_default": preset.depth == DEFAULT_DEPTH,
        })
        out.append(row)
    return out
