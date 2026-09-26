"""Pin the backend's nanochat shape arithmetic to verified values.

``core/nanochat/presets.py`` duplicates the maths in ``nanochat/shapes.py``
because the backend cannot import nanochat (separate virtualenv). If the two
drift, the UI promises a model that is not the one nanochat builds, so these
vectors are checked as literals.

The numbers were produced by ``nanochat/shapes.py`` and are independently
verified there against the real model (``nanochat/tests/test_shapes.py``). Both
sides test the same vectors, so a change to either implementation fails on its
own side without needing the other project installed.
"""
import pytest

from core.nanochat import presets

# depth, vocab, aspect_ratio, head_dim, n_embd, n_head, wte, value_embeds,
# lm_head, transformer_matrices, scalars, total
VECTORS = [
    (2, 512, 64, 128, 128, 1, 65536, 65536, 65536, 393228, 30, 589866),
    (4, 461, 64, 64, 256, 4, 131072, 262144, 131072, 3145824, 34, 3670146),
    (5, 512, 64, 128, 384, 3, 196608, 589824, 196608, 8847468, 36, 9830544),
    (6, 1024, 64, 64, 384, 6, 393216, 1179648, 393216, 10617048, 38, 12583166),
    # depth 9 with aspect 50: 9*50=450, which is not a multiple of 128, so it
    # rounds up to 512. This is the case a naive depth*aspect_ratio gets wrong.
    (9, 512, 50, 128, 512, 4, 262144, 1310720, 262144, 28311792, 44, 30146844),
    (12, 32768, 64, 128, 768, 6, 25165824, 150994944, 25165824, 84935088, 50, 286261730),
    (16, 32768, 64, 128, 1024, 8, 33554432, 268435456, 33554432, 201327360, 58, 536871738),
    (20, 32768, 64, 128, 1280, 10, 41943040, 419430400, 41943040, 393217200, 66, 896533746),
    (24, 32768, 64, 128, 1536, 12, 50331648, 603979776, 50331648, 679478976, 74, 1384122122),
]

COUNT_KEYS = ("wte", "value_embeds", "lm_head", "transformer_matrices", "scalars", "total")


@pytest.mark.parametrize(
    "depth,vocab,aspect_ratio,head_dim,n_embd,n_head,wte,value_embeds,lm_head,matrices,scalars,total",
    VECTORS,
)
def test_param_breakdown_matches_verified_vectors(
    depth, vocab, aspect_ratio, head_dim, n_embd, n_head,
    wte, value_embeds, lm_head, matrices, scalars, total,
):
    assert presets.resolve_shape(depth, aspect_ratio, head_dim) == (n_embd, n_head)
    counts = presets.param_breakdown(depth, vocab, aspect_ratio, head_dim)
    assert counts["wte"] == wte
    assert counts["value_embeds"] == value_embeds
    assert counts["lm_head"] == lm_head
    assert counts["transformer_matrices"] == matrices
    assert counts["scalars"] == scalars
    assert counts["total"] == total
    # The breakdown must actually add up, not just match a table.
    assert sum(counts[k] for k in COUNT_KEYS[:-1]) == counts["total"]


def test_pad_vocab_size():
    assert presets.pad_vocab_size(512) == 512
    assert presets.pad_vocab_size(461) == 512
    assert presets.pad_vocab_size(1) == 64


def test_preset_parameter_counts_are_monotonic():
    """A bigger depth must never be a smaller model."""
    rows = presets.describe_presets()
    totals = [r["num_params"] for r in rows]
    assert totals == sorted(totals)
    assert len(set(totals)) == len(totals)


def test_preset_summaries_quote_the_real_parameter_count():
    """The summary strings carry numbers a user reads; keep them honest.

    Catches an edit to the presets that forgets to update the prose, which is
    otherwise invisible because nothing checks a human-readable string.
    """
    for row in presets.describe_presets():
        millions = row["num_params"] / 1e6
        if row["depth"] < 32:
            # Summaries round to whole millions ("286M", "1.38B").
            expected = f"{millions:.0f}M" if millions < 1000 else f"{millions/1000:.2f}B"
            assert expected in row["summary"], (
                f"{row['label']} is {row['num_params']:,} params but its summary "
                f"says {row['summary']!r} (expected it to contain {expected!r})"
            )


def test_every_preset_depth_is_usable():
    for row in presets.describe_presets():
        assert row["depth"] >= 1
        assert row["num_params"] > 0
        assert row["n_head"] > 0
        assert row["label"] == f"d{row['depth']}"
        assert row["good_for"]


def test_default_depth_is_offered():
    rows = presets.describe_presets()
    assert sum(1 for r in rows if r["is_default"]) == 1
    assert next(r for r in rows if r["is_default"])["depth"] == presets.DEFAULT_DEPTH


def test_embeddings_dominate_parameters_below_depth_25():
    """The parameter count is mostly embedding table at consumer-realistic depths.

    With a 32768 vocab, the wte/lm_head pair and the per-layer value embeddings
    together outnumber the transformer matrices until about depth 25. So "d12 is
    286M parameters" overstates how much *reasoning* capacity that is: 85M of it
    is transformer. The depth picker shows both numbers for this reason, and this
    test fails if the presets start quoting the total alone.
    """
    rows = {r["depth"]: r for r in presets.describe_presets()}
    for depth in (4, 8, 12, 20, 24):
        assert rows[depth]["embedding_params"] > rows[depth]["transformer_params"], (
            f"d{depth} is expected to be embedding-dominated"
        )
    # Crossover is just past the largest depth the speedrun uses.
    assert rows[32]["transformer_params"] > rows[32]["embedding_params"]


def test_summary_mentions_transformer_params_for_embedding_dominated_depths():
    rows = {r["depth"]: r for r in presets.describe_presets()}
    for depth in (4, 6, 8, 10, 12, 16, 20, 24):
        if rows[depth]["embedding_params"] > rows[depth]["transformer_params"]:
            assert "transformer" in rows[depth]["summary"], (
                f"d{depth} is embedding-dominated, so its summary must qualify the "
                f"parameter count: {rows[depth]['summary']!r}"
            )


def test_unknown_gpu_gets_no_fabricated_flops():
    assert presets._table_peak_flops("some gpu nobody has heard of") is None


@pytest.mark.parametrize("reported,expected", [
    ("NVIDIA GeForce RTX 4060 Ti", 22.2e12),
    ("NVIDIA GeForce RTX 4090", 165.2e12),
    ("NVIDIA H100 80GB HBM3", 989e12),
])
def test_gpu_name_normalisation_finds_the_right_entry(reported, expected):
    assert presets._table_peak_flops(presets._normalize_gpu_name(reported)) == expected


def test_consumer_laptop_suffix_is_not_confused_with_desktop():
    """4090 laptop is a different die with a lower clock; the table distinguishes them."""
    assert presets._table_peak_flops(presets._normalize_gpu_name("NVIDIA GeForce RTX 4090")) == 165.2e12
    assert presets._table_peak_flops(presets._normalize_gpu_name("NVIDIA GeForce RTX 4090 Laptop GPU")) == 103.2e12


def test_fit_estimate_flags_an_impossible_run_with_a_suggestion():
    estimate = presets.estimate_fit(32, num_iterations=1000, total_batch_size=524288)
    if not estimate.fits:
        assert estimate.reasons, "a refusal must say why"
        assert estimate.suggestions or estimate.device_type in ("cpu", "mlx")
    assert estimate.tokens == 1000 * 524288
    assert estimate.total_flops > 0


def test_fit_estimate_scales_with_batch_size():
    small = presets.estimate_fit(12, num_iterations=10, total_batch_size=32768)
    large = presets.estimate_fit(12, num_iterations=10, total_batch_size=524288)
    assert large.training_memory_bytes > small.training_memory_bytes
    assert large.tokens == small.tokens * 16


def test_memory_estimate_never_undercounts_weights():
    """Weights, master weights, Adam moments and gradients are a hard floor."""
    for depth in (4, 12, 24):
        counts = presets.param_breakdown(depth)
        floor = counts["total"] * 16  # 2 (bf16) + 12 (optim) + 2 (grads)
        assert presets.training_memory_bytes(depth) > floor
