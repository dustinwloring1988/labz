# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Score materialized items against the model under test, in the backend venv.

Run as ``<this interpreter> -m core.benchmarks.score --config <path>``.

The middle of the three passes (see ``core/benchmarks/__init__.py``). It runs
here because this is the environment that can load an ordinary HF checkpoint:
unsloth, transformers and peft, plus the device and quantisation handling that
``core/inference`` has already solved. The nanochat virtualenv has none of those.

Two scoring paths, and why the difference is recorded rather than smoothed over
-------------------------------------------------------------------------------
``logits`` (safetensors, optionally with a LoRA, optionally 4-bit)
    A categorical benchmark is answered by one forward pass: take the logits at
    the position that follows the prompt, narrow them to the token ids of the
    letters the question actually offers, and take the argmax. The model is
    never asked to emit anything, so it cannot fail to follow instructions, emit
    a preamble, or run past the answer. It is also deterministic, which is the
    property a leaderboard needs: the same checkpoint scores the same number
    twice. This is the same measurement nanochat's own ``run_categorical_eval``
    makes, against a different model.

``generated`` (GGUF, and the fallback when a tokenizer cannot represent letters
as single tokens)
    Asked to produce the answer and read back from the text. Strictly weaker
    than the logit comparison, so it is stored under a different
    ``scoring_mode`` and both the API and the UI show it. A board that mixed the
    two silently would be comparing a measurement to a guess.

Every score records the mode that produced it, and one benchmark never mixes the
two: a score belonging to neither method is worse than a lower score belonging to
one.

Chat template
-------------
A third-party model was trained with its own template, so the prompt is rendered
with that template when the tokenizer has one. nanochat's template is not used,
and should not be: it would measure how well the model does at a format it never
saw, which says something about nanochat rather than about the model. When a
tokenizer has no template at all, the raw user turn is used and the run records
that it did, because a base model scored through a format it was never trained
on is a real caveat rather than a detail.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from core.benchmarks.events import emit, stop_requested

# Problems graded between progress events. Small enough that the bar moves on a
# slow model, large enough that the event file is not the bottleneck.
_PROGRESS_EVERY = 25
# The time trigger for the same reason: see _report_progress.
_PROGRESS_EVERY_SECONDS = 3.0

# Characters kept from a generative answer in the stored prediction. The grader
# reads the file directly, so this copy exists only so a finished run can be
# inspected without re-running it.
_PREDICTION_PREVIEW_CHARS = 2000

# A standalone capital letter, for reading a letter back out of generated text.
# Bounded to the letters offered by the question by the caller.
_LETTER_RE = re.compile(r"\b([A-Z])\b")


def _log(message: str) -> None:
    print(message, flush = True)
    emit("log", line = message)


def _load_config(path: Path) -> dict:
    with path.open("r", encoding = "utf-8") as handle:
        return json.load(handle)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding = "utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated final line means this pass was killed mid-write.
                # Everything before it is valid, so what finished is still
                # reported rather than lost.
                continue
    return rows


# -----------------------------------------------------------------------------
# Conversation accessors
#
# These read the item shape the materializer stored, which is nanochat's own
# Task.get_example output. Kept as three tiny accessors rather than inlined so
# the "user turn is first, gold turn is last" assumption is stated once.
# -----------------------------------------------------------------------------


def _user_text(conversation: dict) -> str:
    for message in conversation.get("messages") or []:
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else json.dumps(content)
    return ""


def _gold_letter(conversation: dict) -> Optional[str]:
    for message in reversed(conversation.get("messages") or []):
        if message.get("role") == "assistant":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return None


def _letters(conversation: dict) -> list[str]:
    letters = conversation.get("letters")
    if isinstance(letters, (list, tuple)) and letters:
        return [str(letter) for letter in letters]
    return ["A", "B", "C", "D"]


def render_prompt(tokenizer, user_text: str) -> tuple[str, bool]:
    """The prompt text for one problem, and whether a chat template was used.

    The bool is reported rather than assumed: a tokenizer without a template
    falls back to the bare turn, and a score produced that way is not the same
    measurement as one produced with a template.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return user_text, False
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_text}],
            tokenize = False,
            add_generation_prompt = True,
        )
    except Exception as exc:  # noqa: BLE001
        # A template that raises is a broken template, not a broken run. Say so
        # and fall back rather than losing the benchmark.
        _log(f"chat template failed ({type(exc).__name__}); using the raw turn")
        return user_text, False
    if not isinstance(rendered, str) or not rendered.strip():
        return user_text, False
    return rendered, True


# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------


def _resolve_dtype(name: str):
    import torch
    if name == "auto":
        # bf16 where the hardware has it, fp16 otherwise. The run is scored on
        # numbers, so matching the hardware is better than matching whichever
        # dtype the checkpoint happened to be saved in.
        return torch.bfloat16 if torch.cuda.is_available() else torch.float16
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(name)


def resolve_load_path(path: str) -> str:
    """The directory a loader can actually open.

    A model in the HF cache is listed by the repo directory it lives in, and that
    directory has no ``config.json`` -- only its snapshot does. Passing the repo
    root to ``from_pretrained`` fails with "Unrecognized model ... should have a
    model_type key", which names the scanner's path rather than a loadable one.

    Resolved through the app's own ``local_load_dir``, the same call the inference
    orchestrator uses, so a model the Chat tab loads is a model this one loads.
    Applied here as well as in the route because this process is the one that
    opens the weights, and a config file could have been written by anything.

    Idempotent: a path that is already a snapshot resolves to itself.
    """
    try:
        from core.inference.local_model_resolver import local_load_dir
        return local_load_dir(path) or path
    except Exception:  # noqa: BLE001
        # Never block a load on a resolution helper. If it cannot answer, the
        # path is passed through and the loader's own error is the honest one.
        return path


def load_safetensors_model(config: dict):
    """Load the model under test, plus its text tokenizer. Returns (model, tokenizer)."""
    from unsloth import FastLanguageModel

    model_path = resolve_load_path(config["model_path"])
    max_seq_length = int(config.get("max_seq_length") or 2048)

    _log(f"loading {model_path}")
    model, container = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        dtype = _resolve_dtype(config.get("dtype") or "auto"),
        load_in_4bit = bool(config.get("load_in_4bit")),
        token = config.get("hf_token") or None,
        trust_remote_code = bool(config.get("trust_remote_code")),
    )

    lora_path = config.get("lora_path")
    if lora_path:
        model = attach_adapter(model, lora_path)

    model.eval()
    tokenizer = resolve_text_tokenizer(container)
    _log(
        f"loaded on {_device_of(model)} ({type(container).__name__} -> {type(tokenizer).__name__})"
    )
    return model, tokenizer


def resolve_text_tokenizer(container):
    """The text tokenizer inside whatever ``from_pretrained`` returned.

    For a vision-language model that is a *processor*, not a tokenizer: it decodes
    and applies a chat template, but has no ``encode``, so every token id lookup
    fails against it. Unwrapping is the app's existing convention -- see the same
    ``getattr(container, "tokenizer", container)`` in
    ``core/inference/inference.py`` -- rather than a new rule.

    Falls back to the container itself when there is no inner tokenizer, so a
    text-only model is unaffected.
    """
    inner = getattr(container, "tokenizer", None)
    if inner is not None and hasattr(inner, "encode"):
        return inner
    return container


def attach_adapter(model, lora_path: str):
    """Attach a LoRA adapter to an already-loaded model, in place.

    Attached rather than merged: merging writes a full copy to disk, which for a
    7B model is tens of gigabytes to answer a question about a few megabytes of
    adapter. peft rather than unsloth's own ``load_adapter`` because that one is
    bound to ``InferenceBackend``'s model registry, and this process has no
    registry -- it loaded one model and is finished with it.
    """
    from peft import PeftModel

    _log(f"attaching adapter {lora_path}")
    merged = PeftModel.from_pretrained(model, lora_path, is_trainable = False)
    merged.eval()
    return merged


def _device_of(model) -> str:
    import torch
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cpu"


def _ensure_padding(tokenizer) -> None:
    """Give the tokenizer a pad token, and left padding, for batching.

    Left padding puts every prompt's last real token at position -1, so one
    forward pass serves the batch and the column read afterwards is a real token
    for every row. A pad token is required to build a batch at all; supplying
    EOS is safe under left padding precisely because that position is never the
    one read.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"


# -----------------------------------------------------------------------------
# Categorical scoring, by logit
# -----------------------------------------------------------------------------


def _letter_token_ids(
    tokenizer,
    letters: list[str],
    prompt: str = "",
) -> Optional[dict[str, int]]:
    """Map each letter to the single token id that would start it.

    Both the bare letter and a space-prefixed one are tried, because which one the
    model will emit depends on the vocabulary and on what precedes the answer.
    Qwen, for instance, has ``"A"`` as one token and ``" A"`` as a different one;
    a prompt ending in "...correct answer." is answered with a space before the
    letter, so reading the bare form would score a token the model would not have
    produced.

    The form preferred is the one whose boundary is clean: if the prompt already
    ends in whitespace, a leading space would be absorbed into the previous
    token, so the bare letter is the correct comparison there.

    Returns None when a letter is not representable as one token under any tried
    form, because this path depends on there being exactly one logit to compare.
    A letter that splits across several tokens has none, and picking one of its
    pieces would make the score depend on an accident of BPE.
    """
    trailing_space = bool(prompt) and prompt[-1].isspace()

    mapping: dict[str, int] = {}
    for letter in letters:
        # The surface forms of *this* letter. Order follows the prompt: a prompt
        # already ending in whitespace absorbs a leading space into the previous
        # token, so the bare letter is the one whose boundary is clean there.
        forms = (letter, f" {letter}") if trailing_space else (f" {letter}", letter)
        chosen: Optional[int] = None
        for form in forms:
            try:
                encoded = tokenizer.encode(form, add_special_tokens = False)
            except Exception:  # noqa: BLE001
                return None
            if len(encoded) == 1:
                chosen = int(encoded[0])
                break
        if chosen is None:
            return None
        mapping[letter] = chosen
    return mapping


def score_categorical(
    model, tokenizer, device, items: list[dict], *, batch_size: int, max_seq_length: int
) -> tuple[list[dict], bool]:
    """One forward pass per problem, narrowed to the letters on offer.

    Returns the predictions and whether the exact path was usable throughout.
    When it was not, the caller re-runs the whole benchmark by generation rather
    than patching the remainder.
    """
    import torch

    previous_side = getattr(tokenizer, "padding_side", "right")
    _ensure_padding(tokenizer)

    predictions: list[dict] = []
    exact = True

    try:
        for start in range(0, len(items), batch_size):
            if stop_requested():
                _log("stop requested; not starting another batch")
                break
            batch = items[start : start + batch_size]

            prompts: list[str] = []
            letter_sets: list[list[str]] = []
            for row in batch:
                conversation = row.get("conversation") or {}
                prompts.append(render_prompt(tokenizer, _user_text(conversation))[0])
                letter_sets.append(_letters(conversation))

            every_letter = sorted({letter for group in letter_sets for letter in group})
            # The first prompt stands in for the batch: every prompt in it ends the
            # same way (they are the same rendered question), and the letter's
            # token boundary depends on what precedes it.
            ids = _letter_token_ids(tokenizer, every_letter, prompts[0] if prompts else "")
            if ids is None:
                _log(
                    f"{key}: this vocabulary has no single token for an answer letter; "
                    "generating instead"
                )
                exact = False
                break

            encoded = tokenizer(
                prompts,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = max_seq_length,
                add_special_tokens = True,
            ).to(device)

            with torch.no_grad():
                outputs = model(**encoded)
            logits = outputs.logits[:, -1, :].float()

            for offset, row in enumerate(batch):
                available = [letter for letter in letter_sets[offset] if letter in ids]
                if not available:
                    # Distinct from the ids-is-None case above: the vocabulary
                    # does have these letters, this particular question just offers
                    # a letter outside the batch's set. Named separately so the
                    # log says which of the two happened.
                    _log(
                        f"{key}: item {row.get('index')} offers no letter this "
                        "vocabulary can score; generating instead"
                    )
                    exact = False
                    break
                scores = {letter: float(logits[offset, ids[letter]]) for letter in available}
                predictions.append(
                    {
                        "index": int(row.get("index", start + offset)),
                        "prediction": max(scores, key = scores.get),
                        "letter_scores": scores,
                    }
                )

            _report_progress(min(start + batch_size, len(items)), len(items), "logits")
    finally:
        tokenizer.padding_side = previous_side

    return predictions, exact


# -----------------------------------------------------------------------------
# Generative scoring
# -----------------------------------------------------------------------------


def score_generative(
    model,
    tokenizer,
    device,
    items: list[dict],
    *,
    batch_size: int,
    max_seq_length: int,
    max_new_tokens: int,
    categorical: bool = False,
) -> list[dict]:
    """Greedy generation, one answer per problem.

    Greedy on purpose. A sampled answer makes the score a function of the seed,
    and a leaderboard entry that moves between two runs of the same checkpoint
    is worse than no entry.

    When ``categorical`` is set the generated text is read back for a letter
    before it is stored, because the grader compares against a bare letter and a
    full sentence would score zero for a model that answered correctly.
    """
    import torch

    previous_side = getattr(tokenizer, "padding_side", "right")
    _ensure_padding(tokenizer)

    predictions: list[dict] = []

    try:
        for start in range(0, len(items), batch_size):
            if stop_requested():
                _log("stop requested; not starting another batch")
                break
            batch = items[start : start + batch_size]

            prompts: list[str] = []
            for row in batch:
                conversation = row.get("conversation") or {}
                prompts.append(render_prompt(tokenizer, _user_text(conversation))[0])

            encoded = tokenizer(
                prompts,
                return_tensors = "pt",
                padding = True,
                truncation = True,
                max_length = max_seq_length,
                add_special_tokens = True,
            ).to(device)

            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens = max_new_tokens,
                    do_sample = False,
                    # Left as None rather than 0: a temperature of 0 divides by
                    # zero inside some sampling kernels, and do_sample=False
                    # already makes this greedy.
                    temperature = None,
                    top_p = None,
                    top_k = None,
                    pad_token_id = tokenizer.pad_token_id,
                )

            prompt_width = encoded["input_ids"].shape[1]
            for offset, row in enumerate(batch):
                text = tokenizer.decode(generated[offset, prompt_width:], skip_special_tokens = True)
                conversation = row.get("conversation") or {}
                prediction: Any = text[:_PREDICTION_PREVIEW_CHARS]
                if categorical:
                    prediction = _extract_letter(text, _letters(conversation))
                predictions.append(
                    {
                        "index": int(row.get("index", start + offset)),
                        "prediction": prediction,
                    }
                )

            _report_progress(min(start + batch_size, len(items)), len(items), "generated")
    finally:
        tokenizer.padding_side = previous_side

    return predictions


def _extract_letter(text: str, letters: list[str]) -> str:
    """The first offered letter appearing on its own in the answer.

    Returns the empty string when none does, which the grader counts as wrong.
    Inventing a letter from a model that answered nothing would turn a wrong
    answer into a right one often enough to matter.
    """
    offered = set(letters)
    for match in _LETTER_RE.finditer(text or ""):
        letter = match.group(1)
        if letter in offered:
            return letter
    return ""


def _report_progress(done: int, total: int, mode: str) -> None:
    """Emit a progress frame, on a count *or* a time trigger.

    The count alone is not enough. A generative benchmark scores one answer at a
    time and each can take twenty seconds, so a suite of eight problems would sit
    on a single frame for minutes and read as a hung run rather than a slow one.
    The time trigger is what keeps the bar honest on exactly the benchmarks where
    waiting is longest.
    """
    now = time.monotonic()
    last = _PROGRESS_LAST.get(mode, 0.0)
    count_triggered = done % _PROGRESS_EVERY == 0
    time_triggered = now - last >= _PROGRESS_EVERY_SECONDS
    if not (count_triggered or time_triggered or done >= total):
        return
    _PROGRESS_LAST[mode] = now
    emit("progress", scored = done, total = total, mode = mode)


# Last emission per scoring mode, for the time trigger above. Process-local and
# per pass, which is the right lifetime: each pass is one process, and a stale
# timestamp from a previous run would only ever make the first frame later.
_PROGRESS_LAST: dict[str, float] = {}


# -----------------------------------------------------------------------------
# GGUF, against the running llama-server
# -----------------------------------------------------------------------------


def score_gguf(item_path: Path, out_path: Path, config: dict) -> dict:
    """Score a categorical benchmark against an already-loaded GGUF.

    A GGUF has no safetensors checkpoint behind it, so it cannot be loaded into
    this process. The benchmark is scored against the llama-server the app is
    already running for it, asking for the top candidates at the first generated
    position and keeping the best letter among them. That is the same
    measurement the safetensors path makes from logits, arrived at through the
    server's ``n_probs`` instead of a local forward pass.

    It is not the same as reading a generated answer, which is why a letter that
    is absent from the returned candidates is treated as a loss rather than
    guessed at: with 20 candidates and a confident model the omitted letters are
    the ones the model did not want, and inventing an answer for them would
    inflate the score.

    The server is reused rather than started, because a second llama-server for
    the same weights is a second copy of the model in VRAM.
    """
    import requests

    from core.inference.llama_cpp import get_llama_cpp_backend

    backend = get_llama_cpp_backend()
    base_url = getattr(backend, "base_url", None)
    if not base_url:
        raise RuntimeError(
            "This GGUF is not loaded. Load it in Chat first, then benchmark it here."
        )

    items = _load_jsonl(item_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    written = 0

    with out_path.open("w", encoding = "utf-8") as handle:
        for row in items:
            if stop_requested():
                _log("stop requested; stopping the GGUF pass")
                break
            conversation = row.get("conversation") or {}
            letters = _letters(conversation)
            prompt = _user_text(conversation)

            scores = _top_candidate_logprobs(base_url, prompt, letters)
            if scores is None:
                raise RuntimeError(
                    "The running llama-server did not return candidate logprobs, "
                    "so a letter cannot be scored. Export this model to "
                    "safetensors and benchmark that instead."
                )
            handle.write(
                json.dumps(
                    {
                        "index": int(row.get("index", -1)),
                        "prediction": max(scores, key = scores.get) if scores else "",
                        "letter_scores": scores,
                    },
                    ensure_ascii = False,
                )
                + "\n"
            )
            written += 1
            if written % _PROGRESS_EVERY == 0:
                handle.flush()
                _report_progress(written, len(items), "generated")

    return {"written": written, "mode": "generated"}


def _top_candidate_logprobs(
    base_url: str, prompt: str, letters: list[str]
) -> Optional[dict[str, float]]:
    """Logprob per offered letter at the first generated position.

    Returns None when the server did not answer with candidates at all, which
    the caller treats as unsupported rather than as a score of zero.
    """
    import requests

    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/completion",
            json = {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": 20,
                "temperature": 0.0,
                "cache_prompt": False,
            },
            timeout = 120,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # noqa: BLE001
        _log(f"llama-server request failed: {type(exc).__name__}: {exc}")
        return None

    candidates = (payload.get("completion_info") or {}).get("completion_probabilities")
    if not candidates:
        return None
    first = candidates[0] or {}
    by_token = {str(entry.get("tok_str", "")): entry for entry in first.get("probs") or []}

    scores: dict[str, float] = {}
    for letter in letters:
        entry = by_token.get(letter)
        if entry is not None and entry.get("logprob") is not None:
            scores[letter] = float(entry["logprob"])
    return scores


# -----------------------------------------------------------------------------


def _kind_for(key: str) -> str:
    from core.benchmarks import catalogue

    rows, _missing = catalogue.resolve_specs([key])
    if rows:
        return str(rows[0].get("kind") or "categorical")
    return "categorical"


def _has_chat_template(tokenizer) -> bool:
    """Whether a usable chat template is present, and worth saying out loud."""
    if not getattr(tokenizer, "chat_template", None):
        _log(
            "this tokenizer has no chat template; problems will be sent as raw "
            "text, so a base model is being scored outside the format it was "
            "trained on"
        )
        return False
    return True


def _device_summary() -> dict:
    try:
        import torch

        info: dict[str, Any] = {"device_type": "cpu"}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.update(
                {
                    "device_type": "cuda",
                    "device_name": props.name,
                    "vram_bytes": int(props.total_memory),
                }
            )
        elif torch.backends.mps.is_available():
            info["device_type"] = "mps"
        return info
    except Exception:  # noqa: BLE001
        return {}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description = "Score benchmark items.")
    parser.add_argument("--config", required = True)
    args = parser.parse_args(argv)

    config = _load_config(Path(args.config))
    items_dir = Path(config["items_dir"])
    predictions_dir = Path(config["predictions_dir"])
    predictions_dir.mkdir(parents = True, exist_ok = True)

    keys = list(config.get("benchmarks") or [])
    batch_size = int(config.get("batch_size") or 8)
    max_seq_length = int(config.get("max_seq_length") or 2048)
    max_new_tokens = int(config.get("max_new_tokens") or 512)
    max_problems = config.get("max_problems")

    emit("device_summary", **_device_summary())

    if (config.get("format") or "safetensors") == "gguf":
        mode = "generated"
        for key in keys:
            try:
                score_gguf(items_dir / f"{key}.jsonl", predictions_dir / f"{key}.jsonl", config)
            except Exception as exc:  # noqa: BLE001
                _log(f"{key}: GGUF scoring failed - {type(exc).__name__}: {exc}")
                emit("score_failed", key = key, error = str(exc))
        emit("scoring_complete", mode = mode)
        return 0

    try:
        model, tokenizer = load_safetensors_model(config)
    except Exception as exc:  # noqa: BLE001
        _log(f"could not load the model: {type(exc).__name__}: {exc}")
        _log(traceback.format_exc())
        emit("load_failed", error = str(exc))
        return 2

    device = _device_of(model)
    emit("model_loaded", device = device, chat_template = _has_chat_template(tokenizer))

    modes: set[str] = set()

    for key in keys:
        items = _load_jsonl(items_dir / f"{key}.jsonl")
        if not items:
            _log(f"{key}: no items; skipping")
            emit("score_skipped", key = key, reason = "no items were materialized")
            continue

        # The cap is the user's, and the item file is shared across runs. If the
        # two disagree the run is trimmed here and the mismatch is said out loud,
        # because silently scoring 400 problems for a run capped at 8 would
        # report a total the user never asked for and a "sampled" flag that would
        # be true for the wrong reason.
        if isinstance(max_problems, int) and len(items) > max_problems:
            _log(
                f"{key}: {len(items)} items are cached but this run is capped at "
                f"{max_problems}; scoring the first {max_problems}"
            )
            items = items[:max_problems]

        out_path = predictions_dir / f"{key}.jsonl"
        kind = _kind_for(key)
        started = time.time()

        try:
            if kind == "categorical":
                predictions, exact = score_categorical(
                    model,
                    tokenizer,
                    device,
                    items,
                    batch_size = batch_size,
                    max_seq_length = max_seq_length,
                )
                if exact:
                    modes.add("logits")
                else:
                    # The whole benchmark is re-run by generation. Mixing a logit
                    # comparison for most problems with a text read for the rest
                    # would produce a number belonging to neither method.
                    # The reason was already logged where it was determined;
                    # repeating a guess here would be wrong whenever the cause was
                    # the question rather than the vocabulary.
                    predictions = score_generative(
                        model,
                        tokenizer,
                        device,
                        items,
                        batch_size = 1,
                        max_seq_length = max_seq_length,
                        max_new_tokens = 8,
                        categorical = True,
                    )
                    modes.add("generated")
            else:
                predictions = score_generative(
                    model,
                    tokenizer,
                    device,
                    items,
                    batch_size = 1,
                    max_seq_length = max_seq_length,
                    max_new_tokens = max_new_tokens,
                )
                modes.add("generated")
        except Exception as exc:  # noqa: BLE001
            _log(f"{key}: scoring failed - {type(exc).__name__}: {exc}")
            emit("score_failed", key = key, error = str(exc))
            continue

        with out_path.open("w", encoding = "utf-8") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii = False) + "\n")

        _log(f"{key}: scored {len(predictions)}/{len(items)} in {time.time() - started:.1f}s")
        emit(
            "scored",
            key = key,
            scored = len(predictions),
            total = len(items),
            elapsed_seconds = time.time() - started,
        )

    mode = "mixed" if len(modes) > 1 else (next(iter(modes)) if modes else None)
    emit("scoring_complete", mode = mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
