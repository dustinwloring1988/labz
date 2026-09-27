# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The generation worker written into an autoresearch checkout.

autoresearch never calls ``model.generate()``. ``train.py`` is a benchmark: it
trains, evaluates one number, saves the weights and exits, and every one of its
outputs is either that number or a file. So there is nothing in the project to
chat with, and the studio cannot add one in its own process either — the weights
are a raw ``state_dict`` for a model class that only exists inside autoresearch's
virtualenv, under a torch version the studio does not have.

So this writes a small server into the copy of the project and runs it there, in
the project's own venv. It is the same shape as nanochat's ``chat_server``: one
long-lived process, newline-delimited JSON on stdin and stdout, a ready
handshake, and cumulative text.

Two constraints shape the code:

  - **It must not import anything the project does not.** It reads ``GPT`` and
    ``build_model_config`` from ``train.py`` and ``Tokenizer`` from
    ``prepare.py``, because those are the definitions the weights were saved
    against. Recompiling the architecture here would be a second source of
    truth that silently disagrees with the project after any edit.

  - **The architecture is recovered from the weights, not from train.py's
    constants.** ``build_model_config`` derives everything from ``DEPTH`` and
    the vocab size, so the depth has to come from the checkpoint rather than from
    whatever the agent's last experiment happened to set it to. Counting the
    transformer blocks recovers it exactly.

The file is gitignored on install and rewritten whenever the studio refreshes the
checkout, so it is treated as generated rather than as project source.
"""

from __future__ import annotations

# The script itself, as text. Deliberately a plain string rather than a file on
# disk in the backend package: it is copied into the user's project directory, and
# a module that exists in two places is a module that can be out of step.
CHAT_SERVER_SOURCE = r'''"""
Generation worker for an autoresearch checkpoint.

Written into this checkout by LABZ Studio. Not part of autoresearch.

Protocol: newline-delimited JSON on stdin and stdout. One request per line, each
carrying an id; replies carry the same id. A reply is either {"text": ...} for a
cumulative snapshot of the response so far, or {"done": true} at the end. Logs
go to stderr, because stdout is the reply stream.

    {"id": 1, "prompt": "...", "max_new_tokens": 256, "temperature": 0.8, "top_k": 50}
    {"id": 1, "abort": true}
    -> {"ready": true}                      (once, after the weights are resident)
    -> {"id": 1, "text": "Once upon"}
    -> {"id": 1, "done": true}
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from prepare import MAX_SEQ_LEN, Tokenizer  # noqa: E402
from train import GPT, build_model_config, detect_runtime  # noqa: E402


def log(message):
    """To stderr, so it can never be mistaken for a protocol frame."""
    print(message, file=sys.stderr, flush=True)


def emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


class Aborted(Exception):
    pass


def infer_depth(state_dict):
    """How many transformer blocks the checkpoint has.

    The depth is the one architectural number the weights record and train.py's
    constants do not: every other shape follows from build_model_config. Reading
    it back is what makes an old experiment's weights loadable after the agent
    has changed DEPTH for a later one.
    """
    depths = set()
    for key in state_dict:
        if key.startswith("transformer.h."):
            parts = key.split(".")
            if len(parts) > 3 and parts[3].isdigit():
                depths.add(int(parts[3]))
    if not depths:
        raise RuntimeError("this checkpoint has no transformer blocks in it")
    return max(depths) + 1


def load_model(weights_path, dataset, device):
    runtime = detect_runtime()
    tokenizer = Tokenizer.from_directory(dataset=dataset)
    vocab_size = tokenizer.get_vocab_size()

    # torch.load with weights_only: this file came out of the same process that
    # wrote it, but it is still a pickle on disk and the default path executes
    # arbitrary code from it.
    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)

    depth = infer_depth(state_dict)
    config = build_model_config(depth, vocab_size, runtime, use_activation_checkpointing=False)
    log(f"checkpoint: depth={depth} vocab_size={vocab_size} "
        f"n_embd={config.n_embd} n_head={config.n_head} seq={config.sequence_len}")

    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except Exception as exc:
        # The usual cause is the agent changing a shape constant (ASPECT_RATIO,
        # HEAD_DIM) after this experiment ran, so the saved weights no longer
        # match what build_model_config produces. Say that rather than letting a
        # bare shape tuple be the error.
        raise RuntimeError(
            f"could not load {weights_path}: {exc}\n"
            f"This usually means train.py's shape constants (ASPECT_RATIO, "
            f"HEAD_DIM) or the tokenizer have changed since this experiment ran. "
            f"Chatting with an older checkpoint needs the train.py that produced it."
        ) from exc

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, tokenizer, config, runtime


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k,
             config, device, should_abort):
    """Sample a continuation, yielding cumulative text.

    A base model trained on raw text continues text rather than replying, so the
    prompt is encoded with a BOS and the model's own idea of what follows is the
    answer. There is no chat template to apply, because there was never a chat
    stage: the weights came out of a pretraining run.
    """
    bos = tokenizer.get_bos_token_id()
    ids = tokenizer.encode(prompt, prepend=bos)[-(config.sequence_len - 1):]

    produced = []
    for _ in range(max(1, int(max_new_tokens))):
        if should_abort():
            raise Aborted()
        window = ids[-config.sequence_len:]
        # A row with a single token still needs a batch dimension and at least
        # one position, and the first generated token has no logits to sample
        # from, so prime with a BOS-only row.
        if len(window) < 2:
            window = [bos, bos]
        inp = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inp)[:, -1, :].float()

        if temperature <= 0:
            next_id = int(torch.argmax(logits, dim=-1).item())
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                threshold = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        ids.append(next_id)
        produced.append(next_id)
        yield tokenizer.decode(produced)

        if should_abort():
            raise Aborted()
        # Keep memory flat on a long generation. autoresearch is tuned for a
        # fixed-size training step, and a chat can run for thousands of tokens.
        if len(ids) > config.sequence_len:
            ids = ids[-config.sequence_len:]


def main():
    parser = argparse.ArgumentParser(description="autoresearch chat worker")
    parser.add_argument("-w", "--weights", required=True, help="path to a .pt state_dict")
    parser.add_argument("-d", "--dataset", default=None, help="dataset profile override")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        emit({"error": f"checkpoint not found: {weights}"})
        return 1

    try:
        runtime = detect_runtime()
        device = runtime.device
        log(f"device: {device}")
        started = time.time()
        model, tokenizer, config, _ = load_model(weights, args.dataset, device)
        log(f"loaded in {time.time() - started:.1f}s")
    except Exception as exc:
        emit({"error": str(exc)})
        return 1

    emit({"ready": True, "params": sum(p.numel() for p in model.parameters())})

    current_id = None
    # Generation is a blocking loop, so cancellation is checked through a flag
    # the main thread flips while a request is in flight. Reading it between
    # tokens is the finest granularity available without a second process.
    abort_flag = {"value": False}

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        request_id = request.get("id")
        if request.get("abort"):
            abort_flag["value"] = True
            continue
        abort_flag["value"] = False
        current_id = request_id

        prompt = str(request.get("prompt") or "")
        if not prompt:
            emit({"id": request_id, "error": "empty prompt"})
            continue

        try:
            for cumulative in generate(
                model, tokenizer, prompt,
                max_new_tokens=int(request.get("max_new_tokens") or 256),
                temperature=float(request.get("temperature", 0.8)),
                top_k=int(request.get("top_k", 50)),
                config=config, device=device,
                should_abort=lambda: abort_flag["value"],
            ):
                if request_id != current_id:
                    # A newer request arrived; this one is abandoned.
                    return
                emit({"id": request_id, "text": cumulative})
            emit({"id": request_id, "done": True})
        except Aborted:
            emit({"id": request_id, "done": True})
        except Exception as exc:
            log(f"generation failed: {exc}")
            emit({"id": request_id, "error": str(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


CHAT_SERVER_FILENAME = "chat_server.py"


def chat_server_path(checkout: "object") -> "object":
    """Where the worker lives inside a checkout."""
    from pathlib import Path

    return Path(checkout) / CHAT_SERVER_FILENAME  # type: ignore[arg-type]


def write_chat_server(checkout, source: str = CHAT_SERVER_SOURCE) -> bool:
    """Put the worker in the checkout, if it is not already there.

    Rewritten whenever its contents differ, so a studio update that fixes the
    worker reaches an existing checkout. Returns whether anything changed.
    """
    from pathlib import Path

    path = Path(checkout) / CHAT_SERVER_FILENAME
    try:
        if path.exists() and path.read_text(encoding="utf-8", errors="replace") == source:
            return False
        path.write_text(source, encoding="utf-8", newline="\n")
    except OSError:
        return False
    return True
