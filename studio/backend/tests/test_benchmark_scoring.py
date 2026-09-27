# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The scoring pass's categorical path, against a tokenizer that behaves like a real one.

The letter-narrowing logic is the part most able to produce a plausible-looking
wrong answer, so it is checked against a stub whose logits are set by the test
rather than by a checkpoint. The stub maps every surface form to a distinct id, the
way a byte-level BPE does, because a stub that only knew the bare letter form
masks the failure where the space-prefixed form is tried against the wrong letter.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)
os.environ.setdefault("UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE", "1")
os.environ.setdefault("UNSLOTH_ALLOW_CPU", "1")

import pytest

torch = pytest.importorskip("torch")

from core.benchmarks import score as scorer  # noqa: E402

PROMPT = "Multiple Choice question: q\n- x=A\n\nRespond only with the letter of the correct answer."


class ByteLevelStubTokenizer:
    """Both surface forms of every letter, each with its own id, as a BPE has."""

    def __init__(self) -> None:
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.unk_token = "</s>"
        self.chat_template = "stub"
        self.bare = {letter: index + 1 for index, letter in enumerate("ABCD")}
        self.spaced = {f" {letter}": 10 + index for index, letter in enumerate("ABCD")}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]:
        if text in self.bare:
            return [self.bare[text]]
        if text in self.spaced:
            return [self.spaced[text]]
        # Anything else splits, so the "no single token" path stays reachable.
        return [99, 98]

    def apply_chat_template(
        self,
        messages,
        tokenize = False,
        add_generation_prompt = True,
    ):
        return "<user> " + messages[0]["content"] + " </user>"

    def __call__(self, prompts, **kwargs):
        width = max(len(p) for p in prompts)
        ids = torch.zeros((len(prompts), width), dtype = torch.long)
        mask = torch.zeros((len(prompts), width), dtype = torch.long)
        for row, prompt in enumerate(prompts):
            parts = prompt.split()
            letter = next(
                (p for p in parts if p in self.bare or p in self.spaced),
                "A",
            )
            # Encoded the same way the scorer resolves letters -- preferring the
            # spaced form, since a rendered prompt ends in prose. Encoding the bare
            # form here instead would put a token id on the input that the model
            # does not recognise as any letter, and every row would score as "A".
            first = self.spaced.get(f" {letter}") or self.bare.get(letter) or 0
            tokens = [first] + [7] * (len(parts) - 1)
            ids[row, : len(tokens)] = torch.tensor(tokens, dtype = torch.long)
            mask[row, : len(tokens)] = 1

        class Encoded(dict):
            """A mapping, because the scorer splats it into the model call."""

            def to(self, device):
                return self

        return Encoded(input_ids = ids, attention_mask = mask)

    def to(self, device):
        return self

    def decode(
        self,
        tokens,
        skip_special_tokens = True,
    ) -> str:
        return ""


class Processor:
    """Stands in for the multimodal processor a vision model returns.

    It decodes and applies a template but has no ``encode``, exactly like
    ``Qwen3VLProcessor``, which is how the scorer first met one.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = "stub"

    def decode(self, *args, **kwargs) -> str:
        return ""

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "<user> hi </user>"


class StubModel:
    """Makes the letter named in the prompt the argmax, by logit value alone."""

    def __init__(
        self,
        token_ids: dict,
        vocab: int = 64,
    ) -> None:
        self._ids = token_ids
        self.vocab = vocab
        self.calls = 0

    def eval(self):
        return self

    def parameters(self):
        yield torch.zeros(1)

    def __call__(
        self,
        input_ids,
        attention_mask = None,
        **kwargs,
    ):
        batch, seq = input_ids.shape
        logits = torch.zeros((batch, seq, self.vocab))
        reverse = {value: letter for letter, value in self._ids.items()}
        for row in range(batch):
            winner = reverse.get(int(input_ids[row, 0]), "A")
            # Ascending across the letters, so the winner is found by the argmax
            # and not by position.
            for index, candidate in enumerate("ABCD"):
                logits[row, -1, self._ids[candidate]] = float(index) + (
                    10.0 if candidate == winner else 0.0
                )
        self.calls += 1
        return type("Out", (), {"logits": logits})()


def items(winners: list[str]) -> list[dict]:
    return [
        {
            "index": index,
            "conversation": {
                "messages": [
                    {"role": "user", "content": f"q {letter}"},
                    {"role": "assistant", "content": "A"},
                ],
                "letters": ["A", "B", "C", "D"],
            },
        }
        for index, letter in enumerate(winners)
    ]


# --- letter resolution ---


def test_each_letter_resolves_to_its_own_token():
    # The one that matters most: if two letters share a token, every logit
    # comparison is the same number and the score is a coin flip dressed up as a
    # measurement. It fails silently, which is why it is asserted on the values.
    tokenizer = ByteLevelStubTokenizer()
    ids = scorer._letter_token_ids(tokenizer, ["A", "B", "C", "D"], PROMPT)
    assert ids is not None
    assert sorted(ids) == ["A", "B", "C", "D"]
    assert len(set(ids.values())) == 4, f"letters collapsed onto one token: {ids}"


def test_the_spaced_form_is_preferred_after_a_prose_prompt():
    # A prompt ending in a full stop is answered with a space before the letter,
    # so the spaced form is the token the model would actually have produced.
    tokenizer = ByteLevelStubTokenizer()
    ids = scorer._letter_token_ids(tokenizer, ["A"], PROMPT)
    assert ids == {"A": tokenizer.spaced[" A"]}


def test_the_bare_form_is_preferred_after_whitespace():
    # A leading space here would be absorbed into the previous token, so the bare
    # letter is the one with a clean boundary.
    tokenizer = ByteLevelStubTokenizer()
    ids = scorer._letter_token_ids(tokenizer, ["A"], PROMPT + " ")
    assert ids == {"A": tokenizer.bare["A"]}


def test_a_letter_with_no_single_token_form_refuses_the_exact_path():
    # No single logit to compare means no exact answer, and guessing which piece
    # of a split token matters would make the score depend on an accident of BPE.
    tokenizer = ByteLevelStubTokenizer()
    assert scorer._letter_token_ids(tokenizer, ["A", "B", "Z"], PROMPT) is None


def test_a_tokenizer_without_encode_refuses_rather_than_raising():
    # A multimodal processor reaches here if it was never unwrapped. Failing
    # closed is what turns that into a fallback instead of an exception.
    assert scorer._letter_token_ids(Processor(None), ["A", "B"], PROMPT) is None


# --- processor unwrapping ---


def test_a_multimodal_processor_is_unwrapped_to_its_tokenizer():
    # What FastLanguageModel returns for a vision model. The processor decodes and
    # templates but has no encode, so scoring against it directly would find every
    # letter unrepresentable and silently drop the benchmark to generation.
    tokenizer = ByteLevelStubTokenizer()
    assert scorer.resolve_text_tokenizer(Processor(tokenizer)) is tokenizer


def test_a_plain_tokenizer_is_passed_through():
    tokenizer = ByteLevelStubTokenizer()
    assert scorer.resolve_text_tokenizer(tokenizer) is tokenizer


# --- scoring ---


def test_the_letter_named_in_the_prompt_wins():
    tokenizer = ByteLevelStubTokenizer()
    ids = scorer._letter_token_ids(tokenizer, ["A", "B", "C", "D"], PROMPT)
    winners = ["C", "A", "D", "B", "A"]
    model = StubModel(ids)
    predictions, exact = scorer.score_categorical(
        model, tokenizer, "cpu", items(winners), batch_size = 2, max_seq_length = 512
    )
    assert exact is True
    assert [row["index"] for row in predictions] == [0, 1, 2, 3, 4]
    assert [row["prediction"] for row in predictions] == winners
    # The logit recorded for each letter must be that letter's own, which is the
    # other half of the collapse check above.
    assert [row["letter_scores"]["B"] for row in predictions] != [
        row["letter_scores"]["A"] for row in predictions
    ]
    # Five items at two per pass, so three forward passes.
    assert model.calls == 3


def test_a_stop_already_pending_scores_nothing():
    tokenizer = ByteLevelStubTokenizer()
    ids = scorer._letter_token_ids(tokenizer, ["A", "B", "C", "D"], PROMPT)
    monkey = scorer.stop_requested
    scorer.stop_requested = lambda: True
    try:
        model = StubModel(ids)
        predictions, _ = scorer.score_categorical(
            model,
            tokenizer,
            "cpu",
            items(["A", "B", "A", "B"]),
            batch_size = 2,
            max_seq_length = 64,
        )
    finally:
        scorer.stop_requested = monkey
    # The check is at the top of the loop, so a pending stop costs no forward pass
    # and returns whatever was already scored, which here is nothing.
    assert predictions == []
    assert model.calls == 0


# --- letter read out of generated text ---


def test_a_letter_is_read_back_from_generated_text():
    assert scorer._extract_letter("B", ["A", "B"]) == "B"
    assert scorer._extract_letter("Sure! The answer is C.", ["A", "B", "C", "D"]) == "C"


def test_no_letter_is_invented_when_the_model_gave_none():
    # Forcing one of the offered letters out of an answer that has none would turn
    # a wrong answer into a right one often enough to matter.
    assert scorer._extract_letter("E", ["A", "B"]) == ""
    assert scorer._extract_letter("I think it is 42", ["A", "B"]) == ""


# --- prompt rendering ---


def test_the_models_own_chat_template_is_used():
    tokenizer = ByteLevelStubTokenizer()
    assert scorer.render_prompt(tokenizer, "hello") == ("<user> hello </user>", True)


def test_a_tokenizer_without_a_template_says_so():
    # A base model scored through a chat format it was never trained on is a real
    # caveat, so the run reports that it happened rather than assuming a template.
    tokenizer = ByteLevelStubTokenizer()
    tokenizer.chat_template = None
    assert scorer.render_prompt(tokenizer, "hello") == ("hello", False)


def test_a_broken_template_falls_back_rather_than_losing_the_benchmark():
    class Broken(ByteLevelStubTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("bad template")

    assert scorer.render_prompt(Broken(), "hello") == ("hello", False)
