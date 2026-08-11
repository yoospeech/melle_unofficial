"""SentencePiece character tokenizer used by MELLE."""

from __future__ import annotations

import json
import os

import sentencepiece as spm


class MelleCharacterTokenizer:
    def __init__(
        self,
        model_path: str = "melle_character_tokenizer.model",
        vocab_size: int = 4000,
    ):
        self.model_path = model_path
        self.requested_vocab_size = vocab_size
        self.processor = spm.SentencePieceProcessor()

    def load_or_train(self, manifest_path: str) -> None:
        if not os.path.exists(self.model_path):
            with open(manifest_path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            texts = (str(item["text"]) for item in records if item.get("text"))
            model_prefix, extension = os.path.splitext(self.model_path)
            if extension != ".model":
                model_prefix = self.model_path
                self.model_path += ".model"
            spm.SentencePieceTrainer.train(
                sentence_iterator=texts,
                model_prefix=model_prefix,
                # SentencePiece keeps whitespace and Unicode normalization
                # consistent between training and inference while emitting
                # character-level pieces rather than learned BPE subwords.
                model_type="char",
                vocab_size=self.requested_vocab_size,
                character_coverage=1.0,
                pad_id=0,
                bos_id=1,
                eos_id=2,
                unk_id=3,
                hard_vocab_limit=False,
            )
        if not self.processor.load(self.model_path):
            raise ValueError(f"failed to load SentencePiece model: {self.model_path}")

    def encode(self, text: str):
        return self.processor.encode(text, out_type=int)

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    @property
    def bos_token_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_token_id(self) -> int:
        return self.processor.eos_id()


# Compatibility for external imports. New MELLE code uses the accurately
# named character-tokenizer class above.
MelleBPETokenizer = MelleCharacterTokenizer
