from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_identity(tokenizer) -> dict[str, object]:
    digest = hashlib.sha256()
    vocab_items = sorted(
        ((int(token_id), str(token)) for token, token_id in tokenizer.get_vocab().items()),
        key=lambda item: (item[0], item[1]),
    )
    for token_id, token in vocab_items:
        digest.update(token_id.to_bytes(8, byteorder="little", signed=False))
        digest.update(token.encode("utf-8", errors="surrogatepass"))
        digest.update(b"\0")
    return {
        "tokenizer_name_or_path": str(tokenizer.name_or_path),
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "tokenizer_vocab_sha256": digest.hexdigest(),
    }


def encode_prompt(tokenizer, prompt: str, *, max_length: int) -> list[int]:
    """Use one prompt-tokenization contract for generation, training, and PPL."""
    token_ids = tokenizer.encode(
        str(prompt),
        add_special_tokens=True,
        truncation=True,
        max_length=int(max_length),
    )
    if token_ids:
        return [int(token_id) for token_id in token_ids]
    fallback = tokenizer.eos_token_id
    if fallback is None:
        raise ValueError("Tokenizer produced an empty prompt and has no EOS fallback.")
    return [int(fallback)]


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def text_sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def files_sha256(root: str | Path, names: Iterable[str]) -> dict[str, str]:
    base = Path(root)
    return {str(name): sha256_file(base / name) for name in names}


def canonical_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def validate_tokenizer_identity(
    metadata: Mapping[str, object], tokenizer, *, context: str
) -> None:
    expected = tokenizer_identity(tokenizer)
    for field in ("tokenizer_vocab_size", "tokenizer_vocab_sha256"):
        actual_value = metadata.get(field)
        if actual_value is None:
            raise ValueError(f"{context} is missing {field} metadata.")
        if str(actual_value) != str(expected[field]):
            raise ValueError(
                f"{context} {field}={actual_value!r} does not match the loaded "
                f"tokenizer value {expected[field]!r}."
            )
