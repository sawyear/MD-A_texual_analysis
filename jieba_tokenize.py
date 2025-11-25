"""
Fast tokenizer using jieba instead of Stanford CoreNLP.

Input:
    data/input/documents.txt        # one document per line
    data/input/document_ids.txt     # optional, keeps alignment

Output:
    data/processed/parsed/documents.txt          # tokenized, space-delimited
    data/processed/parsed/document_sent_ids.txt  # same IDs (one line per doc)

This is a lightweight drop-in for the parsing stage. After running it, you can
continue with clean_and_train.py from the "train and apply a phrase model" step.
"""

import itertools
from pathlib import Path

import jieba

import global_options


def tokenize_line(line: str) -> str:
    """Segment a single line with jieba and return space-joined tokens."""
    # strip to remove trailing newline but keep internal spaces
    tokens = jieba.lcut(line.strip())
    return " ".join(tokens)


def process_largefile(
    input_file: Path,
    output_file: Path,
    input_file_ids: list[str] | None = None,
    output_index_file: Path | None = None,
    chunk_size: int = 10000,
):
    """Stream through the input file, tokenize with jieba, write outputs."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_index_file:
        output_index_file.parent.mkdir(parents=True, exist_ok=True)
    # remove existing outputs to avoid accidental append
    for p in [output_file, output_index_file]:
        if p and p.exists():
            p.unlink()

    with input_file.open(encoding="utf-8", errors="ignore") as f_in:
        idx_iter = iter(input_file_ids) if input_file_ids is not None else None
        while True:
            lines = list(itertools.islice(f_in, chunk_size))
            if not lines:
                break
            if idx_iter:
                ids = list(itertools.islice(idx_iter, len(lines)))
            else:
                ids = []
            tokenized = [tokenize_line(l) for l in lines]
            with output_file.open("a", encoding="utf-8") as f_out:
                f_out.write("\n".join(tokenized) + "\n")
            if output_index_file and ids:
                with output_index_file.open("a", encoding="utf-8") as f_out:
                    f_out.write("\n".join(ids) + "\n")


if __name__ == "__main__":
    in_file = Path(global_options.DATA_FOLDER, "input", "documents.txt")
    in_file_index = Path(global_options.DATA_FOLDER, "input", "document_ids.txt")
    out_file = Path(global_options.DATA_FOLDER, "processed", "parsed", "documents.txt")
    output_index_file = Path(
        global_options.DATA_FOLDER, "processed", "parsed", "document_sent_ids.txt"
    )

    ids = None
    if in_file_index.exists():
        ids = [
            line.strip()
            for line in in_file_index.read_text(encoding="utf-8", errors="ignore").splitlines()
        ]
    process_largefile(
        input_file=in_file,
        output_file=out_file,
        input_file_ids=ids,
        output_index_file=output_index_file if ids is not None else None,
    )

