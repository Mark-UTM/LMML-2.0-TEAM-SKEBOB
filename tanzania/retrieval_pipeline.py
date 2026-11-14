from __future__ import annotations

import argparse
import re
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from flag_utils import derive_flag, load_verification_codes

DOCUMENT_MODEL = "intfloat/e5-small-v2"
QUERY_MODEL = DOCUMENT_MODEL
TITLE_WEIGHT = 0.1

@dataclass(frozen=True)
class Document:

    document_id: str
    title: str
    text: str

@dataclass(frozen=True)
class Chunk:

    document_id: str
    text: str

def load_corpus(corpus_path: Path) -> list[Document]:
    documents: list[Document] = []
    with corpus_path.open(encoding="utf-8") as corpus_file:
        for line in corpus_file:
            row = json.loads(line)
            documents.append(Document(row["document_id"], row["title"], row["text"]))
    return documents

def load_questions(questions_path: Path) -> list[dict[str, str]]:
    with questions_path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("questions.json must contain a list.")
    return questions

def split_into_chunks(documents: list[Document]) -> list[Chunk]:
    return [Chunk(d.document_id, f"{d.title}\n{d.text}") for d in documents]

def retrieve_documents(
    chunks: list[Chunk],
    questions: list[dict[str, str]],
) -> list[dict[str, str]]:
    from sentence_transformers import SentenceTransformer

    document_encoder = SentenceTransformer(DOCUMENT_MODEL)
    query_encoder = SentenceTransformer(QUERY_MODEL)
    chunk_vectors = document_encoder.encode(
        [f"passage: {chunk.text}" for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    query_vectors = query_encoder.encode(
        [f"query: {question['question']}" for question in questions],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    similarities = np.asarray(query_vectors) @ np.asarray(chunk_vectors).T

    def words(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+", text.lower()))

    titles = [words(chunk.text.split("\n", 1)[0]) for chunk in chunks]
    for qi, question in enumerate(questions):
        q_words = words(question["question"])
        bonus = np.array([len(t & q_words) / max(len(t), 1) for t in titles])
        similarities[qi] += TITLE_WEIGHT * bonus

    results: list[dict[str, str]] = []
    for question, row in zip(questions, similarities):
        best_chunk_index = int(np.argmax(row))
        results.append(
            {
                "question_id": question["question_id"],
                "document_id": chunks[best_chunk_index].document_id,
            }
        )
    return results

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("corpus.jsonl"))
    parser.add_argument("--questions", type=Path, default=Path("questions.json"))
    parser.add_argument("--output", type=Path, default=Path("retrieved_ids.json"))
    return parser.parse_args()

def main() -> None:
    arguments = parse_arguments()
    documents = load_corpus(arguments.corpus)
    questions = load_questions(arguments.questions)
    results = retrieve_documents(split_into_chunks(documents), questions)
    arguments.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    flag = derive_flag(results, questions, load_verification_codes(arguments.corpus))
    print(f"Saved {len(results)} results to {arguments.output}.")
    print(flag)

if __name__ == "__main__":
    main()
