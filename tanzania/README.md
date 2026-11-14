# Tanzania (retrieval pipeline repair)

**Flag: `SIGMOID_A69EE28C5B2214A9AD99`**. It is derived by the task's `flag_utils.derive_flag` from `retrieved_ids.json`.

## What was broken and how it was fixed (`retrieval_pipeline.py`)
1. **Different models for documents and queries.** Their vectors lived in different spaces, so cosine similarity was meaningless.
   Both are now embedded with `intfloat/e5-small-v2`.
2. **180-character chunks.** A slice could match an exhibit's name without the title that says whether the notice is CURRENT
   or ARCHIVED, or which city it belongs to. Each document is now kept whole (title + text) as one searchable unit.
3. **Missing e5 prefixes.** e5 models expect `passage: ` on documents and `query: ` on questions. Both are now added.
4. **Title-aware rerank.** Archived background notices repeat the exhibit description almost word for word, so pure embedding
   similarity often preferred them. Every question names the notice type, the exhibit and the city, which are exactly the words
   in the right document's title. So a small bonus (weight 0.1) is added for the share of title words that the question contains.

## Run
Put the task files next to `retrieval_pipeline.py`: `flag_utils.py`, `corpus.jsonl` and `questions.json`.
```
pip install -r tanzania/requirements.txt
python tanzania/retrieval_pipeline.py    # -> retrieved_ids.json and the flag
```
