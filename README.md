# steam-semantic-search

Semantic search engine over the Steam games catalog (~126k titles), using
sentence embeddings (`sentence-transformers`) and Qdrant as the vector store.

## Structure

- `notebooks/` — exploration and data cleaning notebooks
- `data/` — raw and processed datasets (gitignored)
- `qdrant_storage/` — local Qdrant persistence (gitignored)
- `app/`, `scripts/` — application code
