from __future__ import annotations

import argparse
from datetime import datetime, timezone

import pandas as pd
from qdrant_client.models import Distance, HnswConfigDiff, PayloadSchemaType, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from qdrant_utils import ROOT, collection_name, get_client, load_env

PARQUET = ROOT / "data" / "process" / "games_clean.parquet"
API_JSONL = ROOT / "data" / "process" / "steam_api_enrichment.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
BATCH = 128


def ensure_collection(client, name: str, recreate: bool):
    exists = client.collection_exists(name)
    if exists and recreate:
        print(f"suppression de {name}")
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )
        print(f"collection créée : {name}")
    else:
        print(f"collection déjà là : {name}")


def create_indexes(client, name: str):
    indexes = {
        "genres_list": PayloadSchemaType.KEYWORD,
        "is_free": PayloadSchemaType.BOOL,
        "price": PayloadSchemaType.FLOAT,
        "release_year": PayloadSchemaType.INTEGER,
        "win_support": PayloadSchemaType.BOOL,
        "mac_support": PayloadSchemaType.BOOL,
        "linux_support": PayloadSchemaType.BOOL,
        "overall_review_pct": PayloadSchemaType.FLOAT,
        "overall_review_count": PayloadSchemaType.INTEGER,
    }
    for field, schema in indexes.items():
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=schema,
            )
        except Exception as exc:
            print(f"index {field}: {exc}")
    print("indexes payload ok")


def to_payload(row: pd.Series, ingested_at: str) -> dict:
    about = row["about_description"]
    if isinstance(about, str) and len(about) > 600:
        about = about[:600] + "…"

    payload = {
        "app_id": int(row["app_id"]),
        "name": row["title"],
        "about": about,
        "genres": row["genres"],
        "genres_list": list(row["genres_list"]) if row["genres_list"] is not None else [],
        "tags": row["tags"],
        "categories": row["categories"],
        "developer": row["developer"],
        "publisher": row["publisher"],
        "price": None if pd.isna(row["price"]) else float(row["price"]),
        "is_free": bool(row["is_free"]),
        "discount_pct": float(row["discount_pct"]) if pd.notna(row["discount_pct"]) else 0.0,
        "release_date": row["release_date"] if pd.notna(row["release_date"]) else None,
        "release_year": None if pd.isna(row["release_year"]) else int(row["release_year"]),
        "win_support": bool(row["win_support"]),
        "mac_support": bool(row["mac_support"]),
        "linux_support": bool(row["linux_support"]),
        "overall_review_pct": None
        if pd.isna(row["overall_review_pct"])
        else float(row["overall_review_pct"]),
        "overall_review_count": int(row["overall_review_count"]),
        "ingested_at": ingested_at,
    }

    if "current_players" in row.index and pd.notna(row["current_players"]):
        payload["current_players"] = int(row["current_players"])
    if "last_news_title" in row.index and pd.notna(row.get("last_news_title")):
        payload["last_news_title"] = row["last_news_title"]
        payload["last_news_date"] = row.get("last_news_date")
    if "api_ingested_at" in row.index and pd.notna(row.get("api_ingested_at")):
        payload["api_ingested_at"] = row["api_ingested_at"]

    return payload


def load_games(limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET)
    if API_JSONL.exists():
        api = pd.read_json(API_JSONL, lines=True)
        if "ok" in api.columns:
            api = api[api["ok"].eq(True)]
        keep = [
            c
            for c in [
                "app_id",
                "current_players",
                "news_count",
                "last_news_title",
                "last_news_date",
                "last_news_url",
                "ingested_at",
            ]
            if c in api.columns
        ]
        api = api[keep].drop_duplicates("app_id")
        if "ingested_at" in api.columns:
            api = api.rename(columns={"ingested_at": "api_ingested_at"})
        df = df.merge(api, on="app_id", how="left")
        print(f"enrichissement API : {api.shape[0]} jeux")
    else:
        print("pas de fichier Steam API, ingest CSV only")

    if limit:
        df = df.head(limit)
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--batch", type=int, default=BATCH)
    args = parser.parse_args()

    load_env()
    collection = collection_name()

    if not PARQUET.exists():
        raise SystemExit("lance d'abord scripts/clean_games.py")

    df = load_games(args.limit)
    print(f"à ingérer : {len(df)} jeux")

    client = get_client()
    ensure_collection(client, collection, args.recreate)
    create_indexes(client, collection)

    print(f"chargement modèle {MODEL_NAME} …")
    model = SentenceTransformer(MODEL_NAME)
    ingested_at = datetime.now(timezone.utc).isoformat()

    n = len(df)
    for start in tqdm(range(0, n, args.batch), desc="upsert"):
        chunk = df.iloc[start : start + args.batch]
        vectors = model.encode(
            chunk["embedding_text"].tolist(),
            batch_size=min(64, args.batch),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        points = []
        for i, (_, row) in enumerate(chunk.iterrows()):
            points.append(
                PointStruct(
                    id=int(row["app_id"]),
                    vector=vectors[i].tolist(),
                    payload=to_payload(row, ingested_at),
                )
            )
        client.upsert(collection_name=collection, points=points)

    print("count =", client.count(collection).count)


if __name__ == "__main__":
    main()
