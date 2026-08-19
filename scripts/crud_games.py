from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

from qdrant_utils import collection_name, get_client

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def model():
    global _model
    if _model is None:
        print(f"chargement {MODEL_NAME} …")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embedding_text(name: str, about: str, genres: str) -> str:
    parts = [name, about]
    if genres:
        parts.append("Genres: " + genres)
    return ". ".join(p for p in parts if p)


def encode(text: str) -> list[float]:
    return model().encode(text, normalize_embeddings=True).tolist()


def retrieve_one(client, coll: str, app_id: int):
    points = client.retrieve(collection_name=coll, ids=[app_id], with_payload=True)
    return points[0] if points else None


def cmd_get(args):
    client = get_client()
    coll = collection_name()
    point = retrieve_one(client, coll, args.id)
    if point is None:
        raise SystemExit(f"pas trouvé : {args.id}")
    print(json.dumps(point.payload, ensure_ascii=False, indent=2))


def cmd_insert(args):
    client = get_client()
    coll = collection_name()
    if retrieve_one(client, coll, args.id) is not None:
        raise SystemExit(f"{args.id} existe déjà, utilise replace ou update")

    genres_list = [g.strip() for g in args.genres.split(",") if g.strip()]
    vector = encode(embedding_text(args.name, args.about, args.genres))
    payload = {
        "app_id": args.id,
        "name": args.name,
        "about": args.about[:600],
        "genres": args.genres,
        "genres_list": genres_list,
        "tags": "",
        "categories": "",
        "developer": args.developer,
        "publisher": args.publisher,
        "price": args.price,
        "is_free": args.price == 0,
        "discount_pct": 0.0,
        "release_year": args.year,
        "win_support": True,
        "mac_support": False,
        "linux_support": False,
        "overall_review_pct": None,
        "overall_review_count": 0,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(
        collection_name=coll,
        points=[PointStruct(id=args.id, vector=vector, payload=payload)],
    )
    print("insert ok", args.id)


def cmd_update(args):
    client = get_client()
    coll = collection_name()
    if retrieve_one(client, coll, args.id) is None:
        raise SystemExit(f"pas trouvé : {args.id}")

    payload = {}
    if args.name is not None:
        payload["name"] = args.name
    if args.about is not None:
        payload["about"] = args.about[:600]
    if args.genres is not None:
        payload["genres"] = args.genres
        payload["genres_list"] = [g.strip() for g in args.genres.split(",") if g.strip()]
    if args.price is not None:
        payload["price"] = args.price
        payload["is_free"] = args.price == 0
    if args.year is not None:
        payload["release_year"] = args.year

    if not payload:
        raise SystemExit("rien à update, passe --name / --about / --genres / --price / --year")

    client.set_payload(collection_name=coll, payload=payload, points=[args.id])
    print("update ok", args.id, "champs:", ", ".join(payload))


def cmd_replace(args):
    client = get_client()
    coll = collection_name()
    old = retrieve_one(client, coll, args.id)
    if old is None:
        raise SystemExit(f"pas trouvé : {args.id} (pour créer un jeu : insert)")

    payload = dict(old.payload or {})
    payload["name"] = args.name
    payload["about"] = args.about[:600]
    payload["genres"] = args.genres
    payload["genres_list"] = [g.strip() for g in args.genres.split(",") if g.strip()]
    if args.price is not None:
        payload["price"] = args.price
        payload["is_free"] = args.price == 0
    payload["ingested_at"] = datetime.now(timezone.utc).isoformat()

    vector = encode(embedding_text(args.name, args.about, args.genres))
    client.upsert(
        collection_name=coll,
        points=[PointStruct(id=args.id, vector=vector, payload=payload)],
    )
    print("replace ok", args.id)


def cmd_delete(args):
    client = get_client()
    coll = collection_name()
    if retrieve_one(client, coll, args.id) is None:
        raise SystemExit(f"pas trouvé : {args.id}")
    client.delete(collection_name=coll, points_selector=[args.id])
    print("delete ok", args.id)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_get = sub.add_parser("get")
    p_get.add_argument("--id", type=int, required=True)
    p_get.set_defaults(func=cmd_get)

    p_ins = sub.add_parser("insert")
    p_ins.add_argument("--id", type=int, required=True)
    p_ins.add_argument("--name", required=True)
    p_ins.add_argument("--about", required=True)
    p_ins.add_argument("--genres", default="Indie")
    p_ins.add_argument("--price", type=float, default=0.0)
    p_ins.add_argument("--year", type=int, default=None)
    p_ins.add_argument("--developer", default="")
    p_ins.add_argument("--publisher", default="")
    p_ins.set_defaults(func=cmd_insert)

    p_up = sub.add_parser("update")
    p_up.add_argument("--id", type=int, required=True)
    p_up.add_argument("--name", default=None)
    p_up.add_argument("--about", default=None)
    p_up.add_argument("--genres", default=None)
    p_up.add_argument("--price", type=float, default=None)
    p_up.add_argument("--year", type=int, default=None)
    p_up.set_defaults(func=cmd_update)

    p_rep = sub.add_parser("replace")
    p_rep.add_argument("--id", type=int, required=True)
    p_rep.add_argument("--name", required=True)
    p_rep.add_argument("--about", required=True)
    p_rep.add_argument("--genres", default="Indie")
    p_rep.add_argument("--price", type=float, default=None)
    p_rep.set_defaults(func=cmd_replace)

    p_del = sub.add_parser("delete")
    p_del.add_argument("--id", type=int, required=True)
    p_del.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
