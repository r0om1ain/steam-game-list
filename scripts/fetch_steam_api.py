from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from tqdm import tqdm

from qdrant_utils import ROOT, load_env

PARQUET = ROOT / "data" / "process" / "games_clean.parquet"
OUT_JSONL = ROOT / "data" / "process" / "steam_api_enrichment.jsonl"

NEWS_URL = "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/"
PLAYERS_URL = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/"


def already_fetched(path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(int(json.loads(line)["app_id"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return done


def fetch_news(session: requests.Session, app_id: int, key: str | None) -> dict:
    params = {"appid": app_id, "count": 3, "maxlength": 180, "format": "json"}
    if key:
        params["key"] = key
    r = session.get(NEWS_URL, params=params, timeout=20)
    r.raise_for_status()
    items = r.json().get("appnews", {}).get("newsitems", [])
    if not items:
        return {
            "news_count": 0,
            "last_news_title": None,
            "last_news_date": None,
            "last_news_url": None,
        }
    last = items[0]
    ts = last.get("date")
    return {
        "news_count": len(items),
        "last_news_title": last.get("title"),
        "last_news_date": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        if ts
        else None,
        "last_news_url": last.get("url"),
    }


def fetch_players(session: requests.Session, app_id: int, key: str | None) -> int | None:
    params = {"appid": app_id, "format": "json"}
    if key:
        params["key"] = key
    r = session.get(PLAYERS_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json().get("response", {}).get("player_count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--sleep", type=float, default=0.35)
    args = parser.parse_args()

    load_env()
    key = os.getenv("STEAM_API_KEY") or None
    if not key:
        print("warning: STEAM_API_KEY vide, on tente quand même sans clé")

    if not PARQUET.exists():
        raise SystemExit("lance d'abord scripts/clean_games.py")

    games = pd.read_parquet(PARQUET, columns=["app_id", "title", "overall_review_count"])
    games = games.sort_values("overall_review_count", ascending=False)
    targets = games.head(args.limit)

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    done = already_fetched(OUT_JSONL)
    print(f"déjà récupérés : {len(done)} | cible : {len(targets)}")

    session = requests.Session()
    session.headers.update({"User-Agent": "steam-game-list"})

    n_ok = 0
    n_err = 0
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for _, row in tqdm(targets.iterrows(), total=len(targets)):
            app_id = int(row["app_id"])
            if app_id in done:
                continue
            ingested_at = datetime.now(timezone.utc).isoformat()
            record = {
                "app_id": app_id,
                "title": row["title"],
                "current_players": None,
                "news_count": 0,
                "last_news_title": None,
                "last_news_date": None,
                "last_news_url": None,
                "ingested_at": ingested_at,
                "ok": False,
                "error": None,
            }
            try:
                record.update(fetch_news(session, app_id, key))
                time.sleep(0.15)
                record["current_players"] = fetch_players(session, app_id, key)
                record["ok"] = True
                n_ok += 1
            except requests.HTTPError as exc:
                record["error"] = f"HTTP {exc.response.status_code}"
                n_err += 1
                if exc.response is not None and exc.response.status_code in (401, 403):
                    print("\nclé API refusée / manquante -> remplis STEAM_API_KEY dans .env")
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    break
                if exc.response is not None and exc.response.status_code == 429:
                    print("\nrate limit, on attend 20s")
                    time.sleep(20)
            except requests.RequestException as exc:
                record["error"] = str(exc)
                n_err += 1

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            done.add(app_id)
            time.sleep(args.sleep)

    print(f"fini ok={n_ok} erreurs={n_err} fichier={OUT_JSONL}")


if __name__ == "__main__":
    main()
