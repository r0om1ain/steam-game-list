from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "games.csv"
OUT_PARQUET = ROOT / "data" / "process" / "games_clean.parquet"


def load_raw(path: Path) -> pd.DataFrame:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if "DiscountDLC count" in columns:
        pos = columns.index("DiscountDLC count")
        columns[pos : pos + 1] = ["Discount", "DLC count"]
        return pd.read_csv(path, skiprows=1, names=columns, low_memory=False)
    return pd.read_csv(path, low_memory=False)


def split_list(value):
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates(subset=["AppID"])
    out = out.dropna(subset=["AppID", "Name"])

    rename = {
        "AppID": "app_id",
        "Name": "title",
        "About the game": "about_description",
        "Genres": "genres",
        "Tags": "tags",
        "Categories": "categories",
        "Developers": "developer",
        "Publishers": "publisher",
        "Price": "price",
        "Discount": "discount_pct",
        "DLC count": "dlc_count",
        "Release date": "release_date",
        "Windows": "win_support",
        "Mac": "mac_support",
        "Linux": "linux_support",
        "Positive": "positive",
        "Negative": "negative",
        "Recommendations": "recommendations",
        "Metacritic score": "metacritic_score",
    }
    out = out.rename(columns=rename)

    text_cols = [
        "about_description",
        "genres",
        "tags",
        "categories",
        "developer",
        "publisher",
    ]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)

    for col in ["genres", "tags", "categories"]:
        out[col] = out[col].str.replace(",", ", ", regex=False)

    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out["discount_pct"] = pd.to_numeric(out.get("discount_pct"), errors="coerce").fillna(0)
    out["is_free"] = out["price"].fillna(-1) == 0

    out["release_dt"] = pd.to_datetime(out["release_date"], errors="coerce")
    out["release_year"] = out["release_dt"].dt.year

    out["positive"] = pd.to_numeric(out["positive"], errors="coerce").fillna(0).astype(int)
    out["negative"] = pd.to_numeric(out["negative"], errors="coerce").fillna(0).astype(int)
    out["overall_review_count"] = out["positive"] + out["negative"]
    out["overall_review_pct"] = np.where(
        out["overall_review_count"] > 0,
        out["positive"] / out["overall_review_count"] * 100,
        np.nan,
    )

    out["genres_list"] = out["genres"].map(split_list)
    out["tags_list"] = out["tags"].map(split_list)
    out["categories_list"] = out["categories"].map(split_list)

    is_playtest = out["title"].str.contains("Playtest", case=False, na=False)
    no_about = out["about_description"].str.strip() == ""
    out = out[~is_playtest & ~no_about].copy()

    out["embedding_text"] = (
        out["title"]
        + ". "
        + out["about_description"]
        + ". Genres: "
        + out["genres"]
        + ". Tags: "
        + out["tags"]
        + ". Categories: "
        + out["categories"]
    )

    keep = [
        "app_id",
        "title",
        "about_description",
        "genres",
        "genres_list",
        "tags",
        "tags_list",
        "categories",
        "categories_list",
        "developer",
        "publisher",
        "price",
        "discount_pct",
        "is_free",
        "dlc_count",
        "release_date",
        "release_year",
        "win_support",
        "mac_support",
        "linux_support",
        "positive",
        "negative",
        "overall_review_pct",
        "overall_review_count",
        "recommendations",
        "metacritic_score",
        "embedding_text",
    ]
    return out[keep].reset_index(drop=True)


def main():
    if not RAW_CSV.exists():
        raise SystemExit(f"CSV introuvable : {RAW_CSV}")

    print(f"lecture {RAW_CSV}")
    raw = load_raw(RAW_CSV)
    print(f"brut : {raw.shape}")

    cleaned = clean(raw)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(OUT_PARQUET, index=False)
    print(f"clean : {cleaned.shape} -> {OUT_PARQUET}")
    print("jeux gratuits :", round(cleaned["is_free"].mean() * 100, 1), "%")
    paid = cleaned.loc[~cleaned["is_free"], "price"]
    print("prix median USD :", paid.median())


if __name__ == "__main__":
    main()
