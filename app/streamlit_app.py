from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from qdrant_utils import collection_name, get_client

PARQUET = ROOT / "data" / "process" / "games_clean.parquet"
API_JSONL = ROOT / "data" / "process" / "steam_api_enrichment.jsonl"
MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(
    page_title="Steam semantic search",
    page_icon="🎮",
    layout="wide",
)


@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def load_qdrant():
    return get_client()


@st.cache_data
def load_parquet() -> pd.DataFrame | None:
    if not PARQUET.exists():
        return None
    cols = [
        "genres",
        "price",
        "is_free",
        "release_year",
        "win_support",
        "mac_support",
        "linux_support",
    ]
    return pd.read_parquet(PARQUET, columns=cols)


@st.cache_data
def load_api() -> pd.DataFrame | None:
    if not API_JSONL.exists():
        return None
    api = pd.read_json(API_JSONL, lines=True)
    if "ok" in api.columns:
        api = api[api["ok"].eq(True)]
    return api


def collection_count(client) -> int | None:
    try:
        coll = collection_name()
        if not client.collection_exists(coll):
            return 0
        return client.count(coll).count
    except Exception:
        return None


def build_filter(
    max_price: float,
    free_only: bool,
    linux: bool,
    mac: bool,
    min_reviews: int,
    year_min: int,
    year_max: int,
    genres: list[str],
) -> Filter:
    must = []
    if free_only:
        must.append(FieldCondition(key="is_free", match=MatchValue(value=True)))
    else:
        must.append(FieldCondition(key="price", range=Range(lte=float(max_price))))
    if linux:
        must.append(FieldCondition(key="linux_support", match=MatchValue(value=True)))
    if mac:
        must.append(FieldCondition(key="mac_support", match=MatchValue(value=True)))
    if min_reviews > 0:
        must.append(
            FieldCondition(
                key="overall_review_count",
                range=Range(gte=float(min_reviews)),
            )
        )
    must.append(
        FieldCondition(
            key="release_year",
            range=Range(gte=float(year_min), lte=float(year_max)),
        )
    )
    if genres:
        must.append(FieldCondition(key="genres_list", match=MatchAny(any=genres)))
    return Filter(must=must)


def os_label(payload: dict) -> str:
    bits = []
    if payload.get("win_support"):
        bits.append("Win")
    if payload.get("mac_support"):
        bits.append("Mac")
    if payload.get("linux_support"):
        bits.append("Linux")
    return " · ".join(bits) if bits else "?"


def format_price(payload: dict) -> str:
    if payload.get("is_free") or payload.get("price") in (0, 0.0):
        return "Gratuit"
    price = payload.get("price")
    if price is None:
        return "?"
    return f"{price:.2f} $"


def search_tab():
    st.subheader("Recherche sémantique")
    st.caption(
        "Tu décris le jeu, MiniLM encode la phrase, Qdrant renvoie les plus proches. "
        "Les filtres passent par les payload indexes (prix, OS, année, avis)."
    )

    try:
        client = load_qdrant()
        n_points = collection_count(client)
    except Exception as exc:
        st.error(f"Qdrant injoignable ({exc}). Lance `docker compose up -d` puis l'ingest.")
        return

    if not n_points:
        st.warning(
            "Collection vide. Ingest d'abord : `python scripts/ingest_qdrant.py --limit 200`"
        )
        return

    st.sidebar.markdown(f"**Qdrant** · {n_points} jeux indexés")

    query = st.text_input(
        "Décris le jeu que tu cherches",
        placeholder="a relaxing card game with puzzles",
    )
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        k = st.slider("Nombre de résultats", 5, 30, 10)
    with col_b:
        max_price = st.slider("Prix max (USD)", 0.0, 70.0, 70.0, 0.5)
    with col_c:
        min_reviews = st.slider("Avis min.", 0, 5000, 0, 50)

    c1, c2, c3 = st.columns(3)
    with c1:
        free_only = st.checkbox("Gratuits uniquement")
    with c2:
        linux = st.checkbox("Linux")
    with c3:
        mac = st.checkbox("Mac")

    year_min, year_max = st.slider("Année de sortie", 2005, 2026, (2008, 2026))

    genre_opts = []
    catalog = load_parquet()
    if catalog is not None:
        genre_opts = (
            catalog["genres"]
            .fillna("")
            .str.split(",")
            .explode()
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(20)
            .index.tolist()
        )
    picked_genres = st.multiselect("Genres (au moins un)", genre_opts)

    go = st.button("Rechercher", type="primary", disabled=not bool(query.strip()))
    if not go:
        st.info("MiniLM est anglais : une requête en anglais marche généralement mieux.")
        return

    qfilter = build_filter(
        max_price,
        free_only,
        linux,
        mac,
        min_reviews,
        year_min,
        year_max,
        picked_genres,
    )

    with st.spinner("encodage + recherche…"):
        vector = load_model().encode(query, normalize_embeddings=True).tolist()
        results = client.query_points(
            collection_name=collection_name(),
            query=vector,
            query_filter=qfilter,
            limit=k,
            with_payload=True,
        )

    points = results.points
    if not points:
        st.warning("Rien trouvé avec ces filtres. Élargis le prix / l'année.")
        return

    st.write(f"{len(points)} résultat(s)")
    for point in points:
        p = point.payload or {}
        app_id = p.get("app_id")
        store = f"https://store.steampowered.com/app/{app_id}" if app_id else None
        with st.container(border=True):
            left, right = st.columns([4, 1])
            with left:
                title = p.get("name", "(sans titre)")
                if store:
                    st.markdown(f"### [{title}]({store})")
                else:
                    st.markdown(f"### {title}")
                meta = [
                    format_price(p),
                    os_label(p),
                    p.get("genres") or "",
                ]
                if p.get("overall_review_pct") is not None:
                    meta.append(
                        f"{p['overall_review_pct']:.0f}% pos. ({p.get('overall_review_count', 0)} avis)"
                    )
                if p.get("current_players") is not None:
                    meta.append(f"{int(p['current_players']):,} joueurs".replace(",", " "))
                st.caption(" · ".join(str(m) for m in meta if m))
                if p.get("about"):
                    st.write(p["about"][:420] + ("…" if len(p["about"]) > 420 else ""))
                if p.get("last_news_title"):
                    st.caption(f"Dernière news : {p['last_news_title']}")
            with right:
                st.metric("score", f"{point.score:.3f}")
                if p.get("release_year"):
                    st.caption(str(p["release_year"]))


def stats_tab():
    st.subheader("Le catalogue")
    df = load_parquet()
    if df is None:
        st.warning("Pas de parquet. Lance `python scripts/clean_games.py`.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jeux (clean)", f"{len(df):,}".replace(",", " "))
    c2.metric("Gratuits", f"{df['is_free'].mean() * 100:.1f} %")
    paid = df.loc[~df["is_free"], "price"]
    c3.metric("Prix médian", f"{paid.median():.2f} $" if len(paid) else "—")
    c4.metric("Linux", f"{df['linux_support'].mean() * 100:.1f} %")

    genres = (
        df["genres"]
        .fillna("")
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", pd.NA)
        .dropna()
    )
    top = genres.value_counts().head(15).reset_index()
    top.columns = ["genre", "n"]
    fig1 = px.bar(top, x="n", y="genre", orientation="h", title="Top 15 des genres")
    fig1.update_layout(yaxis={"categoryorder": "total ascending"}, height=420)
    st.plotly_chart(fig1, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.histogram(
            df.loc[~df["is_free"], "price"].clip(upper=60),
            nbins=40,
            title="Prix payants (USD, borné à 60)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        os_share = (
            df[["win_support", "mac_support", "linux_support"]]
            .mean()
            .mul(100)
            .rename(
                {
                    "win_support": "Windows",
                    "mac_support": "Mac",
                    "linux_support": "Linux",
                }
            )
            .reset_index()
        )
        os_share.columns = ["os", "pct"]
        fig3 = px.bar(os_share, x="os", y="pct", title="Part des jeux par OS (%)")
        fig3.update_yaxes(range=[0, 100])
        st.plotly_chart(fig3, use_container_width=True)

    by_year = df["release_year"].dropna().astype(int).value_counts().sort_index()
    by_year = by_year[(by_year.index >= 2005) & (by_year.index <= 2026)]
    fig4 = px.line(
        x=by_year.index,
        y=by_year.values,
        markers=True,
        title="Sorties par année",
        labels={"x": "année", "y": "jeux"},
    )
    st.plotly_chart(fig4, use_container_width=True)

    api = load_api()
    st.markdown("#### Steam API (2e source)")
    if api is None or api.empty:
        st.caption("Pas encore d'enrichissement. `python scripts/fetch_steam_api.py --limit 150`")
        return
    show = api.sort_values("current_players", ascending=False).head(15)
    cols = [
        c
        for c in ["title", "current_players", "last_news_title", "last_news_date", "ingested_at"]
        if c in show.columns
    ]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)
    st.caption(f"{len(api)} jeux enrichis · champ ingested_at = date de l'appel API")


st.title("Steam — recherche sémantique")

tab_search, tab_stats = st.tabs(["Recherche", "Catalogue"])
with tab_search:
    search_tab()
with tab_stats:
    stats_tab()
