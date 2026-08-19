# steam-game-list

Recherche sémantique sur le catalogue Steam. Le sujet parlait de MongoDB ou Neo4j, on est partis sur Qdrant : l’idée c’est de taper une phrase du style *« a relaxing card game with puzzles »* et de récupérer des jeux proches, pas de faire du CRUD document juste pour la forme.

Dataset principal : dump Kaggle (~126k jeux) dans `data/raw/games.csv` (gitignoré, trop gros).  
2e source : Steam Web API (news + joueurs en ligne) pour une partie du catalogue, avec un champ `ingested_at`.

Le notebook d’exploration est dans `notebooks/exploration.ipynb` (qualité des données, cleaning, graphes, petit POC Qdrant). Le reste est dans `scripts/` et `app/`.

## Lancer le projet

Python 3.11+ et Docker. Le CSV doit être dans `data/raw/games.csv`.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Dans `.env`, remplir `STEAM_API_KEY` (https://steamcommunity.com/dev/apikey). Les endpoints qu’on utilise marchent parfois sans, mais autant la mettre.

```bash
docker compose up -d
python scripts/clean_games.py
python scripts/fetch_steam_api.py --limit 150
python scripts/ingest_qdrant.py --limit 200
streamlit run app/streamlit_app.py
```

`--limit 200` c’est pour tester. Le catalogue clean fait ~117k jeux, le full ingest (`python scripts/ingest_qdrant.py --recreate`) prend un moment en CPU à cause de MiniLM.

L’app a deux onglets : recherche (Qdrant + filtres prix / OS / année / genres) et un peu de viz sur le parquet.

MiniLM est en anglais, les requêtes FR marchent moins bien. Les prix du dump sont en USD.

## Autres scripts

CRUD (id = app_id Steam) :

```bash
python scripts/crud_games.py get --id 730
python scripts/crud_games.py insert --id 999001 --name "test" --about "un petit roguelike" --price 4.99
python scripts/crud_games.py update --id 999001 --price 0
python scripts/crud_games.py replace --id 999001 --name "test v2" --about "un roguelike plus sombre"
python scripts/crud_games.py delete --id 999001
```

`update` ne touche que le payload. `replace` recalcule l’embedding.

Snapshots Qdrant :

```bash
python scripts/admin_qdrant.py info
python scripts/admin_qdrant.py snapshot
python scripts/admin_qdrant.py list-snapshots
python scripts/admin_qdrant.py restore --name <fichier.snapshot>
```

En local `QDRANT_API_KEY` peut rester vide. Si Qdrant n’est plus sur localhost, il faut en mettre une.

## Liens

- Support de présentation : _à coller_
