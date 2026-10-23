# Détection d'anomalies dans les capteurs industriels

Projet en binôme : **Karati Achraf** et **Ait Boutargante Fatima Zohra**.

**Problématique.** Quels comportements de capteurs s'écartent du fonctionnement habituel ? Quels
signaux précèdent une panne ou une dégradation ? Comment distinguer une anomalie réelle d'un bruit
de mesure ?

## Données

| Rôle | Jeu | Source | Emplacement attendu |
|---|---|---|---|
| Principal | Pump Sensor Data : 52 capteurs à la minute, 7 pannes | [Kaggle `nphantawee/pump-sensor-data`](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) | `data/raw/sensor.csv` |
| Contrôle | NASA C-MAPSS FD002 : 6 conditions opératoires | NASA Prognostics Data Repository (miroir Kaggle `behrad3d/nasa-cmaps`) | `data/raw/CMAPSS/{train,test,RUL}_FD002.txt` |

Les données ne sont **pas versionnées** (voir `.gitignore`).

## Périmètre

- **Noyau** :
  - nettoyage ;
  - segmentation des régimes ;
  - MAD/CUSUM par régime ;
  - PCA avec T² de Hotelling et SPE ;
  - Isolation Forest et LOF ;
  - couche de filtrage anomalie/bruit (persistance, cohérence entre capteurs, seuil par régime).
- **Optionnel** : HMM, autoencodeur LSTM, tableau de bord Streamlit.
- Protocole d'évaluation, figé avant toute modélisation : [reports/protocole_evaluation.md](reports/protocole_evaluation.md).

## Structure

```
data/raw/          données brutes (non versionnées)
data/processed/    données nettoyées (non versionnées)
notebooks/         orchestration et visualisation uniquement
src/               toute la logique réutilisable
  config.py        chemins, paramètres du protocole, seuils de qualité
  data_loading.py  chargement Pump et C-MAPSS
  events.py        pannes, fenêtres pré-panne, zones du protocole
  quality.py       profil de qualité, capteurs morts ou suspects, ADF/KPSS
  viz.py           figures
reports/           protocole, figures, tables
tests/             tests unitaires (pytest, données synthétiques)
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS : source .venv/bin/activate)
pip install -r requirements.txt
pytest                           # vérifie src/ sur données synthétiques
jupyter lab                      # puis notebooks/01_eda_qualite.ipynb
```

## Avancement

| Phase | Contenu | Statut |
|---|---|---|
| 0 | Structure, protocole d'évaluation | ✅ |
| 1 | EDA et qualité (`01_eda_qualite`) | ✅ code, ⏳ à exécuter sur les données |
| 2 | Nettoyage | — |
| 3 | Segmentation des régimes | — |
| 4 | Détection et filtre anomalie/bruit | — |
| 5 | Signaux précurseurs | — |
| 6 | Rapport et soutenance | — |
