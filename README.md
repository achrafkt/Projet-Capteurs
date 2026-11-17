# Détection d'anomalies dans les capteurs industriels

Projet en binôme : **Karati Achraf** et **Ait Boutargante Fatima Zohra**.

Détection d'anomalies et de signaux précurseurs de panne sur des séries temporelles multivariées de capteurs industriels, avec une attention particulière à la distinction entre anomalie réelle et bruit de mesure.

**Questions de recherche**

1. Quels comportements de capteurs s'écartent du fonctionnement habituel ?
2. Quels signaux précèdent une panne ou une dégradation ?
3. Comment distinguer une anomalie réelle d'un bruit de mesure ?

**Approche en bref.** Le projet segmente d'abord les régimes de fonctionnement de la machine, puis détecte les anomalies en combinant méthodes statistiques (MAD, CUSUM, PCA avec T² de Hotelling et SPE) et méthodes non supervisées (Isolation Forest, LOF). Le régime sert de contexte aux détecteurs. Une couche de filtrage écarte ensuite le bruit, à partir de la persistance des anomalies, de leur cohérence entre capteurs et du masquage des transitions de régime. Le protocole d'évaluation a été figé **avant** toute modélisation, pour éviter d'ajuster les méthodes aux résultats. Toute modification ultérieure est datée dans son journal.

**Résultat clé de la phase 3.** Deux régimes nets ressortent : un régime nominal (79 % du temps) et un régime où le bloc `sensor_14`–`sensor_33` est bas (21 %). Ce second régime couvre 73 à 100 % de la fenêtre pré-panne de 3 pannes sur 7. Normaliser les détecteurs par régime effacerait donc ce signal. Les seuils par régime sont pour cette raison évalués en ablation, et non appliqués par défaut.

## Données

| Rôle | Jeu | Source | Emplacement attendu |
|---|---|---|---|
| Principal | Pump Sensor Data : 52 capteurs échantillonnés à la minute, 7 pannes | [Kaggle `nphantawee/pump-sensor-data`](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) | `data/raw/sensor.csv` |
| Contrôle | NASA C-MAPSS FD002 : 6 conditions opératoires | NASA Prognostics Data Repository (miroir Kaggle `behrad3d/nasa-cmaps`) | `data/raw/CMAPSS/{train,test,RUL}_FD002.txt` |

Le jeu C-MAPSS sert de contrôle : ses 6 conditions opératoires connues permettent de vérifier que la segmentation des régimes retrouve une structure réelle.

Les données ne sont **pas versionnées** (voir `.gitignore`). Télécharge-les depuis les sources ci-dessus et place-les aux emplacements indiqués.

## Méthodes

| Étape | Méthodes |
|---|---|
| Qualité des données | Profil de qualité, détection des capteurs morts ou suspects, tests de stationnarité ADF/KPSS |
| Nettoyage | Retrait des capteurs morts, détection des valeurs figées, comblement des trous courts |
| Régimes de fonctionnement | PCA + KMeans (k choisi par silhouette), confirmation causale, détection des transitions, appris dans chaque pli |
| Détection | MAD, CUSUM, PCA (T² de Hotelling, SPE), Isolation Forest, LOF |
| Filtrage anomalie / bruit | Persistance, cohérence entre capteurs, masquage des transitions ; seuils par régime en ablation |
| Évaluation | *Leave-one-failure-out* sur les 7 pannes, métriques par épisode, contrôle chronologique |
| Optionnel | HMM, autoencodeur LSTM, tableau de bord Streamlit |

Protocole d'évaluation (pannes, fenêtres pré-panne, validation croisée *leave-one-failure-out*) : [reports/protocole_evaluation.md](reports/protocole_evaluation.md).

## Structure

```
data/raw/          données brutes (non versionnées)
data/processed/    données nettoyées (non versionnées)
notebooks/         orchestration et visualisation uniquement
src/               toute la logique réutilisable
  config.py        chemins, paramètres du protocole, seuils de qualité
  data_loading.py  chargement Pump et C-MAPSS
  events.py        pannes, fenêtres pré-panne, zones, cycles et plis LOFO
  quality.py       profil de qualité, capteurs morts ou suspects, ADF/KPSS
  cleaning.py      nettoyage (capteurs morts, valeurs figées, trous courts)
  regimes.py       segmentation des régimes (PCA + KMeans), transitions
  viz.py           figures
reports/           protocole, figures, tables
tests/             tests unitaires (pytest, données synthétiques)
```

Choix de conception : toute la logique vit dans `src/` et est couverte par des tests unitaires sur données synthétiques ; les notebooks ne font qu'orchestrer et visualiser.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
pytest                            # vérifie src/ sur données synthétiques
jupyter lab                       # puis notebooks/01_eda_qualite.ipynb
```

Les notebooks s'exécutent dans l'ordre : `01_eda_qualite` → `02_nettoyage` → `03_regimes`.

## Avancement

| Phase | Contenu | Statut |
|---|---|---|
| 0 | Structure, protocole d'évaluation | ✅ Terminé |
| 1 | EDA et qualité (`01_eda_qualite`) | ✅ Terminé |
| 2 | Nettoyage (`02_nettoyage`) | ✅ Terminé |
| 3 | Segmentation des régimes (`03_regimes`) + correctifs phases 1–2 | ✅ Terminé |
| 4 | Détection et filtre anomalie / bruit | 🔄 En cours |
| 5 | Signaux précurseurs | ⏳ À venir |
| 6 | Rapport et soutenance | ⏳ À venir |