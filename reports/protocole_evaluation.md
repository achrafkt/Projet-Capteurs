# Protocole d'évaluation

> Ce protocole est **figé avant toute modélisation**. Toute modification ultérieure doit
> être datée et justifiée dans le [journal des modifications](#7-journal-des-modifications).
> Les valeurs par défaut sont centralisées dans `src/config.py`.

## 1. Objectif évalué

Un détecteur produit un **score d'anomalie** par instant, puis une **alerte binaire** après
seuillage et filtre anomalie/bruit. On évalue sa capacité à **lever une alerte avant chaque
panne** tout en **limitant les fausses alertes** en fonctionnement normal.

## 2. Paramètres

| Paramètre | Défaut | Sensibilité testée | Rôle |
|---|---|---|---|
| `PRE_FAILURE_WINDOW` (W) | **48 h** | 12 h, 24 h, 72 h | Fenêtre pré-panne `[t_f − W, t_f[` = zone positive |
| `LOFO_PURGE` | 24 h | — | Zone tampon retirée de l'entraînement autour du cycle de test |
| `ALERT_MERGE_GAP` | 30 min | — | Deux alertes séparées de moins de 30 min forment un seul épisode |
| `MAX_FALSE_ALARMS_PER_DAY` | 1,0 | — | Contrainte opérationnelle pour le choix du seuil |

`t_f` désigne le premier point `BROKEN` d'une panne. Une panne est une séquence contiguë de `BROKEN`.

## 3. Zones temporelles (Pump)

Chaque instant reçoit une zone unique, calculée par `src/events.py::time_zones` :

| Zone | Définition | Entraînement | Évaluation |
|---|---|---|---|
| `pre_failure` | `[t_f − W, t_f[`, tronquée à la fin de la remise en service précédente | **exclue** (contamination probable par la dégradation) | positif |
| `normal` | `NORMAL` hors fenêtre pré-panne | **seule zone utilisée** | négatif |
| `excluded` | `BROKEN` et **`RECOVERING`** | **exclue** | ignorée (machine arrêtée ou en redémarrage) |
| `censored` | les W dernières heures des données, sans panne observée ensuite | exclue | ignorée (étiquette inconnue) |

Règles :
- Les modèles non supervisés sont entraînés **uniquement sur la zone `normal`** des cycles d'entraînement.
- Toute étape apprise sur les données (imputation, normalisation, PCA, régimes, seuils)
  est ajustée **dans le pli d'entraînement uniquement**, puis appliquée au pli de test.
- Les alertes émises en zone `excluded` ou `censored` ne comptent ni comme vrais ni comme faux positifs.

## 4. Validation croisée : leave-one-failure-out (LOFO)

**Cycle k** = `]fin de la remise en service de la panne k−1, fin de la remise en service de la panne k]`
(le premier cycle commence au début des données). Le segment après la dernière remise en service
est un cycle sans panne, toujours placé à l'entraînement, hors zone `censored`.

Pour chaque panne k (7 plis) :
- **Test** : le cycle k entier. Sa fenêtre pré-panne fournit les positifs, sa zone `normal` les négatifs.
- **Entraînement** : la zone `normal` de tous les autres cycles, **moins une purge de 24 h** de part
  et d'autre des bornes du cycle k. La purge limite la fuite d'information due à l'autocorrélation.
- **Seuil** : choisi sur les scores d'entraînement. Par défaut, c'est le quantile q tel que le taux
  d'épisodes d'alerte sur l'entraînement reste ≤ 1 par jour. Il n'est **jamais** ajusté sur le cycle de test.
- **Hyperparamètres** (persistance k, fenêtres, nombre de composantes PCA…) : fixés a priori, ou
  réglés par un LOFO interne sur les 6 cycles d'entraînement.

Limite assumée : le LOFO entraîne sur des cycles postérieurs au cycle de test. On ajoute donc un
**contrôle chronologique** (entraînement sur les cycles < k, test sur le cycle k, pour k = 2…7), présenté à côté.

## 5. Métriques (par événement)

Les alertes binaires sont regroupées en **épisodes** : séquences contiguës d'alertes, fusionnées
si elles sont séparées de moins de `ALERT_MERGE_GAP`.

Classement d'un épisode :
- **TP** s'il recoupe la fenêtre pré-panne d'une panne.
- **FP** s'il recoupe la zone `normal` sans recouper aucune fenêtre pré-panne.
- **Ignoré** s'il est entièrement en zone `excluded` ou `censored`.
- Les FP situés dans `[t_f − 2W, t_f − W[` sont aussi comptés à part comme **alertes précoces**.
  Ce sont des signaux précurseurs possibles, à analyser en phase 5. Ils restent des FP pour les métriques.

| Métrique | Définition |
|---|---|
| **Rappel** | pannes détectées (au moins un épisode TP) / pannes testées |
| **Précision** | épisodes TP / (épisodes TP + FP) |
| **F1 événementiel** | moyenne harmonique de la précision et du rappel ci-dessus |
| **Délai d'anticipation** | `t_f − début du premier épisode TP`, borné à W. Médiane, min et max sur les pannes détectées |
| **Fausses alertes par jour** | épisodes FP / durée de la zone `normal` évaluée (en jours) |
| **PR-AUC** | `average_precision_score` **par point**, sur le score continu, en zones `normal` et `pre_failure` uniquement. Indépendante du seuil. Toujours la comparer à la **prévalence** des positifs (score d'un modèle aléatoire) |

Agrégation sur les 7 plis : les épisodes TP et FP et les durées sont **additionnés sur tous les plis**
avant de calculer précision, rappel et taux de fausses alertes. Les métriques par pli sont aussi
publiées : avec 7 pannes, la dispersion compte autant que la moyenne.

**Critère principal de comparaison** : F1 événementiel sous la contrainte fausses alertes par jour ≤ 1.
Ensuite, à F1 égal : délai d'anticipation, puis PR-AUC.

**Ablation obligatoire (anomalie ou bruit)** : chaque détecteur est évalué **avec et sans** la couche
de filtrage (persistance, cohérence entre capteurs, seuil par régime, masquage des transitions).
Un filtre utile réduit fortement les fausses alertes par jour pour une perte de rappel nulle ou faible.

## 6. Jeu de contrôle : NASA C-MAPSS FD002

| Élément | Choix |
|---|---|
| Unité de temps | le cycle |
| Événement | la fin de vie de chaque moteur du jeu `train` (run-to-failure) |
| Fenêtre pré-panne | les **30 derniers cycles** (`CMAPSS_PRE_FAILURE_CYCLES`), sensibilité testée à 15 et 50 |
| Données saines d'entraînement | cycles où RUL > 125 (`CMAPSS_HEALTHY_RUL`, convention usuelle de RUL plafonnée) |
| Régimes | 6 conditions opératoires connues (`op_1..3`) : normalisation **par condition**, ajustée sur l'entraînement |
| Validation | `GroupKFold(5)` par moteur : un moteur n'est jamais à la fois en entraînement et en test |
| Test final | `test_FD002` + `RUL_FD002`, utilisé **une seule fois**, à la fin |
| Métriques | les mêmes qu'en section 5, avec les fausses alertes rapportées **pour 100 cycles** au lieu de par jour |

## 7. Journal des modifications

| Date | Modification | Justification |
|---|---|---|
| 2026-10-23 | Version initiale | — |
