# Dynamic Pricing & Competition in E-Commerce — Agent 1 : prévision de la demande (LSTM)

Projet universitaire n°8. L'**Agent 1** prévoit la demande hebdomadaire de chaque produit, 1 à 4 semaines à l'avance, avec un réseau LSTM (TensorFlow/Keras). Sa prévision est transmise à l'**Agent 3** (optimisation des prix) dans le fichier `outputs/demand_forecast.csv`.

| Agent | Rôle | Technique |
|---|---|---|
| **Agent 1 (ce dépôt)** | Prévision de la demande hebdomadaire | LSTM |
| Agent 2 | Réputation à partir des avis | Transformer |
| Agent 3 | Prix optimal, intégration, démo | Régression linéaire |

Documents associés : `RAPPORT_AGENT1.md` (section du rapport technique), `CONTRACT.md` (contrat d'intégration entre agents).

## Résultats en bref

Erreur en % (WAPE, plus bas = mieux), 32 produits, 4 horizons regroupés.

| Modèle | Validation | Test (évalué une seule fois) |
|---|---|---|
| Ensemble de 5 LSTM | 8,77 | 7,01 |
| Hybride (0,5 × LSTM + 0,5 × saisonnier naïf) | 7,32 (diagnostic) | 8,63 |
| Naïf (dernière semaine) | 11,42 | 7,95 |
| Moyenne mobile (4 semaines) | 10,00 | **6,20** |
| Saisonnier naïf (même semaine l'an passé) | 8,48 | 12,66 |

Le LSTM bat le modèle naïf sur les deux périodes, mais **la moyenne mobile fait mieux sur le test** : les écarts d'environ un point ne sont pas concluants (un seul test de 26 semaines). Détails et limites : `RAPPORT_AGENT1.md`.

## Structure du projet

```
dynamic-pricing-ecommerce/
├── config.yaml              # tous les paramètres (chemins, fenêtre, réseau, graine...)
├── requirements.txt
├── data/
│   ├── raw/                 # CSV Kaggle (NON fournis dans le dépôt)
│   └── processed/           # variables, séquences (générés)
├── notebooks/               # 01 exploration, 02 nettoyage, 03 entraînement/résultats
├── models/                  # 5 réseaux .keras, model_meta*.json, ensemble_meta.json, feature_scaler.json
├── outputs/                 # demand_forecast.csv, évaluations (json/csv), graphiques
├── src/forecast/            # code du pipeline (un module par étape)
└── tests/                   # tests unitaires (pytest)
```

Modules de `src/forecast/` : `data_loader`, `preprocessing`, `feature_engineering`, `sequence_generator`, `model`, `train`, `evaluate`, `experiments`, `ensemble`, `predict`, `export_forecast`, `final_evaluation`.

## Installation (Windows 11, PowerShell, CPU)

```powershell
cd dynamic-pricing-ecommerce
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --default-timeout=120 --retries 10 -r requirements.txt
```

Version testée : Python 3.13.5, TensorFlow 2.21.0. `requirements.txt` ne fixe que des versions minimales. Pour figer votre environnement : `pip freeze > requirements-lock.txt`.

## Données

Jeu de données Kaggle **« Store Sales - Time Series Forecasting »** (Favorita). Télécharger les fichiers et les placer dans `data/raw/` : `train.csv`, `holidays_events.csv` (utilisés par le modèle), `stores.csv`, `oil.csv`, `transactions.csv` (explorés seulement).

Choix importants : `product_id` = colonne `family` (une **catégorie** de produits, pas un produit individuel) ; les ventes sont additionnées sur les 54 magasins ; il n'y a **pas de colonne prix**.

## Exécution du pipeline (dans l'ordre)

```powershell
python -m src.forecast.preprocessing        # étape 4 : nettoyage + agrégation hebdomadaire
python -m src.forecast.sequence_generator   # étape 5 : variables, mise à l'échelle, fenêtres
python -m src.forecast.train                # étape 6 : modèle principal (graine 42)
python -m src.forecast.evaluate --split val # étape 7 : comparaison aux références (validation)
python -m src.forecast.ensemble             # étape 6d : 5 réseaux, validation seulement
python -m src.forecast.export_forecast      # étape 9 : écrit et valide demand_forecast.csv
python -m src.forecast.final_evaluation --final   # UNE SEULE FOIS : évaluation sur le test
```

Optionnel : `python -m src.forecast.experiments` (comparaison des variantes sur la validation).

**Attention** : ne relancez pas `sequence_generator` sur de nouvelles données sans réentraîner les réseaux, car il réécrit `models/feature_scaler.json` et recalcule les dates de découpage. Le jeu de test n'est utilisé que par `final_evaluation`, qui refuse de s'exécuter une seconde fois.

## Utiliser la prévision

Depuis Python (charge les 5 réseaux, ce qui demande TensorFlow) :

```python
from src.forecast.predict import predict_demand

predict_demand("GROCERY I", 1)             # semaine suivante, en unités
predict_demand("GROCERY I", "2017-08-21")  # ou n'importe quelle date de la semaine voulue
```

`week` = 1 à 4 (semaines après la dernière semaine des données) ou une date. Une erreur `ValueError` explicite est levée pour un produit inconnu ou une semaine hors plage.

En ligne de commande :

```powershell
python -m src.forecast.predict                                  # tableau de tous les produits
python -m src.forecast.predict --product "GROCERY I" --week 1
```

## Fichier d'intégration pour l'Agent 3

`outputs/demand_forecast.csv` : une ligne par (produit, semaine), 32 × 4 = 128 lignes.

| Colonne | Type | Signification |
|---|---|---|
| `product_id` | texte | famille de produits (ex. `GROCERY I`) |
| `week` | date `AAAA-MM-JJ` | **lundi** de la semaine prévue |
| `demand_pred` | nombre ≥ 0 | unités vendues par semaine, tous magasins |
| `horizon` | entier 1 à 4 | semaines après la dernière semaine observée |

Vérification : `python -m src.forecast.export_forecast --check`. Le contrat complet (validation, versions, questions ouvertes) est dans `CONTRACT.md`.

## Configuration

Tout se règle dans `config.yaml`. Bloc optionnel qui choisit le modèle livré dans `demand_forecast.csv` :

```yaml
prediction:
  use_ensemble: true      # moyenne des 5 réseaux (models/ensemble_meta.json)
  seasonal_weight: 0.5    # 0.0 = LSTM pur ; 0.5 = hybride (valeur par défaut si le bloc est absent)
```

Après modification, relancer `python -m src.forecast.export_forecast`.

## Tests

```powershell
python -m pytest tests -q        # 94 tests
```

Les tests utilisent des données synthétiques : ils ne lisent ni le jeu Kaggle ni le jeu de test.

## Reproductibilité

Graine 42 (réseaux de l'ensemble : graines 42 à 46), exécution déterministe sur CPU. Un autre ordinateur ou une autre version de TensorFlow peut donner des chiffres légèrement différents.

## Limites connues

- Pas de colonne prix dans les données : la prévision ne dépend pas du prix.
- Granularité = famille de produits (32), pas de SKU. La clé commune avec les Agents 2 et 3 reste à définir (voir `CONTRACT.md`, question Q1).
- Un seul jeu de test de 26 semaines ; forte variabilité entre graines (WAPE de validation de 8,05 % à 10,64 % selon la graine).
- Les réseaux sont entraînés jusqu'à environ août 2016 et ne sont pas réentraînés sur la validation et le test.
- Pas d'intervalle de confiance.

## Modèle livré : décision en attente

Le fichier `demand_forecast.csv` actuel provient de l'**hybride** (`seasonal_weight: 0.5`), choisi sur la validation avant d'ouvrir le test. Sur le test, le LSTM pur (7,01 %) est meilleur que l'hybride (8,63 %). Recommandation : `seasonal_weight: 0.0`, avec mention dans le rapport que ce choix a été fait après avoir vu le test.
