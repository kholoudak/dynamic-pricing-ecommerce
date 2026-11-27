# Agent 2 – Analyse de la réputation des produits

## 1. Introduction:

L'objectif de l'Agent 2 est d'analyser les avis des clients afin de calculer un score de réputation pour chaque produit.

Ce score représente le niveau de satisfaction des clients et constitue un facteur supplémentaire pouvant être utilisé par le système de tarification dynamique pour prendre des décisions sur les prix.

## 2. Jeu de données utilisé:

L'Agent 2 utilise le jeu de données *Flipkart Product Reviews*.

- **Fichier source :** `data/raw/Dataset-SA.csv`  fichier utilisé localement , non versionné sur GitHub .
- **Nombre total d'avis disponibles :** 205 052
- **Principales colonnes :**
  - `product_name` : nom du produit.
  - `product_price` : prix du produit.
  - `Rate` : note attribuée par le client.
  - `Review` : texte de l'avis.
  - `Summary` : résumé de l'avis.
  - `Sentiment` : sentiment associé à l'avis.

Pour cette expérimentation, 20 000 avis ont été traités afin de générer les scores de réputation.

## 3. Prétraitement des données:

Le prétraitement est réalisé dans le fichier `src/nlp/preprocessing.py`.

Les principales opérations effectuées sont :

- Chargement et validation du jeu de données.
- Suppression des avis dont le nom du produit est manquant.
- Conversion des notes en valeurs numériques.
- Gestion des textes manquants en utilisant le résumé de l'avis lorsqu'il est disponible.
- Préparation des textes et des étiquettes de sentiment pour l'analyse.

## 4. Analyse des sentiments:

L'analyse des sentiments est implémentée dans `src/nlp/sentiment.py`.

Le projet utilise un modèle Transformer pré-entraîné :

`cardiffnlp/twitter-roberta-base-sentiment-latest`

Ce modèle analyse les avis des clients et estime trois probabilités :

- **Positive :** sentiment positif.
- **Neutral :** sentiment neutre.
- **Negative :** sentiment négatif.

Ces probabilités permettent d'évaluer les opinions des clients concernant chaque produit.

## 5. Calcul du score de réputation:

Le calcul de la réputation est réalisé dans `src/nlp/reputation.py`.

Chaque avis reçoit un score calculé à partir de ses probabilités de sentiment selon la formule suivante :

**Score de l'avis = P(positif) + 0,5 × P(neutre)**

Le score de réputation d'un produit correspond à la moyenne des scores de ses avis.

Ainsi :

- Les avis positifs contribuent pleinement au score.
- Les avis neutres contribuent partiellement au score.
- Les avis négatifs ne contribuent pas positivement au score.

Le module calcule également :

- `n_reviews` : nombre d'avis par produit.
- `positive_ratio` : proportion d'avis positifs.
- `neutral_ratio` : proportion d'avis neutres.
- `negative_ratio` : proportion d'avis négatifs.

## 6. Résultats obtenus:

Les résultats sont enregistrés dans le fichier :

`data/processed/reputation_scores.csv`

Pour l'expérimentation réalisée sur 20 000 avis, le fichier généré contient 94 produits et 6 colonnes :

- `product_name`
- `reputation_score`
- `n_reviews`
- `positive_ratio`
- `neutral_ratio`
- `negative_ratio`

Ce fichier fournit les informations de réputation nécessaires à l'étape suivante du système de tarification dynamique.

## 7. Tests et validation:

Les tests de l'Agent 2 sont regroupés dans :

`tests/test_nlp.py`

Trois tests ont été exécutés avec succès :

1. `test_clean_reviews` : vérifie le fonctionnement du prétraitement des avis.
2. `test_calculate_reputation` : vérifie le calcul du score de réputation.
3. `test_invalid_prediction_count` : vérifie la détection d'un nombre incorrect de prédictions.

**Résultat : les trois tests ont réussi sans erreur.**

## 8. Limites et contraintes techniques:

L'expérimentation a été réalisée sur un échantillon de 20 000 avis parmi les 205 052 avis disponibles dans le jeu de données original.

Le traitement de la totalité des avis nécessitait des ressources de calcul importantes, notamment en raison de l'utilisation d'un modèle Transformer pour l'analyse des sentiments.

En raison des limitations matérielles rencontrées, notamment la lenteur du processeur et le temps d'exécution élevé, il n'a pas été possible de réaliser le traitement complet, aussi bien sur Google Colab que dans l'environnement local avec VS Code.

Le choix d'un échantillon de 20 000 avis a donc permis de réaliser l'expérimentation avec les ressources disponibles, de vérifier le fonctionnement du pipeline et de générer les scores de réputation nécessaires à l'intégration dans le système global.

D'autres limites doivent également être prises en compte :

- Le score de réputation dépend de la qualité et de la quantité des avis disponibles.
- Le nombre d'avis peut varier d'un produit à l'autre, ce qui peut influencer la fiabilité des scores obtenus.
- Le traitement d'un volume plus important nécessiterait des ressources matérielles plus performantes ou une optimisation du temps d'exécution.

## 9. Conclusion:

L'Agent 2 permet d'analyser la réputation des produits à partir des avis clients en utilisant un modèle Transformer pré-entraîné.

Il transforme les opinions exprimées dans les avis en indicateurs numériques de réputation qui peuvent être exploités par le système de tarification dynamique.

Les résultats générés et la réussite des trois tests confirment le bon fonctionnement des principaux composants de l'Agent 2.
