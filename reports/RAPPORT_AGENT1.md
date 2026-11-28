# Rapport technique — Section Agent 1 : prévision de la demande hebdomadaire par LSTM

Projet n°8 « Tarification dynamique et concurrence en e-commerce ».

## 1. Rôle de l'Agent 1 et objectif

L'Agent 1 prévoit, pour chaque produit, la demande des **4 semaines suivantes** (+1 à +4) à partir de l'historique des ventes. Sa sortie (`outputs/demand_forecast.csv`) alimente l'Agent 3, qui s'en sert comme demande de base pour calculer le prix optimal. Contraintes du cahier des charges : un réseau LSTM, aucune fuite de données, découpage chronologique, comparaison à des références simples, prévisions en unités d'origine, exécution sur CPU.

## 2. Données

**Source.** Jeu Kaggle « Store Sales - Time Series Forecasting » (Corporación Favorita, Équateur).

| Fichier | Rôle |
|---|---|
| `train.csv` (3 000 888 lignes) | données principales : `id`, `date`, `store_nbr`, `family`, `sales`, `onpromotion` |
| `holidays_events.csv` | jours fériés et événements nationaux (variables calendaires) |
| `stores.csv`, `oil.csv`, `transactions.csv` | explorés, non utilisés comme variables |

**Identifiant produit.** `product_id` = `family`, une **catégorie** de produits (33 familles, 32 retenues). Les ventes sont additionnées sur les 54 magasins, ce qui donne une série nationale par famille.

**Absence de prix.** Le jeu ne contient aucune colonne prix ni identifiant de produit individuel. Conséquence : la prévision est une demande observée, non conditionnée au prix. C'est la limite la plus importante pour l'intégration avec l'Agent 3 (section 10).

## 3. Prétraitement

1. **Agrégation hebdomadaire** : semaines commençant le lundi ; les semaines incomplètes (début et fin) sont supprimées ; les jours calendaires manquants (par exemple le 25 décembre, magasins fermés) sont ajoutés avant l'agrégation.
2. **Valeurs nulles.** D'après l'exploration (notebook 01), environ 31 % des observations de `train.csv` ont `sales = 0`. Une partie vient de magasins fermés ou de familles pas encore vendues dans un magasin ; les zéros de début de série sont retirés, les autres sont conservés (ce sont de vraies ventes nulles ou des fermetures).
3. **Sélection des produits** : au moins 104 semaines d'historique. `BOOKS` (6 438 unités au total) est exclu ; 32 produits restent (liste dans `outputs/excluded_products.csv`).
4. **Valeurs aberrantes** : environ 244 semaines sont **signalées** (`is_outlier`, fenêtre médiane centrée de 9 semaines, seuil 6 écarts-types robustes) mais **ni supprimées ni modifiées**. Ce signalement n'est pas une variable du modèle car il regarde le futur.
5. **Promotions** : `onpromotion` est peu fiable en 2013 (beaucoup de zéros) ; une variable causale `promo_available` indique si les promotions existent déjà.

Résultat : `outputs/sales_weekly.csv` (`product_id`, `week`, `sales`, `onpromotion`, `open_stores`, `is_outlier`) et un rapport qualité `outputs/data_quality_report.csv`.

## 4. Variables (18)

Calculées avec la semaine courante et le **passé uniquement** :

| Groupe | Variables |
|---|---|
| Ventes | `log_sales` (log1p), `lag_1`, `lag_2`, `lag_4`, `roll_mean_4`, `roll_std_4`, `roll_mean_8`, `roll_std_8` |
| Promotion et magasins | `log_promo`, `promo_available`, `open_stores` |
| Calendrier | `woy_sin`, `woy_cos`, `month_sin`, `month_cos`, `is_christmas_week` |
| Fêtes | `n_holidays`, `n_events` (jours fériés nationaux seulement ; les jours « transférés » sont ignorés) |

**Mise à l'échelle** : standardisation par produit sur la log-vente ; les paramètres sont ajustés **uniquement sur les semaines d'entraînement** et sauvegardés dans `models/feature_scaler.json`. Les prévisions sont reconverties en unités d'origine (et bornées à 0).

## 5. Découpage et prévention des fuites

- **Découpage chronologique global** : test = 26 dernières semaines, validation = 26 semaines précédentes, entraînement = le reste. Dates approximatives : entraînement jusqu'à août 2016, validation jusqu'à février 2017, test jusqu'au 7 août 2017 (dates exactes dans `outputs/split_info.json`).
- Un échantillon appartient à un ensemble selon ses **semaines cibles** ; ceux qui chevauchent une frontière sont écartés. La fenêtre d'entrée peut remonter dans l'ensemble précédent (c'est du passé).
- **Échantillons** : 4 785 (entraînement), 736 (validation), 736 (test).
- Fenêtre de **12 semaines** en entrée, **4 semaines** en sortie (prévision directe multi-pas : un seul modèle donne +1 à +4).
- Discipline : tous les choix (architecture, variantes, ensemble) sont faits sur la **validation** ; le test n'est lu que par `final_evaluation`, une seule fois (le script refuse une seconde exécution).

## 6. Modèle

**Architecture** (10 756 paramètres par réseau) : deux couches LSTM `[32, 16]`, dropout 0,3, un plongement appris de dimension 4 pour chaque produit, une couche dense de 32 neurones, 4 sorties.

**Deux idées qui ont corrigé un premier modèle perdant.** Le premier LSTM (cible = niveau de ventes, 35 108 paramètres, dropout 0,2) obtenait un WAPE de **13,30 %** sur la validation, pire que les trois références (8,48 % à 11,42 %), avec un net sur-apprentissage (perte d'entraînement 0,15 contre 0,236 en validation). Corrections, testées sur la validation seulement :
- **cible « delta »** : le réseau prédit la variation par rapport à la dernière semaine observée ; une sortie nulle équivaut à la prévision naïve, le réseau n'apprend que des corrections ;
- **entrée saisonnière** : ce qui s'est passé un an plus tôt (52 semaines) sur les mêmes semaines, plus un indicateur de disponibilité ;
- réseau plus petit et dropout plus fort.

Le modèle retenu atteint 9,14 % sur la validation (graine 42).

**Ensemble.** Les résultats varient avec la graine aléatoire (WAPE de validation de 8,05 % à 10,64 % pour cinq graines). Cinq réseaux (graines 42 à 46) sont donc entraînés et leurs prévisions **moyennées** en unités d'origine. Aucun réseau n'est sélectionné après coup.

**Entraînement.** Perte de Huber (robuste aux pics), Adam (pas 0,001), lots de 32, 100 époques au plus, arrêt anticipé (patience 10, meilleurs poids restaurés), réduction du pas sur plateau, graine fixe. Un réseau s'entraîne en 1 à 4 minutes sur CPU.

## 7. Références de comparaison

- **Naïf** : la dernière semaine observée, répétée.
- **Moyenne mobile** : moyenne des 4 dernières semaines.
- **Saisonnier naïf** : la même semaine un an plus tôt (repli sur le naïf si absente).

Toutes les références voient exactement la même information que le LSTM (historique jusqu'à la semaine d'origine).

## 8. Résultats

**Métriques** : MAE, RMSE, MAPE (ventes > 0), sMAPE, biais, et surtout **WAPE** = somme des erreurs absolues / somme des ventes. MAE et RMSE en unités sont dominées par les plus gros produits (par exemple `GROCERY I`, environ 1,5 million d'unités par semaine contre quelques dizaines pour `BABY CARE`) ; le WAPE et le nombre de produits battus complètent la lecture.

### 8.1 Validation (32 produits)

| Modèle | MAE | RMSE | MAPE | WAPE | sMAPE | Biais |
|---|---|---|---|---|---|---|
| LSTM, une seule graine (42) | 16 519,64 | 53 696,65 | 15,63 | 9,14 | 14,09 | −11 053 |
| **Ensemble de 5 LSTM** | 15 866,02 | 49 789,01 | 15,05 | **8,77** | 13,54 | −10 407 |
| Naïf | 20 647,55 | 65 499,25 | 18,78 | 11,42 | 15,01 | −2 883 |
| Moyenne mobile | 18 074,64 | 58 768,69 | 20,05 | 10,00 | 15,15 | −5 200 |
| Saisonnier naïf | 15 340,73 | 46 367,11 | 20,67 | 8,48 | 19,51 | −8 468 |
| Hybride 50/50 (diagnostic) | 13 236,51 | 42 479,95 | 15,54 | 7,32 | 14,12 | −9 437 |

Sur la validation, l'ensemble bat le naïf et la moyenne mobile mais pas le saisonnier naïf en WAPE global (8,77 % contre 8,48 %). Il est le meilleur à +1 semaine (7,5 % contre 8,2 %) et se dégrade pour les horizons plus longs.

### 8.2 Test final (évalué une seule fois)

| Modèle | MAE | RMSE | MAPE | WAPE | sMAPE | Biais |
|---|---|---|---|---|---|---|
| Hybride 50/50 | 16 244,10 | 54 911,42 | 12,58 | 8,63 | 13,54 | −10 636 |
| **Ensemble de 5 LSTM** | 13 185,32 | 45 327,87 | 11,95 | **7,01** | 11,73 | −3 382 |
| LSTM, une seule graine (42) | 13 290,55 | 44 191,46 | 12,44 | 7,06 | 12,20 | −1 963 |
| Naïf | 14 970,81 | 45 210,66 | 13,14 | 7,95 | 12,32 | −622 |
| **Moyenne mobile** | 11 660,76 | 34 185,94 | 13,39 | **6,20** | 11,50 | −522 |
| Saisonnier naïf | 23 833,03 | 74 859,07 | 17,82 | 12,66 | 20,82 | −17 889 |

WAPE (%) par horizon sur le test :

| Horizon | Hybride | Ensemble LSTM | Naïf | Moyenne mobile | Saisonnier naïf |
|---|---|---|---|---|---|
| +1 semaine | 8,1 | 6,4 | 7,7 | 6,2 | 12,3 |
| +2 semaines | 8,6 | 7,0 | 8,6 | 6,0 | 12,6 |
| +3 semaines | 9,3 | 7,8 | 9,2 | 6,3 | 12,9 |
| +4 semaines | 8,6 | 6,8 | 6,3 | 6,2 | 12,9 |

WAPE moyen par produit : hybride 13,2 %, ensemble 11,9 %, naïf 12,7 %, moyenne mobile 11,8 %, saisonnier naïf 18,7 %. WAPE des cinq réseaux pris seuls sur le test : 7,06, 7,39, 7,25, 7,15 et 7,57 %. L'hybride bat le saisonnier naïf sur 31 produits sur 32 mais le LSTM pur sur seulement 14 sur 32.

### 8.3 Interprétation

- Le LSTM pur bat le naïf sur la validation et sur le test (la moyenne mobile aussi), et il n'est jamais le pire modèle. L'hybride et le saisonnier naïf ne battent le naïf que sur la validation.
- La moyenne mobile est la meilleure sur le test (6,20 %) mais perd contre le LSTM sur la validation (10,00 % contre 8,77 %) : l'ordre entre les deux s'inverse selon la période. On ne peut pas conclure à une supériorité durable de l'un ou de l'autre.
- Le composant saisonnier naïf est instable (8,48 % puis 12,66 %). Il entraîne l'hybride vers le bas sur le test. **Hypothèse non vérifiée** : l'année de référence des semaines de test (2016) contient le séisme d'avril 2016, ce qui rend « ce qui s'est passé l'an dernier » peu fiable. Elle peut être testée à partir de `outputs/predictions_test.csv`.
- À +4 semaines, le LSTM (6,8 %) ne bat ni le naïf (6,3 %) ni la moyenne mobile (6,2 %).
- Biais négatif : tous les modèles sous-estiment. L'hybride sous-estime d'environ 5,7 % (−10 636 pour une demande moyenne de 188 000 par produit et par semaine, valeur déduite de MAE ÷ WAPE) ; le LSTM d'environ 1,8 %.

## 9. Modèle livré à l'Agent 3

Le fichier `outputs/demand_forecast.csv` actuel a été produit par l'**hybride** : `demand_pred = max(0, 0,5 × moyenne des 5 LSTM + 0,5 × saisonnier naïf)`. Ce choix avait été fait sur la validation (7,32 %), **avant** l'ouverture du jeu de test.

**Recommandation : livrer le LSTM pur** (`prediction.seasonal_weight: 0.0`), car c'est le modèle le plus stable sur les deux périodes et le plus conforme à la mission « Agent 1 = LSTM ». Ce changement est décidé **après avoir vu** le test : le chiffre de test (7,01 %) n'est donc plus une estimation sans biais de cette décision. Le rapport doit l'écrire explicitement.

> **[DÉCISION À CONFIRMER PAR L'AUTEUR]** Garder l'hybride (conforme au protocole, moins bon sur le test) ou passer au LSTM pur (plus stable, choix post-test). Mettre à jour cette section en conséquence.

## 10. Intégration avec les Agents 2 et 3

Le format du fichier, les règles de validation et les questions ouvertes sont dans `CONTRACT.md`. Points importants :
- **Clé commune non définie** : le `product_id` de l'Agent 1 est une famille de produits Favorita ; il n'existe pas de table de correspondance avec les produits des avis (Agent 2) ni avec les prix et coûts (Agent 3).
- **Pas de prix** : l'Agent 3 devra trouver ailleurs un historique prix et quantité pour estimer l'élasticité. La prévision de l'Agent 1 sert de **demande de base** au prix actuel.
- **Période** : les prévisions couvrent du 14 août au 4 septembre 2017 ; l'alignement temporel avec les autres données est à confirmer.

## 11. Limites

1. Données sans prix ; la prévision n'est pas conditionnée au prix.
2. Granularité « famille de produits », pas de SKU.
3. Un seul test de 26 semaines (736 fenêtres qui se recouvrent) : les petits écarts peuvent être du bruit.
4. Variabilité entre graines importante (validation : 8,05 à 10,64 %) ; l'ensemble la réduit sans la supprimer.
5. Le saisonnier naïf dépend d'une année de référence atypique (séisme de 2016, hypothèse non vérifiée).
6. Réseaux entraînés seulement jusqu'à environ août 2016 ; pas de réentraînement sur validation et test.
7. Pas d'intervalle de prévision ; biais négatif (sous-estimation).
8. `onpromotion` peu fiable en 2013 ; environ 244 semaines aberrantes signalées mais conservées.
9. Les versions de bibliothèques ne sont pas figées dans `requirements.txt` ; les chiffres peuvent varier légèrement selon l'environnement.
10. Le choix du modèle livré a été arrêté en partie après la lecture du test (section 9).

## 12. Perspectives

Réentraîner le modèle final sur entraînement + validation avec un nombre d'époques fixé ; ajouter des intervalles de prévision ; tester d'autres variables (pétrole, transactions) sur la validation ; évaluer la correction du biais ; valider avec plusieurs fenêtres de test glissantes plutôt qu'une seule ; adapter le niveau de granularité à celui de l'Agent 3 dès que la clé commune est définie.

## 13. Reproduire les résultats

Voir `README.md`. Fichiers de résultats : `outputs/ensemble_val.json` (validation), `outputs/evaluation_test_final.json` (test), `outputs/predictions_test.csv`, `outputs/plots/`. Le notebook `notebooks/03_lstm_training.ipynb` les affiche sans rien réentraîner.
