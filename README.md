# Détection de Fraude Bancaire — Pipeline ML sur données déséquilibrées (Dataset Kaggle)

## Présentation
Ce projet traite le problème classique de **détection de fraude bancaire** à partir du dataset Kaggle `creditcard.csv`.  
Plutôt que de viser directement des modèles très puissants (XGBoost, LightGBM), l’objectif principal est de construire un **pipeline Machine Learning robuste, réaliste et exploitable**, en mettant l’accent sur :

- le déséquilibre extrême des classes
- le choix de métriques adaptées (PR-AUC plutôt qu’Accuracy)
- un protocole strict anti-data leakage
- l’ajustement du seuil de décision selon une logique métier

En détection de fraude, obtenir un bon ROC-AUC n’est pas suffisant : le modèle doit être utilisable par une équipe opérationnelle, avec un volume d’alertes raisonnable.

---

## Dataset
**Source :** Kaggle — Credit Card Fraud Detection

- **Transactions :** 284 807  
- **Fraudes :** 492 (0.17%)  
- **Features :** 31  
  - `Time`
  - `Amount`
  - `V1 ... V28` (composantes PCA anonymisées)
  - `Class` (variable cible)

---

## Problématique
Dans un problème fortement déséquilibré, l’**Accuracy** est une métrique trompeuse.  
Un modèle prédisant uniquement "Non fraude" peut dépasser 99% d’accuracy tout en détectant 0 fraude.

L’objectif est donc de :
- maximiser la détection de fraude (**Recall**)
- tout en limitant les fausses alertes (**Precision / Faux positifs**)

---

## Méthodologie

### 1. Split Train/Test (Anti-fuite de données)
Un protocole strict anti-data leakage est appliqué :

- split **stratifié 80/20** réalisé avant toute transformation
- toutes les statistiques (scaling, quantiles, transformations) sont calculées uniquement sur le **train**
- le test reste totalement isolé pour garantir une évaluation réaliste

**Règle essentielle : SMOTE n’est jamais appliqué sur le test.**

---

### 2. Feature Engineering orienté métier
La fraude se concentre souvent sur des comportements extrêmes.

Features ajoutées :
- `Hour` : extraction de l’heure à partir de `Time`
- indicateurs binaires basés sur les quantiles du train :
  - `is_very_small_amount` (Q10)
  - `is_very_large_amount` (Q90)

Ces variables permettent de capturer :
- les micro-transactions (tests de carte)
- les transactions très élevées (cash-out)

---

### 3. Transformations mathématiques
Pour gérer les outliers et l’asymétrie :

- transformation `log1p(Amount)`
- `RobustScaler` (basé sur médiane et IQR)
- `PowerTransformer (Yeo-Johnson)` pour corriger la skewness des composantes PCA (compatible avec valeurs négatives)

---

### 4. Stratégies contre le déséquilibre testées
Plusieurs approches ont été comparées :

- Baseline Logistic Regression
- Class Weight Balanced
- Random Under-sampling
- SMOTE Over-sampling

---

## Modèle
Le modèle principal est une **Régression Logistique**.

Pourquoi ce choix ?
- baseline interprétable
- rapide et stable
- bon benchmark pour les pipelines de fraude
- permet d’analyser clairement l’impact du rééchantillonnage et du seuil

---

## Métriques d’évaluation
Les métriques utilisées sont adaptées aux classes déséquilibrées :

- **PR-AUC (Precision-Recall AUC)** (métrique prioritaire)
- Precision / Recall
- F1-score
- Confusion Matrix
- analyse selon différents seuils

ROC-AUC est également reporté, mais PR-AUC est privilégié car plus représentatif de la performance sur la classe minoritaire.

---

## Ajustement du seuil (Decision Threshold Tuning)
Une partie essentielle du projet est l’optimisation du seuil de décision.

Le seuil standard (0.50) génère trop de faux positifs, rendant le modèle inutilisable en contexte réel.  
Le seuil a donc été ajusté afin de maximiser le **F1-score** et de réduire la charge opérationnelle.

**Seuil final sélectionné : 0.95**

---

## Résultats finaux (Seuil = 0.95)
- **Recall :** 0.89  
- **Precision :** 0.25  
- **F1-score :** 0.39  
- **Faux positifs :** 1567 → 262 (réduction par 6)

Ce réglage permet de détecter la majorité des fraudes tout en maintenant un volume d’alertes réaliste pour une équipe de contrôle.

---



