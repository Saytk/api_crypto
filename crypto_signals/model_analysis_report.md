# Analyse du modèle de prédiction crypto : Pourquoi les probabilités ne dépassent jamais 0.5

## Résumé des observations

Après analyse approfondie du modèle de prédiction, nous avons identifié plusieurs facteurs qui expliquent pourquoi les probabilités prédites (`prob_up`) ne dépassent jamais 0.5 :

1. **Déséquilibre extrême des classes** : Seulement 7.71% des exemples sont positifs (hausse de 0.16% en 5 minutes)
2. **Calibration problématique** : La calibration logistique empire le log-loss (0.2230 → 0.2725)
3. **Paramètres conservateurs** : Arbres peu profonds et techniques de régularisation compressent les prédictions
4. **Double sigmoid** : La calibration applique une seconde fonction sigmoid qui écrase davantage les probabilités

## Analyse détaillée

### 1. Distribution des probabilités prédites

Statistiques des prédictions du modèle actuel :
- **Min** : 0.0256
- **Max** : 0.5950 (très peu de valeurs > 0.5)
- **Moyenne** : 0.0806 (proche du taux de base de 7.71%)
- **Médiane** : 0.0542
- **Écart-type** : 0.0692

Distribution :
- Prédictions > 0.5 : 0.11% (extrêmement rare)
- Prédictions > 0.4 : 0.77%
- Prédictions > 0.3 : 2.20%

### 2. Performance à différents seuils

| Seuil | Précision | Rappel |
|-------|-----------|--------|
| 0.3   | 64.71%    | 18.50% |
| 0.35  | 69.52%    | 12.36% |
| 0.4   | 74.53%    | 7.42%  |
| 0.45  | 83.90%    | 2.64%  |
| 0.5   | 90.91%    | 1.34%  |

Observation clé : **Le modèle est précis mais très conservateur**. À 0.5, la précision est excellente (90.91%) mais le rappel est minuscule (1.34%).

### 3. Tests de différentes configurations

Nous avons testé plusieurs configurations pour tenter d'obtenir des probabilités > 0.5 :

| Configuration | Max prob | Moyenne | % > 0.5 |
|---------------|----------|---------|---------|
| Baseline | 0.2172 | 0.0718 | 0% |
| Avec poids de classe | 0.1246 | 0.0982 | 0% |
| Modèle plus complexe | 0.2761 | 0.0747 | 0% |
| Complexe + poids | 0.1398 | 0.0987 | 0% |

Aucune configuration n'a produit de prédictions > 0.5 dans nos tests.

## Recommandations

Pour résoudre ce problème, voici les solutions recommandées par ordre de priorité :

### 1. Adapter le seuil de décision (solution immédiate)

```python
# Dans predict.py
signal = "LONG" if p_up >= 0.3 else "SHORT"  # Utiliser 0.3 au lieu de 0.5
```

Cette solution est la plus simple et peut être implémentée immédiatement. Un seuil de 0.3 offre un bon compromis entre précision (64.71%) et rappel (18.50%).

### 2. Désactiver la calibration logistique

La calibration actuelle empire le log-loss. Désactivez-la ou utilisez une méthode de calibration différente :

```python
# Dans train_lgbm.py
CALIBRATE = False  # Désactiver la calibration
```

### 3. Utiliser les log-odds plutôt que les probabilités

Si vous souhaitez conserver la calibration, calibrez sur les log-odds plutôt que sur des probabilités déjà compressées :

```python
# Dans train_lgbm.py, fonction calibrate_probabilities
calibrator.fit(train_preds_2d, y_train, raw_score=True)  # Utiliser raw_score=True
```

### 4. Rééquilibrer les classes plus agressivement

Augmentez le poids des exemples positifs pour forcer le modèle à prédire des probabilités plus élevées :

```python
# Dans train_lgbm.py
params["scale_pos_weight"] = 1.0 / pos_rate  # Inverse du taux positif
```

### 5. Autoriser plus de complexité

Permettez au modèle de capturer des patterns plus spécifiques :

```python
# Dans train_lgbm.py
params.update({
    "num_leaves": 127,  # Au lieu de 31
    "min_child_samples": 10,  # Au lieu de 20
    "bagging_fraction": 1.0,  # Désactiver le bagging
})
```

### 6. Évaluer l'espérance de gain plutôt que la probabilité brute

Plutôt que d'utiliser un seuil fixe, calculez l'espérance de gain :

```python
# Dans predict.py
tp_pct = 0.004  # 0.4% take profit
sl_pct = 0.002  # 0.2% stop loss
ev = p_up * tp_pct - (1 - p_up) * sl_pct  # Espérance de gain
signal = "LONG" if ev > 0 else "SHORT"
```

## Conclusion

Le problème des probabilités toujours inférieures à 0.5 est principalement dû à la rareté de l'événement prédit (7.71%) et aux paramètres conservateurs du modèle. La solution la plus simple est d'adapter le seuil de décision à 0.3 ou 0.35, ce qui permettra d'obtenir un meilleur équilibre entre précision et rappel.

Pour une solution plus robuste à long terme, combinez l'adaptation du seuil avec la désactivation de la calibration problématique et l'augmentation de la complexité du modèle.