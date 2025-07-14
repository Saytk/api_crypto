# Résumé des modifications pour résoudre les problèmes de calibration et de seuil

## Problèmes identifiés

1. **Calibration problématique** : Application d'une sigmoid sur des probabilités déjà transformées par sigmoid
2. **Classe positive rare** : Le taux de cible est souvent ≤ 10%, rendant le seuil de 0.5 irréaliste
3. **Paramètres conservateurs** : Arbres peu profonds et techniques de régularisation compressant les prédictions
4. **Calcul de cible incluant la bougie courante** : Biais dans les données d'entraînement

## Modifications apportées

### 1. Calibration sur les scores bruts (log-odds)

Dans `train_lgbm.py`, nous avons modifié la fonction `calibrate_probabilities` pour utiliser les scores bruts :

```python
# Avant
train_preds = model.predict(X_train)
calibrator.fit(train_preds_2d, y_train)

# Après
train_raw = model.predict(X_train, raw_score=True)
calibrator.fit(train_raw_2d, y_train)
```

Dans `predict.py`, nous avons mis à jour la fonction `apply_calibration` et son utilisation :

```python
# Obtenir le score brut pour la calibration
raw_score = MODELS[symbol].predict(feats_last[FEATURE_ORDER], raw_score=True)[0]
p_up = apply_calibration(raw_score, weights, bias)
```

### 2. Gestion du déséquilibre des classes

Nous avons activé la gestion automatique du déséquilibre des classes dans LightGBM :

```python
params = {
    # ...autres paramètres...
    "is_unbalance": True,  # Gestion automatique du déséquilibre des classes
}
```

### 3. Ajustement du seuil de décision

Nous avons modifié le code pour utiliser un seuil plus réaliste (0.3 par défaut) :

```python
# Utiliser le seuil de décision du metadata ou un seuil adapté par défaut (0.3)
decision_threshold = metadata.get("pred_thresh", 0.3)
signal = "LONG" if p_up >= decision_threshold else "SHORT"
```

Et nous avons ajusté le calcul de confiance pour qu'il soit relatif au seuil :

```python
confidence = min(abs(p_up - decision_threshold) / decision_threshold * 2, 1.0)
```

### 4. Correction du calcul de la cible

Nous avons modifié le calcul de la cible pour exclure la bougie courante :

```python
# Avant
df["future_max"] = df["high"].rolling(HORIZON).max().shift(-HORIZON)

# Après
df["future_max"] = df["high"].shift(-1).rolling(HORIZON).max().shift(-(HORIZON-1))
```

### 5. Suppression de la recalibration manuelle

Nous avons supprimé la logique de recalibration manuelle qui était utilisée en fallback :

```python
# Avant
if p_up_original >= 0.4:
    p_up = (p_up_original - 0.4) / (1 - 0.4) * 0.5 + 0.5
else:
    p_up = p_up_original / 0.4 * 0.5

# Après
p_up = p_up_original  # Utiliser la probabilité originale si pas de calibration
```

### 6. Amélioration de la journalisation

Nous avons ajouté des logs pour mieux comprendre le comportement du modèle :

```python
# Log du taux de classe positive
pos_rate = y_train.mean()
log.info(f"[calibrate] Taux de classe positive: {pos_rate:.4f} ({pos_rate*100:.2f}%)")

# Log des décisions de prédiction
log.info(f"Decision threshold: {decision_threshold}, Probability: {p_up:.4f}, Signal: {signal}")
```

## Résultats attendus

Ces modifications devraient résoudre les problèmes suivants :

1. **Amélioration de la calibration** : Les probabilités seront mieux calibrées grâce à l'utilisation des scores bruts
2. **Meilleure gestion des classes déséquilibrées** : Le modèle donnera plus de poids aux exemples positifs
3. **Seuil de décision plus réaliste** : Le seuil de 0.3 est plus adapté à la distribution des probabilités
4. **Données d'entraînement non biaisées** : L'exclusion de la bougie courante évite les fuites de données
5. **Confiance plus précise** : Le calcul de confiance est maintenant relatif au seuil de décision

## Prochaines étapes recommandées

1. **Réentraîner le modèle** avec ces modifications pour générer de nouveaux fichiers de modèle et de métadonnées
2. **Surveiller les métriques** après déploiement pour vérifier l'amélioration des performances
3. **Ajuster le seuil de décision** en fonction des résultats observés
4. **Envisager l'approche par espérance de gain** comme alternative au seuil fixe