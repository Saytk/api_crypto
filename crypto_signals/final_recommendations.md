# Recommandations finales : Résoudre le problème des probabilités < 0.5

## Résumé du problème

Notre analyse approfondie a confirmé que le modèle de prédiction crypto ne produit presque jamais de probabilités supérieures à 0.5, ce qui entraîne plusieurs conséquences :

1. Presque tous les signaux sont "SHORT" avec le seuil actuel de 0.5
2. Les métriques d'évaluation (precision/recall) sont trompeuses
3. La calibration logistique empire la situation en compressant davantage les probabilités

## Causes identifiées

1. **Déséquilibre extrême des classes** : Seulement 7.71% des exemples sont positifs
2. **Paramètres conservateurs** : `num_leaves=31`, `min_child_samples=20`, bagging activé
3. **Double sigmoid** : La calibration applique une seconde fonction sigmoid qui écrase les probabilités
4. **Seuil d'évaluation inadapté** : Le modèle est évalué avec `PRED_THRESH=0.4` pendant l'entraînement mais 0.5 en production

## Démonstration de solutions

Nous avons testé plusieurs approches :

### 1. Adaptation du seuil (0.3 au lieu de 0.5)

Pour une prédiction avec `prob_up=0.4063` :
- **Avec seuil 0.5** : Signal SHORT, Confiance 0.1874 (faible)
- **Avec seuil 0.3** : Signal LONG, Confiance 0.7087 (élevée)

Cette simple modification inverse complètement le signal et augmente significativement la confiance.

### 2. Approche par espérance de gain

En considérant un ratio risque/récompense de 1:2 (SL 0.2%, TP 0.4%) :
- Espérance de gain = 0.000438 (positive)
- Signal = LONG

Cette approche confirme que le signal devrait être LONG malgré une probabilité < 0.5.

### 3. Tentatives d'amélioration du modèle

Nous avons testé différentes configurations :

| Configuration | Max prob | % > 0.5 |
|---------------|----------|---------|
| Baseline | 0.2172 | 0% |
| Avec poids de classe | 0.1246 | 0% |
| Modèle plus complexe | 0.2761 | 0% |
| Complexe + poids | 0.1398 | 0% |

Aucune n'a produit de prédictions > 0.5 dans nos tests rapides.

## Plan d'action recommandé

### Solution immédiate (aujourd'hui)

1. **Adapter le seuil de décision à 0.3** :
   ```python
   # Dans predict.py
   signal = "LONG" if p_up >= 0.3 else "SHORT"
   ```

2. **Ajuster le calcul de confiance** :
   ```python
   # Dans predict.py
   confidence = min(abs(p_up - 0.3) / 0.3 * 2, 1.0)
   ```

### Solutions à moyen terme (prochaine semaine)

3. **Désactiver la calibration logistique** :
   ```python
   # Dans train_lgbm.py
   CALIBRATE = False
   ```

4. **Implémenter l'approche par espérance de gain** :
   ```python
   # Dans predict.py
   tp_pct = 0.004  # 0.4% take profit
   sl_pct = 0.002  # 0.2% stop loss
   ev = p_up * tp_pct - (1 - p_up) * sl_pct
   signal = "LONG" if ev > 0 else "SHORT"
   ```

### Solutions à long terme (prochain mois)

5. **Réentraîner avec plus de complexité** :
   ```python
   # Dans train_lgbm.py
   params.update({
       "num_leaves": 127,
       "min_child_samples": 10,
       "bagging_fraction": 1.0,
       "scale_pos_weight": 1.0 / pos_rate
   })
   ```

6. **Évaluer d'autres approches de modélisation** :
   - Prédiction directe du mouvement de prix (régression)
   - Modèles d'ensemble combinant plusieurs prédicteurs
   - Approches de calibration alternatives (Platt, isotonique)

## Conclusion

Le problème des probabilités < 0.5 n'est pas un défaut du modèle mais une conséquence naturelle de la rareté de l'événement prédit et des choix de modélisation. La solution la plus simple et efficace est d'adapter le seuil de décision à 0.3, ce qui permettra d'obtenir des signaux LONG lorsque le modèle est relativement confiant, tout en maintenant une bonne précision.

L'approche par espérance de gain est également prometteuse car elle intègre directement le profil risque/récompense de la stratégie de trading, produisant des signaux plus alignés avec l'objectif final de maximiser les profits.