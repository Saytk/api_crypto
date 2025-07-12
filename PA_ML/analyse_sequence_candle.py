import pandas as pd
import numpy as np
from collections import Counter

# === Paramètres ===
CSV_PATH = "BTCUSDT_labeled_candles.csv"
MAX_SEQ_LEN = 3
TARGET_HORIZON = 5  # minutes après la fin de la séquence
TOLERANCE_PCT = 0.1  # seuil neutre (ex: ±0.1%)

# === Charger les données ===
df = pd.read_csv(CSV_PATH)
df = df.sort_values("timestamp_utc").reset_index(drop=True)

# === Générer les séquences de 1 à 3 bougies ===
records = []
candle_ids = df['candle_type'].tolist()
close_prices = df['close'].tolist()

for i in range(len(df) - TARGET_HORIZON):
    for seq_len in range(1, MAX_SEQ_LEN + 1):
        end = i + seq_len
        if end + TARGET_HORIZON >= len(df):
            continue

        seq = tuple(candle_ids[i:end])
        now_price = close_prices[end - 1]
        fut_price = close_prices[end - 1 + TARGET_HORIZON]
        change_pct = (fut_price - now_price) / now_price

        if change_pct > TOLERANCE_PCT / 100:
            label = 1
        elif change_pct < -TOLERANCE_PCT / 100:
            label = -1
        else:
            label = 0

        records.append((seq, label))

# === Statistiques : quelles séquences mènent à quoi ? ===
stats = {}
for seq, label in records:
    if seq not in stats:
        stats[seq] = Counter()
    stats[seq][label] += 1

# === Convertir en DataFrame ===
rows = []
for seq, counter in stats.items():
    total = sum(counter.values())
    bullish = counter[1]
    bearish = counter[-1]
    neutral = counter[0]
    rows.append({
        'sequence': seq,
        'total': total,
        'bullish': bullish,
        'bearish': bearish,
        'neutral': neutral,
        'bullish_ratio': bullish / total,
        'bearish_ratio': bearish / total,
        'neutral_ratio': neutral / total,
        'bias': (bullish - bearish) / total
    })

result_df = pd.DataFrame(rows).sort_values(by='total', ascending=False)

# === Afficher les séquences les plus fréquentes et les plus biaisées ===
pd.set_option('display.max_rows', 50)
print("\nTop séquences par fréquence :")
print(result_df.head(20)[['sequence', 'total', 'bullish_ratio', 'bearish_ratio', 'bias']])

print("\nTop séquences haussières :")
print(result_df.sort_values(by='bias', ascending=False).head(20)[['sequence', 'total', 'bullish_ratio', 'bias']])

print("\nTop séquences baissières :")
print(result_df.sort_values(by='bias').head(20)[['sequence', 'total', 'bearish_ratio', 'bias']])
# === Paramètres du filtre intelligent ===
MIN_OCCURRENCES = 30     # filtre les séquences trop rares
MIN_BIAS = 0.1           # au moins 10 % de biais net

# === Filtrer les séquences puissantes ===
filtered_df = result_df[
    (result_df['total'] >= MIN_OCCURRENCES) &
    (result_df['bias'].abs() >= MIN_BIAS)
].copy()

# === Trier selon la force du biais (positif ou négatif) ===
filtered_df = filtered_df.sort_values(by='bias', ascending=False)

# === Affichage final ===
print(f"\n🧠 Séquences directionnelles solides (≥{MIN_OCCURRENCES} cas, |bias| ≥ {MIN_BIAS}):")
print(filtered_df[['sequence', 'total', 'bullish_ratio', 'bearish_ratio', 'bias']].head(30))
filtered_df.to_csv("patterns_significatifs.csv", index=False)
print("📦 Exporté vers patterns_significatifs.csv")
