import sys
from pathlib import Path
import numpy as np

# Veriyi yukle (repo kokunden veya lab klasorunden calistirilinca da calisir)
csv_path = Path(__file__).resolve().parent.parent / "data" / "ccpp.csv"
data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
cols = ["AT", "V", "AP", "RH", "PE"]

# --- SECTION 3: Data-quality checks ---
n_rows, n_cols = data.shape
missing = int(np.isnan(data).sum())
duplicates = n_rows - len(np.unique(data, axis=0))

print("=== SECTION 3: DATA-QUALITY CHECKS ===")
print(f"Rows: {n_rows}, Columns: {n_cols}")
print(f"Missing values: {missing}")
print(f"Exact duplicate rows: {duplicates}")
for i, name in enumerate(cols):
    print(f"{name}: min={data[:, i].min():.2f}, max={data[:, i].max():.2f}")

# --- SECTION 4: Summary statistics and correlation ---
print("\n=== SECTION 4: SUMMARY STATISTICS & CORRELATION ===")
for name, idx in [("AT", 0), ("PE", 4)]:
    col = data[:, idx]
    print(f"{name}: mean={np.mean(col):.2f}, median={np.median(col):.3f}, std={np.std(col):.2f}")

pe = data[:, 4]
for i, name in enumerate(cols[:4]):
    r = np.corrcoef(data[:, i], pe)[0, 1]
    print(f"Corr({name}, PE): {r:.3f}")

# --- SECTION 5: Models and test error (80/20 split, keep all rows) ---
split = int(0.8 * n_rows)
train = data[:split]
test = data[split:]
print("\n=== SECTION 5: MODELS & TEST ERROR ===")
print(f"Split (80/20): train={len(train)}, test={len(test)}")

# 5.1 Baseline
train_mean_pe = np.mean(train[:, 4])
baseline_mae = np.mean(np.abs(test[:, 4] - train_mean_pe))
print(f"Baseline — train mean PE: {train_mean_pe:.2f} MW, test MAE: {baseline_mae:.2f} MW")

# Referans: AT dogrusu
a_at, b_at = np.polyfit(train[:, 0], train[:, 4], 1)
at_mae = np.mean(np.abs(test[:, 4] - (a_at * test[:, 0] + b_at)))
print(f"Line from AT: a={a_at:.4f}, b={b_at:.4f}, test MAE: {at_mae:.2f} MW")

# 5.2 Secilen ikinci ozellik: V (Exhaust Vacuum, cm Hg)
a_v, b_v = np.polyfit(train[:, 1], train[:, 4], 1)
v_mae = np.mean(np.abs(test[:, 4] - (a_v * test[:, 1] + b_v)))
print(f"Line from V: a={a_v:.4f}, b={b_v:.4f}, test MAE: {v_mae:.2f} MW")

# 5.5 Elle kontrol (test setinin ilk satiri: dosya satiri 7656)
row_idx = split
file_line = row_idx + 2  # baslik satir 1, ilk veri satir 2
x_val = data[row_idx, 1]
true_pe = data[row_idx, 4]
pred_pe = a_v * x_val + b_v
abs_err = abs(true_pe - pred_pe)
print(f"\n=== 5.5 HAND CHECK (FILE LINE {file_line}) ===")
print(f"V={x_val:.2f} cm Hg, Predicted PE={pred_pe:.2f} MW, True PE={true_pe:.2f} MW, Abs error={abs_err:.2f} MW")

# 6.4 Yazilim surumleri
print("\n=== VERSIONS ===")
print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")