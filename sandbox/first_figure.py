from pathlib import Path

import matplotlib.pyplot as plt

from crosspeak import asynchronous, plot_contour, read_series, synchronous

DATA_DIR = Path.home() / "crosspeak" / "sandbox" / "des"

files = {
    0: DATA_DIR / "ma50w_0_original.csv",
    1: DATA_DIR / "ma50w_1_original.csv",
    2: DATA_DIR / "ma50w_2_original.csv",
    5: DATA_DIR / "ma50w_5_original.csv",
    10: DATA_DIR / "ma50w_10_original.csv",
    15: DATA_DIR / "ma50w_15_original.csv",
    20: DATA_DIR / "ma50w_20_original.csv",
    30: DATA_DIR / "ma50w_30_original.csv",
}

s = read_series(files, name="MA50W", skiprows=1)

phi = synchronous(s)
psi = asynchronous(s)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
plot_contour(phi, s.wavenumbers, ax=axes[0], title="MA50W synchronous")
plot_contour(psi, s.wavenumbers, ax=axes[1], title="MA50W asynchronous")
plt.savefig("ma50w_first_figure.png", dpi=300, bbox_inches="tight")
