from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path(__file__).resolve().parent / "analysis_results"
OUTPUT_DIR.mkdir(exist_ok=True)

OUT_PATH = OUTPUT_DIR / "sleep_model_hypnogram_plot.png"


# Replace these example values with your real sleep-stage data.
# Common coding:
# Wake = 0, N1 = 1, N2 = 2, N3 = 3, REM = 4
time_minutes = np.arange(0, 120, 5)
sleep_stage = np.array([
    0, 0, 1, 2, 2, 3, 3, 2, 2, 4, 4, 2,
    3, 3, 2, 4, 4, 2, 1, 0, 0, 1, 2, 2,
])

stage_labels = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}


fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

ax.step(time_minutes, sleep_stage, where="post", color="#2E86AB", linewidth=2)
ax.scatter(time_minutes, sleep_stage, color="#2E86AB", s=18)

ax.set_title("Sleep Model Hypnogram", fontsize=14, fontweight="bold")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Sleep stage")
ax.set_yticks(list(stage_labels.keys()))
ax.set_yticklabels(list(stage_labels.values()))
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)

plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PATH}")
