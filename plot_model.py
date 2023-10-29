#!/usr/bin/env python3
"""Util module for printing JKTEBOP output phase folded model."""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

out_file = Path.cwd() / "staging/zeta_phe/zeta_phe_s0002.out"

# We now need to load the data
df = pd.read_csv(out_file, names=["time", "mag", "mag_err", "phase", "model_mag", "model_res"], header=None, comment="#", delim_whitespace=True)
df["phase"][df["phase"]>0.75] -= 1.

matplotlib.use("TkAgg")
fig = plt.figure(figsize=(6, 4), constrained_layout=True)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df["phase"], df["model_mag"], marker=".", s=.05)
ax.invert_yaxis()

ax.set(ylabel="Relative magnitude [mag]", xlabel="Orbital Phase")

ax.set_xticks(np.arange(-.25, .8, 0.25))
ax.tick_params(axis="both", direction="in", top=True, bottom=True, left=True, right=True)

ax.vlines(x=[.0, .5], ymin=np.min(ax.get_ylim()), ymax=np.max(ax.get_ylim()), color="k", linestyle="--", linewidth=0.5, alpha=0.5)


plt.savefig(f"{out_file}.mdl.png", dpi=300)