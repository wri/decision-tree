# ── visuals_config.py ──

import seaborn as sns

# 1) define your canonical decision order
DECISION_ORDER = [
    "strong remote",
    "weak remote",
    "strong field",
    "weak field",
    "review required",
    "mangrove",
]

# 2) pull exactly that many colors from Set2
_PALETTE = sns.color_palette("Set2", n_colors=len(DECISION_ORDER))

# 3) build a dict mapping each decision → its color
DECISION_COLORS = dict(zip(DECISION_ORDER, _PALETTE))
