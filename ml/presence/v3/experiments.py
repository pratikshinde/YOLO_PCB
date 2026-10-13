"""Experiment definitions. Every experiment = 5-fold leave-boards-out cross-validation (+ a final model on all boards when asked)."""
BASE_AUG = dict(fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=5, translate=0.04, scale=0.15, erasing=0.0)
STRONG_AUG = dict(fliplr=0.5, flipud=0.0, hsv_h=0.02, hsv_s=0.8, hsv_v=0.7, degrees=10, translate=0.06, scale=0.25, erasing=0.0)

_BASE = dict(model="yolo11n-cls.pt", margin=1.5, epochs=20, aug=BASE_AUG,
             jitter_frac=0.20, jitter_max=0.6,         # present crops with the window moved 0 - jitter_max mm (tolerance)
             syn_ratio=1.0, syn_min=1.3, syn_max=2.5,  # missing crops: window moved syn_min - syn_max mm, count = syn_ratio x real missing
             degrade_frac=0.0,                         # share of real training crops duplicated with random blur / exposure / noise / JPEG
             exclude_missing_types=[])                 # part types whose REAL missing crops are removed from training (transfer test)
NON_R = ["R2K2", "C100N", "D4148", "C", "D", "Q", "U"]

EXPS = {
    "E0_base": dict(_BASE),                                                   # = Antigravity v2 settings, corrected labels, 20 epochs
    "E1_small": dict(_BASE, model="yolo11s-cls.pt"),                          # bigger network
    "E2_margin25": dict(_BASE, margin=2.5),                                   # more context around the part
    "E3_strongaug": dict(_BASE, aug=STRONG_AUG),                              # stronger light / geometry augmentation
    "E4_nosyn": dict(_BASE, syn_ratio=0.0),                                   # ablation: no shifted-window 'missing' crops
    "E5_tol09": dict(_BASE, jitter_max=0.9, syn_min=1.5, syn_max=2.6),        # explicitly teach "up to 0.9 mm off is present"
    "E6_degrade": dict(_BASE, degrade_frac=0.3),                              # blur / exposure / noise / JPEG copies of the training crops
}
EXPS["E8_norm"] = dict(_BASE, colornorm=True)                                  # per-photo colour / exposure normalisation before cutting crops
EXPS["E11_expnorm_strong"] = dict(_BASE, colornorm="exp", aug=STRONG_AUG)       # exposure normalisation + strong augmentation (E10 + E3)
EXPS["E10_expnorm"] =dict(_BASE, colornorm="exp")                             # exposure-only normalisation (one gain per photo, no colour cast)
EXPS["E9_norm_degrade"] =dict(_BASE, colornorm=True, degrade_frac=0.3)        # normalisation + degraded copies
EXPS["E7_combo"] = dict(EXPS["E6_degrade"], aug=STRONG_AUG, jitter_max=0.9, syn_min=1.5, syn_max=2.6)   # E3 + E5 + E6 together

# TRANSFER TEST: same settings, but the training data contains real missing crops of resistors (R) ONLY. Recall on the missing capacitors,
# diodes and transistors of the held-out boards then shows how well "empty pads" transfers to part types that were never removed.
for _k in ("E0_base", "E2_margin25", "E3_strongaug", "E6_degrade", "E7_combo"):
    EXPS["T" + _k[1:]] = dict(EXPS[_k], exclude_missing_types=NON_R)

# order of the queue (most promising first)
QUEUE = ["E7_combo", "E6_degrade", "E3_strongaug", "E5_tol09", "E2_margin25", "E1_small", "E4_nosyn",
         "T0_base", "T7_combo", "T3_strongaug", "T6_degrade", "T2_margin25"]
