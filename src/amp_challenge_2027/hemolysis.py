"""Hemolysis / selectivity predictor: P(hemolytic) from physicochemical descriptors.

The competition's *Optimal Selectivity* category scores the safety window HC50/MIC50, so a
peptide that is potent **and** non-hemolytic wins there while also counting as active in the
activity categories. APEX predicts only antibacterial MIC and tends to favour hyper-cationic,
hydrophobic peptides -- exactly the hemolysis-prone kind -- so an orthogonal hemolysis signal
both opens the selectivity category and removes likely-toxic peptides from the top-100.

The model is deliberately lean: a small MLP over the 11 :mod:`physchem` descriptors, trained on
HemoPI-1 (``training/train_hemolysis.py``). Descriptor models are the established strong
baseline for hemolysis (Plisson et al. 2020), so this ships in the main environment with only
``torch`` -- no ESM/transformers -- and runs deterministically on CPU inside ``generate``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .physchem import FEATURE_NAMES, feature_matrix

DEFAULT_HEMOLYSIS_CHECKPOINT = "checkpoint/hemolysis.pt"


def build_mlp(n_features: int, hidden: int):
    """The classifier architecture, shared by training and inference."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(n_features, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )


class HemolysisScorer:
    """Predict P(hemolytic) in [0, 1] for peptides; higher = more likely to lyse red cells.

    Loads the shipped checkpoint (weights + the feature standardisation fitted on the training
    set). Deterministic: ``.eval()`` on CPU over deterministic descriptors.
    """

    name = "hemolysis-mlp"

    def __init__(self, checkpoint: str | Path = DEFAULT_HEMOLYSIS_CHECKPOINT) -> None:
        import torch

        from .paths import resolve_repo_path

        self._torch = torch
        ckpt = torch.load(resolve_repo_path(checkpoint), map_location="cpu", weights_only=False)
        if list(ckpt["feature_names"]) != list(FEATURE_NAMES):
            raise ValueError("hemolysis checkpoint feature set does not match physchem.FEATURE_NAMES")
        self._mean = np.asarray(ckpt["mean"], dtype=float)
        self._std = np.asarray(ckpt["std"], dtype=float)
        self.val_auroc = ckpt.get("val_auroc")
        self.model = build_mlp(len(FEATURE_NAMES), ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """Return P(hemolytic) for each sequence, aligned to input order."""
        if not sequences:
            return np.empty((0,), dtype=float)
        feats = (feature_matrix(sequences) - self._mean) / self._std
        with self._torch.no_grad():
            logits = self.model(self._torch.tensor(feats, dtype=self._torch.float32)).squeeze(1)
            return self._torch.sigmoid(logits).numpy()
