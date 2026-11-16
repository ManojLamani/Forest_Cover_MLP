"""Predict forest cover types from a CSV of the 54 original UCI input columns."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

from forest_cover import CONT_COLS, FEATURES, MLP, engineer_features, impute_continuous


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "final_mlp.pt"


class ForestCoverPredictor:
    """Restore a saved model and apply its fitted preprocessing to new records."""

    def __init__(self, model_path=DEFAULT_MODEL):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2:
            raise ValueError("Expected checkpoint format 2; regenerate it with the current notebook.")
        if checkpoint["features"] != FEATURES or checkpoint["cont_cols"] != CONT_COLS:
            raise ValueError("Checkpoint feature order does not match the input pipeline.")
        self.class_names = checkpoint["class_names"]
        self.medians = pd.Series(checkpoint["imputer_median"], index=CONT_COLS)
        self.low = np.asarray(checkpoint["clip_low"], dtype=float)
        self.high = np.asarray(checkpoint["clip_high"], dtype=float)
        self.mean = np.asarray(checkpoint["scaler_mean"], dtype=float)
        self.scale = np.asarray(checkpoint["scaler_scale"], dtype=float)
        cfg = checkpoint["config"]
        self.model = MLP(
            len(FEATURES), len(self.class_names), hidden=cfg["hidden"],
            activation=cfg["activation"], dropout=cfg["dropout"],
            batch_norm=cfg["batch_norm"], init=cfg["init"],
        )
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    def transform(self, frame):
        """Engineer, impute, clip, and scale features using saved training statistics."""
        engineered = impute_continuous(engineer_features(frame), self.medians)
        continuous = engineered[CONT_COLS].to_numpy(dtype=float)
        continuous = (np.clip(continuous, self.low, self.high) - self.mean) / self.scale
        binary = engineered[FEATURES[len(CONT_COLS):]].to_numpy(dtype=float)
        matrix = np.hstack([continuous, binary]).astype(np.float32)
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessed inputs must be finite.")
        return matrix

    def predict(self, frame, batch_size=8192):
        """Return UCI labels (1-7), names, confidence, and probabilities in row order."""
        if batch_size < 1:
            raise ValueError("Batch size must be positive.")
        matrix = self.transform(frame)
        batches = []
        with torch.inference_mode():
            for start in range(0, len(matrix), batch_size):
                logits = self.model(torch.from_numpy(matrix[start:start + batch_size]))
                batches.append(torch.softmax(logits, dim=1).numpy())
        probabilities = np.concatenate(batches)
        indices = probabilities.argmax(axis=1)
        result = pd.DataFrame({
            "cover_type": indices + 1,
            "class_name": [self.class_names[i] for i in indices],
            "confidence": probabilities.max(axis=1),
        })
        for i in range(len(self.class_names)):
            result[f"probability_{i + 1}"] = probabilities[:, i]
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV with the 54 named UCI input features")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Saved model checkpoint")
    parser.add_argument("--output", type=Path, help="Output CSV (default: standard output)")
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args(argv)
    try:
        if args.output and args.output.resolve() in {args.input.resolve(), args.model.resolve()}:
            raise ValueError("Output must be different from the input CSV and model checkpoint.")
        torch.set_num_threads(1)
        frame = pd.read_csv(args.input)
        result = ForestCoverPredictor(args.model).predict(frame, args.batch_size)
        if args.output:
            result.to_csv(args.output, index=False)
        else:
            result.to_csv(sys.stdout, index=False)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
