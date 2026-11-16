"""Reusable MLP and preprocessing helpers; importing this module never trains a model."""

import numpy as np
import pandas as pd
import torch.nn as nn


CONT_RAW = [
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILD_COLS = [f"Wilderness_Area_{i}" for i in range(4)]
SOIL_COLS = [f"Soil_Type_{i}" for i in range(40)]
RAW_FEATURES = CONT_RAW + WILD_COLS + SOIL_COLS
CONT_COLS = [c for c in CONT_RAW if c != "Aspect"] + [
    "Aspect_sin", "Aspect_cos", "Euclid_Distance_To_Hydrology",
]
FEATURES = CONT_COLS + WILD_COLS + SOIL_COLS


def engineer_features(frame):
    """Validate 54 raw UCI features and derive the 56 model inputs.

    Continuous NaNs are retained for later imputation with training medians.
    Categorical indicators must be present, binary, and one-hot within each group.
    Extra columns, including Cover_Type, are retained but never used as inputs.
    """
    missing = sorted(set(RAW_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing input columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Input must contain at least one record.")
    result = frame.copy()
    try:
        numeric = result[RAW_FEATURES].apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("All input features must be numeric.") from exc
    if np.isinf(numeric.to_numpy(dtype=float)).any():
        raise ValueError("Input features must not contain infinity.")
    # Assign columns separately so integer input columns can safely become floats.
    for col in RAW_FEATURES:
        result[col] = numeric[col]
    for name, cols in [("Wilderness_Area", WILD_COLS), ("Soil_Type", SOIL_COLS)]:
        group = numeric[cols]
        if not (group.isin([0, 1]).all(axis=1) & group.sum(axis=1).eq(1)).all():
            raise ValueError(f"{name} must contain exactly one 1 and otherwise 0 in every row.")
    ranges = {
        "Aspect": (0, 360), "Slope": (0, np.inf),
        "Hillshade_9am": (0, 255), "Hillshade_Noon": (0, 255),
        "Hillshade_3pm": (0, 255),
        **{c: (0, np.inf) for c in CONT_RAW if c.startswith("Horizontal_Distance")},
    }
    for col, (low, high) in ranges.items():
        values = numeric[col]
        if ((values < low) | (values > high)).any():
            raise ValueError(f"{col} contains values outside [{low}, {high}].")
    result["Aspect"] = numeric["Aspect"] % 360
    radians = np.deg2rad(result["Aspect"])
    result["Aspect_sin"] = np.sin(radians)
    result["Aspect_cos"] = np.cos(radians)
    result["Euclid_Distance_To_Hydrology"] = np.sqrt(
        numeric["Horizontal_Distance_To_Hydrology"] ** 2
        + numeric["Vertical_Distance_To_Hydrology"] ** 2
    )
    return result


def fit_training_medians(training_frame, columns=CONT_COLS):
    """Learn continuous-feature medians from the training split only."""
    medians = training_frame[columns].median()
    if not np.isfinite(medians.to_numpy(dtype=float)).all():
        raise ValueError("Every continuous feature needs a finite training median.")
    return medians


def impute_continuous(frame, medians):
    """Apply fitted medians without learning anything from this frame."""
    result = frame.copy()
    for col in medians.index:
        result[col] = result[col].fillna(medians[col])
    return result


ACTIVATIONS = {
    "relu": nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Fully connected classifier with optional BatchNorm and Dropout.

    Hidden layers apply Linear -> optional BatchNorm -> activation -> optional
    Dropout. The output layer produces logits for CrossEntropyLoss; inference
    applies softmax. Initialization supports He, Xavier, small normal, and the
    PyTorch default. Parameter names match the original assignment checkpoint.
    """

    def __init__(self, n_in, n_out, hidden=(64, 32), activation="relu", dropout=0.0,
                 batch_norm=False, init="he"):
        super().__init__()
        layers, prev = [], n_in
        for width in hidden:
            layers.append(nn.Linear(prev, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)
        self._init_weights(init)

    def _init_weights(self, init):
        if init == "default":
            return
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if init == "he":
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                elif init == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif init == "normal":
                    nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                else:
                    raise ValueError(f"Unknown initialization: {init}")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)
