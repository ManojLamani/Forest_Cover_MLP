"""Checks for train-only imputation, input validation, and saved-model inference."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from forest_cover import (
    CONT_COLS, FEATURES, SOIL_COLS, WILD_COLS, engineer_features,
    fit_training_medians, impute_continuous,
)
from predict import ForestCoverPredictor


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.raw = pd.read_csv(ROOT / "examples" / "sample_input.csv")
        cls.predictor = ForestCoverPredictor(ROOT / "models" / "final_mlp.pt")

    def test_train_only_imputation_does_not_use_validation_distribution(self):
        training = engineer_features(self.raw.iloc[:3]).copy()
        training["Elevation"] = [100.0, 300.0, np.nan]
        medians = fit_training_medians(training)
        validation = engineer_features(self.raw.iloc[:2]).copy()
        validation["Elevation"] = [100000.0, np.nan]
        filled = impute_continuous(validation, medians)
        self.assertEqual(medians["Elevation"], 200.0)
        self.assertEqual(filled["Elevation"].iloc[1], 200.0)
        self.assertTrue(pd.isna(validation["Elevation"].iloc[1]))

    def test_all_missing_training_column_is_rejected(self):
        training = engineer_features(self.raw)
        training["Elevation"] = np.nan
        with self.assertRaisesRegex(ValueError, "training median"):
            fit_training_medians(training)

    def test_feature_engineering_wraps_aspect_and_keeps_signed_vertical_distance(self):
        raw = self.raw.iloc[:2].copy()
        raw["Aspect"] = [0.0, 360.0]
        raw["Horizontal_Distance_To_Hydrology"] = 3.0
        raw["Vertical_Distance_To_Hydrology"] = -4.0
        features = engineer_features(raw)
        np.testing.assert_allclose(features["Aspect_sin"], 0.0, atol=1e-12)
        np.testing.assert_allclose(features["Aspect_cos"], 1.0)
        np.testing.assert_allclose(features["Euclid_Distance_To_Hydrology"], 5.0)

    def test_missing_input_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing input columns: Elevation"):
            self.predictor.predict(self.raw.drop(columns="Elevation"))

    def test_invalid_categorical_indicators_are_rejected(self):
        for columns in [WILD_COLS, SOIL_COLS]:
            for invalid in [0.0, np.nan, 0.5, 1.0]:
                with self.subTest(group=columns[0], invalid=invalid):
                    raw = self.raw.copy()
                    raw.loc[0, columns] = invalid
                    with self.assertRaisesRegex(ValueError, "exactly one 1"):
                        self.predictor.predict(raw)

    def test_invalid_numeric_values_are_rejected(self):
        for column, value in [("Elevation", np.inf), ("Aspect", 361.0),
                              ("Hillshade_9am", 256.0), ("Slope", -1.0),
                              ("Horizontal_Distance_To_Hydrology", -1.0),
                              ("Elevation", "not-a-number")]:
            with self.subTest(column=column, value=value):
                raw = self.raw.copy()
                raw[column] = raw[column].astype(object)
                raw.loc[0, column] = value
                with self.assertRaises(ValueError):
                    self.predictor.predict(raw)

    def test_missing_continuous_values_use_saved_statistics(self):
        raw = self.raw.copy()
        raw.loc[0, ["Elevation", "Aspect", "Horizontal_Distance_To_Hydrology"]] = np.nan
        matrix = self.predictor.transform(raw)
        self.assertTrue(np.isfinite(matrix).all())
        checkpoint = torch.load(ROOT / "models" / "final_mlp.pt", weights_only=True)
        for column in ["Elevation", "Aspect_sin", "Aspect_cos", "Euclid_Distance_To_Hydrology"]:
            i = CONT_COLS.index(column)
            value = np.clip(checkpoint["imputer_median"][i],
                            checkpoint["clip_low"][i], checkpoint["clip_high"][i])
            expected = (value - checkpoint["scaler_mean"][i]) / checkpoint["scaler_scale"][i]
            self.assertAlmostEqual(float(matrix[0, i]), expected, places=6)

    def test_predictions_are_valid_and_independent_of_batching_and_column_order(self):
        result = self.predictor.predict(self.raw, batch_size=1)
        reordered = self.predictor.predict(self.raw[self.raw.columns[::-1]], batch_size=4)
        np.testing.assert_array_equal(result["cover_type"], reordered["cover_type"])
        probs = result.filter(like="probability_").to_numpy()
        np.testing.assert_allclose(probs, reordered.filter(like="probability_"), atol=1e-6)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(result["cover_type"].between(1, 7).all())
        np.testing.assert_allclose(result["confidence"], probs.max(axis=1))
        self.assertEqual(self.predictor.transform(self.raw).shape, (len(self.raw), len(FEATURES)))

    def test_optional_target_is_ignored(self):
        raw = self.raw.copy()
        raw["Cover_Type"] = 999
        np.testing.assert_array_equal(self.predictor.transform(raw), self.predictor.transform(self.raw))

    def test_empty_input_and_nonpositive_batch_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one record"):
            self.predictor.predict(self.raw.iloc[:0])
        with self.assertRaisesRegex(ValueError, "Batch size must be positive"):
            self.predictor.predict(self.raw, batch_size=0)

    def test_cli_writes_predictions_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.csv"
            completed = subprocess.run(
                [sys.executable, str(ROOT / "predict.py"),
                 str(ROOT / "examples" / "sample_input.csv"), "--output", str(output)],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = pd.read_csv(output)
            self.assertEqual(len(result), len(self.raw))
            self.assertIn("probability_7", result.columns)

    def test_cli_reports_input_errors_without_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            bad_input = Path(directory) / "bad.csv"
            output = Path(directory) / "predictions.csv"
            self.raw.drop(columns="Elevation").to_csv(bad_input, index=False)
            completed = subprocess.run(
                [sys.executable, str(ROOT / "predict.py"), str(bad_input), "--output", str(output)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Missing input columns: Elevation", completed.stderr)
            self.assertFalse(output.exists())

    def test_cli_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.csv"
            self.raw.to_csv(input_path, index=False)
            original = input_path.read_bytes()
            completed = subprocess.run(
                [sys.executable, str(ROOT / "predict.py"), str(input_path),
                 "--output", str(input_path)], capture_output=True, text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("Output must be different", completed.stderr)
            self.assertEqual(input_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
