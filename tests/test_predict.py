import unittest

import pandas as pd

from app import app, feature_names, pipeline


class PredictContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()
        cls.sample = pd.read_csv("data.csv")[feature_names].iloc[0].to_dict()

    def test_bad_probability_and_threshold(self):
        frame = pd.DataFrame([self.sample], columns=feature_names)
        bad_column = list(pipeline.classes_).index(0)
        expected = float(pipeline.predict_proba(frame)[0, bad_column])
        for threshold, decision in ((0, "REJECT"), (1, "APPROVE")):
            with self.subTest(threshold=threshold):
                response = self.client.post(
                    "/predict", json={**self.sample, "threshold": threshold}
                )
                self.assertEqual(response.status_code, 200)
                self.assertAlmostEqual(response.json["proba_bad"], expected, places=6)
                self.assertEqual(response.json["decision"], decision)

    def test_invalid_request_returns_400(self):
        for payload in ([], {}, {**self.sample, "threshold": "x"},
                        {**self.sample, "threshold": 2},
                        {**self.sample, feature_names[0]: "x"}):
            with self.subTest(payload=payload):
                self.assertEqual(self.client.post("/predict", json=payload).status_code, 400)


if __name__ == "__main__":
    unittest.main()
