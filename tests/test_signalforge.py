import unittest

from signalforge.decision import choose_action
from signalforge.models import model_bundle, score_models
from signalforge.monitoring import active_learning_priority, drift_report
from signalforge.service import build_report, score_event
from signalforge.synthetic import generate_events


class SignalForgeTests(unittest.TestCase):
    def test_model_quality_on_synthetic_holdout(self):
        metrics = model_bundle().metrics
        self.assertGreater(metrics["fraud_classifier"]["roc_auc"], 0.70)
        self.assertLess(metrics["fraud_classifier"]["brier_score"], 0.25)
        self.assertGreater(metrics["graph_ring"]["roc_auc"], 0.80)

    def test_score_contract_and_bounds(self):
        event = generate_events(1, 999)[0]
        result = score_event(event)
        self.assertIn(result["decision"], {"ALLOW", "REVIEW", "BLOCK"})
        self.assertGreaterEqual(result["fraud_probability"], 0.0)
        self.assertLessEqual(result["fraud_probability"], 1.0)
        self.assertGreaterEqual(result["graph_risk"], 0.0)
        self.assertLessEqual(result["graph_risk"], 1.0)
        self.assertGreaterEqual(result["anomaly_percentile"], 0.0)
        self.assertLessEqual(result["anomaly_percentile"], 100.0)

    def test_cost_policy_can_choose_each_action(self):
        self.assertEqual(choose_action(0.05, 4.0)["decision"], "ALLOW")
        self.assertEqual(choose_action(0.55, 50.0)["decision"], "REVIEW")
        self.assertEqual(choose_action(0.96, 600.0)["decision"], "BLOCK")

    def test_shifted_population_surfaces_drift(self):
        reference = generate_events(600, 120)
        shifted = generate_events(600, 121, adversarial_shift=True)
        report = drift_report(reference, shifted)
        self.assertGreater(report["max_psi"], 0.05)
        self.assertEqual(len(report["features"]), 6)

    def test_active_learning_priority_is_bounded(self):
        value = active_learning_priority(0.51, 0.78, 85.0)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_report_contains_all_decision_science_layers(self):
        report = build_report()
        self.assertIn("evaluation", report["model"])
        self.assertIn("drift", report)
        self.assertIn("cohorts", report)
        self.assertIn("graph", report)
        self.assertEqual(report["boundary"], "all metrics are generated from deterministic synthetic data")


if __name__ == "__main__":
    unittest.main()
