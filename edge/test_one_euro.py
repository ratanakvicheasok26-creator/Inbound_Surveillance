"""Unit tests for OneEuroFilter and KeypointStabilizer."""

from __future__ import annotations

import unittest
import numpy as np

from one_euro import LowPassFilter, OneEuroFilter, KeypointStabilizer


class OneEuroFilterTests(unittest.TestCase):
    def test_low_pass_filter_basic(self):
        lpf = LowPassFilter(alpha=0.5)
        self.assertEqual(lpf.filter(10.0), 10.0)
        self.assertEqual(lpf.filter(20.0), 15.0)
        lpf.reset()
        self.assertIsNone(lpf.s)

    def test_stationary_noise_reduction(self):
        """Stationary noisy signal should have significantly lower variance after filtering."""
        np.random.seed(42)
        base_val = 100.0
        noise = np.random.normal(0.0, 2.0, size=100)
        raw_vals = [base_val + float(n) for n in noise]

        filter_oe = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.01)
        filtered_vals = []
        for i, val in enumerate(raw_vals):
            filtered_vals.append(filter_oe.filter(val, timestamp=i * (1.0 / 30.0)))

        raw_var = float(np.var(raw_vals))
        filt_var = float(np.var(filtered_vals[10:]))  # skip initial settling
        self.assertLess(filt_var, raw_var * 0.35, f"Expected >65% variance drop, got raw={raw_var:.2f}, filt={filt_var:.2f}")

    def test_step_response_low_lag(self):
        """A 40px sudden step should adapt quickly (within <= 2 frames)."""
        filter_oe = OneEuroFilter(freq=30.0, min_cutoff=0.5, beta=0.01)
        # Settle at 100
        for i in range(10):
            filter_oe.filter(100.0, timestamp=i * (1.0 / 30.0))

        # Sudden jump to 140 at frame 10
        f10 = filter_oe.filter(140.0, timestamp=10 * (1.0 / 30.0))
        f11 = filter_oe.filter(140.0, timestamp=11 * (1.0 / 30.0))
        f12 = filter_oe.filter(140.0, timestamp=12 * (1.0 / 30.0))

        # Within 2 frames after the step, the filter should reach >75% of the step (i.e. > 130)
        self.assertGreater(f12, 130.0, f"Expected step to settle near 140 within 2 frames, got f12={f12:.1f}")


class KeypointStabilizerTests(unittest.TestCase):
    def test_stabilizer_filters_17_keypoints(self):
        stabilizer = KeypointStabilizer(num_keypoints=17, conf_floor=0.15)
        raw_kpts = [(100.0 + i, 200.0 + i, 0.8) for i in range(17)]
        out = stabilizer.filter_keypoints(raw_kpts, timestamp=0.0)
        self.assertEqual(len(out), 17)
        for i in range(17):
            self.assertAlmostEqual(out[i][0], raw_kpts[i][0], places=3)
            self.assertAlmostEqual(out[i][1], raw_kpts[i][1], places=3)
            self.assertEqual(out[i][2], raw_kpts[i][2])

    def test_stabilizer_skips_low_confidence_joints(self):
        stabilizer = KeypointStabilizer(num_keypoints=17, conf_floor=0.15)
        raw_kpts = [(100.0, 200.0, 0.10)] * 17  # conf=0.10 < 0.15
        # Feed first frame
        stabilizer.filter_keypoints(raw_kpts, timestamp=0.0)
        # Feed jump on low conf joint
        raw_jump = [(200.0, 300.0, 0.10)] * 17
        out = stabilizer.filter_keypoints(raw_jump, timestamp=0.033)
        # Should NOT filter or smooth; should pass raw coordinates directly
        self.assertEqual(out[0][0], 200.0)
        self.assertEqual(out[0][1], 300.0)
        self.assertEqual(out[0][2], 0.10)


if __name__ == "__main__":
    unittest.main()
