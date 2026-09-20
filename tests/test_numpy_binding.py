import unittest

import numpy as np
import tracker


class NumpyBindingTests(unittest.TestCase):
    def setUp(self):
        self.tracker = tracker.ByteTracker()
        self.detections = np.array([
            [10, 20, 40, 60, 0.9, 7],
            [100, 200, 40, 60, 0.8, 3],
        ], dtype=np.float32)

    def test_output_and_identity_across_frames(self):
        result = self.tracker.update(self.detections)
        self.assertEqual(result.shape, (2, 7))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[:, [0, 1, 2, 3, 5, 6]], self.detections)
        np.testing.assert_array_equal(result[:, 4], [1, 2])
        second = self.tracker.update(self.detections[::-1])
        np.testing.assert_array_equal(second[:, 4], result[:, 4])
        np.testing.assert_array_equal(second[:, 6], result[:, 6])
        self.tracker.reset()
        np.testing.assert_array_equal(self.tracker.update(self.detections)[:, 4], [1, 2])

    def test_empty_frame(self):
        self.tracker.update(self.detections)
        result = self.tracker.update(np.empty((0, 6), dtype=np.float32))
        self.assertEqual(result.shape, (0, 7))
        self.assertEqual(result.dtype, np.float32)

    def test_dtype_and_layout_conversion(self):
        backing = np.zeros((2, 12), dtype=np.float64)
        backing[:, ::2] = self.detections
        for values in (backing[:, ::2], np.asfortranarray(self.detections),
                       self.detections.astype(np.float64)):
            with self.subTest(strides=values.strides, dtype=values.dtype):
                self.tracker.reset()
                result = self.tracker.update(values)
                np.testing.assert_allclose(result[:, [0, 1, 2, 3, 5, 6]], self.detections)

    def test_invalid_shapes(self):
        for shape in ((), (6,), (2, 5), (2, 7), (0, 5), (1, 2, 6)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, r"shape \(N, 6\)"):
                    self.tracker.update(np.zeros(shape, dtype=np.float32))

    def test_invalid_values_do_not_advance_tracker(self):
        for col, value in ((5, np.nan), (5, np.inf), (5, 1.5), (5, 2**31),
                           (5, -2**32), (2, -1), (4, 1.5), (0, np.nan)):
            with self.subTest(column=col, value=value):
                values = self.detections.copy()
                values[1, col] = value
                with self.assertRaises(ValueError):
                    self.tracker.update(values)
        # Rejected frames must not consume the first-frame confirmation or IDs.
        np.testing.assert_array_equal(self.tracker.update(self.detections)[:, 4], [1, 2])


if __name__ == "__main__":
    unittest.main()
