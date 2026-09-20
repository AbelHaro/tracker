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

    def test_box_formats_produce_equivalent_tracks(self):
        # Same two boxes, including negative coordinates, in all three formats.
        boxes_by_format = (
            (tracker.BoxFormat.TLWH, [[-40, -30, 20, 10], [100, 200, 40, 60]]),
            (tracker.BoxFormat.CXCYWH, [[-30, -25, 20, 10], [120, 230, 40, 60]]),
            (tracker.BoxFormat.XYXY, [[-40, -30, -20, -20], [100, 200, 140, 260]]),
        )
        canonical = self.detections.copy()
        canonical[:, :4] = boxes_by_format[0][1]
        for box_format, coordinates in boxes_by_format:
            with self.subTest(format=box_format):
                config = tracker.ByteTrackerConfig()
                config.inputFormat = box_format
                engine = tracker.ByteTracker(config)
                baseline = tracker.ByteTracker()
                values = self.detections.copy()
                values[:, :4] = coordinates
                original = values.copy()
                first = engine.update(values)
                np.testing.assert_allclose(first, baseline.update(canonical))
                np.testing.assert_array_equal(values, original)
                # Motion and reordering must produce identical associations and Kalman updates.
                moved = canonical.copy()
                moved[:, :2] += [1, 2]
                values[:, :2] += [1, 2]
                if box_format == tracker.BoxFormat.XYXY:
                    values[:, 2:4] += [1, 2]
                np.testing.assert_allclose(engine.update(values[::-1]), baseline.update(moved[::-1]))
                empty = np.empty((0, 6), dtype=np.float32)
                self.assertEqual(engine.update(empty).shape, (0, 7))
                baseline.update(empty)
                np.testing.assert_allclose(engine.update(values), baseline.update(moved))
                engine.reset()
                np.testing.assert_allclose(engine.update(original), first)

    def test_input_format_is_fixed_at_construction(self):
        self.assertEqual(self.tracker.inputFormat, tracker.BoxFormat.TLWH)
        config = tracker.ByteTrackerConfig()
        config.inputFormat = tracker.BoxFormat.XYXY
        engine = tracker.ByteTracker(config)
        config.inputFormat = tracker.BoxFormat.CXCYWH
        engine.reset()
        self.assertEqual(engine.inputFormat, tracker.BoxFormat.XYXY)
        with self.assertRaises(AttributeError):
            engine.inputFormat = tracker.BoxFormat.TLWH

    def test_invalid_boxes_in_each_format_do_not_advance_tracker(self):
        for box_format, valid_box, invalid_box in (
            (tracker.BoxFormat.TLWH, [10, 20, 40, 60], [10, 20, 0, 60]),
            (tracker.BoxFormat.CXCYWH, [30, 50, 40, 60], [30, 50, 40, -1]),
            (tracker.BoxFormat.XYXY, [10, 20, 50, 80], [10, 20, 9, 80]),
            (tracker.BoxFormat.XYXY, [10, 20, 50, 80], [10, 20, 50, 20]),
        ):
            with self.subTest(format=box_format, invalid_box=invalid_box):
                config = tracker.ByteTrackerConfig()
                config.inputFormat = box_format
                engine = tracker.ByteTracker(config)
                valid = np.array([[*valid_box, .9, 7]], dtype=np.float32)
                invalid = np.array([[*invalid_box, .9, 7]], dtype=np.float32)
                rejected = [invalid]
                for column in range(4):
                    for value in (np.nan, np.inf, -np.inf):
                        bad = valid.copy()
                        bad[0, column] = value
                        rejected.append(bad)
                for bad in rejected:
                    with self.assertRaises(ValueError):
                        engine.update(np.concatenate([valid, bad]))
                first = engine.update(valid)
                np.testing.assert_allclose(first[0], [10, 20, 40, 60, 1, .9, 7])

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
