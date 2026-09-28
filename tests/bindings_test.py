"""Run against a built extension: python tests/bindings_test.py path/to/tracker.so."""
import importlib.util
import sys
import unittest

import numpy as np

# Load the requested build even when an older editable package is installed.
if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
    spec = importlib.util.spec_from_file_location('tracker', sys.argv.pop(1))
    tracker = importlib.util.module_from_spec(spec)
    sys.modules['tracker'] = tracker
    spec.loader.exec_module(tracker)
else:
    import tracker


class BindingsTest(unittest.TestCase):
    def test_shared_interface(self):
        for implementation in (tracker.Sort, tracker.ByteTracker):
            for fmt, coords in (
                (tracker.BoxFormat.TLWH, [10, 20, 40, 60]),
                (tracker.BoxFormat.CXCYWH, [30, 50, 40, 60]),
                (tracker.BoxFormat.XYXY, [10, 20, 50, 80]),
            ):
                with self.subTest(implementation=implementation, format=fmt):
                    model = implementation(input_format=fmt, max_lost_frames=1)
                    self.assertIsInstance(model, tracker.Tracker)
                    self.assertEqual(model.input_format, fmt)
                    detections = np.array([coords + [0.9, 7]], dtype=np.float64)
                    output = model.update(detections)
                    self.assertEqual(output.dtype, np.float32)
                    np.testing.assert_allclose(output, [[10, 20, 40, 60, 1, 0.9, 7]])
                    self.assertEqual(model.update(np.empty((0, 6))).shape, (0, 7))
                    self.assertEqual(model.update(detections)[0, 4], 1)
                    model.reset()
                    np.testing.assert_allclose(model.update(detections), output)
                    with self.assertRaises(AttributeError):
                        model.input_format = tracker.BoxFormat.TLWH

    def test_constructor_parameters(self):
        tracker.ByteTracker(low_confidence=0.2, high_confidence=0.5,
                            new_track_confidence=0.6, first_match_cost=0.9,
                            second_match_cost=0.4, tentative_match_cost=0.6, threads=1)
        model = tracker.Sort(max_match_cost=0, max_lost_frames=0)
        model.update([[0, 0, 20, 20, 0.01, 1]])
        self.assertEqual(model.update([[10, 0, 20, 20, 0.01, 1]]).shape, (0, 7))
        for implementation in (tracker.Sort, tracker.ByteTracker):
            with self.assertRaises(TypeError):
                implementation(0.7)
            with self.assertRaises(TypeError):
                implementation(config={})
            with self.assertRaises(ValueError):
                implementation(max_lost_frames=-1)
        for name in ('low_confidence', 'high_confidence', 'new_track_confidence',
                     'first_match_cost', 'second_match_cost', 'tentative_match_cost'):
            with self.subTest(parameter=name), self.assertRaises(ValueError):
                tracker.ByteTracker(**{name: float('nan')})
        with self.assertRaises(ValueError):
            tracker.ByteTracker(low_confidence=0.6, high_confidence=0.6)
        with self.assertRaises(ValueError):
            tracker.ByteTracker(threads=-1)
        with self.assertRaises(ValueError):
            tracker.Sort(max_match_cost=float('nan'))

    def test_input_validation_is_shared(self):
        for implementation in (tracker.Sort, tracker.ByteTracker):
            model = implementation()
            for invalid in (np.empty((0, 5)), np.zeros((6,)),
                            [[0, 0, 20, 20, 0.9, 1.5]], [[0, 0, -1, 20, 0.9, 1]]):
                with self.subTest(implementation=implementation), self.assertRaises(ValueError):
                    model.update(invalid)
            # Failed validation must not advance the frame or consume an ID.
            self.assertEqual(model.update([[0, 0, 20, 20, 0.9, 1]])[0, 4], 1)
            model.reset()
            noncontiguous = np.array([[0, 0, 20, 20, 0.9, 1]] * 4)[::2]
            self.assertFalse(noncontiguous.flags.c_contiguous)
            self.assertEqual(model.update(noncontiguous).shape, (2, 7))

    def test_public_names(self):
        self.assertFalse(hasattr(tracker, 'SortConfig'))
        self.assertFalse(hasattr(tracker, 'ByteTrackerConfig'))
        box = tracker.Detection(x=0, y=0, width=20, height=20, confidence=0.4, class_id=7)
        self.assertEqual(box.confidence, 0.4)
        self.assertEqual(box.class_id, 7)


if __name__ == '__main__':
    unittest.main()
