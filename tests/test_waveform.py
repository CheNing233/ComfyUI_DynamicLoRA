import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "waveform.py"
spec = importlib.util.spec_from_file_location("xcn_waveform", MODULE_PATH)
waveform = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(waveform)


class WaveformTests(unittest.TestCase):
    def test_oscillation_square_is_discrete(self):
        values = waveform.generate_oscillation_schedule(8, "square", x_cycles=4, y_offset=0.5, amplitude=0.5)
        self.assertEqual(values, (1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0))

    def test_oscillation_active_window_is_zero_outside(self):
        values = waveform.generate_oscillation_schedule(6, "cosine", x_cycles=1, start_step=2, end_step=5)
        self.assertEqual(values[0], 0.0)
        self.assertEqual(values[-1], 0.0)
        self.assertEqual(len(values), 6)

    def test_monotonic_curves_hold_left_and_right_limits(self):
        values = waveform.generate_monotonic_schedule(6, "linear", 0.2, 0.8, start_step=2, end_step=5)
        self.assertEqual(values, (0.2, 0.2, 0.4, 0.6, 0.8, 0.8))
        self.assertEqual(waveform.generate_monotonic_schedule(5, "cosine", 0.0, 1.0), (0.0, 0.146447, 0.5, 0.853553, 1.0))

    def test_monotonic_curve_names(self):
        for curve in ("linear", "cosine", "exponential", "logarithmic"):
            values = waveform.generate_monotonic_schedule(7, curve)
            self.assertEqual(len(values), 7)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in values))

    def test_oscillation_rejects_bad_window(self):
        with self.assertRaisesRegex(ValueError, "start_step"):
            waveform.generate_oscillation_schedule(4, "square", start_step=4, end_step=2)

    def test_format_and_parse(self):
        text = waveform.format_schedule((1.0, 0.5, 0.0), 4)
        self.assertEqual(text, "1,0.5,0")
        self.assertEqual(waveform.parse_numeric_schedule(text, 0.0, 1.0), (1.0, 0.5, 0.0))

    def test_preview_render_contains_image(self):
        image = waveform.render_schedule_preview((1.0, 0.5, 0.0), 4)
        self.assertGreater(image.width, 300)
        self.assertGreater(image.height, 300)

    def test_parse_reports_invalid_step_and_range(self):
        with self.assertRaisesRegex(ValueError, "step 2"):
            waveform.parse_numeric_schedule("1,abc,0", 0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "allowed range"):
            waveform.parse_numeric_schedule("1,2,0", 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
