import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "waveform.py"
spec = importlib.util.spec_from_file_location("dynamic_lora_waveform", MODULE_PATH)
waveform = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(waveform)


class WaveformTests(unittest.TestCase):
    def test_linear_up_is_discrete_and_exact_length(self):
        values = waveform.generate_waveform(5, "linear_up", decimals=6)
        self.assertEqual(values, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(len(values), 5)

    def test_linear_down_and_constant(self):
        self.assertEqual(waveform.generate_waveform(4, "linear_down", decimals=6), (1.0, 0.666667, 0.333333, 0.0))
        self.assertEqual(waveform.generate_waveform(3, "constant", start_value=0.5, decimals=6), (0.5, 0.5, 0.5))

    def test_periodic_waveforms_use_step_values_not_smoothing(self):
        self.assertEqual(waveform.generate_waveform(5, "triangle", cycles=1, decimals=6), (0.0, 0.4, 0.8, 0.8, 0.4))
        self.assertEqual(waveform.generate_waveform(4, "square", cycles=2, decimals=6), (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(waveform.generate_waveform(4, "pulse", cycles=1, pulse_start=0.0, pulse_end=0.5, decimals=6), (1.0, 1.0, 0.0, 0.0))

    def test_format_and_parse(self):
        text = waveform.format_schedule((1.0, 0.5, 0.0), 4)
        self.assertEqual(text, "1,0.5,0")
        self.assertEqual(waveform.parse_numeric_schedule(text), (1.0, 0.5, 0.0))

    def test_parse_reports_invalid_step(self):
        with self.assertRaisesRegex(ValueError, "step 2"):
            waveform.parse_numeric_schedule("1,abc,0")
        with self.assertRaisesRegex(ValueError, "empty value"):
            waveform.parse_numeric_schedule("1,,0")
        with self.assertRaisesRegex(ValueError, "allowed range"):
            waveform.parse_numeric_schedule("1,2,0", 0.0, 1.0)

    def test_waveform_output_is_loader_compatible(self):
        values = waveform.generate_waveform(16, "square", cycles=8, decimals=6)
        schedule = waveform.format_schedule(values, 6)
        self.assertEqual(waveform.parse_numeric_schedule(schedule, 0.0, 1.0), values)

    def test_step_preview_is_not_smoothed(self):
        text = waveform.format_step_preview((1.0, 0.5, 0.0), 4)
        self.assertIn("step 0001: 1", text)
        self.assertIn("step 0002: 0.5", text)
        self.assertIn("step 0003: 0", text)
        self.assertNotIn("0.75", text)


if __name__ == "__main__":
    unittest.main()
