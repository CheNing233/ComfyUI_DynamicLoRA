import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "schedule.py"
spec = importlib.util.spec_from_file_location("dynamic_lora_rank_schedule", MODULE_PATH)
schedule = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(schedule)


class DynamicLoraRankScheduleTests(unittest.TestCase):
    def test_parse_rank_and_strength_ratios(self):
        self.assertEqual(schedule.parse_ratio_schedule("1, 0.5, 1, 0", "rank_schedule"), (1.0, 0.5, 1.0, 0.0))
        self.assertEqual(schedule.parse_ratio_schedule("1, 0.25", "strength_schedule"), (1.0, 0.25))

    def test_parse_rejects_out_of_range_and_empty_values(self):
        with self.assertRaisesRegex(ValueError, "strength_schedule"):
            schedule.parse_ratio_schedule("1,1.1", "strength_schedule")
        with self.assertRaisesRegex(ValueError, "rank_schedule"):
            schedule.parse_ratio_schedule("1,,0", "rank_schedule")

    def test_last_value_repeats_after_schedule(self):
        values = schedule.parse_ratio_schedule("1,0.5")
        self.assertEqual(schedule.ratio_for_step(values, 0), 1.0)
        self.assertEqual(schedule.ratio_for_step(values, 1), 0.5)
        self.assertEqual(schedule.ratio_for_step(values, 99), 0.5)

    def test_rank_ratio_is_relative_to_total_rank(self):
        self.assertEqual(schedule.active_rank(8, 1.0), 8)
        self.assertEqual(schedule.active_rank(8, 0.5), 4)
        self.assertEqual(schedule.active_rank(8, 0.0), 0)
        self.assertEqual(schedule.active_rank(3, 0.5), 2)

    def test_terminal_sigma_is_not_a_denoise_step(self):
        sigmas = [10.0, 7.0, 3.0, 0.5, 0.0]
        self.assertEqual(schedule.step_index_for_sigma(10.0, sigmas), 0)
        self.assertEqual(schedule.step_index_for_sigma(0.5, sigmas), 3)
        self.assertEqual(schedule.step_index_for_sigma(0.0, sigmas), 3)

    def test_tensor_like_sigma_sequence_is_supported(self):
        class TensorLikeSequence:
            def __len__(self):
                return 4

            def __getitem__(self, item):
                return (10.0, 7.0, 3.0, 0.0)[item]

        self.assertEqual(schedule.step_index_for_sigma(7.0, TensorLikeSequence()), 1)


if __name__ == "__main__":
    unittest.main()
