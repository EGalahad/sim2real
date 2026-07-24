import math
import unittest

from scripts.dh116s_keyboard_control import next_grip


class KeyboardGripTest(unittest.TestCase):
    def test_period_closes_and_comma_opens(self) -> None:
        self.assertAlmostEqual(next_grip(0.50, ".", 0.05), 0.55)
        self.assertAlmostEqual(next_grip(0.50, ",", 0.05), 0.45)

    def test_commands_clip_to_safe_normalized_range(self) -> None:
        self.assertEqual(next_grip(0.98, ".", 0.05), 1.0)
        self.assertEqual(next_grip(0.02, ",", 0.05), 0.0)
        self.assertEqual(next_grip(0.75, "0", 0.05), 0.0)

    def test_unknown_key_holds_position(self) -> None:
        self.assertEqual(next_grip(0.25, "x", 0.05), 0.25)

    def test_invalid_values_are_rejected(self) -> None:
        for current in (-0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                next_grip(current, ".", 0.05)
        for step in (0.0, -0.1, 1.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                next_grip(0.0, ".", step)


if __name__ == "__main__":
    unittest.main()
