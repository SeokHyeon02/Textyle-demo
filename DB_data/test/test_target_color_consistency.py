import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parent / "target_color_consistency.py"
spec = importlib.util.spec_from_file_location("target_color_consistency", SCRIPT_PATH)
target_color_consistency = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = target_color_consistency
spec.loader.exec_module(target_color_consistency)


class TargetColorConsistencyTests(unittest.TestCase):
    def test_default_target_is_top_sweater(self):
        args = target_color_consistency.parse_args([])

        self.assertEqual(args.category, "상의")
        self.assertEqual(args.sub_category_contains, "스웨터")
        self.assertEqual(args.target_label, "top_sweater")

    def test_target_preset_changes_category_filters(self):
        args = target_color_consistency.parse_args(["--target", "jeans"])

        self.assertEqual(args.category, "하의")
        self.assertEqual(args.sub_category_contains, "데님")
        self.assertEqual(args.target_label, "jeans")

    def test_explicit_filters_override_target_preset(self):
        args = target_color_consistency.parse_args([
            "--target",
            "jeans",
            "--category",
            "아우터",
            "--sub-category-contains",
            "레더",
            "--target-label",
            "leather_jacket",
        ])

        self.assertEqual(args.category, "아우터")
        self.assertEqual(args.sub_category_contains, "레더")
        self.assertEqual(args.target_label, "leather_jacket")


if __name__ == "__main__":
    unittest.main()
