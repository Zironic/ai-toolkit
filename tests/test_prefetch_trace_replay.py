import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "replay_prefetch_trace", ROOT / "scripts" / "replay_prefetch_trace.py"
)
replay_prefetch_trace = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = replay_prefetch_trace
SPEC.loader.exec_module(replay_prefetch_trace)


class PrefetchTraceReplayTests(unittest.TestCase):
    def test_replay_resyncs_exact_semantic_access(self):
        stats = replay_prefetch_trace.replay(
            [
                ("a", "forward", 0),
                ("b", "forward", 0),
                ("a", "backward", 1),
            ],
            [("b", "forward", 0)],
            lookahead=4,
        )
        self.assertEqual(stats.resyncs, 1)
        self.assertEqual(stats.mismatches, 0)
        self.assertEqual(stats.consume_pos, 2)

    def test_replay_blocks_duplicate_layer_wrong_occurrence(self):
        stats = replay_prefetch_trace.replay(
            [
                ("a", "forward", 0),
                ("c", "forward", 0),
                ("a", "backward", 1),
            ],
            [("a", "backward", 0)],
            lookahead=4,
        )
        self.assertEqual(stats.resyncs, 0)
        self.assertEqual(stats.mismatches, 1)
        self.assertEqual(stats.duplicate_key_resync_blocked, 1)

    def test_replay_assigns_observed_occurrences_when_missing(self):
        stats = replay_prefetch_trace.replay(
            [("a", "forward", 0), ("a", "backward", 1)],
            [
                {"layer_key": "a", "operation": "forward"},
                {"layer_key": "a", "operation": "backward"},
            ],
            lookahead=2,
        )
        self.assertEqual(stats.aligned, 2)
        self.assertEqual(stats.mismatches, 0)

    def test_loads_jsonl_captures(self):
        records = [
            {
                "schedule": [["a", "forward", 0], ["b", "forward", 0]],
                "observed": [["b", "forward", 0]],
                "lookahead": 4,
            },
            {
                "schedule": [["a", "forward", 0]],
                "observed": [["a", "forward", 0]],
                "lookahead": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "capture.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records),
                encoding="utf-8",
            )

            captures = replay_prefetch_trace.load_captures(path)

        self.assertEqual(len(captures), 2)
        first = replay_prefetch_trace.replay(captures[0][0], captures[0][1], lookahead=4)
        self.assertEqual(first.resyncs, 1)


if __name__ == "__main__":
    unittest.main()