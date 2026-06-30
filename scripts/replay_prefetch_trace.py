#!/usr/bin/env python3
"""Replay captured prefetch trace schedules against observed access streams.

This is the offline harness for Prefetch 2.0 threshold tuning. It deliberately
does not allocate CUDA tensors or start bounce-pool workers; it replays only the
cursor alignment policy: exact semantic match, bounded lookahead resync, and
duplicate-layer resync blocking.

Input JSON shape:

    {
      "schedule": [["a", "forward", 0], ["b", "forward", 0]],
      "observed": [["b", "forward", 0], ["a", "backward", 1]]
    }

Entries may also be objects with layer_key/operation/occurrence fields. Observed
entries may omit occurrence; the harness assigns the same per-layer occurrence
counter used by the live bounce pool.
"""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ReplayStats:
    accesses: int = 0
    aligned: int = 0
    resyncs: int = 0
    mismatches: int = 0
    duplicate_key_resync_blocked: int = 0
    consume_pos: int = 0
    schedule_len: int = 0
    lookahead: int = 0

    @property
    def resync_rate(self) -> float:
        return self.resyncs / max(1, self.accesses)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / max(1, self.accesses)

    @property
    def duplicate_key_resync_block_rate(self) -> float:
        return self.duplicate_key_resync_blocked / max(1, self.accesses)

    @property
    def consumed_past_schedule(self) -> bool:
        return self.consume_pos > self.schedule_len + max(1, self.lookahead)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            resync_rate=self.resync_rate,
            mismatch_rate=self.mismatch_rate,
            duplicate_key_resync_block_rate=self.duplicate_key_resync_block_rate,
            consumed_past_schedule=self.consumed_past_schedule,
        )
        return data


def _entry_layer(entry):
    if isinstance(entry, dict):
        return entry.get("layer_key", entry.get("layer"))
    if isinstance(entry, (list, tuple)):
        return entry[0] if entry else None
    return entry


def _normalize_entry(entry, occurrence_counters=None):
    if isinstance(entry, dict):
        layer = entry.get("layer_key", entry.get("layer"))
        operation = entry.get("operation", "forward")
        occurrence = entry.get("occurrence")
    elif isinstance(entry, (list, tuple)):
        layer = entry[0]
        operation = entry[1] if len(entry) > 1 else "forward"
        occurrence = entry[2] if len(entry) > 2 else None
    else:
        layer = entry
        operation = "forward"
        occurrence = None

    if occurrence is None and occurrence_counters is not None:
        occurrence = occurrence_counters[layer]
        occurrence_counters[layer] += 1
    if occurrence is None:
        return layer
    return (layer, operation, int(occurrence))


def _matches(schedule_entry, access_key) -> bool:
    if isinstance(schedule_entry, tuple):
        return schedule_entry == access_key
    return schedule_entry == access_key[0]


def replay(schedule, observed, *, lookahead: int = 16) -> ReplayStats:
    normalized_schedule = [_normalize_entry(entry) for entry in schedule]
    counters = collections.Counter()
    normalized_observed = [
        _normalize_entry(entry, occurrence_counters=counters) for entry in observed
    ]
    stats = ReplayStats(schedule_len=len(normalized_schedule), lookahead=max(1, int(lookahead)))
    consume_pos = 0

    for access_key in normalized_observed:
        stats.accesses += 1
        pos = consume_pos
        if pos < len(normalized_schedule) and _matches(normalized_schedule[pos], access_key):
            stats.aligned += 1
            consume_pos = pos + 1
            continue

        layer = access_key[0] if isinstance(access_key, tuple) else access_key
        end = min(len(normalized_schedule), pos + 1 + stats.lookahead)
        duplicate_layer_seen = False
        found = None
        for index in range(pos + 1, end):
            entry = normalized_schedule[index]
            if _matches(entry, access_key):
                found = index
                break
            if _entry_layer(entry) == layer:
                duplicate_layer_seen = True

        if found is not None:
            stats.resyncs += 1
            consume_pos = found + 1
        else:
            stats.mismatches += 1
            if duplicate_layer_seen:
                stats.duplicate_key_resync_blocked += 1
            consume_pos = pos + 1

    stats.consume_pos = consume_pos
    return stats


def _capture_streams(data: dict) -> tuple[list[Any], list[Any]]:
    return data.get("schedule", []), data.get("observed", data.get("accesses", []))


def load_captures(path: Path) -> list[tuple[list[Any], list[Any], dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            if not isinstance(data, dict):
                raise ValueError("capture must be a JSON object or JSONL records")
            schedule, observed = _capture_streams(data)
            return [(schedule, observed, data)]

    captures = []
    for line_no, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError(f"line {line_no}: expected JSON object")
        schedule, observed = _capture_streams(data)
        captures.append((schedule, observed, data))
    return captures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="JSON or JSONL capture containing schedule and observed access streams")
    parser.add_argument("--lookahead", type=int, default=16)
    parser.add_argument("--json", action="store_true", help="print machine-readable stats")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    captures = load_captures(args.path)
    per_step = []
    total = ReplayStats(lookahead=max(1, int(args.lookahead)))
    for schedule, observed, metadata in captures:
        lookahead = int(metadata.get("lookahead", args.lookahead))
        step_stats = replay(schedule, observed, lookahead=lookahead)
        item = step_stats.to_dict()
        item["schedule_confidence"] = metadata.get("schedule_confidence")
        item["schedule_shape_key"] = metadata.get("schedule_shape_key")
        per_step.append(item)
        total.accesses += step_stats.accesses
        total.aligned += step_stats.aligned
        total.resyncs += step_stats.resyncs
        total.mismatches += step_stats.mismatches
        total.duplicate_key_resync_blocked += step_stats.duplicate_key_resync_blocked
        total.schedule_len += step_stats.schedule_len
        total.consume_pos += step_stats.consume_pos
    stats = total.to_dict()
    stats["captures"] = len(captures)
    if args.json:
        print(json.dumps({"summary": stats, "steps": per_step}, indent=2, sort_keys=True))
    else:
        print(
            "captures={captures} accesses={accesses} aligned={aligned} resyncs={resyncs} "
            "mismatches={mismatches} dup_block={duplicate_key_resync_blocked} "
            "resync_rate={resync_rate:.3f} mismatch_rate={mismatch_rate:.3f}".format(**stats)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())