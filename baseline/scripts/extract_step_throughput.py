#!/usr/bin/env python3
"""Extract per-step token throughput from baseline run logs."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


TOKENS_PER_S = re.compile(r"(?:tokens_per_s=|Throughput:\s*)(?P<value>[0-9.]+)\s*(?:tok/s)?")
STEP = re.compile(r"(?:^|\s)(?:step=|Iter\s+)(?P<value>[0-9]+)")
LOSS = re.compile(r"(?:loss=|Loss:\s*)(?P<value>[-+0-9.eE]+)")
STEP_TIME = re.compile(r"(?:step_time_s=|Step:\s*)(?P<value>[0-9.]+)\s*(?:ms)?")


def _match_float(pattern: re.Pattern[str], text: str) -> float | None:
    match = pattern.search(text)
    if match is None:
        return None
    value = float(match.group("value"))
    if "Step:" in match.group(0) and "ms" in text[match.start() : match.end() + 8]:
        value /= 1000.0
    return value


def _match_int(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group("value")) if match else None


def parse_log(path: Path) -> list[dict[str, str]]:
    backend = path.parent.name
    rows: list[dict[str, str]] = []
    for raw_line in path.read_text(errors="replace").splitlines():
        tokens_per_s = _match_float(TOKENS_PER_S, raw_line)
        if tokens_per_s is None:
            continue
        rows.append(
            {
                "backend": backend,
                "log": str(path),
                "step": "" if (step := _match_int(STEP, raw_line)) is None else str(step),
                "loss": "" if (loss := _match_float(LOSS, raw_line)) is None else f"{loss:.6g}",
                "step_time_s": ""
                if (step_time := _match_float(STEP_TIME, raw_line)) is None
                else f"{step_time:.6g}",
                "tokens_per_s": f"{tokens_per_s:.6g}",
                "line": raw_line,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for log in args.logs:
        if log.is_dir():
            rows.extend(parse_log(log / "run.log"))
        else:
            rows.extend(parse_log(log))

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["backend", "step", "loss", "step_time_s", "tokens_per_s", "log", "line"],
    )
    writer.writeheader()
    writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
