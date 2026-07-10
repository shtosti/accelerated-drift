from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json


def select_micro_records(
    records: list[dict],
    *,
    years: list[int],
    papers_per_year: int,
    seed: str,
) -> list[dict]:
    by_year: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        try:
            year = int(record.get("year"))
        except (TypeError, ValueError):
            continue
        if year in years:
            by_year[year].append(record)

    selected = []
    for year in years:
        candidates = by_year.get(year, [])
        if len(candidates) < papers_per_year:
            raise ValueError(
                f"Year {year} has {len(candidates)} records; "
                f"need {papers_per_year}"
            )
        candidates.sort(
            key=lambda record: sha256(
                f"{seed}:{record.get('paperId', '')}".encode()
            ).hexdigest()
        )
        selected.extend(candidates[:papers_per_year])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a deterministic, year-balanced JSONL micro dataset."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", required=True)
    parser.add_argument("--papers-per-year", type=int, default=2)
    parser.add_argument("--seed", default="not-an-llm-arxiv-micro-v1")
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = select_micro_records(
        records,
        years=args.years,
        papers_per_year=args.papers_per_year,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(selected)} records to {args.output}")


if __name__ == "__main__":
    main()
