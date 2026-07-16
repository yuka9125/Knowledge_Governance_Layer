#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate fixed conflict detection on the Phase F-0 synthetic dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge_distillation.knowledge_output_utils import (
    P32_REVIEW_THRESHOLD,
    has_conflict_signal,
)


DEFAULT_DATASET = Path(__file__).resolve().parent / "synthetic_conflict_dataset.json"


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("synthetic_data"):
        raise ValueError(f"dataset must be explicitly marked synthetic: {path}")
    cases = data.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"dataset has no cases: {path}")
    return cases


def _detect_conflict(case: Dict[str, Any]) -> bool:
    """Fixed Phase F-0 conflict rule.

    A conflict is counted only when a candidate is close enough to an existing
    FAQ to require P3-2 review and the answer pair contains an opposite-signal
    term such as old/new, disabled/enabled, or impossible/possible.
    """
    similarity = float(case.get("similarity_large", 0.0))
    existing_answer = str(case.get("existing_faq", {}).get("answer", ""))
    candidate_answer = str(case.get("candidate", {}).get("answer", ""))
    return similarity >= P32_REVIEW_THRESHOLD and has_conflict_signal(
        candidate_answer, existing_answer
    )


def _evaluate(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    tp = fp = tn = fn = 0
    for case in cases:
        expected = bool(case.get("expected_conflict"))
        actual = _detect_conflict(case)
        if actual and expected:
            tp += 1
            bucket = "TP"
        elif actual and not expected:
            fp += 1
            bucket = "FP"
        elif not actual and expected:
            fn += 1
            bucket = "FN"
        else:
            tn += 1
            bucket = "TN"
        rows.append(
            {
                "id": case.get("id", ""),
                "label": case.get("label", ""),
                "similarity_large": case.get("similarity_large"),
                "expected_conflict": expected,
                "actual_conflict": actual,
                "bucket": bucket,
            }
        )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "thresholds": {
            "embedding_model_assumption": "text-embedding-3-large",
            "p32_review_threshold": P32_REVIEW_THRESHOLD,
        },
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(cases)},
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "rows": rows,
    }


def _print_report(report: Dict[str, Any]) -> None:
    print("Phase F-0 synthetic conflict benchmark")
    print("Synthetic data: true")
    print(
        "Model assumption:",
        report["thresholds"]["embedding_model_assumption"],
    )
    print("P3-2 review threshold:", report["thresholds"]["p32_review_threshold"])
    print()
    for row in report["rows"]:
        print(
            f"[{row['bucket']}] {row['id']} "
            f"label={row['label']} "
            f"sim={row['similarity_large']} "
            f"expected={row['expected_conflict']} "
            f"actual={row['actual_conflict']}"
        )
    print()
    counts = report["counts"]
    metrics = report["metrics"]
    print(
        f"total={counts['total']} tp={counts['tp']} fp={counts['fp']} "
        f"tn={counts['tn']} fn={counts['fn']}"
    )
    print(
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase F-0 synthetic conflict detection."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    args = parser.parse_args()

    try:
        report = _evaluate(_load_cases(args.dataset))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to evaluate dataset: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["counts"]["fn"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
