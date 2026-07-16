#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate Phase F synthetic demo scenarios against a running API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx


DEFAULT_SCENARIOS = Path(__file__).resolve().parent / "phase_f_demo_scenarios.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUIRED_SCENES = {"before", "governance", "after", "no_match"}


def _load_scenarios(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("synthetic_data"):
        raise ValueError(f"scenario file must be explicitly synthetic: {path}")
    scenarios = data.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError(f"scenario file has no scenarios: {path}")
    scenes = {str(item.get("scene", "")) for item in scenarios}
    missing = REQUIRED_SCENES - scenes
    if missing:
        raise ValueError(f"scenario file is missing scenes: {sorted(missing)}")
    return data


def _check_health(client: httpx.Client, base_url: str) -> None:
    response = client.get(f"{base_url}/health")
    response.raise_for_status()


def _evaluate_case(
    client: httpx.Client,
    base_url: str,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    response = client.get(
        f"{base_url}/knowledge/search",
        params={"q": case["query"]},
    )
    response.raise_for_status()
    body = response.json()

    reasons: List[str] = []
    expected_answerable = bool(case.get("expect_answerable", False))
    actual_answerable = bool(body.get("answerable", False))
    if actual_answerable != expected_answerable:
        reasons.append(
            f"answerable expected={expected_answerable} actual={actual_answerable}"
        )

    expected_id = case.get("expect_knowledge_id")
    if expected_answerable and expected_id and body.get("knowledge_id") != expected_id:
        reasons.append(
            f"knowledge_id expected={expected_id} actual={body.get('knowledge_id')}"
        )

    expected_text = case.get("expect_answer_contains")
    if expected_answerable and expected_text:
        answer = str(body.get("answer", ""))
        if expected_text not in answer:
            reasons.append(f"answer does not contain expected text: {expected_text}")

    expected_fallback = case.get("expect_fallback")
    if not expected_answerable and expected_fallback:
        if body.get("fallback") != expected_fallback:
            reasons.append(
                f"fallback expected={expected_fallback} actual={body.get('fallback')}"
            )

    return {
        "id": case.get("id", ""),
        "scene": case.get("scene", ""),
        "expected_state": case.get("expected_state", ""),
        "passed": not reasons,
        "reason": "; ".join(reasons),
        "actual": body,
    }


def _print_validation_summary(data: Dict[str, Any]) -> None:
    print("Phase F synthetic demo scenarios")
    print("Synthetic data: true")
    print("Fixtures:")
    for name, path in data.get("fixtures", {}).items():
        print(f"  {name}: {path}")
    print("Scenes:")
    for item in data["scenarios"]:
        marker = "api" if item.get("api_check") else "manual"
        print(f"  [{marker}] {item['scene']}: {item['id']}")


def _print_report(results: List[Dict[str, Any]]) -> None:
    print("Phase F demo scenario API check")
    for result in results:
        mark = "PASS" if result["passed"] else "FAIL"
        print(
            f"[{mark}] {result['id']} "
            f"scene={result['scene']} state={result['expected_state']}"
        )
        if result["reason"]:
            print(f"       {result['reason']}")
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    print(f"total={total} passed={passed} failed={total - passed}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Phase F synthetic demo scenarios."
    )
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--expected-state",
        choices=["before", "after"],
        default="after",
        help="Run API checks for the fixture currently loaded by the API.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate scenario structure; do not call the API.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    try:
        data = _load_scenarios(args.scenarios)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to load scenarios: {exc}", file=sys.stderr)
        return 2

    _print_validation_summary(data)
    if args.validate_only:
        return 0

    cases = [
        item
        for item in data["scenarios"]
        if item.get("api_check") and item.get("expected_state") == args.expected_state
    ]
    if not cases:
        print(f"no API-check scenarios for state: {args.expected_state}", file=sys.stderr)
        return 2

    base_url = args.base_url.rstrip("/")
    try:
        with httpx.Client(timeout=args.timeout) as client:
            _check_health(client, base_url)
            results = [_evaluate_case(client, base_url, case) for case in cases]
    except httpx.HTTPError as exc:
        print(f"failed to call API: {exc}", file=sys.stderr)
        return 2

    _print_report(results)
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
