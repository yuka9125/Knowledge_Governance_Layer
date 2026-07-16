#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
golden query 評価スクリプト（Phase C / Governed Knowledge API）。

起動中の Governed Knowledge API に対して、合成 golden query セットを投げ、
各 query の期待結果（answerable / knowledge_id）と実結果を突き合わせて
pass/fail と正解率をレポートする。全件passなら exit 0、失敗があれば exit 1。
APIへ接続できない場合は exit 2。

前提：APIを golden 用 approved_knowledge フィクスチャで起動しておく。

  # repo root で実行（PowerShell）
  $env:APPROVED_KNOWLEDGE_PATH = "benchmark/eval/golden_approved_knowledge.json"
  python -m uvicorn serving.governed_knowledge_api:app --port 8000

  # 別ターミナルで（repo root で実行）
  python eval/run_golden_queries.py --base-url http://127.0.0.1:8000

注意：golden_queries.json / golden_approved_knowledge.json は「合成データ」。
実顧客データではない。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx


DEFAULT_GOLDEN = Path(__file__).resolve().parent / "golden_queries.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _load_golden(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    cases = data.get("cases", [])
    if not cases:
        raise ValueError(f"goldenケースが空です: {path}")
    return cases


def _check_health(client: httpx.Client, base_url: str) -> None:
    """/health が200を返すか確認する。失敗時は例外。"""
    response = client.get(f"{base_url}/health")
    response.raise_for_status()


def _evaluate_case(
    client: httpx.Client, base_url: str, case: Dict[str, Any]
) -> Dict[str, Any]:
    """1ケースを実行し、判定結果を返す。"""
    query = case["query"]
    expect_answerable = bool(case.get("expect_answerable", False))
    expect_knowledge_id = case.get("expect_knowledge_id")

    response = client.get(
        f"{base_url}/knowledge/search", params={"q": query}
    )
    response.raise_for_status()
    body = response.json()
    actual_answerable = bool(body.get("answerable", False))
    actual_knowledge_id = body.get("knowledge_id", "")

    reasons: List[str] = []
    passed = True

    if actual_answerable != expect_answerable:
        passed = False
        reasons.append(
            f"answerable 期待={expect_answerable} 実={actual_answerable}"
        )
    elif expect_answerable:
        # 該当ありを期待する場合のみ knowledge_id を検証する
        if expect_knowledge_id and actual_knowledge_id != expect_knowledge_id:
            passed = False
            reasons.append(
                f"knowledge_id 期待={expect_knowledge_id} 実={actual_knowledge_id}"
            )
    else:
        # 該当なしを期待する場合は fallback が出ているか軽く確認する
        if body.get("fallback") != "human_review":
            passed = False
            reasons.append(
                f"fallback 期待=human_review 実={body.get('fallback')}"
            )

    return {
        "id": case.get("id", query),
        "query": query,
        "expect_answerable": expect_answerable,
        "actual_answerable": actual_answerable,
        "expect_knowledge_id": expect_knowledge_id or "",
        "actual_knowledge_id": actual_knowledge_id,
        "passed": passed,
        "reason": "; ".join(reasons),
    }


def _print_report(results: List[Dict[str, Any]]) -> None:
    print("=" * 78)
    print("golden query 評価レポート（合成データ）")
    print("=" * 78)
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['id']}")
        print(f"       query : {r['query']}")
        print(
            f"       expect: answerable={r['expect_answerable']}"
            + (f", id={r['expect_knowledge_id']}" if r["expect_knowledge_id"] else "")
        )
        print(
            f"       actual: answerable={r['actual_answerable']}"
            + (f", id={r['actual_knowledge_id']}" if r["actual_knowledge_id"] else "")
        )
        if not r["passed"]:
            print(f"       --> {r['reason']}")
    print("-" * 78)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    accuracy = (passed / total * 100) if total else 0.0
    print(f"合計 {total} 件 / pass {passed} 件 / fail {total - passed} 件")
    print(f"正解率: {accuracy:.1f}%")
    print("=" * 78)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Governed Knowledge API に golden query を投げて評価する。"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"APIのベースURL（既定: {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=DEFAULT_GOLDEN,
        help="golden query JSON のパス",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="リクエストのタイムアウト秒（既定: 10）",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    try:
        cases = _load_golden(args.golden)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"goldenデータ読み込み失敗: {exc}", file=sys.stderr)
        return 2

    try:
        with httpx.Client(timeout=args.timeout) as client:
            _check_health(client, base_url)
            results = [_evaluate_case(client, base_url, case) for case in cases]
    except httpx.HTTPError as exc:
        print(f"APIへの接続/呼び出しに失敗: {exc}", file=sys.stderr)
        print(
            "ヒント: golden用フィクスチャでAPIを起動しているか確認してください。",
            file=sys.stderr,
        )
        return 2

    _print_report(results)
    all_passed = all(r["passed"] for r in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
