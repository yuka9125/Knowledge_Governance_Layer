from __future__ import annotations

import argparse
import json
import sys

from knowledge_distillation.pipeline import export, ingest, normalize


def parse_comma_separated_cols(value: str | None) -> list[str] | None:
    """カンマ区切り列指定を配列へ変換する。"""
    if value is None:
        return None
    cols = [part.strip() for part in value.split(",")]
    cols = [c for c in cols if c]
    return cols or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_distillation",
        description="Knowledge distillation CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="問い合わせデータを取り込み")
    ingest_parser.add_argument(
        "--adapter",
        required=True,
        choices=["csv", "servicenow", "graphmail"],
        help="入力アダプタ",
    )
    ingest_parser.add_argument("--path", required=True, help="入力ファイルパス")
    ingest_parser.add_argument(
        "--question-col",
        default=None,
        help="質問列を明示指定（指定時は自動解決より優先）",
    )
    ingest_parser.add_argument(
        "--answer-cols",
        default=None,
        help="回答列をカンマ区切りで指定（指定時は自動解決より優先）",
    )
    ingest_parser.add_argument(
        "--source-text-cols",
        default=None,
        help="source_text列をカンマ区切りで指定（指定時は自動解決より優先）",
    )
    ingest_parser.add_argument(
        "--db-path",
        default="data/knowledge/knowledge.db",
        help="SQLite保存先",
    )
    ingest_parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Azure OpenAIを使わずローカルフォールバックで生成",
    )

    export_parser = subparsers.add_parser("export", help="ナレッジを出力")
    export_parser.add_argument(
        "--db-path",
        default="data/knowledge/knowledge.db",
        help="SQLite保存先",
    )
    export_parser.add_argument(
        "--out-dir",
        default="data/outputs",
        help="出力ディレクトリ",
    )

    normalize_parser = subparsers.add_parser(
        "normalize", help="既存SNOW互換CSVへ正規化"
    )
    normalize_parser.add_argument(
        "--adapter",
        required=True,
        choices=["csv", "servicenow", "graphmail"],
        help="入力アダプタ",
    )
    normalize_parser.add_argument("--path", required=True, help="入力ファイルパス")
    normalize_parser.add_argument(
        "--out",
        required=True,
        help="正規化CSV出力パス",
    )
    normalize_parser.add_argument(
        "--question-col",
        default=None,
        help="概要の優先入力列（指定時は自動解決より優先）",
    )
    normalize_parser.add_argument(
        "--answer-cols",
        default=None,
        help="対応結果に連結する列をカンマ区切りで指定",
    )
    normalize_parser.add_argument(
        "--source-text-cols",
        default=None,
        help="source_text列をカンマ区切りで指定（補助情報）",
    )
    normalize_parser.add_argument(
        "--title-col",
        default=None,
        help="件名の優先入力列",
    )
    normalize_parser.add_argument(
        "--category-col",
        default=None,
        help="カテゴリの優先入力列",
    )
    normalize_parser.add_argument(
        "--link-col",
        default=None,
        help="リンク名の優先入力列",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        answer_cols = parse_comma_separated_cols(args.answer_cols)
        source_text_cols = parse_comma_separated_cols(args.source_text_cols)
        result = ingest(
            adapter_name=args.adapter,
            path=args.path,
            db_path=args.db_path,
            use_openai=not args.no_openai,
            question_col=args.question_col,
            answer_cols=answer_cols,
            source_text_cols=source_text_cols,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "export":
        result = export(db_path=args.db_path, out_dir=args.out_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    if args.command == "normalize":
        answer_cols = parse_comma_separated_cols(args.answer_cols)
        source_text_cols = parse_comma_separated_cols(args.source_text_cols)
        result = normalize(
            adapter_name=args.adapter,
            path=args.path,
            out_path=args.out,
            question_col=args.question_col,
            answer_cols=answer_cols,
            source_text_cols=source_text_cols,
            title_col=args.title_col,
            category_col=args.category_col,
            link_col=args.link_col,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
