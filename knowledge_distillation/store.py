from __future__ import annotations

import csv
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Sequence

from knowledge_distillation.models import FAQCandidate
from knowledge_distillation.xlsx_writer import write_xlsx
from knowledge_distillation.knowledge_output_utils import determine_risk_level


# JSON / CSV / Excel で共通の出力項目（この順で並べる）
KNOWLEDGE_EXPORT_COLUMNS: Sequence[str] = (
    "knowledge_id",
    "cluster_id",
    "question",
    "answer",
    "category",
    "source_logs",
    "similar_logs_count",
    "existing_faq_diff_reason",
    "risk_level",
    "review_status",
    "confidence",
)


class KnowledgeStore(ABC):
    """ナレッジ保存先の抽象インターフェース。"""

    @abstractmethod
    def save_faqs(self, faqs: List[FAQCandidate]) -> int:
        """FAQ一覧を保存する。戻り値は保存件数。"""

    @abstractmethod
    def list_faqs(self) -> List[Dict]:
        """保存済みFAQを返す。"""

    @abstractmethod
    def export_json(self, path: str | Path) -> Path:
        """JSONへエクスポートする。"""

    @abstractmethod
    def export_csv(self, path: str | Path) -> Path:
        """CSVへエクスポートする（JSONと同一項目）。"""

    @abstractmethod
    def export_excel(self, path: str | Path) -> Path:
        """Excelへエクスポートする。"""


class SQLiteKnowledgeStore(KnowledgeStore):
    """SQLiteにFAQナレッジを保存する。"""

    def __init__(self, db_path: str | Path = "data/knowledge/knowledge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faqs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sources_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_faq_question ON faqs(question)"
            )
            conn.commit()

    def save_faqs(self, faqs: List[FAQCandidate]) -> int:
        if not faqs:
            return 0
        rows = [
            (
                faq.question,
                faq.answer,
                faq.category,
                faq.confidence,
                json.dumps(faq.sources, ensure_ascii=False),
                json.dumps(faq.metadata, ensure_ascii=False),
            )
            for faq in faqs
        ]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO faqs (
                    question, answer, category, confidence, sources_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def list_faqs(self) -> List[Dict]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT id, question, answer, category, confidence, sources_json,
                       metadata_json, created_at
                FROM faqs
                ORDER BY id
                """
            )
            rows = cur.fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "id": row[0],
                    "question": row[1],
                    "answer": row[2],
                    "category": row[3],
                    "confidence": row[4],
                    "sources": json.loads(row[5]),
                    "metadata": json.loads(row[6]),
                    "created_at": row[7],
                }
            )
        return result

    def export_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self._build_knowledge_candidates(self.list_faqs())
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def export_csv(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self._build_export_rows()
        # Excelでそのまま開けるようBOM付きUTF-8で書き出す
        with target.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(KNOWLEDGE_EXPORT_COLUMNS)
            writer.writerows(rows)
        return target

    def export_excel(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rows = self._build_export_rows()
        return write_xlsx(
            target, headers=list(KNOWLEDGE_EXPORT_COLUMNS), rows=rows
        )

    def _build_export_rows(self) -> List[List[str]]:
        """CSV / Excel共通の行データを作る（列順はJSONの項目順と同じ）。"""
        data = self._build_knowledge_candidates(self.list_faqs())
        rows: List[List[str]] = []
        for knowledge in data:
            row: List[str] = []
            for column in KNOWLEDGE_EXPORT_COLUMNS:
                value = knowledge.get(column, "")
                if isinstance(value, list):
                    # source_logs は配列なので情報を落とさずJSON文字列にする
                    value = json.dumps(value, ensure_ascii=False)
                row.append(str(value))
            rows.append(row)
        return rows

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _build_knowledge_candidates(self, faqs: List[Dict]) -> List[Dict]:
        """保存済みFAQをナレッジ候補形式に変換する。"""
        knowledge_candidates: List[Dict] = []
        for idx, faq in enumerate(faqs, start=1):
            source_logs = self._extract_source_logs(
                faq.get("sources", []), fallback_id=faq.get("id", idx)
            )
            question = str(faq.get("question", ""))
            answer = str(faq.get("answer", ""))
            category = str(faq.get("category", ""))

            knowledge_candidates.append(
                {
                    "knowledge_id": f"k-{idx:03d}",
                    "cluster_id": f"c-{idx:03d}",
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "source_logs": source_logs,
                    "similar_logs_count": len(source_logs),
                    "existing_faq_diff_reason": "既存FAQ照合未実施（CLI出力）",
                    "risk_level": determine_risk_level(answer, category),
                    "review_status": "draft",
                    "confidence": float(faq.get("confidence", 0.0)),
                }
            )
        return knowledge_candidates

    def _extract_source_logs(self, sources: List[Dict], fallback_id: int) -> List[str]:
        """sources配列から表示用source_logsを抽出する。"""
        source_logs: List[str] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_name = str(source.get("source", "")).strip()
            if source_name:
                source_logs.append(source_name)

        if source_logs:
            return source_logs
        return [f"idx:{fallback_id}"]
