from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from knowledge_distillation.adapters import (
    CSVAdapter,
    GraphMailAdapter,
    InputAdapter,
    ServiceNowAdapter,
)
from knowledge_distillation.generator import FAQGenerator
from knowledge_distillation.models import FAQCandidate, NormalizedItem
from knowledge_distillation.normalizer import normalize_items_to_snow_csv
from knowledge_distillation.store import SQLiteKnowledgeStore


def create_adapter(
    adapter_name: str,
    path: str,
    question_col: str | None = None,
    answer_cols: Sequence[str] | None = None,
    source_text_cols: Sequence[str] | None = None,
) -> InputAdapter:
    name = adapter_name.lower().strip()
    if name == "csv":
        return CSVAdapter(
            path=path,
            question_col=question_col,
            answer_cols=answer_cols,
            source_text_cols=source_text_cols,
        )
    if name == "servicenow":
        return ServiceNowAdapter(
            path=path,
            question_col=question_col,
            answer_cols=answer_cols,
            source_text_cols=source_text_cols,
        )
    if name == "graphmail":
        return GraphMailAdapter(path=path)
    raise ValueError(f"未対応adapterです: {adapter_name}")


def ingest(
    adapter_name: str,
    path: str,
    db_path: str = "data/knowledge/knowledge.db",
    use_openai: bool = True,
    question_col: str | None = None,
    answer_cols: Sequence[str] | None = None,
    source_text_cols: Sequence[str] | None = None,
) -> dict:
    adapter = create_adapter(
        adapter_name=adapter_name,
        path=path,
        question_col=question_col,
        answer_cols=answer_cols,
        source_text_cols=source_text_cols,
    )
    normalized_items: List[NormalizedItem] = adapter.load()
    generator = FAQGenerator(use_openai=use_openai)
    faqs: List[FAQCandidate] = generator.generate(normalized_items)
    store = SQLiteKnowledgeStore(db_path=db_path)
    saved_count = store.save_faqs(faqs)
    return {
        "adapter": adapter_name,
        "input_count": len(normalized_items),
        "faq_count": len(faqs),
        "saved_count": saved_count,
        "db_path": str(Path(db_path)),
    }


def export(
    db_path: str = "data/knowledge/knowledge.db",
    out_dir: str = "data/outputs",
    formats: Sequence[str] | None = None,
) -> dict:
    """ナレッジをJSON / CSVで出力する（既定は両方）。"""
    selected = [fmt.lower().strip() for fmt in (formats or ("json", "csv"))]
    unsupported = [fmt for fmt in selected if fmt not in ("json", "csv")]
    if unsupported:
        raise ValueError(f"未対応の出力形式です: {', '.join(unsupported)}")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    store = SQLiteKnowledgeStore(db_path=db_path)

    result: dict = {"count": len(store.list_faqs())}
    if "json" in selected:
        result["json_path"] = str(store.export_json(out / "knowledge_export.json"))
    if "csv" in selected:
        result["csv_path"] = str(store.export_csv(out / "knowledge_export.csv"))
    return result


def normalize(
    adapter_name: str,
    path: str,
    out_path: str,
    question_col: str | None = None,
    answer_cols: Sequence[str] | None = None,
    source_text_cols: Sequence[str] | None = None,
    title_col: str | None = None,
    category_col: str | None = None,
    link_col: str | None = None,
) -> dict:
    adapter = create_adapter(
        adapter_name=adapter_name,
        path=path,
        question_col=question_col,
        answer_cols=answer_cols,
        source_text_cols=source_text_cols,
    )
    items = adapter.load()
    output = normalize_items_to_snow_csv(
        items=items,
        out_path=out_path,
        question_col=question_col,
        answer_cols=answer_cols,
        title_col=title_col,
        category_col=category_col,
        link_col=link_col,
    )
    return {
        "adapter": adapter_name,
        "input_count": len(items),
        "output_count": output["count"],
        "out_path": output["out_path"],
    }
