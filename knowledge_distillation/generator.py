from __future__ import annotations

import json
import os
from typing import List, Optional

from knowledge_distillation.deduplicator import EmbeddingDeduplicator
from knowledge_distillation.models import FAQCandidate, NormalizedItem


class FAQGenerator:
    """問い合わせデータからFAQ候補を生成する。"""

    def __init__(
        self,
        use_openai: bool = True,
        dedupe_threshold: float = 0.88,
    ):
        self.use_openai = use_openai
        self._chat_client = None
        self._chat_model = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4.1")
        self._deduplicator = EmbeddingDeduplicator(threshold=dedupe_threshold)

    def generate(self, items: List[NormalizedItem]) -> List[FAQCandidate]:
        candidates: List[FAQCandidate] = []
        for item in items:
            candidate = self._generate_one(item)
            if candidate:
                candidates.append(candidate)
        return self._deduplicator.deduplicate(candidates)

    def _generate_one(self, item: NormalizedItem) -> Optional[FAQCandidate]:
        ai_candidate = self._generate_with_openai(item)
        if ai_candidate:
            return ai_candidate

        # OpenAIが使えない場合のMVPフォールバック。
        answer = item.answer.strip() or "お問い合わせ内容を確認し、担当窓口へエスカレーションしてください。"
        category = item.category.strip() or "未分類"
        return FAQCandidate(
            question=item.question.strip(),
            answer=answer,
            category=category,
            confidence=0.6,
            sources=[
                {
                    "source": item.source,
                    "created_at": item.created_at,
                    "metadata": item.metadata,
                }
            ],
            metadata={"generator": "fallback"},
        )

    def _generate_with_openai(self, item: NormalizedItem) -> Optional[FAQCandidate]:
        if not self.use_openai:
            return None
        client = self._get_chat_client()
        if not client:
            return None

        prompt = self._build_prompt(item)
        try:
            response = client.chat.completions.create(
                model=self._chat_model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "あなたは問い合わせログからナレッジ候補の核となる質問・回答・カテゴリを抽出するアシスタントです。JSONのみ返してください。",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
        except Exception:  # noqa: BLE001
            return None

        question = str(data.get("question", "") or "").strip() or item.question.strip()
        answer = str(data.get("answer", "") or "").strip() or item.answer.strip()
        category = (
            str(data.get("category", "") or "").strip()
            or item.category.strip()
            or "未分類"
        )
        confidence = self._safe_float(data.get("confidence", 0.8), default=0.8)
        confidence = max(0.0, min(1.0, confidence))
        return FAQCandidate(
            question=question,
            answer=answer,
            category=category,
            confidence=confidence,
            sources=[
                {
                    "source": item.source,
                    "created_at": item.created_at,
                    "metadata": item.metadata,
                }
            ],
            metadata={"generator": "azure_openai"},
        )

    def _get_chat_client(self):
        if self._chat_client is not None:
            return self._chat_client
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        if not endpoint or not api_key:
            return None
        try:
            from openai import AzureOpenAI

            self._chat_client = AzureOpenAI(
                api_key=api_key,
                api_version="2025-04-01-preview",
                azure_endpoint=endpoint,
            )
            return self._chat_client
        except Exception:  # noqa: BLE001
            return None

    def _build_prompt(self, item: NormalizedItem) -> str:
        return f"""
次の問い合わせ情報からナレッジ候補のベースデータを1件作成してください。

- question: {item.question}
- source_text: {item.source_text}
- answer: {item.answer}
- category: {item.category}
- source: {item.source}

注意:
- この段階では question / answer / category / confidence のみを返してください。
- knowledge_id / cluster_id / source_logs / similar_logs_count / existing_faq_diff_reason / risk_level / review_status は後続の出力整形処理で付与されます。

出力JSONスキーマ:
{{
  "question": "ナレッジ候補の質問文",
  "answer": "ナレッジ候補の回答文",
  "category": "カテゴリ",
  "confidence": 0.0
}}
"""

    def _safe_float(self, value, default: float) -> float:
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return default
