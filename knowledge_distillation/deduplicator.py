from __future__ import annotations

import os
from difflib import SequenceMatcher
from typing import List

from knowledge_distillation.models import FAQCandidate


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingDeduplicator:
    """EmbeddingベースでFAQ候補を重複排除する。"""

    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold
        self._embedding_client = None
        self._embedding_model = os.getenv(
            "AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"
        )

    def deduplicate(self, candidates: List[FAQCandidate]) -> List[FAQCandidate]:
        if len(candidates) <= 1:
            return candidates

        texts = [c.question for c in candidates]
        embeddings = self._build_embeddings(texts)
        if embeddings:
            return self._dedupe_with_embeddings(candidates, embeddings)
        return self._dedupe_with_text_similarity(candidates)

    def _build_embeddings(self, texts: List[str]) -> List[List[float]]:
        client = self._get_embedding_client()
        if not client:
            return []
        try:
            response = client.embeddings.create(model=self._embedding_model, input=texts)
            return [d.embedding for d in response.data]
        except Exception:  # noqa: BLE001
            return []

    def _dedupe_with_embeddings(
        self, candidates: List[FAQCandidate], embeddings: List[List[float]]
    ) -> List[FAQCandidate]:
        kept: List[FAQCandidate] = []
        kept_embeddings: List[List[float]] = []
        for candidate, emb in zip(candidates, embeddings):
            dup_idx = -1
            best_score = -1.0
            for i, k_emb in enumerate(kept_embeddings):
                score = _cosine_similarity(emb, k_emb)
                if score > best_score:
                    best_score = score
                    dup_idx = i

            if best_score >= self.threshold and dup_idx >= 0:
                self._merge_candidate(kept[dup_idx], candidate, best_score)
            else:
                kept.append(candidate)
                kept_embeddings.append(emb)
        return kept

    def _dedupe_with_text_similarity(
        self, candidates: List[FAQCandidate]
    ) -> List[FAQCandidate]:
        kept: List[FAQCandidate] = []
        for candidate in candidates:
            dup_idx = -1
            best_score = -1.0
            for i, existing in enumerate(kept):
                score = SequenceMatcher(
                    None, candidate.question, existing.question
                ).ratio()
                if score > best_score:
                    best_score = score
                    dup_idx = i

            if best_score >= self.threshold and dup_idx >= 0:
                self._merge_candidate(kept[dup_idx], candidate, best_score)
            else:
                kept.append(candidate)
        return kept

    def _merge_candidate(
        self, base: FAQCandidate, new_item: FAQCandidate, score: float
    ) -> None:
        base.confidence = max(base.confidence, new_item.confidence * score)
        if len(new_item.answer) > len(base.answer):
            base.answer = new_item.answer
        if not base.category and new_item.category:
            base.category = new_item.category
        base.sources.extend(new_item.sources)
        base.metadata["merged_count"] = int(base.metadata.get("merged_count", 0)) + 1

    def _get_embedding_client(self):
        if self._embedding_client is not None:
            return self._embedding_client
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
        api_key = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")
        if not endpoint or not api_key:
            return None
        try:
            from openai import AzureOpenAI

            self._embedding_client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint,
            )
            return self._embedding_client
        except Exception:  # noqa: BLE001
            return None

