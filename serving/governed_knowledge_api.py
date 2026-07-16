#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Governed Knowledge API（Phase C）。

承認済みKnowledge（approved_knowledge.json）だけを参照する読み取り専用API。
- GET /health                 : 稼働確認（200）
- GET /knowledge/search?q=...  : 承認済みのみ検索。該当があれば answerable=true、
                                 無ければ answerable=false（fallback=human_review）。

検索は Embedding（Azure OpenAI）優先・テキスト類似度フォールバック。
APIキーが無い／呼び出し失敗時はテキスト類似度のみで動作する（実キー無しでも動く）。
"""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from fastapi import FastAPI, Query

try:  # JSON読み込みは標準ライブラリのみ
    import json
except ImportError:  # pragma: no cover - 標準ライブラリなので実質発生しない
    raise


DEFAULT_APPROVED_PATH = "data/approved_knowledge.json"
# Embedding一致のしきい値（cosine類似度）
EMBEDDING_THRESHOLD = 0.80
# テキスト類似度（SequenceMatcher）のしきい値
TEXT_THRESHOLD = 0.86
MIN_CONTAINED_QUESTION_LENGTH = 8


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """2ベクトルのコサイン類似度。"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize(text: str) -> str:
    """比較用に空白を畳んで小文字化する。"""
    return " ".join(str(text or "").split()).lower()


def _is_approved(item: Dict[str, Any]) -> bool:
    """API側でも承認済みステータスだけに絞る。"""
    return str(item.get("approved_status", "")).strip().lower() == "approved"


class EmbeddingBackend(Protocol):
    """Embeddingベクトル化のインターフェース（テストで差し替え可能）。"""

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        """テキスト群をベクトル化する。利用不可なら None を返す。"""
        ...


class AzureEmbeddingBackend:
    """Azure OpenAI Embeddingを使うバックエンド。キー未設定時は None を返す。"""

    def __init__(self) -> None:
        self._client = None
        self._deployment = os.getenv(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
            os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        endpoint = os.getenv(
            "AZURE_OPENAI_ENDPOINT_EMBEDDING",
            os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        api_key = os.getenv(
            "AZURE_OPENAI_API_KEY_EMBEDDING",
            os.getenv("AZURE_OPENAI_API_KEY"),
        )
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        if not endpoint or not api_key:
            return None
        try:
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint,
            )
            return self._client
        except Exception:  # noqa: BLE001
            return None

    def embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        client = self._get_client()
        if not client or not texts:
            return None
        try:
            response = client.embeddings.create(model=self._deployment, input=texts)
            return [d.embedding for d in response.data]
        except Exception:  # noqa: BLE001
            return None


# --- 承認済みナレッジの取得元（ファイル / Azure Blob） -------------------
class KnowledgeSource(Protocol):
    """承認済みナレッジの取得元インターフェース。"""

    def signature(self) -> Optional[tuple]:
        """変更検知用の軽量トークン。取得元が無い/不達なら None。"""
        ...

    def load(self) -> List[Dict[str, Any]]:
        """承認済みナレッジの生データ（dictのリスト）を返す。"""
        ...


class FileKnowledgeSource:
    """ローカルファイルから approved_knowledge.json を読む。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def signature(self) -> Optional[tuple]:
        if not self.path.exists():
            return None
        stat = self.path.stat()
        # mtime解像度が粗い環境でも取りこぼさないよう size も含める
        return ("file", stat.st_mtime, stat.st_size)

    def load(self) -> List[Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
        return data if isinstance(data, list) else []


class BlobKnowledgeSource:
    """Azure Blob Storage から approved_knowledge.json を読む。

    再デプロイ無しでナレッジを差し替えられる。ETag/最終更新で変更検知し、
    変わった時だけダウンロードする。テスト用に blob_client を注入できる。
    """

    def __init__(
        self,
        container: str,
        blob_name: str = "approved_knowledge.json",
        connection_string: str | None = None,
        account_url: str | None = None,
        blob_client: Any | None = None,
    ) -> None:
        self.container = container
        self.blob_name = blob_name
        self._connection_string = connection_string
        self._account_url = account_url
        self._blob_client = blob_client

    def _client(self) -> Any:
        if self._blob_client is not None:
            return self._blob_client
        from azure.storage.blob import BlobClient

        if self._connection_string:
            self._blob_client = BlobClient.from_connection_string(
                self._connection_string, self.container, self.blob_name
            )
        elif self._account_url:
            from azure.identity import DefaultAzureCredential

            self._blob_client = BlobClient(
                self._account_url,
                self.container,
                self.blob_name,
                credential=DefaultAzureCredential(),
            )
        else:
            raise RuntimeError(
                "Blob接続情報が未設定です（接続文字列 or アカウントURLが必要）。"
            )
        return self._blob_client

    def signature(self) -> Optional[tuple]:
        try:
            props = self._client().get_blob_properties()
        except Exception:  # noqa: BLE001 - 不達時は「取得元なし」扱い
            return None
        return ("blob", str(getattr(props, "etag", "")), str(getattr(props, "last_modified", "")))

    def load(self) -> List[Dict[str, Any]]:
        try:
            raw = self._client().download_blob().readall()
        except Exception:  # noqa: BLE001
            return []
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []
        return data if isinstance(data, list) else []


def build_default_source() -> KnowledgeSource:
    """環境変数から既定の取得元を選ぶ。

    Blob設定（コンテナ名＋接続文字列 or アカウントURL）があれば Blob、
    無ければローカルファイル。
    """
    container = os.getenv("APPROVED_KNOWLEDGE_BLOB_CONTAINER")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    blob_name = os.getenv("APPROVED_KNOWLEDGE_BLOB_NAME", "approved_knowledge.json")
    if container and (connection_string or account_url):
        return BlobKnowledgeSource(
            container=container,
            blob_name=blob_name,
            connection_string=connection_string,
            account_url=account_url,
        )
    return FileKnowledgeSource(
        os.getenv("APPROVED_KNOWLEDGE_PATH", DEFAULT_APPROVED_PATH)
    )


class GovernedKnowledgeService:
    """承認済みKnowledgeの読み込みと検索を担う。"""

    def __init__(
        self,
        approved_path: str | Path | None = None,
        embedding_backend: EmbeddingBackend | None = None,
        embedding_threshold: float = EMBEDDING_THRESHOLD,
        text_threshold: float = TEXT_THRESHOLD,
        source: KnowledgeSource | None = None,
    ) -> None:
        if source is not None:
            self.source: KnowledgeSource = source
        elif approved_path is not None:
            self.source = FileKnowledgeSource(approved_path)
        else:
            self.source = build_default_source()
        self.embedding_backend = embedding_backend or AzureEmbeddingBackend()
        self.embedding_threshold = embedding_threshold
        self.text_threshold = text_threshold
        self._items: List[Dict[str, Any]] = []
        self._signature: Optional[tuple] = None

    # --- データ読み込み ---------------------------------------------------
    def _load_if_needed(self) -> None:
        """取得元（ファイル/Blob）を変更検知付きで読み込む。"""
        signature = self.source.signature()
        if signature is None:
            self._items = []
            self._signature = None
            return
        if self._signature is not None and signature == self._signature:
            return
        data = self.source.load()
        self._items = [
            item for item in data if isinstance(item, dict) and _is_approved(item)
        ]
        self._signature = signature

    # --- 検索 -------------------------------------------------------------
    def search(self, query: str) -> Dict[str, Any]:
        """承認済みのみ検索し、answerable分岐の結果を返す。"""
        self._load_if_needed()
        normalized_query = _normalize(query)
        if not normalized_query or not self._items:
            return self._not_found()

        questions = [str(item.get("question", "")) for item in self._items]

        # 1) 完全一致（正規化後）は最優先
        for idx, question in enumerate(questions):
            if _normalize(question) == normalized_query:
                return self._answer(self._items[idx])

        # 2) Embedding一致（利用可能なとき）
        best_idx = self._search_by_embedding(query, questions)
        if best_idx is None:
            # 3) テキスト類似度フォールバック
            best_idx = self._search_by_text(normalized_query, questions)

        if best_idx is None:
            return self._not_found()
        return self._answer(self._items[best_idx])

    def _search_by_embedding(
        self, query: str, questions: List[str]
    ) -> Optional[int]:
        embeddings = self.embedding_backend.embed([query, *questions])
        if not embeddings or len(embeddings) != len(questions) + 1:
            return None
        query_vec = embeddings[0]
        best_idx: Optional[int] = None
        best_score = -1.0
        for idx, vec in enumerate(embeddings[1:]):
            score = _cosine_similarity(query_vec, vec)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= self.embedding_threshold:
            return best_idx
        return None

    def _search_by_text(
        self, normalized_query: str, questions: List[str]
    ) -> Optional[int]:
        best_idx: Optional[int] = None
        best_score = -1.0
        for idx, question in enumerate(questions):
            normalized_question = _normalize(question)
            if not normalized_question:
                continue
            if (
                len(normalized_question) >= MIN_CONTAINED_QUESTION_LENGTH
                and normalized_question in normalized_query
            ):
                score = 1.0
            else:
                score = SequenceMatcher(
                    None, normalized_query, normalized_question
                ).ratio()
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None and best_score >= self.text_threshold:
            return best_idx
        return None

    # --- レスポンス整形 ---------------------------------------------------
    @staticmethod
    def _answer(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "answerable": True,
            "knowledge_id": str(item.get("knowledge_id", "")),
            "question": str(item.get("question", "")),
            "answer": str(item.get("answer", "")),
            "source": "approved_knowledge",
        }

    @staticmethod
    def _not_found() -> Dict[str, Any]:
        return {
            "answerable": False,
            "reason": "No approved knowledge found",
            "fallback": "human_review",
        }


def create_app(service: GovernedKnowledgeService | None = None) -> FastAPI:
    """FastAPIアプリを生成する。serviceを渡せばテストで差し替え可能。"""
    app = FastAPI(title="Governed Knowledge API", version="1.0.0")
    knowledge_service = service or GovernedKnowledgeService()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/knowledge/search")
    def knowledge_search(
        q: str = Query(..., description="検索したい質問文")
    ) -> Dict[str, Any]:
        return knowledge_service.search(q)

    return app


app = create_app()
