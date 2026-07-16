#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Governed Knowledge Plugin（Phase D）。

Semantic Kernel エージェントが使う「ツール（関数）」。
Governed Knowledge API の /knowledge/search を呼び、承認済みKnowledgeの
検索結果（answerable / answer / reason）をJSON文字列で返す。

HTTPクライアントを注入できるようにしてあり、テストでは FastAPI の
TestClient を差し込むことで実キー・実サーバ無しで検証できる。
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from semantic_kernel.functions import kernel_function


DEFAULT_API_BASE = "http://127.0.0.1:8000"


class GovernedKnowledgePlugin:
    """承認済みKnowledge検索をエージェントのツールとして提供する。"""

    def __init__(
        self,
        api_base: str | None = None,
        http_client: Optional[Any] = None,
        timeout: float = 10.0,
    ) -> None:
        # http_client を渡した場合（TestClient等）は api_base="" でも動く
        self.api_base = (
            api_base
            if api_base is not None
            else os.getenv("GOVERNED_API_BASE", DEFAULT_API_BASE)
        ).rstrip("/")
        self._http_client = http_client
        self.timeout = timeout

    @kernel_function(
        name="search_approved_knowledge",
        description=(
            "社内の承認済みKnowledgeのみを検索する。未承認・Draftは返らない。"
            "answerable=true のとき answer に承認済み回答が入る。"
            "answerable=false のときは該当なし（推測せずエスカレーションすること）。"
        ),
    )
    def search_approved_knowledge(self, question: str) -> str:
        """承認済みKnowledgeを検索し、結果JSON文字列を返す。"""
        url = f"{self.api_base}/knowledge/search"
        params = {"q": question}

        client = self._http_client
        owns_client = False
        if client is None:
            client = httpx.Client(timeout=self.timeout)
            owns_client = True
        try:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as exc:
            # APIに届かない/エラー時も、推測させないため answerable=false を返す
            return json.dumps(
                {
                    "answerable": False,
                    "reason": f"knowledge api error: {exc}",
                    "fallback": "human_review",
                },
                ensure_ascii=False,
            )
        finally:
            if owns_client:
                client.close()
