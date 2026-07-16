#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web アプリ（Phase E / デプロイのエントリポイント）。

1つの FastAPI アプリに以下を集約し、App Service にそのままデプロイできる。
- GET  /health           : 稼働確認（Governed Knowledge API 由来）
- GET  /knowledge/search : 承認済みKnowledge検索（同上）
- GET  /                 : 最小チャットUI（HTML）
- POST /chat             : 質問→回答エージェント応答

回答エージェント(SupportAgent)は Azure OpenAI(chat) が必要。未設定でも
アプリは起動し、/chat は500で落とさず「設定が必要」を返す。
エージェントの tool は GOVERNED_API_BASE 経由で自身の /knowledge/search を呼ぶ。
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from serving.governed_knowledge_api import GovernedKnowledgeService, create_app
from serving.support_agent import SupportAgent


WEB_DIR = Path(__file__).resolve().parent / "web"
INDEX_HTML = WEB_DIR / "index.html"
logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)


class ChatRequest(BaseModel):
    question: str


class _LocalKnowledgePlugin:
    """In-process plugin used by the web app to avoid self-HTTP calls."""

    def __init__(self, service: GovernedKnowledgeService) -> None:
        self._service = service

    def search_approved_knowledge(self, question: str) -> str:
        return json.dumps(self._service.search(question), ensure_ascii=False)


def _load_index_html() -> str:
    try:
        return INDEX_HTML.read_text(encoding="utf-8")
    except OSError:
        return "<html><body><h1>Knowledge Governance Chat</h1>"\
               "<p>UIファイルが見つかりません。</p></body></html>"


def create_web_app(
    knowledge_service: GovernedKnowledgeService | None = None,
    support_agent: Optional[SupportAgent] = None,
) -> FastAPI:
    """チャットUIを含む統合アプリを生成する。

    knowledge_service / support_agent を渡せばテストで差し替え可能。
    """
    local_service = knowledge_service or GovernedKnowledgeService()
    app = create_app(local_service)
    # support_agent はリクエスト時に遅延生成する（Azure未設定でも起動可能に）
    injected_agent = support_agent

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse(content=_load_index_html())

    @app.post("/chat")
    async def chat(req: ChatRequest) -> Any:
        question = (req.question or "").strip()
        if not question:
            return JSONResponse(
                status_code=400,
                content={"error": "質問が空です。"},
            )
        agent = injected_agent or SupportAgent(
            plugin=_LocalKnowledgePlugin(local_service)
        )
        try:
            answer = await agent.answer(question)
        except Exception as exc:  # noqa: BLE001
            # Azure OpenAI 未設定・デプロイ名不一致・APIエラー時もUIを500にしない。
            logger.warning("Support agent failed to answer: %s", exc)
            return JSONResponse(
                status_code=503,
                content={
                    "error": "回答エージェントが利用できません。",
                    "detail": "Azure OpenAI の設定またはデプロイ状態を確認してください。",
                },
            )
        return {"answer": answer}

    return app


app = create_web_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("serving.web_app:app", host="0.0.0.0", port=port)
