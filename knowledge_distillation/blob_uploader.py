#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
approved_knowledge.json を Azure Blob Storage へアップロードする。

serving 側は Blob を ETag で監視して自動再読込するため、ここでアップロードすれば
**再デプロイ無し**でナレッジが反映される。

Blob 設定（serving と同じ環境変数）が無い場合は何もしない（ローカル出力のみ）。
- APPROVED_KNOWLEDGE_BLOB_CONTAINER : コンテナ名（必須）
- APPROVED_KNOWLEDGE_BLOB_NAME      : blob名（既定 approved_knowledge.json）
- AZURE_STORAGE_CONNECTION_STRING   : 接続文字列（どちらか必須）
- AZURE_STORAGE_ACCOUNT_URL         : アカウントURL（マネージドID利用時）
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def blob_upload_configured() -> bool:
    """Blobアップロードに必要な設定が揃っているか。"""
    container = os.getenv("APPROVED_KNOWLEDGE_BLOB_CONTAINER")
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    return bool(container and (conn or account_url))


def _build_blob_client(container: str, blob_name: str) -> Any:
    conn = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    from azure.storage.blob import BlobClient

    if conn:
        return BlobClient.from_connection_string(conn, container, blob_name)
    if account_url:
        from azure.identity import DefaultAzureCredential

        return BlobClient(
            account_url, container, blob_name, credential=DefaultAzureCredential()
        )
    raise RuntimeError("Blob接続情報が未設定です（接続文字列 or アカウントURL）。")


def upload_approved_knowledge_to_blob(
    local_path: str | Path,
    blob_client: Optional[Any] = None,
) -> Optional[str]:
    """approved_knowledge.json を Blob にアップロードする。

    Blob設定が無ければ None を返す（何もしない）。成功時は "container/blob" を返す。
    blob_client を渡せばテストで差し替え可能。
    """
    container = os.getenv("APPROVED_KNOWLEDGE_BLOB_CONTAINER")
    blob_name = os.getenv("APPROVED_KNOWLEDGE_BLOB_NAME", "approved_knowledge.json")

    if blob_client is None:
        if not blob_upload_configured():
            return None
        blob_client = _build_blob_client(container, blob_name)

    data = Path(local_path).read_bytes()
    blob_client.upload_blob(data, overwrite=True)
    return f"{container or '(injected)'}/{blob_name}"
