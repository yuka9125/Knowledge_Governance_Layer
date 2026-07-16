#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
承認済みナレッジ取得元（ファイル / Blob）のテスト。

Blob は fake クライアントを注入し、実Azure・実キー無しで検証する。
再デプロイ無しの差し替え（ETag変化で再読込）も確認する。
"""

import json
import unittest

from serving.governed_knowledge_api import (
    BlobKnowledgeSource,
    GovernedKnowledgeService,
    build_default_source,
    FileKnowledgeSource,
)


APPROVED = [
    {
        "knowledge_id": "k-001",
        "question": "VPNに接続できません",
        "answer": "最新版VPNクライアントへ更新してください。",
        "category": "質問",
        "approved_status": "approved",
        "approved_at": "2026-05-29T10:00:00+09:00",
    }
]


class _FakeProps:
    def __init__(self, etag, last_modified):
        self.etag = etag
        self.last_modified = last_modified


class _FakeDownload:
    def __init__(self, raw: bytes):
        self._raw = raw

    def readall(self) -> bytes:
        return self._raw


class _FakeBlobClient:
    """get_blob_properties / download_blob だけ持つダミー。"""

    def __init__(self, items, etag="etag-1"):
        self.items = items
        self.etag = etag
        self.last_modified = "2026-05-30T00:00:00Z"
        self.download_calls = 0

    def set_items(self, items, etag):
        self.items = items
        self.etag = etag

    def get_blob_properties(self):
        return _FakeProps(self.etag, self.last_modified)

    def download_blob(self):
        self.download_calls += 1
        raw = json.dumps(self.items, ensure_ascii=False).encode("utf-8")
        return _FakeDownload(raw)


class _FakeEmbeddingBackend:
    def embed(self, texts):
        return None


class BlobKnowledgeSourceTest(unittest.TestCase):
    def test_signature_and_load(self):
        fake = _FakeBlobClient(APPROVED)
        src = BlobKnowledgeSource(container="c", blob_client=fake)
        self.assertIsNotNone(src.signature())
        data = src.load()
        self.assertEqual(data[0]["knowledge_id"], "k-001")

    def test_unreachable_blob_returns_none_and_empty(self):
        class _Broken:
            def get_blob_properties(self):
                raise RuntimeError("boom")

            def download_blob(self):
                raise RuntimeError("boom")

        src = BlobKnowledgeSource(container="c", blob_client=_Broken())
        self.assertIsNone(src.signature())
        self.assertEqual(src.load(), [])


class ServiceWithBlobTest(unittest.TestCase):
    def test_search_via_blob_source(self):
        fake = _FakeBlobClient(APPROVED)
        svc = GovernedKnowledgeService(
            source=BlobKnowledgeSource(container="c", blob_client=fake),
            embedding_backend=_FakeEmbeddingBackend(),
        )
        r = svc.search("VPNに接続できません")
        self.assertTrue(r["answerable"])
        self.assertEqual(r["knowledge_id"], "k-001")

    def test_hot_swap_without_redeploy(self):
        # 最初は空。後からBlobを差し替える（ETag変化）と再デプロイ無しで反映される
        fake = _FakeBlobClient([], etag="etag-empty")
        svc = GovernedKnowledgeService(
            source=BlobKnowledgeSource(container="c", blob_client=fake),
            embedding_backend=_FakeEmbeddingBackend(),
        )
        self.assertFalse(svc.search("VPNに接続できません")["answerable"])

        fake.set_items(APPROVED, etag="etag-2")
        self.assertTrue(svc.search("VPNに接続できません")["answerable"])

    def test_no_redownload_when_unchanged(self):
        # ETagが変わらなければ再ダウンロードしない（毎回叩かない）
        fake = _FakeBlobClient(APPROVED)
        svc = GovernedKnowledgeService(
            source=BlobKnowledgeSource(container="c", blob_client=fake),
            embedding_backend=_FakeEmbeddingBackend(),
        )
        svc.search("VPNに接続できません")
        svc.search("VPNに接続できません")
        self.assertEqual(fake.download_calls, 1)


class BuildDefaultSourceTest(unittest.TestCase):
    def test_defaults_to_file_without_blob_env(self):
        # Blob環境変数が無ければファイル取得元
        src = build_default_source()
        self.assertIsInstance(src, FileKnowledgeSource)


if __name__ == "__main__":
    unittest.main()
