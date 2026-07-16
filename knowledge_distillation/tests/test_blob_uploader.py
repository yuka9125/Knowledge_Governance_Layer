#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
approved_knowledge の Blob アップロードのテスト。

fake クライアント注入で実Azure無しに検証する。
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from knowledge_distillation.blob_uploader import (
    blob_upload_configured,
    upload_approved_knowledge_to_blob,
)


class _FakeBlobClient:
    def __init__(self):
        self.uploaded = None
        self.overwrite = None

    def upload_blob(self, data, overwrite=False):
        self.uploaded = bytes(data)
        self.overwrite = overwrite


class BlobUploaderTest(unittest.TestCase):
    def setUp(self):
        # Blob環境変数を退避してクリア
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (
                "APPROVED_KNOWLEDGE_BLOB_CONTAINER",
                "AZURE_STORAGE_CONNECTION_STRING",
                "AZURE_STORAGE_ACCOUNT_URL",
                "APPROVED_KNOWLEDGE_BLOB_NAME",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v
            else:
                os.environ.pop(k, None)

    def _tmp_json(self):
        d = Path(tempfile.mkdtemp())
        p = d / "approved_knowledge.json"
        p.write_text(json.dumps([{"knowledge_id": "k-1"}], ensure_ascii=False), encoding="utf-8")
        return p

    def test_not_configured_returns_none(self):
        self.assertFalse(blob_upload_configured())
        self.assertIsNone(upload_approved_knowledge_to_blob(self._tmp_json()))

    def test_configured_detection(self):
        os.environ["APPROVED_KNOWLEDGE_BLOB_CONTAINER"] = "c"
        os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "conn"
        self.assertTrue(blob_upload_configured())

    def test_upload_with_injected_client(self):
        os.environ["APPROVED_KNOWLEDGE_BLOB_CONTAINER"] = "approved-knowledge"
        fake = _FakeBlobClient()
        p = self._tmp_json()
        target = upload_approved_knowledge_to_blob(p, blob_client=fake)
        self.assertEqual(target, "approved-knowledge/approved_knowledge.json")
        self.assertTrue(fake.overwrite)
        self.assertEqual(json.loads(fake.uploaded.decode("utf-8"))[0]["knowledge_id"], "k-1")


if __name__ == "__main__":
    unittest.main()
