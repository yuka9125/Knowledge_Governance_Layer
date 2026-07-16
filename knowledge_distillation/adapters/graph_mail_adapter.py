from __future__ import annotations

import json
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Dict, Iterable, List

from knowledge_distillation.adapters.base import InputAdapter
from knowledge_distillation.models import NormalizedItem
from knowledge_distillation.models import utc_now_iso


class GraphMailAdapter(InputAdapter):
    """Graph Mail(JSON/EML)入力アダプタ。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def fetch_items(self) -> Iterable[Dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"メール入力が見つかりません: {self.path}")

        suffix = self.path.suffix.lower()
        if suffix == ".json":
            return self._read_json()
        if suffix == ".eml":
            return [self._read_eml()]
        raise ValueError("GraphMailAdapterは .json または .eml のみ対応です。")

    def normalize(self, items: Iterable[Dict[str, Any]]) -> List[NormalizedItem]:
        normalized: List[NormalizedItem] = []
        for i, item in enumerate(items):
            subject = str(item.get("subject", "") or "").strip()
            body = str(item.get("body", "") or "").strip()
            if not subject and not body:
                continue

            question = subject or (body[:120] + "..." if len(body) > 120 else body)
            source_text = "\n\n".join([p for p in [subject, body] if p]).strip()
            created_at = (
                str(item.get("received_at", "") or "").strip()
                or str(item.get("created_at", "") or "").strip()
                or ""
            )
            metadata = {
                "row_index": i,
                "path": str(self.path),
                "message_id": item.get("message_id", ""),
                "from": item.get("from", ""),
                "to": item.get("to", ""),
                "received_at": item.get("received_at", ""),
                "subject": subject,
                "raw": item,
            }
            normalized.append(
                NormalizedItem(
                    question=question,
                    source_text=source_text or question,
                    answer="",
                    category=str(item.get("category", "") or "").strip(),
                    source="graph_mail",
                    created_at=created_at or utc_now_iso(),
                    metadata=metadata,
                )
            )
        return normalized

    def _read_json(self) -> List[Dict[str, Any]]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "value" in data and isinstance(data["value"], list):
                data = data["value"]
            else:
                data = [data]
        if not isinstance(data, list):
            raise ValueError("JSONは配列またはオブジェクトである必要があります。")

        normalized: List[Dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            normalized.append(self._normalize_json_mail(item))
        return normalized

    def _normalize_json_mail(self, item: Dict[str, Any]) -> Dict[str, Any]:
        # Graphレスポンス形式と簡易形式の両方を吸収する。
        subject = item.get("subject") or item.get("title") or ""
        body = item.get("body", "")
        if isinstance(body, dict):
            body = body.get("content", "")
        sender = item.get("from", "")
        if isinstance(sender, dict):
            sender = (
                sender.get("emailAddress", {}).get("address")
                or sender.get("address")
                or ""
            )
        recipients = item.get("toRecipients", item.get("to", ""))
        if isinstance(recipients, list):
            resolved = []
            for r in recipients:
                if isinstance(r, dict):
                    resolved.append(
                        r.get("emailAddress", {}).get("address")
                        or r.get("address")
                        or ""
                    )
                else:
                    resolved.append(str(r))
            recipients = ", ".join([x for x in resolved if x])

        return {
            "message_id": item.get("id", item.get("message_id", "")),
            "subject": str(subject),
            "body": str(body),
            "from": str(sender),
            "to": str(recipients),
            "received_at": str(
                item.get("receivedDateTime", item.get("received_at", ""))
            ),
            "category": str(item.get("category", "")),
        }

    def _read_eml(self) -> Dict[str, Any]:
        with self.path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        to = msg.get("to", "")
        date = msg.get("date", "")
        body = self._extract_eml_text(msg)

        return {
            "message_id": msg.get("message-id", ""),
            "subject": subject,
            "body": body,
            "from": sender,
            "to": to,
            "received_at": date,
            "category": "",
        }

    def _extract_eml_text(self, msg: Any) -> str:
        if msg.is_multipart():
            parts: List[str] = []
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/plain":
                    try:
                        parts.append(part.get_content())
                    except Exception:  # noqa: BLE001
                        continue
            return "\n".join(parts).strip()
        try:
            return str(msg.get_content()).strip()
        except Exception:  # noqa: BLE001
            return ""
