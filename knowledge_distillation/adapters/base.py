from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List

from knowledge_distillation.models import NormalizedItem


class InputAdapter(ABC):
    """入力アダプタの抽象基底クラス。"""

    @abstractmethod
    def fetch_items(self) -> Iterable[Any]:
        """入力ソースから生データを取得する。"""

    @abstractmethod
    def normalize(self, items: Iterable[Any]) -> List[NormalizedItem]:
        """生データを共通形式へ変換する。"""

    def load(self) -> List[NormalizedItem]:
        """取得と正規化をまとめて実行する。"""
        return self.normalize(self.fetch_items())

