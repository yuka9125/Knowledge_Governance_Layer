#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 3: Embedding重複除去システム

【機能】
- Phase 3-1: Q内重複除去（Embeddingベースの階層的クラスタリング）
- Phase 3-2: 既存FAQとの重複除去（Embeddingで照合）
- Phase0のGIDを引き継ぎ、統合時にマッピング更新
- 最終結果Excel出力（3シート構成）
- ★修正: FAQ除外データ（回答=「-」）はEmbedding対象外
- ★修正: 最終結果CSV/JSONからFAQ除外データを除外
- ★修正: 信頼度（confidence_score）を引き継ぎ

【使用タイミング】
- Phase 2（文字列類似度）の後
"""

import os
import re
import html
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from knowledge_distillation.knowledge_output_utils import (
    build_judgement_reason,
    build_existing_faq_comparison_label,
    build_existing_faq_diff_reason,
    build_recommended_action,
    classify_p32_result,
    determine_risk_level,
    should_output_to_sheet1,
    build_source_logs,
    build_review_reason,
)

from knowledge_distillation.verification_excel import (
    VerificationExcelWriter,
    DuplicateGroup,
    GroupCandidate,
    ProcessingRecord,
    GIDTracker,
    get_adopted_indices,
)


# ================================================================================
# ログ設定
# ================================================================================
def setup_logging():
    """ログ設定の初期化"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ================================================================================
# Azure OpenAI クライアント初期化
# ================================================================================
def initialize_azure_openai_client():
    """AzureOpenAIクライアントの初期化"""
    try:
        from openai import AzureOpenAI

        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT_EMBEDDING")
        api_key = os.getenv("AZURE_OPENAI_API_KEY_EMBEDDING")

        if not azure_endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI環境変数が未設定です。\n"
                "以下の環境変数を設定してください:\n"
                "  - AZURE_OPENAI_ENDPOINT_EMBEDDING\n"
                "  - AZURE_OPENAI_API_KEY_EMBEDDING"
            )

        client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-02-01",
            azure_endpoint=azure_endpoint,
        )

        logger.info("AzureOpenAI APIクライアント初期化完了")
        return client

    except ImportError:
        raise ImportError(
            "openai ライブラリが未インストールです。\n"
            "以下のコマンドでインストールしてください:\n"
            "  pip install openai"
        )
    except Exception as e:
        raise RuntimeError(f"AzureOpenAI初期化エラー: {e}")


# ================================================================================
# ユーティリティ関数
# ================================================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """コサイン類似度計算"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def preprocess_text(text: str) -> str:
    """テキスト前処理"""
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_get_confidence(row, col_name="信頼度"):
    """信頼度を安全に取得"""
    val = row.get(col_name)
    if pd.isna(val) or val == "" or val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def safe_get_text(row, col_name: str) -> str:
    """DataFrame行から文字列を安全に取得する。"""
    if row is None or col_name not in row.index:
        return ""
    val = row.get(col_name)
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def extract_faq_id(faq_row, fallback_row_number: Optional[int] = None) -> str:
    """既存FAQ行からID候補を取得する。無ければExcel行番号で補完する。"""
    if faq_row is not None:
        for col_name in [
            "FAQ_ID",
            "faq_id",
            "FAQID",
            "id",
            "ID",
            "ナレッジID",
            "knowledge_id",
        ]:
            if col_name in faq_row.index:
                value = faq_row.get(col_name)
                if not pd.isna(value) and str(value).strip():
                    return str(value).strip()
    if fallback_row_number is not None:
        return f"FAQ行:{fallback_row_number}"
    return ""


# ================================================================================
# Phase 3-1: Q内重複除去
# ================================================================================
class Phase31Deduplicator:
    """Phase 3-1: Q内重複除去（Embeddingベース）"""

    def __init__(
        self,
        threshold: float = 0.75,
        embedding_batch_size: int = 100,
        sleep_between_batches: float = 0.1,
    ):
        self.threshold = threshold
        self.embedding_batch_size = embedding_batch_size
        self.sleep_between_batches = sleep_between_batches
        self.client = None
        self.similarity_matrix = None
        self.groups: List[DuplicateGroup] = []
        self.processing_records: Dict[int, ProcessingRecord] = {}
        self.gid_tracker: Optional[GIDTracker] = None

    def _init_client(self):
        """クライアント初期化"""
        if self.client is None:
            self.client = initialize_azure_openai_client()

    def _create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Embeddingベクトル作成"""
        self._init_client()

        embeddings = []
        total_batches = (
            len(texts) + self.embedding_batch_size - 1
        ) // self.embedding_batch_size

        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            batch_num = (i // self.embedding_batch_size) + 1

            logger.info(
                f"  Embeddingバッチ {batch_num}/{total_batches} ({len(batch)}件)"
            )

            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if batch_num < total_batches:
                time.sleep(self.sleep_between_batches)

        return np.array(embeddings)

    def _hierarchical_clustering(
        self,
        similarity_matrix: np.ndarray,
        threshold: float = 0.75,
    ) -> np.ndarray:
        """階層的クラスタリング"""
        sim_clipped = np.clip(similarity_matrix, -1.0, 1.0)
        np.fill_diagonal(sim_clipped, 1.0)

        distance_matrix = 1 - sim_clipped
        distance_matrix = np.clip(distance_matrix, 0.0, None)
        np.fill_diagonal(distance_matrix, 0.0)

        condensed = squareform(distance_matrix, checks=False)
        Z = linkage(condensed, method="complete")

        # thresholdを使用（類似度threshold以上でグループ化）
        labels = fcluster(Z, t=1 - threshold, criterion="distance")

        return labels

    def find_duplicates(
        self,
        df: pd.DataFrame,
        question_col: str = "質問",
        answer_col: str = "回答",
        raw_overview_col: str = "概要",
        raw_response_col: str = "対応結果",
        category_col: str = "カテゴリ",
        keywords_col: str = "キーワード",
        link_names_col: str = "リンク名",
        processing_records: Optional[Dict[int, ProcessingRecord]] = None,
        gid_tracker: Optional[GIDTracker] = None,
    ) -> List[DuplicateGroup]:
        """Q内重複を検出"""
        print("=" * 50)
        print("📋 Phase 3-1: Q内重複除去（Embedding）開始")
        print(f"  対象件数: {len(df)}件")
        print(f"  採用閾値: {self.threshold}")
        print("=" * 50)

        self.processing_records = processing_records or {}
        self.gid_tracker = gid_tracker or GIDTracker()

        # ★修正: FAQ除外データを分離
        # 回答が「-」のデータはEmbedding対象外
        faq_excluded_mask = df[answer_col].astype(str) == "-"
        df_normal = df[~faq_excluded_mask].copy()
        df_excluded = df[faq_excluded_mask].copy()

        excluded_count = len(df_excluded)
        if excluded_count > 0:
            print(f"  📌 FAQ除外データ: {excluded_count}件（Embedding対象外）")

        # 通常データのみで処理
        if len(df_normal) > 0:
            # テキスト抽出
            texts = [
                preprocess_text(str(row[question_col]))
                for _, row in df_normal.iterrows()
            ]

            # Embedding作成
            logger.info("Embedding生成中...")
            embeddings = self._create_embeddings(texts)

            # 類似度行列計算
            logger.info("類似度行列計算中...")
            self.similarity_matrix = cosine_similarity(embeddings, embeddings)
            np.fill_diagonal(self.similarity_matrix, 0)

            # 階層的クラスタリング
            logger.info("階層的クラスタリング実行中...")
            labels = self._hierarchical_clustering(
                self.similarity_matrix, threshold=self.threshold
            )

            # グループ作成（通常データのみ）
            self.groups = self._create_groups(
                df=df_normal,
                labels=labels,
                question_col=question_col,
                answer_col=answer_col,
                raw_overview_col=raw_overview_col,
                raw_response_col=raw_response_col,
                category_col=category_col,
                keywords_col=keywords_col,
                link_names_col=link_names_col,
            )
        else:
            self.groups = []

        # ★修正: FAQ除外データを単独グループとして追加
        for idx in df_excluded.index:
            row = df_excluded.loc[idx]
            question = str(row.get(question_col, ""))
            answer = str(row.get(answer_col, ""))
            raw_overview = preprocess_text(str(row.get(raw_overview_col, "")))
            raw_response = preprocess_text(str(row.get(raw_response_col, "")))
            category = str(row.get(category_col, ""))
            keywords = str(row.get(keywords_col, ""))
            link_names = (
                str(row.get(link_names_col, ""))
                if link_names_col in df_excluded.columns
                else ""
            )
            user_role = (
                str(row.get("立場", ""))
                if "立場" in df_excluded.columns
                else ""
            )
            confidence_score = safe_get_confidence(row, "信頼度")

            if idx in self.processing_records:
                group_id = self.processing_records[idx].p0_gid
            else:
                group_id = idx

            candidate = GroupCandidate(
                original_idx=idx,
                rank=1,
                is_adopted=True,
                similarity=None,
                raw_overview=raw_overview,
                raw_response=raw_response,
                raw_response_length=len(raw_response),
                question=question,
                answer=answer,
                answer_length=len(answer),
                category=category,
                keywords=keywords,
                link_names=link_names,
                user_role=user_role,
                confidence_score=confidence_score,
            )

            # 処理履歴を更新（既に「FAQ対象外」の場合はそのまま）
            if idx in self.processing_records:
                record = self.processing_records[idx]
                record.confidence_score = confidence_score
                # 既にFAQ対象外なら変更しない
                if record.final_result != "FAQ対象外":
                    record.final_result = "FAQ対象外"
            else:
                self.processing_records[idx] = ProcessingRecord(
                    original_idx=idx,
                    p0_gid=group_id,
                    final_gid=group_id,
                    final_result="FAQ対象外",
                    raw_overview=raw_overview,
                    raw_response=raw_response,
                    question=question,
                    answer=answer,
                    confidence_score=confidence_score,
                )

            group = DuplicateGroup(group_id=group_id, candidates=[candidate])
            self.groups.append(group)

        # 全レコードのfinal_gidを最終統合先に更新
        self.gid_tracker.update_all_records(self.processing_records)

        # 統計
        adopted_count = sum(
            1
            for g in self.groups
            for c in g.candidates
            if c.is_adopted and c.answer != "-"
        )
        deleted_count = len(df) - adopted_count

        print("=" * 50)
        print("📊 Phase 3-1: 結果")
        print(f"  グループ数: {len(self.groups)}")
        print(f"  採用: {adopted_count}件")
        print(f"  削除: {deleted_count}件")
        print(f"  FAQ対象外: {excluded_count}件")
        print("=" * 50)

        return self.groups

    def _create_groups(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        question_col: str,
        answer_col: str,
        raw_overview_col: str,
        raw_response_col: str,
        category_col: str,
        keywords_col: str,
        link_names_col: str,
    ) -> List[DuplicateGroup]:
        """グループ作成"""
        groups = []
        unique_labels = sorted(set(labels))

        # df.indexからdf内の位置へのマッピング
        idx_list = list(df.index)

        for label in unique_labels:
            member_positions = [i for i, l in enumerate(labels) if l == label]

            if len(member_positions) == 0:
                continue

            # 候補データ作成
            candidates_data = []
            for pos in member_positions:
                original_idx = idx_list[pos]
                row = df.iloc[pos]

                answer = str(row.get(answer_col, ""))
                answer_length = len(answer)
                confidence_score = safe_get_confidence(row, "信頼度")

                candidates_data.append(
                    {
                        "pos": pos,
                        "original_idx": original_idx,
                        "question": str(row.get(question_col, "")),
                        "answer": answer,
                        "answer_length": answer_length,
                        "raw_overview": str(row.get(raw_overview_col, "")),
                        "raw_response": str(row.get(raw_response_col, "")),
                        "category": str(row.get(category_col, "")),
                        "keywords": str(row.get(keywords_col, "")),
                        "link_names": (
                            str(row.get(link_names_col, ""))
                            if link_names_col in df.columns
                            else ""
                        ),
                        "user_role": (
                            str(row.get("立場", ""))
                            if "立場" in df.columns
                            else ""
                        ),
                        "confidence_score": confidence_score,
                    }
                )
            # 回答文字数でソート
            candidates_data.sort(
                key=lambda x: x["answer_length"], reverse=True
            )

            # 代表のP0_GIDをグループIDとして使用
            representative_original_idx = candidates_data[0]["original_idx"]
            if representative_original_idx in self.processing_records:
                group_id = self.processing_records[
                    representative_original_idx
                ].p0_gid
            else:
                group_id = representative_original_idx

            # 他のメンバーのP0_GIDを代表のGIDに統合
            for cdata in candidates_data[1:]:
                if cdata["original_idx"] in self.processing_records:
                    member_p0_gid = self.processing_records[
                        cdata["original_idx"]
                    ].p0_gid
                    if member_p0_gid is not None and member_p0_gid != group_id:
                        self.gid_tracker.merge(member_p0_gid, group_id)

            representative_pos = candidates_data[0]["pos"]

            # GroupCandidate作成
            candidates = []
            for rank, cdata in enumerate(candidates_data, 1):
                if cdata["pos"] == representative_pos:
                    similarity = None
                else:
                    similarity = float(
                        self.similarity_matrix[
                            representative_pos, cdata["pos"]
                        ]
                    )

                # 採用判定
                if rank == 1:
                    is_adopted = True
                elif similarity is not None and similarity < self.threshold:
                    is_adopted = True
                else:
                    is_adopted = False

                candidate = GroupCandidate(
                    original_idx=cdata["original_idx"],
                    rank=rank,
                    is_adopted=is_adopted,
                    similarity=similarity,
                    raw_overview=cdata["raw_overview"],
                    raw_response=cdata["raw_response"],
                    raw_response_length=len(cdata["raw_response"]),
                    question=cdata["question"],
                    answer=cdata["answer"],
                    answer_length=cdata["answer_length"],
                    category=cdata["category"],
                    keywords=cdata["keywords"],
                    link_names=cdata["link_names"],
                    user_role=cdata["user_role"],
                    confidence_score=cdata["confidence_score"],
                )
                candidates.append(candidate)

                # 処理履歴更新
                if cdata["original_idx"] in self.processing_records:
                    record = self.processing_records[cdata["original_idx"]]
                    record.p3_1_similarity = similarity
                    record.confidence_score = cdata["confidence_score"]
                    if not is_adopted:
                        record.final_result = "P3-1削除"
                else:
                    self.processing_records[cdata["original_idx"]] = (
                        ProcessingRecord(
                            original_idx=cdata["original_idx"],
                            p0_gid=group_id,
                            p3_1_similarity=similarity,
                            final_gid=group_id,
                            final_result="◯採用" if is_adopted else "P3-1削除",
                            raw_overview=cdata["raw_overview"],
                            raw_response=cdata["raw_response"],
                            question=cdata["question"],
                            answer=cdata["answer"],
                            confidence_score=cdata["confidence_score"],
                        )
                    )

            groups.append(
                DuplicateGroup(group_id=group_id, candidates=candidates)
            )

        return groups

    def get_adopted_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """採用データのみ取得（インデックス保持）"""
        adopted_indices = get_adopted_indices(self.groups)
        # 渡されたDataFrameに存在するインデックスのみフィルタリング
        valid_indices = [idx for idx in adopted_indices if idx in df.index]
        return df.loc[valid_indices].copy()

    def get_processing_records(self) -> Dict[int, ProcessingRecord]:
        """処理履歴取得"""
        return self.processing_records

    def get_gid_tracker(self) -> GIDTracker:
        """GIDトラッカー取得"""
        return self.gid_tracker


# ================================================================================
# Phase 3-2: 既存FAQとの重複除去
# ================================================================================
class Phase32Deduplicator:
    """Phase 3-2: 既存FAQとの重複除去"""

    def __init__(
        self,
        threshold: float = 0.75,
        embedding_batch_size: int = 100,
        sleep_between_batches: float = 0.1,
    ):
        self.threshold = threshold
        self.embedding_batch_size = embedding_batch_size
        self.sleep_between_batches = sleep_between_batches
        self.client = None
        self.cross_similarity = None
        self.groups: List[DuplicateGroup] = []
        self.processing_records: Dict[int, ProcessingRecord] = {}
        self.gid_tracker: Optional[GIDTracker] = None

    def _init_client(self):
        """クライアント初期化"""
        if self.client is None:
            self.client = initialize_azure_openai_client()

    def _create_embeddings(
        self, texts: List[str], text_type: str = "Q"
    ) -> np.ndarray:
        """Embedding作成"""
        self._init_client()

        embeddings = []
        total_batches = (
            len(texts) + self.embedding_batch_size - 1
        ) // self.embedding_batch_size

        for i in range(0, len(texts), self.embedding_batch_size):
            batch = texts[i : i + self.embedding_batch_size]
            batch_num = (i // self.embedding_batch_size) + 1

            logger.info(
                f"  {text_type} Embeddingバッチ {batch_num}/{total_batches} ({len(batch)}件)"
            )

            response = self.client.embeddings.create(
                model="text-embedding-3-large",
                input=batch,
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

            if batch_num < total_batches:
                time.sleep(self.sleep_between_batches)

        return np.array(embeddings)

    def find_duplicates(
        self,
        df: pd.DataFrame,
        faq_df: pd.DataFrame,
        question_col: str = "質問",
        answer_col: str = "回答",
        raw_overview_col: str = "概要",
        raw_response_col: str = "対応結果",
        category_col: str = "カテゴリ",
        keywords_col: str = "キーワード",
        link_names_col: str = "リンク名",
        processing_records: Optional[Dict[int, ProcessingRecord]] = None,
        gid_tracker: Optional[GIDTracker] = None,
        phase31_groups: Optional[List[DuplicateGroup]] = None,
    ) -> List[DuplicateGroup]:
        """既存FAQとの重複を検出"""
        print("=" * 50)
        print("📋 Phase 3-2: 既存FAQとの重複除去開始")
        print(f"  Q件数: {len(df)}件")
        print(f"  FAQ件数: {len(faq_df)}件")
        print(f"  採用閾値: {self.threshold}")
        print("=" * 50)

        self.processing_records = processing_records or {}
        self.gid_tracker = gid_tracker or GIDTracker()
        self.faq_df = faq_df

        # ★修正: FAQ除外データを分離
        faq_excluded_mask = df[answer_col].astype(str) == "-"
        df_normal = df[~faq_excluded_mask].copy()
        df_excluded = df[faq_excluded_mask].copy()

        excluded_count = len(df_excluded)
        if excluded_count > 0:
            print(f"  📌 FAQ除外データ: {excluded_count}件（Embedding対象外）")

        # 通常データのみで処理
        if len(df_normal) > 0:
            # Qテキスト（質問で照合）。更新/矛盾は「回答が違う」のが本質なので、
            # 回答を混ぜると同一質問の類似度が下がり取りこぼす。質問のみで照合する。
            q_texts = [
                preprocess_text(str(row[question_col]))
                for _, row in df_normal.iterrows()
            ]

            # FAQテキスト（質問）
            faq_texts = [
                preprocess_text(str(row["質問"]))
                for _, row in faq_df.iterrows()
            ]

            # Embedding作成
            logger.info("Q Embedding生成中...")
            q_embeddings = self._create_embeddings(q_texts, "Q")

            logger.info("FAQ Embedding生成中...")
            faq_embeddings = self._create_embeddings(faq_texts, "FAQ")

            # クロス類似度計算
            logger.info("クロス類似度計算中...")
            self.cross_similarity = cosine_similarity(
                q_embeddings, faq_embeddings
            )

            # df_normalのindexからdf_normal内の位置へのマッピング
            idx_to_pos = {idx: pos for pos, idx in enumerate(df_normal.index)}
        else:
            idx_to_pos = {}

        # グループを更新
        if phase31_groups:
            self.groups = self._update_groups_with_faq(
                df=df_normal,
                phase31_groups=phase31_groups,
                idx_to_pos=idx_to_pos,
            )
        else:
            self.groups = []

        # 全レコードのfinal_gidを最終統合先に更新
        self.gid_tracker.update_all_records(self.processing_records)

        # 統計
        adopted_count = sum(
            1
            for g in self.groups
            for c in g.candidates
            if c.is_adopted and c.answer != "-"
        )
        total_count = sum(len(g.candidates) for g in self.groups)
        deleted_count = total_count - adopted_count

        print("=" * 50)
        print("📊 Phase 3-2: 結果")
        print(f"  採用: {adopted_count}件")
        print(f"  削除: {deleted_count}件")
        print(f"  FAQ対象外: {excluded_count}件")
        print("=" * 50)

        return self.groups

    def _update_groups_with_faq(
        self,
        df: pd.DataFrame,
        phase31_groups: List[DuplicateGroup],
        idx_to_pos: Dict[int, int],
    ) -> List[DuplicateGroup]:
        """Phase 3-1のグループを更新"""
        updated_groups = []

        for group in phase31_groups:
            updated_candidates = []

            for candidate in group.candidates:
                # FAQ除外データはそのまま通過
                if candidate.answer == "-":
                    updated_candidates.append(candidate)
                    continue

                # このデータがdfに存在するか確認
                if candidate.original_idx not in idx_to_pos:
                    # Phase 3-1で不採用になったデータはそのまま
                    updated_candidates.append(candidate)
                    continue

                pos = idx_to_pos[candidate.original_idx]

                # FAQとの最大類似度を取得
                max_faq_similarity = float(np.max(self.cross_similarity[pos]))
                max_faq_idx = int(np.argmax(self.cross_similarity[pos]))

                # 処理履歴更新
                if candidate.original_idx in self.processing_records:
                    record = self.processing_records[candidate.original_idx]
                    record.p3_2_similarity = max_faq_similarity
                    # 一致したFAQ情報を記録
                    if max_faq_similarity >= self.threshold:
                        record.matched_faq_question = str(
                            self.faq_df.iloc[max_faq_idx]["質問"]
                        )
                        record.matched_faq_answer = str(
                            self.faq_df.iloc[max_faq_idx]["回答"]
                        )
                        record.matched_faq_row = max_faq_idx + 2
                        record.matched_faq_id = extract_faq_id(
                            self.faq_df.iloc[max_faq_idx],
                            record.matched_faq_row,
                        )

                # 採用判定更新
                if max_faq_similarity >= self.threshold:
                    # FAQと近い候補は削除ではなく、人間レビュー対象として確認に回す
                    new_candidate = GroupCandidate(
                        original_idx=candidate.original_idx,
                        rank=candidate.rank,
                        is_adopted=False,
                        similarity=candidate.similarity,
                        raw_overview=candidate.raw_overview,
                        raw_response=candidate.raw_response,
                        raw_response_length=candidate.raw_response_length,
                        question=candidate.question,
                        answer=candidate.answer,
                        answer_length=candidate.answer_length,
                        category=candidate.category,
                        keywords=candidate.keywords,
                        link_names=candidate.link_names,
                        confidence_score=candidate.confidence_score,
                    )

                    if candidate.original_idx in self.processing_records:
                        matched_answer = str(self.faq_df.iloc[max_faq_idx]["回答"])
                        self.processing_records[
                            candidate.original_idx
                        ].final_result = classify_p32_result(
                            max_similarity=max_faq_similarity,
                            candidate_answer=candidate.answer,
                            matched_faq_answer=matched_answer,
                        )

                    updated_candidates.append(new_candidate)
                else:
                    # FAQと重複なし → 採用状態維持
                    if (
                        candidate.is_adopted
                        and candidate.original_idx in self.processing_records
                    ):
                        self.processing_records[
                            candidate.original_idx
                        ].final_result = "◯採用"

                    updated_candidates.append(candidate)

            updated_groups.append(
                DuplicateGroup(
                    group_id=group.group_id,
                    candidates=updated_candidates,
                )
            )

        return updated_groups

    def get_adopted_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """採用データのみ取得（インデックス保持）"""
        adopted_indices = get_adopted_indices(self.groups)
        # 渡されたDataFrameに存在するインデックスのみフィルタリング
        # （Phase 3-1で既に不採用になったデータは df に含まれない）
        valid_indices = [idx for idx in adopted_indices if idx in df.index]
        return df.loc[valid_indices].copy()

    def get_processing_records(self) -> Dict[int, ProcessingRecord]:
        """処理履歴取得"""
        return self.processing_records

    def get_gid_tracker(self) -> GIDTracker:
        """GIDトラッカー取得"""
        return self.gid_tracker


# ================================================================================
# Phase 3 メイン処理 + 最終Excel出力
# ================================================================================
class Phase3Processor:
    """Phase 3全体の処理を管理"""

    def __init__(
        self,
        threshold_q: float = 0.75,
        threshold_faq: float = 0.70,
        embedding_batch_size: int = 100,
        output_dir: str = "data/outputs",
    ):
        self.threshold_q = threshold_q
        self.threshold_faq = threshold_faq
        self.embedding_batch_size = embedding_batch_size
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.phase31_deduplicator = None
        self.phase32_deduplicator = None
        self.final_groups: List[DuplicateGroup] = []
        self.processing_records: Dict[int, ProcessingRecord] = {}
        self.gid_tracker: Optional[GIDTracker] = None
        self.source_df: Optional[pd.DataFrame] = None
        self.faq_checked: bool = False
        # 最終結果の付随出力（Excelと同じ内容をJSON/CSVでも残す）
        self.final_json_path: Optional[str] = None
        self.final_csv_path: Optional[str] = None

    def run(
        self,
        df: pd.DataFrame,
        faq_df: Optional[pd.DataFrame] = None,
        question_col: str = "質問",
        answer_col: str = "回答",
        raw_overview_col: str = "概要",
        raw_response_col: str = "対応結果",
        category_col: str = "カテゴリ",
        keywords_col: str = "キーワード",
        link_names_col: str = "リンク名",
        processing_records: Optional[Dict[int, ProcessingRecord]] = None,
        gid_tracker: Optional[GIDTracker] = None,
    ) -> Tuple[pd.DataFrame, Dict[int, ProcessingRecord], GIDTracker, str]:
        """
        Phase 3を実行

        Returns:
            (最終採用DataFrame, 処理履歴, GIDトラッカー, 最終Excelパス)
        """
        self.processing_records = processing_records or {}
        self.gid_tracker = gid_tracker or GIDTracker()
        self.source_df = df.copy()
        self.faq_checked = faq_df is not None and len(faq_df) > 0

        # Phase 3-1: Q内重複除去
        self.phase31_deduplicator = Phase31Deduplicator(
            threshold=self.threshold_q,
            embedding_batch_size=self.embedding_batch_size,
        )

        self.phase31_deduplicator.find_duplicates(
            df=df,
            question_col=question_col,
            answer_col=answer_col,
            raw_overview_col=raw_overview_col,
            raw_response_col=raw_response_col,
            category_col=category_col,
            keywords_col=keywords_col,
            link_names_col=link_names_col,
            processing_records=self.processing_records,
            gid_tracker=self.gid_tracker,
        )

        self.processing_records = (
            self.phase31_deduplicator.get_processing_records()
        )
        self.gid_tracker = self.phase31_deduplicator.get_gid_tracker()
        phase31_groups = self.phase31_deduplicator.groups

        # Phase 3-1の採用データを取得
        df_after_31 = self.phase31_deduplicator.get_adopted_dataframe(df)

        # Phase 3-2: 既存FAQとの重複除去
        if faq_df is not None and len(faq_df) > 0:
            self.phase32_deduplicator = Phase32Deduplicator(
                threshold=self.threshold_faq,
                embedding_batch_size=self.embedding_batch_size,
            )

            self.phase32_deduplicator.find_duplicates(
                df=df_after_31,
                faq_df=faq_df,
                question_col=question_col,
                answer_col=answer_col,
                raw_overview_col=raw_overview_col,
                raw_response_col=raw_response_col,
                category_col=category_col,
                keywords_col=keywords_col,
                link_names_col=link_names_col,
                processing_records=self.processing_records,
                gid_tracker=self.gid_tracker,
                phase31_groups=phase31_groups,
            )

            self.processing_records = (
                self.phase32_deduplicator.get_processing_records()
            )
            self.gid_tracker = self.phase32_deduplicator.get_gid_tracker()
            self.final_groups = self.phase32_deduplicator.groups
            final_df = self.phase32_deduplicator.get_adopted_dataframe(
                df_after_31
            )
        else:
            print("📌 既存FAQなし: Phase 3-2をスキップ")
            self.final_groups = phase31_groups
            final_df = df_after_31

        # 最終Excel出力
        knowledge_candidates = self._build_knowledge_candidates(final_df)

        excel_path = self._export_final_excel(knowledge_candidates)

        # Excelと同じナレッジ候補をJSON / CSVでも出力する
        # （CSVはExcel以外のツールやスクリプトからの再利用向け）
        self._export_final_results(knowledge_candidates)

        return final_df, self.processing_records, self.gid_tracker, excel_path

    def _build_knowledge_candidates(
        self, final_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """最終出力用のナレッジ候補リストを構築"""
        # 回答が「-」のデータは最終ナレッジに含めない
        faq_only_df = final_df[final_df["回答"].astype(str) != "-"].copy()

        excluded_count = len(final_df) - len(faq_only_df)
        if excluded_count > 0:
            logger.info(f"最終出力からFAQ除外データを除外: {excluded_count}件")

        # 参照元データはPhase3入力（Phase2出力）を優先
        source_df = self.source_df if self.source_df is not None else final_df

        # クラスタIDはgroup_idの安定ソート順で採番
        sorted_groups = sorted(self.final_groups, key=lambda g: g.group_id)
        cluster_id_map = {
            group.group_id: f"c-{idx:03d}"
            for idx, group in enumerate(sorted_groups, start=1)
        }

        # 同一クラスタに属する元ログidxを収集（source_dfに残っている代表のみ）
        cluster_members: Dict[Any, List[Any]] = {}
        for original_idx, row in source_df.iterrows():
            if str(row.get("回答", "")) == "-":
                continue
            record = self.processing_records.get(original_idx)
            cluster_key = (
                record.final_gid
                if record is not None and record.final_gid is not None
                else original_idx
            )
            cluster_members.setdefault(cluster_key, []).append(original_idx)

        # 統合件数・候補参考行用：P0/P2/P3-1で削除された重複も含めた
        # 最終GIDごとの元ログidx一覧。削除側はsource_dfに残らないためrecordsから復元する。
        # （source_dfには残らないが processing_records には残るため、ここで数える）
        gid_total_counts: Dict[Any, int] = {}
        gid_record_indices: Dict[Any, List[Any]] = {}
        for original_idx, record in self.processing_records.items():
            if str(getattr(record, "answer", "") or "").strip() == "-":
                continue
            cluster_key = (
                record.final_gid
                if getattr(record, "final_gid", None) is not None
                else original_idx
            )
            gid_total_counts[cluster_key] = gid_total_counts.get(cluster_key, 0) + 1
            gid_record_indices.setdefault(cluster_key, []).append(original_idx)

        # 出力用に索引化
        faq_rows_by_idx = {idx: row for idx, row in faq_only_df.iterrows()}
        source_rows_by_idx = {idx: row for idx, row in source_df.iterrows()}

        knowledge_candidates: List[Dict[str, Any]] = []
        knowledge_seq = 1

        for group in sorted_groups:
            # Sheet1は採用候補とP3-2確認候補をレビュー対象として出力する。
            # 既存FAQとほぼ一致する候補は既存FAQ維持でよいためSheet1には出さない。
            review_candidates = []
            for candidate in group.candidates:
                if candidate.answer == "-":
                    continue
                record = self.processing_records.get(candidate.original_idx)
                final_result = (
                    record.final_result
                    if record is not None and record.final_result
                    else ("◯採用" if candidate.is_adopted else "")
                )
                max_similarity = (
                    float(record.p3_2_similarity)
                    if record is not None and record.p3_2_similarity is not None
                    else None
                )
                faq_comparison = build_existing_faq_comparison_label(
                    max_similarity=max_similarity,
                    faq_checked=self.faq_checked,
                )
                if should_output_to_sheet1(final_result, faq_comparison):
                    review_candidates.append(candidate)
            if not review_candidates:
                continue

            group_keys = {group.group_id}
            for candidate in group.candidates:
                record = self.processing_records.get(candidate.original_idx)
                if record is not None and record.final_gid is not None:
                    group_keys.add(record.final_gid)

            member_indices: List[Any] = []
            for key in group_keys:
                for member_idx in gid_record_indices.get(key, []):
                    if member_idx not in member_indices:
                        member_indices.append(member_idx)
            if not member_indices:
                # 保険: records側にクラスタ情報が見つからない場合はsource_df側を使う
                for key in group_keys:
                    for member_idx in cluster_members.get(key, []):
                        if member_idx not in member_indices:
                            member_indices.append(member_idx)
            if not member_indices:
                # 最終保険: 採用候補のidxを使用
                member_indices = [
                    candidate.original_idx for candidate in review_candidates
                ]

            # 統合件数：P0/P2削除の重複も含めた総数（無ければ代表数にフォールバック）
            member_total = len(member_indices)

            # 既存仕様に合わせて「回答候補1〜3」を保持（Excel表示用）
            # 候補1: 採用回答、候補2/3: P0/P2/P3-1で統合された削除側の回答。
            review_candidate_indices = {
                candidate.original_idx for candidate in review_candidates
            }
            main_answers = {
                str(candidate.answer).strip()
                for candidate in review_candidates
                if str(candidate.answer).strip()
            }
            other_answers: List[str] = []
            duplicate_results = ("P0削除", "P2削除", "P3-1削除")
            for member_idx in sorted(
                member_indices,
                key=lambda idx: (
                    -len(str(getattr(self.processing_records.get(idx), "answer", "") or "")),
                    idx,
                ),
            ):
                if member_idx in review_candidate_indices:
                    continue
                member_record = self.processing_records.get(member_idx)
                if member_record is None:
                    continue
                if not str(member_record.final_result or "").startswith(
                    duplicate_results
                ):
                    continue
                answer_text = str(member_record.answer or "").strip()
                if not answer_text or answer_text == "-" or answer_text in main_answers:
                    continue
                if answer_text not in other_answers:
                    other_answers.append(answer_text)
                if len(other_answers) >= 2:
                    break

            titles_by_index: Dict[Any, str] = {}
            if "件名" in source_df.columns:
                for member_idx in member_indices:
                    if member_idx in source_df.index:
                        titles_by_index[member_idx] = str(
                            source_df.at[member_idx, "件名"]
                        )
                    elif member_idx in self.processing_records:
                        record = self.processing_records[member_idx]
                        titles_by_index[member_idx] = (
                            str(record.raw_overview or record.question or "").strip()
                        )
            source_logs = build_source_logs(titles_by_index, member_indices)

            for adopted in review_candidates:
                row = faq_rows_by_idx.get(adopted.original_idx)
                if row is None:
                    row = source_rows_by_idx.get(adopted.original_idx)
                question = (
                    str(row.get("質問", adopted.question))
                    if row is not None
                    else str(adopted.question)
                )
                answer = (
                    str(row.get("回答", adopted.answer))
                    if row is not None
                    else str(adopted.answer)
                )
                category = (
                    str(row.get("カテゴリ", adopted.category))
                    if row is not None
                    else str(adopted.category)
                )
                link_names = (
                    safe_get_text(row, "リンク名") if row is not None else ""
                )
                if not link_names and row is not None:
                    # Sheet1で表示名が「参考リンク・資料名」の場合も拾う
                    link_names = safe_get_text(row, "参考リンク・資料名")
                if not link_names:
                    link_names = str(adopted.link_names)
                confidence = (
                    safe_get_confidence(row, "信頼度")
                    if row is not None
                    else float(adopted.confidence_score)
                )
                confidence_reason = safe_get_text(row, "信頼度理由")
                ai_risk_level = safe_get_text(row, "AIリスクレベル")
                ai_risk_reason = safe_get_text(row, "リスク理由")
                record = self.processing_records.get(adopted.original_idx)
                final_result = (
                    record.final_result
                    if record is not None and record.final_result
                    else "◯採用"
                )
                max_similarity = (
                    float(record.p3_2_similarity)
                    if record is not None and record.p3_2_similarity is not None
                    else None
                )
                existing_faq_comparison = build_existing_faq_comparison_label(
                    max_similarity=max_similarity,
                    faq_checked=self.faq_checked,
                )
                existing_faq_diff_reason = build_existing_faq_diff_reason(
                    max_similarity=max_similarity,
                    threshold=self.threshold_faq,
                    faq_checked=self.faq_checked,
                )
                risk_level = ai_risk_level or determine_risk_level(answer, category)
                review_reason = build_review_reason(
                    existing_faq_diff_reason=existing_faq_diff_reason,
                    risk_level=risk_level,
                    confidence=confidence,
                )

                knowledge_candidates.append(
                    {
                        "knowledge_id": f"k-{knowledge_seq:03d}",
                        "cluster_id": cluster_id_map.get(
                            group.group_id, f"c-{knowledge_seq:03d}"
                        ),
                        "group_id": group.group_id,
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "link_names": link_names,
                        "source_logs": source_logs,
                        "similar_logs_count": member_total,
                        "existing_faq_diff_reason": existing_faq_diff_reason,
                        "matched_faq_question": (
                            record.matched_faq_question
                            if record is not None
                            else ""
                        ),
                        "matched_faq_answer": (
                            record.matched_faq_answer if record is not None else ""
                        ),
                        "matched_faq_id": (
                            record.matched_faq_id if record is not None else ""
                        ),
                        "matched_faq_similarity": max_similarity,
                        "existing_faq_comparison": existing_faq_comparison,
                        "final_result": final_result,
                        "recommended_action": build_recommended_action(
                            final_result
                        ),
                        "judgement_reason": build_judgement_reason(
                            confidence=confidence,
                            similar_logs_count=member_total,
                            faq_comparison=existing_faq_comparison,
                            answer=answer,
                            category=category,
                            risk_level=risk_level,
                            final_result=final_result,
                            confidence_reason=confidence_reason,
                            risk_reason=ai_risk_reason,
                        ),
                        "risk_level": risk_level,
                        "review_status": "draft",
                        "confidence": confidence,
                        "review_reason": review_reason,
                        "review_result": "",
                        "answer_candidate_1": answer,
                        "answer_candidate_2": (
                            other_answers[0] if len(other_answers) > 0 else ""
                        ),
                        "answer_candidate_3": (
                            other_answers[1] if len(other_answers) > 1 else ""
                        ),
                    }
                )
                knowledge_seq += 1

        return knowledge_candidates

    def _export_final_excel(
        self, knowledge_candidates: List[Dict[str, Any]]
    ) -> str:
        """最終結果Excelを出力"""
        writer = VerificationExcelWriter(
            output_dir=self.output_dir,
            phase="FAQ_final_result",
            timestamp=self.timestamp,
        )

        # シート1: 最終ナレッジ候補一覧
        writer.add_final_faq_sheet(
            sheet_name="最終ナレッジ候補一覧",
            groups=self.final_groups,
            knowledge_candidates=knowledge_candidates,
            include_raw=True,
            processing_records=self.processing_records,
        )

        # シート2: 全データ処理履歴
        records_list = list(self.processing_records.values())

        writer.add_processing_history_sheet(
            sheet_name="全データ処理履歴",
            records=records_list,
        )

        return writer.save()

    def _export_final_results(
        self, knowledge_candidates: List[Dict[str, Any]]
    ) -> Tuple[str, str]:
        """最終結果をJSON/CSVで出力する（同一項目・同一列順）。"""
        os.makedirs(self.output_dir, exist_ok=True)
        csv_columns = [
            "knowledge_id",
            "cluster_id",
            "group_id",
            "question",
            "answer",
            "category",
            "source_logs",
            "similar_logs_count",
            "matched_faq_id",
            "matched_faq_question",
            "matched_faq_answer",
            "matched_faq_similarity",
            "existing_faq_comparison",
            "final_result",
            "recommended_action",
            "judgement_reason",
            "existing_faq_diff_reason",
            "risk_level",
            "review_status",
            "review_result",
            "confidence",
        ]

        # JSON出力
        json_path = os.path.join(
            self.output_dir, f"deduplicated_questions_{self.timestamp}.json"
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {key: item.get(key) for key in csv_columns}
                    for item in knowledge_candidates
                ],
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"最終JSON出力: {json_path}")

        # CSV出力（JSONと同じ列順）
        csv_path = os.path.join(
            self.output_dir, f"deduplicated_questions_{self.timestamp}.csv"
        )
        csv_rows = []
        for item in knowledge_candidates:
            csv_item = dict(item)
            # source_logs は配列なのでCSVではJSON文字列にして情報を落とさない
            csv_item["source_logs"] = json.dumps(
                csv_item.get("source_logs", []), ensure_ascii=False
            )
            csv_rows.append(csv_item)
        pd.DataFrame(csv_rows, columns=csv_columns).to_csv(
            csv_path, index=False, encoding="utf-8-sig"
        )
        logger.info(f"最終CSV出力: {csv_path}")

        self.final_json_path = json_path
        self.final_csv_path = csv_path
        return json_path, csv_path


# ================================================================================
# 便利関数
# ================================================================================
def run_phase3(
    df: pd.DataFrame,
    faq_df: Optional[pd.DataFrame] = None,
    threshold_q: float = 0.75,
    threshold_faq: float = 0.70,
    embedding_batch_size: int = 100,
    output_dir: str = "data/outputs",
    question_col: str = "質問",
    answer_col: str = "回答",
    raw_overview_col: str = "概要",
    raw_response_col: str = "対応結果",
    category_col: str = "カテゴリ",
    keywords_col: str = "キーワード",
    link_names_col: str = "リンク名",
    processing_records: Optional[Dict[int, ProcessingRecord]] = None,
    gid_tracker: Optional[GIDTracker] = None,
) -> Tuple[pd.DataFrame, Dict[int, ProcessingRecord], GIDTracker, str]:
    """
    Phase 3を実行する便利関数

    Returns:
        (最終採用DataFrame, 処理履歴, GIDトラッカー, 最終Excelパス)
    """
    processor = Phase3Processor(
        threshold_q=threshold_q,
        threshold_faq=threshold_faq,
        embedding_batch_size=embedding_batch_size,
        output_dir=output_dir,
    )

    return processor.run(
        df=df,
        faq_df=faq_df,
        question_col=question_col,
        answer_col=answer_col,
        raw_overview_col=raw_overview_col,
        raw_response_col=raw_response_col,
        category_col=category_col,
        keywords_col=keywords_col,
        link_names_col=link_names_col,
        processing_records=processing_records,
        gid_tracker=gid_tracker,
    )


# ================================================================================
# テスト用
# ================================================================================
if __name__ == "__main__":
    print("Phase 3モジュールのテスト")
    print("実際のテストにはAzure OpenAI APIキーが必要です")
