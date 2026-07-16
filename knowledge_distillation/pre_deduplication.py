#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テキスト重複除去モジュール（汎用版）

【機能】
- 完全一致チェック（ハッシュベース）
- 文字列類似度チェック（difflib + 階層的クラスタリング）
- Phase 0: 概要の完全一致＆類似度チェック（閾値0.9固定）
- Phase 2: 質問の完全一致＆類似度チェック（閾値は可変）
- FAQ除外データ（回答=「-」）はグループ化対象外
"""

import os
import re
import html
import hashlib
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
from janome.tokenizer import Tokenizer
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import (
    cosine_similarity as sklearn_cosine_similarity,
)
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from knowledge_distillation.verification_excel import (
    DuplicateGroup,
    GroupCandidate,
    ProcessingRecord,
    GIDTracker,
    get_adopted_indices,
)


class TextDeduplicator:
    """テキスト重複除去クラス（完全一致＆文字列類似度）"""

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        min_response_length: int = 30,
        output_dir: str = "data/intermediate",
    ):
        """
        初期化

        Args:
            similarity_threshold: 類似度閾値（これ以上は重複として統合）
            min_response_length: 最小回答文字数（これ以下は短文として除外）
            output_dir: 出力ディレクトリ
        """
        self.similarity_threshold = similarity_threshold
        self.min_response_length = min_response_length
        self.output_dir = output_dir
        self.groups: List[DuplicateGroup] = []
        self.similarity_matrix: Optional[np.ndarray] = None
        self.processing_records: Dict[int, ProcessingRecord] = {}
        self.gid_tracker: Optional[GIDTracker] = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.tokenizer = Tokenizer()

    def _preprocess_text(self, text: str) -> str:
        """テキスト前処理（正規化）"""
        if pd.isna(text) or not text:
            return ""
        text = str(text)

        # HTMLタグを除去
        text = re.sub(r"<[^>]+>", "", text)

        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        text = text.replace("　", " ")
        return text.lower()

    def _tokenize(self, text: str) -> str:
        """日本語をトークン化（スペース区切りに変換）"""
        tokens = self.tokenizer.tokenize(text, wakati=True)
        return " ".join(tokens)

    def _get_text_hash(self, text: str) -> str:
        """テキストのハッシュを取得"""
        normalized = self._preprocess_text(text)
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def _build_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """TF-IDF + コサイン類似度で類似度行列を構築"""
        # トークン化
        tokenized_texts = [self._tokenize(t) for t in texts]

        # TF-IDF行列作成
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(tokenized_texts)

        # コサイン類似度計算
        similarity_matrix = sklearn_cosine_similarity(tfidf_matrix)

        return similarity_matrix

    def _hierarchical_clustering(
        self,
        similarity_matrix: np.ndarray,
        cluster_threshold: float,
    ) -> np.ndarray:
        """階層的クラスタリングを実行"""
        # 類似度を0〜1にクリップ（浮動小数点誤差対策）
        similarity_clipped = np.clip(similarity_matrix, 0.0, 1.0)

        distance_matrix = 1 - similarity_clipped
        np.fill_diagonal(distance_matrix, 0)

        # 負の値をゼロにクリップ（念のため）
        distance_matrix = np.clip(distance_matrix, 0.0, None)

        condensed = squareform(distance_matrix)
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=1 - cluster_threshold, criterion="distance")

        return labels

    def find_duplicates(
        self,
        df: pd.DataFrame,
        target_col: str,
        sort_by_col: str,
        phase_name: str = "P0",
        question_col: Optional[str] = None,
        answer_col: Optional[str] = None,
        category_col: str = "カテゴリ",
        keywords_col: str = "キーワード",
        link_names_col: str = "リンク名",
        processing_records: Optional[Dict[int, ProcessingRecord]] = None,
        gid_tracker: Optional[GIDTracker] = None,
        check_short_response: bool = True,
    ) -> List[DuplicateGroup]:
        """
        重複を検出（完全一致＆文字列類似度）

        Args:
            df: 対象DataFrame
            target_col: チェック対象カラム（"概要" or "質問"）
            sort_by_col: ソート基準カラム（"対応結果" or "回答"）
            phase_name: フェーズ名（"P0" or "P2"）
            question_col: 質問カラム（Phase2用）
            answer_col: 回答カラム（Phase2用、FAQ除外判定に使用）
            category_col: カテゴリカラム
            keywords_col: キーワードカラム
            link_names_col: リンク名カラム
            processing_records: 既存の処理履歴
            gid_tracker: 既存のGIDトラッカー
            check_short_response: 短文回答チェックを行うか

        Returns:
            重複グループのリスト
        """
        print("=" * 50)
        print(f"📋 {phase_name}: 完全一致＆文字列類似度チェック開始")
        print(f"  対象件数: {len(df)}件")
        print(f"  対象カラム: {target_col}")
        print(f"  類似度閾値: {self.similarity_threshold}")
        print("=" * 50)

        self.processing_records = processing_records or {}
        self.gid_tracker = gid_tracker or GIDTracker()

        # FAQ除外データを分離（Phase 2のみ）
        if answer_col and answer_col in df.columns:
            faq_excluded_mask = df[answer_col].astype(str) == "-"
            df_normal = df[~faq_excluded_mask].copy()
            df_excluded = df[faq_excluded_mask].copy()
            excluded_count = len(df_excluded)
            if excluded_count > 0:
                print(
                    f"  📌 FAQ除外データ: {excluded_count}件（チェック対象外）"
                )
        else:
            df_normal = df.copy()
            df_excluded = pd.DataFrame()
            excluded_count = 0

        if len(df_normal) == 0:
            self.groups = []
            return self.groups

        # テキスト抽出・前処理
        texts = [
            self._preprocess_text(str(row[target_col]))
            for _, row in df_normal.iterrows()
        ]

        # ハッシュでグループ化（完全一致）
        df_normal = df_normal.copy()
        df_normal["_hash"] = [
            self._get_text_hash(str(row[target_col]))
            for _, row in df_normal.iterrows()
        ]

        hash_groups = {}
        for idx, hash_val in zip(df_normal.index, df_normal["_hash"]):
            if hash_val not in hash_groups:
                hash_groups[hash_val] = []
            hash_groups[hash_val].append(idx)

        # 類似度行列を構築
        print("  類似度行列を計算中...")
        self.similarity_matrix = self._build_similarity_matrix(texts)

        # 階層的クラスタリング
        print("  階層的クラスタリング実行中...")
        labels = self._hierarchical_clustering(
            self.similarity_matrix,
            cluster_threshold=self.similarity_threshold,
        )

        # インデックス→位置のマッピング
        idx_to_pos = {idx: pos for pos, idx in enumerate(df_normal.index)}

        # クラスタごとにグループ化
        cluster_groups = {}
        for idx, label in zip(df_normal.index, labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(idx)

        # グループ作成
        self.groups = self._create_groups(
            df=df_normal,
            cluster_groups=cluster_groups,
            hash_groups=hash_groups,
            idx_to_pos=idx_to_pos,
            target_col=target_col,
            sort_by_col=sort_by_col,
            phase_name=phase_name,
            question_col=question_col,
            answer_col=answer_col,
            category_col=category_col,
            keywords_col=keywords_col,
            link_names_col=link_names_col,
            check_short_response=check_short_response,
        )

        df_normal.drop("_hash", axis=1, inplace=True)

        # FAQ除外データを単独グループとして追加（Phase 2のみ）
        if len(df_excluded) > 0:
            self._add_excluded_data(
                df_excluded=df_excluded,
                target_col=target_col,
                sort_by_col=sort_by_col,
                question_col=question_col,
                answer_col=answer_col,
                category_col=category_col,
                keywords_col=keywords_col,
                link_names_col=link_names_col,
            )

        # GIDトラッカーを更新
        self.gid_tracker.update_all_records(self.processing_records)

        # 統計
        total_items = sum(len(g.candidates) for g in self.groups)
        duplicate_groups = [g for g in self.groups if len(g.candidates) > 1]
        adopted_count = sum(
            1
            for g in self.groups
            for c in g.candidates
            if c.is_adopted and c.answer != "-"
        )
        deleted_count = total_items - adopted_count - excluded_count

        print("=" * 50)
        print(f"📊 {phase_name}: 結果")
        print(f"  グループ総数: {len(self.groups)}")
        print(f"  重複グループ数: {len(duplicate_groups)}")
        print(f"  採用: {adopted_count}件")
        print(f"  削除: {deleted_count}件")
        if excluded_count > 0:
            print(f"  FAQ対象外: {excluded_count}件")
        print("=" * 50)

        return self.groups

    def _create_groups(
        self,
        df: pd.DataFrame,
        cluster_groups: Dict[int, List[int]],
        hash_groups: Dict[str, List[int]],
        idx_to_pos: Dict[int, int],
        target_col: str,
        sort_by_col: str,
        phase_name: str,
        question_col: Optional[str],
        answer_col: Optional[str],
        category_col: str,
        keywords_col: str,
        link_names_col: str,
        check_short_response: bool,
    ) -> List[DuplicateGroup]:
        """クラスタからグループを作成"""
        groups = []

        # ハッシュ→完全一致フラグのマッピング
        exact_match_hashes = {
            h for h, indices in hash_groups.items() if len(indices) > 1
        }

        for cluster_label, indices in cluster_groups.items():
            if len(indices) == 0:
                continue

            # 候補データを収集
            candidates_data = []
            for idx in indices:
                row = df.loc[idx]
                pos = idx_to_pos[idx]

                target_text = self._preprocess_text(
                    str(row.get(target_col, ""))
                )
                sort_by_text = self._preprocess_text(
                    str(row.get(sort_by_col, ""))
                )
                sort_by_length = len(sort_by_text)

                # 質問・回答（Phase 2用）
                question = (
                    str(row.get(question_col, "")) if question_col else ""
                )
                answer = str(row.get(answer_col, "")) if answer_col else ""

                # ハッシュを確認
                text_hash = df.loc[idx, "_hash"]
                is_exact_match = text_hash in exact_match_hashes

                # 信頼度
                confidence_score = 0.0
                if "信頼度" in df.columns:
                    val = row.get("信頼度")
                    if pd.notna(val) and val != "":
                        try:
                            confidence_score = float(val)
                        except (ValueError, TypeError):
                            pass

                candidates_data.append(
                    {
                        "pos": pos,
                        "original_idx": idx,
                        "target_text": target_text,
                        "sort_by_text": sort_by_text,
                        "sort_by_length": sort_by_length,
                        "text_hash": text_hash,
                        "is_exact_match": is_exact_match,
                        "raw_overview": self._preprocess_text(
                            str(row.get("概要", ""))
                        ),
                        "raw_response": self._preprocess_text(
                            str(row.get("対応結果", ""))
                        ),
                        "question": question,
                        "answer": answer,
                        "category": str(row.get(category_col, "")),
                        "keywords": (
                            str(row.get(keywords_col, ""))
                            if keywords_col in df.columns
                            else ""
                        ),
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

            # ソート基準の文字数でソート（降順）
            candidates_data.sort(
                key=lambda x: x["sort_by_length"], reverse=True
            )

            # グループIDの決定
            representative_idx = candidates_data[0]["original_idx"]
            if representative_idx in self.processing_records:
                group_id = self.processing_records[representative_idx].p0_gid
            else:
                group_id = representative_idx

            self.gid_tracker.register(group_id)

            # 他のメンバーのGIDを統合
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
                # 類似度計算
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
                    # 代表データ
                    if (
                        check_short_response
                        and cdata["sort_by_length"] <= self.min_response_length
                    ):
                        is_adopted = False
                        similarity_display = "短文"
                    else:
                        is_adopted = True
                        similarity_display = "-"
                else:
                    is_adopted = False
                    # 完全一致か類似か
                    if (
                        cdata["is_exact_match"]
                        and similarity is not None
                        and similarity >= 0.99
                    ):
                        similarity_display = 1.00
                    else:
                        similarity_display = similarity

                candidate = GroupCandidate(
                    original_idx=cdata["original_idx"],
                    rank=rank,
                    is_adopted=is_adopted,
                    similarity=(
                        similarity_display
                        if similarity_display != "-"
                        else None
                    ),
                    raw_overview=cdata["raw_overview"],
                    raw_response=cdata["raw_response"],
                    raw_response_length=len(cdata["raw_response"]),
                    question=cdata["question"],
                    answer=cdata["answer"],
                    answer_length=len(cdata["answer"]),
                    category=cdata["category"],
                    keywords=cdata["keywords"],
                    link_names=cdata["link_names"],
                    user_role=cdata["user_role"],
                    confidence_score=cdata["confidence_score"],
                )
                candidates.append(candidate)

                # 処理履歴を更新/作成
                final_result = self._determine_final_result(
                    is_adopted, similarity_display, phase_name
                )

                if cdata["original_idx"] in self.processing_records:
                    record = self.processing_records[cdata["original_idx"]]
                    if phase_name == "P0":
                        record.p0_similarity = (
                            similarity_display
                            if similarity_display not in ["-", "短文"]
                            else None
                        )
                    else:  # P2
                        record.p2_similarity = (
                            similarity_display
                            if similarity_display not in ["-", "短文"]
                            else None
                        )
                        record.question = cdata["question"]
                        record.answer = cdata["answer"]
                    record.confidence_score = cdata["confidence_score"]
                    if not is_adopted:
                        record.final_result = final_result
                else:
                    sim_val = (
                        similarity_display
                        if similarity_display not in ["-", "短文"]
                        else None
                    )
                    self.processing_records[cdata["original_idx"]] = (
                        ProcessingRecord(
                            original_idx=cdata["original_idx"],
                            p0_gid=group_id,
                            p0_similarity=(
                                sim_val if phase_name == "P0" else None
                            ),
                            p2_similarity=(
                                sim_val if phase_name == "P2" else None
                            ),
                            final_gid=group_id,
                            final_result=final_result,
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

    def _determine_final_result(
        self,
        is_adopted: bool,
        similarity_display,
        phase_name: str,
    ) -> str:
        """最終結果を決定"""
        if is_adopted:
            return "◯採用"

        if similarity_display == "短文":
            return f"{phase_name}削除（短文）"
        elif similarity_display == 1.00:
            return f"{phase_name}削除（完全一致）"
        elif similarity_display is not None:
            return f"{phase_name}削除（類似）"
        else:
            return f"{phase_name}削除"

    def _add_excluded_data(
        self,
        df_excluded: pd.DataFrame,
        target_col: str,
        sort_by_col: str,
        question_col: Optional[str],
        answer_col: Optional[str],
        category_col: str,
        keywords_col: str,
        link_names_col: str,
    ):
        """FAQ除外データを単独グループとして追加"""
        for idx in df_excluded.index:
            row = df_excluded.loc[idx]

            question = str(row.get(question_col, "")) if question_col else ""
            answer = str(row.get(answer_col, "")) if answer_col else ""
            raw_overview = self._preprocess_text(str(row.get("概要", "")))
            raw_response = self._preprocess_text(str(row.get("対応結果", "")))
            category = str(row.get(category_col, ""))
            user_role = (
                str(row.get("立場", ""))
                if "立場" in df_excluded.columns
                else ""
            )
            confidence_score = 0.0
            if "信頼度" in df_excluded.columns:
                val = row.get("信頼度")
                if pd.notna(val) and val != "":
                    try:
                        confidence_score = float(val)
                    except (ValueError, TypeError):
                        pass

            # グループIDの決定
            if idx in self.processing_records:
                group_id = self.processing_records[idx].p0_gid
            else:
                group_id = idx

            candidate = GroupCandidate(
                original_idx=idx,
                rank=1,
                is_adopted=True,  # 採用扱い（後続処理でスルーさせる）
                similarity=None,
                raw_overview=raw_overview,
                raw_response=raw_response,
                raw_response_length=len(raw_response),
                question=question,
                answer=answer,
                answer_length=len(answer),
                category=category,
                keywords="",
                link_names="",
                user_role=user_role,
                confidence_score=confidence_score,
            )

            # 処理履歴を更新
            if idx in self.processing_records:
                record = self.processing_records[idx]
                record.question = question
                record.answer = answer
                record.confidence_score = confidence_score
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

    def get_adopted_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """採用されたデータのみのDataFrameを取得（インデックス保持）"""
        adopted_indices = get_adopted_indices(self.groups)
        valid_indices = [idx for idx in adopted_indices if idx in df.index]
        return df.loc[valid_indices].copy()

    def get_representative_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """代表データのみのDataFrameを取得（グループごとにrank=1のデータ）"""
        representative_indices = []
        for group in self.groups:
            for candidate in group.candidates:
                if candidate.rank == 1 and candidate.is_adopted:
                    representative_indices.append(candidate.original_idx)
        valid_indices = [
            idx for idx in representative_indices if idx in df.index
        ]
        return df.loc[valid_indices].copy()

    def get_group_mapping(self) -> Dict[int, List[int]]:
        """代表→メンバーのマッピングを取得"""
        mapping = {}
        for group in self.groups:
            representative_idx = None
            member_indices = []
            for candidate in group.candidates:
                if candidate.rank == 1:
                    representative_idx = candidate.original_idx
                member_indices.append(candidate.original_idx)
            if representative_idx is not None:
                mapping[representative_idx] = member_indices
        return mapping

    def get_processing_records(self) -> Dict[int, ProcessingRecord]:
        """処理履歴を取得"""
        return self.processing_records

    def get_gid_tracker(self) -> GIDTracker:
        """GIDトラッカーを取得"""
        return self.gid_tracker


# =============================================================================
# 実行関数
# =============================================================================
def run_phase0(
    df: pd.DataFrame,
    output_dir: str = "data/intermediate",
    overview_col: str = "概要",
    response_col: str = "対応結果",
    link_names_col: str = "リンク名",
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Dict[int, List[int]],
    Dict[int, ProcessingRecord],
    GIDTracker,
]:
    """
    Phase 0を実行（概要の完全一致＆類似度チェック）

    Returns:
        (採用DataFrame, 代表DataFrame, グループマッピング, 処理履歴, GIDトラッカー)
    """
    deduplicator = TextDeduplicator(
        similarity_threshold=0.90,  # Phase 0は0.9固定
        min_response_length=30,
        output_dir=output_dir,
    )

    deduplicator.find_duplicates(
        df=df,
        target_col=overview_col,
        sort_by_col=response_col,
        phase_name="P0",
        link_names_col=link_names_col,
        check_short_response=True,
    )

    adopted_df = deduplicator.get_adopted_dataframe(df)
    representative_df = deduplicator.get_representative_dataframe(df)
    group_mapping = deduplicator.get_group_mapping()
    records = deduplicator.get_processing_records()
    gid_tracker = deduplicator.get_gid_tracker()

    return adopted_df, representative_df, group_mapping, records, gid_tracker


def run_phase2(
    df: pd.DataFrame,
    threshold: float = 0.90,
    output_dir: str = "data/intermediate",
    question_col: str = "質問",
    answer_col: str = "回答",
    category_col: str = "カテゴリ",
    keywords_col: str = "キーワード",
    link_names_col: str = "リンク名",
    processing_records: Optional[Dict[int, ProcessingRecord]] = None,
    gid_tracker: Optional[GIDTracker] = None,
) -> Tuple[pd.DataFrame, Dict[int, ProcessingRecord], GIDTracker]:
    """
    Phase 2を実行（質問の完全一致＆類似度チェック）

    Returns:
        (採用DataFrame, 処理履歴, GIDトラッカー)
    """
    deduplicator = TextDeduplicator(
        similarity_threshold=threshold,
        min_response_length=0,  # Phase 2では短文チェックしない
        output_dir=output_dir,
    )

    deduplicator.find_duplicates(
        df=df,
        target_col=question_col,
        sort_by_col=answer_col,
        phase_name="P2",
        question_col=question_col,
        answer_col=answer_col,
        category_col=category_col,
        keywords_col=keywords_col,
        link_names_col=link_names_col,
        processing_records=processing_records,
        gid_tracker=gid_tracker,
        check_short_response=False,
    )

    adopted_df = deduplicator.get_adopted_dataframe(df)
    records = deduplicator.get_processing_records()
    tracker = deduplicator.get_gid_tracker()

    return adopted_df, records, tracker


# =============================================================================
# テスト用
# =============================================================================
if __name__ == "__main__":
    # テストデータ
    test_data = {
        "件名": [
            "ログインできない",
            "ログインできない",
            "パスワード忘れ",
            "画面エラー",
        ],
        "カテゴリ": ["認証", "認証", "認証", "表示"],
        "概要": [
            "システムにログインできません",
            "システムにログインできません",
            "パスワードを忘れました",
            "画面が真っ白です",
        ],
        "対応結果": [
            "パスワードリセットを実施。再度ログインを確認済み。問題なく動作。",
            "パスワードリセットを実施",
            "パスワードリセット手順をご案内しました。",
            "キャッシュクリアで解決",
        ],
        "リンク名": ["手順書", "手順書;FAQ", "手順書", ""],
    }

    df = pd.DataFrame(test_data)
    print("テストデータ:")
    print(df)
    print()

    # Phase 0実行
    adopted_df, representative_df, group_mapping, records, gid_tracker = (
        run_phase0(df, output_dir="/tmp/test_output")
    )

    print(f"\nPhase 0 採用: {len(adopted_df)}件")
    print(f"Phase 0 代表: {len(representative_df)}件")
    print(f"グループマッピング: {group_mapping}")
    for idx, record in records.items():
        print(
            f"  idx={idx}: {record.final_result} (P0類似度={record.p0_similarity})"
        )
