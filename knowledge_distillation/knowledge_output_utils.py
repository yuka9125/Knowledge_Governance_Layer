"""
ナレッジ候補出力の共通ロジック。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


P32_EXACT_MATCH_THRESHOLD = 0.95
P32_SIMILAR_THRESHOLD = 0.85
# 更新候補のしきい値。質問＋回答での照合に合わせ、同テーマ・別表現の更新を
# 取りこぼさないよう 0.75 → 0.70 に緩める（取りこぼし減・人間レビューで担保）。
P32_REVIEW_THRESHOLD = 0.70


def normalize_text_for_conflict(text: str) -> str:
    """矛盾判定用にテキストを正規化する。"""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def build_existing_faq_diff_analysis(
    max_similarity: float | None,
    threshold: float,
    faq_checked: bool,
) -> str:
    """既存FAQとの差分分析結果を返す。"""
    if not faq_checked:
        return "既存FAQ照合未実施（FAQ未指定）"
    if max_similarity is None:
        return "既存FAQとの差分判定不可"
    if max_similarity >= threshold:
        return f"既存FAQ重複候補（最大類似度: {max_similarity:.3f}）"
    return f"既存FAQとの差分あり（最大類似度: {max_similarity:.3f}）"


def build_existing_faq_diff_reason(
    max_similarity: float | None,
    threshold: float,
    faq_checked: bool,
) -> str:
    """既存FAQ差分の短い理由を返す。"""
    if not faq_checked:
        return "FAQ未指定"
    if max_similarity is None:
        return "判定情報なし"
    if max_similarity >= threshold + 0.10:
        return "既存FAQとほぼ同一"
    if max_similarity >= threshold:
        return "既存FAQと類似度高"
    if max_similarity >= max(0.0, threshold - 0.10):
        return "回答内容に一部変更あり"
    return "回答内容に大きく変更あり"


def has_conflict_signal(candidate_answer: str, matched_faq_answer: str) -> bool:
    """回答同士に相反する運用の可能性を示す語があるか判定する。"""
    candidate = normalize_text_for_conflict(candidate_answer)
    faq = normalize_text_for_conflict(matched_faq_answer)
    conflict_pairs = [
        ("旧", "新"),
        ("古い", "最新"),
        ("古い", "新しい"),
        ("旧システム", "新システム"),
        ("廃止", "利用"),
        ("使用不可", "使用可能"),
        ("利用不可", "利用可能"),
        ("利用できません", "利用できます"),
        ("使用できません", "使用できます"),
        ("対応していません", "対応しています"),
        ("できません", "できます"),
        ("不要", "必要"),
    ]
    return any(
        (left in candidate and right in faq)
        or (right in candidate and left in faq)
        for left, right in conflict_pairs
    )


def classify_p32_result(
    max_similarity: float | None,
    candidate_answer: str,
    matched_faq_answer: str,
) -> str:
    """既存FAQ照合結果をP3-2確認ステータスへ分類する。"""
    if max_similarity is None:
        return "P3-2確認（既存FAQ更新候補）"

    # しきい値未満は「一致なし」＝新規
    if max_similarity < P32_REVIEW_THRESHOLD:
        return "◯採用"

    # ここから先は「質問が十分近い＝一致」とみなし、**回答の関係**で判定する。
    # （高類似度でも、回答が矛盾/更新なら見逃さない）
    candidate = normalize_text_for_conflict(candidate_answer)
    faq = normalize_text_for_conflict(matched_faq_answer)

    # 回答がほぼ同一 → 完全一致 / 類似（維持・統合）
    if candidate and faq and candidate == faq:
        if max_similarity >= P32_EXACT_MATCH_THRESHOLD:
            return "P3-2確認（既存FAQ完全一致）"
        return "P3-2確認（既存FAQ類似）"

    # 回答が矛盾（使用不可↔使用可能 等） → 矛盾可能性（人間レビュー必須）
    if has_conflict_signal(candidate_answer, matched_faq_answer):
        return "P3-2確認（既存FAQ矛盾可能性）"

    # 回答が異なる → 更新候補（既存FAQ更新）
    if candidate and faq and candidate != faq:
        return "P3-2確認（既存FAQ更新候補）"

    # 片方が空などの判断不能 → 類似扱い
    return "P3-2確認（既存FAQ類似）"


def build_recommended_action(final_result: str) -> str:
    """最終結果からSheet1向けの推奨アクションを返す。"""
    if final_result == "◯採用":
        return "新規FAQ作成"
    if final_result == "P3-2確認（既存FAQ完全一致）":
        return "既存FAQ維持"
    if final_result == "P3-2確認（既存FAQ類似）":
        return "既存FAQに統合"
    if final_result == "P3-2確認（既存FAQ更新候補）":
        return "既存FAQ更新"
    if final_result == "P3-2確認（既存FAQ矛盾可能性）":
        return "人間レビュー必須"
    if final_result == "FAQ対象外":
        return "FAQ対象外"
    return ""


def build_existing_faq_comparison_label(
    max_similarity: float | None,
    faq_checked: bool,
) -> str:
    """既存FAQとの近さをレビュー向けの説明文で返す。"""
    if not faq_checked or max_similarity is None:
        return "既存FAQなし/未照合"
    if max_similarity >= P32_EXACT_MATCH_THRESHOLD:
        return "既存FAQとほぼ一致"
    if max_similarity >= P32_SIMILAR_THRESHOLD:
        return "既存FAQと内容が近い（一部差分あり）"
    if max_similarity >= P32_REVIEW_THRESHOLD:
        return "既存FAQと内容が一部異なる"
    return "既存FAQと内容が異なる"


def should_output_to_sheet1(final_result: str, faq_comparison: str) -> bool:
    """Sheet1のレビュー対象として出力するか判定する。"""
    if faq_comparison == "既存FAQとほぼ一致":
        return False
    if final_result == "P3-2確認（既存FAQ完全一致）":
        return False
    if final_result == "P3-2確認（既存FAQ類似）":
        return False
    return final_result == "◯採用" or final_result.startswith("P3-2確認")


def _answer_clarity_label(answer: str) -> str:
    """回答の明確さを短く評価する。"""
    text = normalize_text_for_conflict(answer)
    if not text or text == "-":
        return "回答が空または対象外"
    if len(text) >= 80:
        return "回答は具体的"
    if len(text) >= 35:
        return "回答は要点あり"
    return "回答が短く補足確認が必要"


def _risk_reason(risk_level: str, answer: str, category: str) -> str:
    """リスクレベルの理由を説明する。"""
    target_text = f"{answer} {category}"
    critical_keywords = ["認証", "権限", "管理者"]
    high_keywords = ["人事", "給与", "セキュリティ", "パスワード", "個人情報"]
    accounting_keywords = ["会計", "経理", "請求", "支払", "精算", "承認", "申請"]
    operation_keywords = ["運用", "手順", "システム", "ワークフロー", "workflow"]

    found_critical = [kw for kw in critical_keywords if kw in target_text]
    found_high = [kw for kw in high_keywords if kw in target_text]
    found_accounting = [kw for kw in accounting_keywords if kw in target_text]
    found_operation = [kw for kw in operation_keywords if kw in target_text]

    if risk_level == "critical":
        return (
            f"リスクレベル{risk_level}。"
            f"{'・'.join(found_critical) or '認証・権限系'}に関係し、"
            "権限管理への影響が大きいため。"
        )
    if risk_level == "high":
        return (
            f"リスクレベル{risk_level}。"
            f"{'・'.join(found_high) or '人事・セキュリティ系'}に関係し、"
            "個人情報や運用へ影響し得るため。"
        )
    if found_accounting:
        return (
            f"リスクレベル{risk_level}。"
            f"{'・'.join(found_accounting)}に関係し、"
            "会計・承認フローの手戻りが起き得るため。"
        )
    if found_operation:
        return (
            f"リスクレベル{risk_level}。"
            f"{'・'.join(found_operation)}に関係するが、直接影響は限定的なため。"
        )
    return (
        f"リスクレベル{risk_level}。"
        "権限・会計・承認フローへの直接影響が小さいため。"
    )


def build_judgement_reason(
    confidence: float,
    similar_logs_count: int,
    faq_comparison: str,
    answer: str,
    category: str,
    risk_level: str,
    final_result: str,
    confidence_reason: str = "",
    risk_reason: str = "",
) -> str:
    """Sheet1向けに信頼度理由とリスク理由を2行形式で返す。"""
    if confidence_reason and risk_reason:
        return (
            f"信頼度理由: {normalize_text_for_conflict(confidence_reason)}"
            f"\nリスク理由: {normalize_text_for_conflict(risk_reason)}"
        )

    clarity = _answer_clarity_label(answer)
    if final_result.startswith("P3-2確認"):
        pending = "FAQ更新要否は未確認"
    elif final_result == "◯採用":
        pending = "公開前確認は未実施"
    else:
        pending = "レビュー未確認"

    confidence_reason = (
        f"信頼度理由: {confidence:.2f}。"
        f"元ログ{similar_logs_count}件、{faq_comparison}。"
        f"{clarity}。{pending}。"
    )
    risk_reason = f"リスク理由: {_risk_reason(risk_level, answer, category)}"
    return f"{confidence_reason}\n{risk_reason}"


def determine_risk_level(answer: str, category: str) -> str:
    """回答文とカテゴリからリスクレベルを判定する。"""
    target_text = f"{answer} {category}"
    critical_keywords = ["認証", "権限", "管理者"]
    high_keywords = ["人事", "給与", "セキュリティ", "パスワード", "個人情報"]

    if any(keyword in target_text for keyword in critical_keywords):
        return "critical"
    if any(keyword in target_text for keyword in high_keywords):
        return "high"
    return "low"


def build_review_reason(
    existing_faq_diff_reason: str,
    risk_level: str,
    confidence: float,
) -> str:
    """Excel向けのレビュー理由を生成する。"""
    if existing_faq_diff_reason in {"既存FAQとほぼ同一", "既存FAQと類似度高"}:
        return "既存FAQと重複する可能性があるため、既存記事との統合要否を確認してください。"
    if risk_level in {"critical", "high"}:
        return f"リスクレベルが{risk_level}のため、公開前レビューを推奨します。"
    if confidence < 0.7:
        return "信頼度が低めのため、内容確認を推奨します。"
    return "ドラフトの初稿です。公開前に内容確認してください。"


def build_source_logs(
    titles_by_index: Dict[Any, str], member_indices: List[Any]
) -> List[str]:
    """クラスタメンバーの件名一覧を生成（空欄時はidx補完）。"""
    source_logs: List[str] = []
    for member_idx in member_indices:
        title = str(titles_by_index.get(member_idx, "")).strip()
        source_logs.append(title if title else f"idx:{member_idx}")
    return source_logs
