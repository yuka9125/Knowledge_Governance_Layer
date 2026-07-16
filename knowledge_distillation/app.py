#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAQ自動生成システム - Streamlit アプリケーション

処理フロー:
  Phase 0: 完全一致 & 短文回答チェック（クレンジング前）→ 検証Excel出力
  Phase 1: クレンジング（SNOW_Cleansing.py）
  Phase 1.5: 完全一致チェック（クレンジング後）→ 検証Excel出力 ★追加
  Phase 2: 文字列類似度チェック → 検証Excel出力
  Phase 3-1: Q内重複除去（Embedding）
  Phase 3-2: 既存FAQとの照合 → 最終結果Excel出力

★修正:
- インデックスを正しく保持してPhase間でデータを受け渡し
- Phase 1.5を追加（AIによる正規化後の重複を除去）
"""

import glob
import json
import os
import pandas as pd
import re
import streamlit as st
import sys
import tempfile
import time
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from knowledge_distillation.display_labels import display_label


def configure_console_encoding():
    """Windowsのcp932端末でも処理ログの絵文字で落ちないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_console_encoding()

# app.py の場所を基準にパスを解決（knowledge_distillation/ 配下に置いても動作するよう）
BASE_DIR = Path(__file__).parent
os.chdir(BASE_DIR)

# .envファイル読み込み（knowledge_distillation/ の1つ上のフォルダを優先、なければ同階層）
_env_path = (
    BASE_DIR.parent / ".env"
    if (BASE_DIR.parent / ".env").exists()
    else BASE_DIR / ".env"
)
load_dotenv(_env_path)

# 承認済みKnowledgeパス（既存FAQ照合の参照元）。
# 環境変数 APPROVED_KNOWLEDGE_PATH で上書き可（デモ用ベースラインに切替できる）。
# 注意: アプリは起動時に os.chdir(BASE_DIR) するため、相対パスは「リポジトリルート基準」で
# 解決する（cwd基準だと knowledge_distillation/ 配下を見て読み込めなくなる）。
_REPO_ROOT = BASE_DIR.parent
_approved_env = os.getenv("APPROVED_KNOWLEDGE_PATH")
if _approved_env:
    _approved_path = Path(_approved_env)
    if not _approved_path.is_absolute():
        _approved_path = _REPO_ROOT / _approved_path
    APPROVED_KNOWLEDGE_PATH = _approved_path
else:
    APPROVED_KNOWLEDGE_PATH = _REPO_ROOT / "data" / "approved_knowledge.json"


OUTPUT_DIR = Path("data/outputs")
INTERMEDIATE_DIR = Path("data/intermediate")
DEBUG_OUTPUT = False

# ページ設定
st.set_page_config(
    page_title="FAQ自動生成システム", page_icon="📋", layout="wide"
)


# ================================================================================
# ユーティリティ関数
# ================================================================================
def check_environment():
    """環境変数が正しく設定されているか確認"""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY_EMBEDDING",
        "AZURE_OPENAI_ENDPOINT_EMBEDDING",
    ]

    missing = [v for v in required_vars if not os.getenv(v)]
    return len(missing) == 0, missing


def check_approved_knowledge():
    """承認済みKnowledgeファイルの存在確認"""
    return APPROVED_KNOWLEDGE_PATH.exists()


def load_approved_knowledge_as_faq_df(path=APPROVED_KNOWLEDGE_PATH):
    """approved_knowledge.jsonをPhase 3-2比較用DataFrameへ変換する。"""
    target = Path(path)
    if not target.exists():
        return None

    with target.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("approved_knowledge.json は配列形式である必要があります。")

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if str(item.get("approved_status", "")).strip().lower() != "approved":
            continue

        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if not question or not answer:
            continue

        rows.append(
            {
                "質問": question,
                "回答": answer,
                "カテゴリ": str(item.get("category", "")).strip(),
                "knowledge_id": str(item.get("knowledge_id", "")).strip(),
                "approved_at": str(item.get("approved_at", "")).strip(),
            }
        )

    return pd.DataFrame(rows)


STANDARD_COLUMN_LABELS = {
    "inquiry_title": display_label("件名"),
    "inquiry_category": display_label("カテゴリ"),
    "inquiry_text": display_label("概要"),
    "resolution_text": display_label("対応結果"),
}

INQUIRY_TEXT_CANDIDATES = [
    "問合せ内容",
    "問い合わせ内容",
    "問い合わせ",
    "質問",
    "本文",
    "概要",
    "description",
    "body",
]

RESOLUTION_TEXT_CANDIDATES = [
    "対応結果",
    "対応内容",
    "対応内容(一次)",
    "対応内容(二次)",
    "第1対応",
    "第2対応",
    "最終結果",
    "回答",
    "回答内容",
    "解決内容",
    "resolution",
    "response",
    "close_notes",
]

INQUIRY_TITLE_CANDIDATES = [
    "コールタイトル",
    "件名",
    "タイトル",
    "subject",
    "title",
]

INQUIRY_CATEGORY_CANDIDATES = [
    "種別",
    "カテゴリ名",
    "分類",
    "カテゴリ",
    "category",
]

LINK_NAME_CANDIDATES = [
    "リンク名",
    "関連リンク",
    "link",
    "link_name",
    "url",
]

TITLE_FALLBACK_LEN = 60


def canonicalize_header(name: str) -> str:
    """列名を比較用に正規化する。"""
    text = str(name).strip().lower().replace("\u3000", " ")
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "").replace("-", "")
    return text


def build_header_index(headers):
    """正規化済み列名から実列名へのマップを作る。"""
    index = {}
    for header in headers:
        key = canonicalize_header(header)
        if key and key not in index:
            index[key] = header
    return index


def resolve_candidate_columns(df, candidates):
    """候補列のうち存在する実列名を優先順で返す。"""
    index = build_header_index(df.columns)
    resolved = []
    seen = set()
    for candidate in candidates:
        key = canonicalize_header(candidate)
        actual = index.get(key)
        if actual and actual not in seen:
            seen.add(actual)
            resolved.append(actual)
    return resolved


def first_non_empty_from_columns(row, columns):
    """指定列を順に見て最初の非空文字を返す。"""
    for col in columns:
        value = str(row.get(col, "") or "").strip()
        if value:
            return value
    return ""


def join_labeled_values(row, columns):
    """空欄を除外してラベル付きで連結する。"""
    blocks = []
    for col in columns:
        value = str(row.get(col, "") or "").strip()
        if not value:
            continue
        blocks.append(f"【{col}】\n{value}")
    return "\n\n".join(blocks).strip()


def normalize_uploaded_csv_to_standard_schema(df):
    """
    アップロードCSVを標準スキーマへ正規化する。

    Returns:
        (standard_df, detect_info, errors)
    """
    inquiry_text_cols = resolve_candidate_columns(df, INQUIRY_TEXT_CANDIDATES)
    resolution_text_cols = resolve_candidate_columns(df, RESOLUTION_TEXT_CANDIDATES)
    inquiry_title_cols = resolve_candidate_columns(df, INQUIRY_TITLE_CANDIDATES)
    inquiry_category_cols = resolve_candidate_columns(df, INQUIRY_CATEGORY_CANDIDATES)
    link_name_cols = resolve_candidate_columns(df, LINK_NAME_CANDIDATES)

    detect_info = {
        "inquiry_text_cols": inquiry_text_cols,
        "resolution_text_cols": resolution_text_cols,
        "inquiry_title_cols": inquiry_title_cols,
        "inquiry_category_cols": inquiry_category_cols,
        "link_name_cols": link_name_cols,
    }

    errors = []
    if not inquiry_text_cols:
        errors.append("問い合わせ内容に相当する列が見つかりません。")
    if not resolution_text_cols:
        errors.append("回答内容に相当する列が見つかりません。")

    if errors:
        return None, detect_info, errors

    normalized_rows = []
    for _, row in df.iterrows():
        inquiry_text = first_non_empty_from_columns(row, inquiry_text_cols)
        resolution_text = join_labeled_values(row, resolution_text_cols)
        inquiry_title = first_non_empty_from_columns(row, inquiry_title_cols)
        inquiry_category = first_non_empty_from_columns(row, inquiry_category_cols)
        link_name = first_non_empty_from_columns(row, link_name_cols)

        if not inquiry_title:
            inquiry_title = (
                inquiry_text[:TITLE_FALLBACK_LEN] if inquiry_text else ""
            )
        if not inquiry_category:
            inquiry_category = "未分類"

        normalized_rows.append(
            {
                "inquiry_title": inquiry_title,
                "inquiry_category": inquiry_category,
                "inquiry_text": inquiry_text,
                "resolution_text": resolution_text,
                "link_name": link_name,
            }
        )

    standard_df = pd.DataFrame(normalized_rows)
    if standard_df["resolution_text"].astype(str).str.strip().eq("").all():
        errors.append(
            "回答内容に相当する列は見つかりましたが、全行が空欄です。"
        )
        return None, detect_info, errors

    return standard_df, detect_info, errors


def to_legacy_schema(standard_df):
    """既存処理接続用にLegacy列名へ変換する。"""
    legacy_df = standard_df.rename(
        columns={
            "inquiry_title": "件名",
            "inquiry_category": "カテゴリ",
            "inquiry_text": "概要",
            "resolution_text": "対応結果",
            "link_name": "リンク名",
        }
    )
    return legacy_df[["件名", "カテゴリ", "概要", "対応結果", "リンク名"]]


def validate_faq_csv(df):
    """既存FAQ CSVの必須カラムチェック"""
    required_columns = ["質問", "回答"]
    missing = [col for col in required_columns if col not in df.columns]
    return len(missing) == 0, missing


# ファイル一時保存
def save_uploaded_file(uploaded_file, suffix=".csv"):
    """アップロードファイルを一時ファイルとして保存"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name


def read_csv_flexible(file_or_path, **kwargs):
    """
    UTF-8、UTF-8-SIG、Shift-JISに対応したCSV読み込み

    Args:
        file_or_path: ファイルパスまたはアップロードファイルオブジェクト
        **kwargs: pd.read_csvに渡すその他の引数

    Returns:
        pd.DataFrame
    """
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift-jis"]

    for encoding in encodings:
        try:
            # Streamlitのアップロードファイルの場合
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding=encoding, **kwargs)
            return df
        except (UnicodeDecodeError, UnicodeError):
            continue
        except Exception as e:
            # エンコーディング以外のエラーは即座に送出
            raise e

    # すべて失敗した場合
    raise ValueError(
        f"ファイルの読み込みに失敗しました。対応エンコーディング: {encodings}"
    )


# ================================================================================
# 結果表示関数
# ================================================================================
def summarize_review_targets(records):
    """ナレッジ候補数と既存FAQほぼ一致の除外件数を集計する。"""
    review_target_count = 0
    exact_faq_count = 0
    for record in records.values():
        if record.answer == "-":
            continue
        final_result = record.final_result or ""
        if final_result in {
            "P3-2確認（既存FAQ完全一致）",
            "P3-2確認（既存FAQ類似）",
        }:
            exact_faq_count += 1
        elif final_result == "◯採用" or final_result.startswith("P3-2確認"):
            review_target_count += 1
    return review_target_count, exact_faq_count


def display_results():
    """処理結果を表示"""
    output_dir = st.session_state.get("output_dir")

    if not output_dir or not os.path.exists(output_dir):
        st.warning("結果ファイルが見つかりません")
        return

    st.header("処理完了")

    # 処理時間の表示
    if st.session_state.get("processing_time"):
        st.info(f"⏱️ 総処理時間: {st.session_state['processing_time']}")

    if st.session_state.get("reduction_summary"):
        summary = st.session_state["reduction_summary"]
        st.subheader("出力サマリー")

        col1, col2, col3 = st.columns(3)
        col1.metric("元ログ", f"{summary.get('original', 0)}件")
        col2.metric(
            "Phase 0後",
            f"{summary.get('after_p0', 0)}件",
            delta=f"代表{summary.get('representative', 0)}件",
            delta_color="off",
        )
        col3.metric("FAQ候補", f"{summary.get('after_p1', 0)}件")
        col4, col5, col6 = st.columns(3)
        col4.metric("Phase 2後", f"{summary.get('after_p2', 0)}件")
        col5.metric("ナレッジ候補数", f"{summary.get('final', 0)}件")
        col6.metric("FAQほぼ一致除外", f"{summary.get('exact_faq', 0)}件")

        review_target_rate = (
            summary.get("final", 0) / max(summary.get("original", 1), 1) * 100
        )
        st.progress(
            min(review_target_rate / 100, 1.0),
            text=f"ナレッジ候補率: {review_target_rate:.1f}%",
        )

        # API節約率も表示
        original_api_calls = summary.get("original", 0)
        actual_api_calls = summary.get("representative", 0)
        api_saving_rate = (
            (1 - actual_api_calls / max(original_api_calls, 1)) * 100
            if original_api_calls > 0
            else 0
        )
        st.caption(
            f"推定APIコスト: 約{summary.get('estimated_cost', 0):,.1f}円 / "
            f"クレンジングAPI節約率: {api_saving_rate:.1f}%"
        )

    st.divider()

    st.subheader("結果ファイル")

    final_excel_files = sorted(
        glob.glob(os.path.join(output_dir, "FAQ_final_result_*.xlsx")),
        reverse=True,
    )
    if final_excel_files:
        with open(final_excel_files[0], "rb") as f:
            st.download_button(
                label="最終結果Excel（FAQ_final_result.xlsx）",
                data=f.read(),
                file_name="FAQ_final_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

    st.divider()

    # 新しい処理を開始するボタン
    if st.button("🔄 新しい処理を開始", type="primary"):
        keys_to_clear = [
            "processing_done",
            "reduction_summary",
            "output_dir",
            "processing_time",
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


def display_approved_knowledge_exporter():
    """承認済みKnowledge出力専用画面を表示する。"""
    st.header("承認済みKnowledge出力")
    st.caption(
        "レビュー済みの FAQ_final_result.xlsx をアップロードすると、"
        "レビュー結果が「採用」の行だけ approved_knowledge.json に出力します。"
    )

    reviewed_excel = st.file_uploader(
        "レビュー結果を入力した FAQ_final_result.xlsx",
        type=["xlsx"],
        key="reviewed_excel_for_approved_knowledge",
        help="Sheet1のレビュー結果が「採用」の行だけ approved_knowledge.json に出力します",
    )
    if reviewed_excel is not None:
        try:
            from knowledge_distillation.approved_knowledge_exporter import (
                export_approved_knowledge_from_excel,
                load_approved_knowledge_from_excel,
            )

            approved_items = load_approved_knowledge_from_excel(reviewed_excel)
            reviewed_excel.seek(0)
            approved_path = APPROVED_KNOWLEDGE_PATH
            export_approved_knowledge_from_excel(reviewed_excel, approved_path)

            try:
                with open(approved_path, encoding="utf-8") as f:
                    total_count = len(json.load(f))
            except Exception:
                total_count = len(approved_items)
            st.success(
                f"今回採用: {len(approved_items)}件 / 統合後 合計: {total_count}件"
                "（既存FAQ更新は上書き・新規は追加）"
            )

            # serving（本番URL）への反映は、ローカル出力とは分け、**明示ボタンを押した時だけ**
            # 実行する。プレビュー目的の出力で本番が勝手に書き換わらないための安全策。
            try:
                from knowledge_distillation.blob_uploader import (
                    upload_approved_knowledge_to_blob,
                    blob_upload_configured,
                )

                if blob_upload_configured():
                    st.warning(
                        "⚠️ 下のボタンを押すと、この内容で **本番（serving / 公開URL）に即時反映** されます。"
                        "内容を確認してから押してください。"
                    )
                    if st.button(
                        "☁️ serving へ反映する（Blobへアップロード）",
                        type="primary",
                        key="publish_to_serving",
                    ):
                        target = upload_approved_knowledge_to_blob(approved_path)
                        st.success(
                            f"☁️ Blobへアップロードしました（{target}）。"
                            "serving が自動で読み直すため、再デプロイは不要です。"
                        )
                else:
                    st.caption(
                        "（Blob未設定のためローカル出力のみ。serving反映には再デプロイ、"
                        "または APPROVED_KNOWLEDGE_BLOB_CONTAINER / AZURE_STORAGE_CONNECTION_STRING を設定）"
                    )
            except Exception as e:  # noqa: BLE001
                st.warning(f"serving への反映に失敗しました（ローカル出力は成功）: {e}")
            with open(approved_path, "rb") as f:
                st.download_button(
                    label="⬇️ approved_knowledge.json をダウンロード（ローカル保存・本番反映なし）",
                    data=f.read(),
                    file_name="approved_knowledge.json",
                    mime="application/json",
                    type="secondary",
                )
        except Exception as e:
            st.error(f"approved_knowledge.json の出力に失敗しました: {e}")


# ================================================================================
# メインUI
# ================================================================================

st.title("📋 FAQ自動生成システム")
st.markdown("問い合わせデータからFAQナレッジを自動生成")

if "current_view" not in st.session_state:
    st.session_state["current_view"] = "faq_generation"

nav_col1, nav_col2 = st.columns(2)
with nav_col1:
    if st.button(
        "FAQ生成",
        type=(
            "primary"
            if st.session_state["current_view"] == "faq_generation"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state["current_view"] = "faq_generation"
        st.rerun()
with nav_col2:
    if st.button(
        "承認済みKnowledge出力",
        type=(
            "primary"
            if st.session_state["current_view"] == "approved_export"
            else "secondary"
        ),
        use_container_width=True,
    ):
        st.session_state["current_view"] = "approved_export"
        st.rerun()

st.divider()

if st.session_state["current_view"] == "approved_export":
    display_approved_knowledge_exporter()
    st.stop()

# 環境変数チェック
env_ok, missing_vars = check_environment()
if not env_ok:
    st.error("❌ 環境変数が未設定です")
    with st.expander("📝 設定方法", expanded=True):
        st.markdown(
            f"""
        ### 未設定の環境変数
        - {', '.join(f'`{v}`' for v in missing_vars)}

        ### 設定手順
        1. **アプリを終了**（このウィンドウを閉じる）
        2. **.envファイルを作成・編集**
           - `.env.example`ファイルを`.env`にコピー
           - メモ帳で開く
        3. **APIキーとエンドポイントを入力**
           ```
           AZURE_OPENAI_API_KEY=実際のAPIキー
           AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
           ...
           ```
        4. **ファイルを保存**
        5. **アプリを再起動**

        ### APIキーの取得方法
        Azure Portal → Azure OpenAI → キーとエンドポイント
        """
        )
    st.stop()

# 処理完了後の結果表示
if st.session_state.get("processing_done", False):
    display_results()
    st.stop()

# ================================================================================
# サイドバー
# ================================================================================
with st.sidebar:
    st.header("📁 ファイルアップロード")

    # ===========================
    # 1. 問い合わせ履歴CSV
    # ===========================
    st.subheader("1️⃣ 問い合わせ履歴CSV（必須）")
    snow_file = st.file_uploader(
        "CSVファイルをアップロード",
        type=["csv"],
        help="必須: 問い合わせ内容列 + 回答内容列（列名ゆれ対応）",
    )

    # 問い合わせCSVのバリデーション（snow_dfは標準スキーマで保持）
    snow_df = None
    if snow_file:
        try:
            uploaded_df = read_csv_flexible(snow_file)
            standard_df, detect_info, errors = normalize_uploaded_csv_to_standard_schema(
                uploaded_df
            )
            if not errors:
                snow_df = standard_df
                st.success(f"✅ 有効なCSV（{len(snow_df)}件）")

                with st.expander("検出列"):
                    st.write(
                        f"- 問い合わせ内容: {', '.join(detect_info['inquiry_text_cols'])}"
                    )
                    st.write(
                        f"- 回答内容: {', '.join(detect_info['resolution_text_cols'])}"
                    )
                    st.write(
                        "- 問い合わせタイトル: "
                        + (
                            ", ".join(detect_info["inquiry_title_cols"])
                            if detect_info["inquiry_title_cols"]
                            else "未検出（問い合わせ内容から補完）"
                        )
                    )
                    st.write(
                        "- 問い合わせ分類: "
                        + (
                            ", ".join(detect_info["inquiry_category_cols"])
                            if detect_info["inquiry_category_cols"]
                            else "未検出（未分類で補完）"
                        )
                    )

                with st.expander("標準スキーマプレビュー"):
                    preview_df = snow_df[
                        [
                            "inquiry_title",
                            "inquiry_category",
                            "inquiry_text",
                            "resolution_text",
                        ]
                    ].rename(columns=STANDARD_COLUMN_LABELS)
                    st.dataframe(preview_df.head(3))
            else:
                st.error("❌ 入力CSVの検証に失敗しました。")
                for err in errors:
                    st.error(f"- {err}")
                st.info(f"入力CSVの列名一覧: {', '.join(map(str, uploaded_df.columns))}")
                st.info(
                    "問い合わせ内容の候補: "
                    + ", ".join(INQUIRY_TEXT_CANDIDATES)
                )
                st.info(
                    "回答内容の候補: "
                    + ", ".join(RESOLUTION_TEXT_CANDIDATES)
                )
                snow_df = None
        except Exception as e:
            st.error(f"❌ 読込エラー: {e}")
            snow_df = None

    st.divider()

    # ===========================
    # 2. 承認済みKnowledge（Phase 3-2比較対象）
    # ===========================
    st.subheader("2️⃣ 承認済みKnowledge（比較対象）")

    faq_df = None
    faq_source = None

    if check_approved_knowledge():
        try:
            faq_df = load_approved_knowledge_as_faq_df()
            if faq_df is not None and len(faq_df) > 0:
                faq_source = f"approved_knowledge.json ({APPROVED_KNOWLEDGE_PATH})"
                st.success("✅ 承認済みKnowledgeを比較対象として読み込みました")
                st.metric("承認済みKnowledge数", f"{len(faq_df)}件")

                with st.expander("承認済みKnowledgeプレビュー"):
                    preview_cols = [
                        col
                        for col in [
                            "knowledge_id",
                            "質問",
                            "回答",
                            "カテゴリ",
                            "approved_at",
                        ]
                        if col in faq_df.columns
                    ]
                    st.dataframe(faq_df[preview_cols].head(5))
            else:
                st.warning(
                    "approved_knowledge.json は存在しますが、"
                    "approved_status=approved の有効なKnowledgeがありません。"
                )
                faq_df = None
        except Exception as e:
            st.error(f"❌ approved_knowledge.json の読込に失敗しました: {e}")
            faq_df = None
    else:
        st.info(
            f"approved_knowledge.json がまだありません（{APPROVED_KNOWLEDGE_PATH}）。"
            "先にレビュー済みExcelから承認済みKnowledgeを出力してください。"
        )

    st.caption(
        "この画面では既存FAQ CSVのアップロードは不要です。"
        "Phase 3-2は承認済みKnowledgeだけを既存FAQとして照合します。"
    )

    st.divider()

    # ===========================
    # 3. パラメータ設定
    # ===========================
    st.subheader("3️⃣ 詳細設定")

    with st.expander("⚙️ Phase別閾値設定", expanded=True):
        st.markdown(
            "**採用閾値**: この値以上の類似度を持つデータを重複として統合"
        )

        p2_threshold = st.slider(
            "Phase 2（文字列類似度）閾値",
            min_value=0.50,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="文字列類似度がこの値以上なら重複として統合",
        )

        p3_1_threshold = st.slider(
            "Phase 3-1（Q内Embedding）閾値",
            min_value=0.50,
            max_value=0.95,
            value=0.75,
            step=0.05,
            help="Embedding類似度がこの値以上なら重複として統合",
        )

        p3_2_threshold = st.slider(
            "Phase 3-2（承認済みKnowledge）閾値",
            min_value=0.50,
            max_value=0.95,
            value=0.70,
            step=0.05,
            help="承認済みKnowledgeとの類似度がこの値以上なら確認対象としてSheet1に出力（質問＋回答で照合）",
        )

    with st.expander("⚙️ 処理設定"):
        max_concurrent = st.slider(
            "AI並列数（Phase 1）",
            min_value=10,
            max_value=100,
            value=50,
            step=10,
        )

        embedding_batch_size = st.number_input(
            "Embeddingバッチサイズ",
            min_value=10,
            max_value=200,
            value=100,
            step=10,
            help="Embedding API の一度に処理する件数",
        )

    st.divider()
    st.caption("Version 3.0.0 - API節約版（Phase 0類似度チェック追加）")

# ================================================================================
# メインエリア
# ================================================================================

if snow_df is None:
    st.info("👈 サイドバーからCSVをアップロードしてください（問い合わせ内容/回答内容は必須）")

    # タブUI
    tab1, tab2, tab3 = st.tabs(["📖 使い方", "⚙️ 環境情報", "❓ FAQ"])

    with tab1:
        st.subheader("📋 処理の流れ")
        st.markdown(
            """
### Step 1: ファイル準備
- **問い合わせ履歴CSV**をアップロード（必須）
- **承認済みKnowledge**を確認（任意）

入力CSVはまず以下の標準項目へ正規化されます。
- 問い合わせタイトル（`inquiry_title`）
- 問い合わせ分類（`inquiry_category`）
- 問い合わせ内容（`inquiry_text`）
- 回答内容（`resolution_text`）

### Step 2: 閾値設定
- Phase 2（文字列類似度）: デフォルト 0.75
- Phase 3-1（Q内Embedding）: デフォルト 0.75
- Phase 3-2（承認済みKnowledge）: デフォルト 0.75

### Step 3: 処理実行
| Phase | 処理内容 | API |
|-------|---------|------|
| Phase 0 | 完全一致＆類似度チェック（問い合わせ内容） | なし |
| Phase 1 | AIクレンジング（**代表のみ**） | GPT-4.1 |
| Phase 2 | 完全一致＆類似度チェック（質問） | なし |
| Phase 3-1 | Q内Embedding重複除去 | Embedding |
| Phase 3-2 | 承認済みKnowledgeとの照合・確認対象化 | Embedding |

### Step 4: 結果ダウンロード
- **FAQ_final_result.xlsx**: 2シート構成
  - 最終ナレッジ候補一覧
  - 全データ処理履歴（P0_類似度、P2_類似度追跡）
        """
        )

    with tab2:
        st.subheader("⚙️ 現在のシステム環境")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**API設定**")
            st.markdown(f"- {'✅' if env_ok else '❌'} Azure OpenAI API")

            if env_ok:
                st.success("環境変数: 設定済み")
            else:
                st.error("環境変数: 未設定")

        with col2:
            st.markdown("**データ設定**")
            st.markdown(
                f"- {'✅' if check_approved_knowledge() else '❌'} 承認済みKnowledge"
            )

            if check_approved_knowledge():
                st.success(f"承認済みKnowledgeあり: {APPROVED_KNOWLEDGE_PATH}")
            else:
                st.warning("承認済みKnowledgeなし")

        st.divider()

        st.markdown("**Python環境**")
        st.code(f"Python {sys.version}", language="text")

    with tab3:
        st.subheader("❓ よくある質問")

        with st.expander("Q1: Phase 0での類似度チェックとは？"):
            st.markdown(
                """
クレンジング前の「問い合わせ内容」列で完全一致＆文字列類似度（0.9以上）をチェックし、
類似グループの**代表データのみ**をAIクレンジングに送ります。
- **大幅なAPI節約**（類似データは代表のみクレンジング）
- 例：同じ問い合わせが10件 → 1件のみAI処理
            """
            )

        with st.expander("Q2: 処理結果を確認するには？"):
            st.markdown(
                """
**全データ処理履歴シート**の「最終結果」カラムで確認できます：
- `◯採用`: ナレッジ候補として出力される
- `P0削除（完全一致）`: Phase 0で削除（問い合わせ内容が完全一致）
- `P0削除（類似）`: Phase 0で削除（問い合わせ内容の類似度0.9以上）
- `P0削除（短文）`: Phase 0で削除（回答内容が30文字以内）
- `P2削除（完全一致）`: Phase 2で削除（質問が完全一致）
- `P2削除（類似）`: Phase 2で削除（質問の類似度が閾値以上）
- `P3-1削除`: Phase 3-1で削除（Embedding類似度）
- `P3-2確認（既存FAQ完全一致）`: 既存FAQとほぼ一致するためSheet1には出さない
- `P3-2確認（既存FAQ類似）`: 既存FAQへの統合候補のためSheet1には出さない
- `P3-2確認（既存FAQ更新候補/矛盾可能性）`: ナレッジ候補として出力される
- `FAQ対象外`: AIがナレッジ候補に適さないと判断
        """
            )

        with st.expander("Q3: 比較対象のKnowledgeはどこから読み込まれる？"):
            st.markdown(
                f"""
            Phase 3-2の比較対象は、レビュー済みExcelから出力した承認済みKnowledgeです。

            読み込み先:
            ```
            {APPROVED_KNOWLEDGE_PATH}
            ```

            画面上部の「承認済みKnowledge出力」から生成した後、この処理画面で自動的に読み込まれます。
            """
            )


else:
    st.subheader("🚀 処理実行")

    st.info(f"📄 処理対象: {len(snow_df)}件")

    if faq_df is not None:
        st.info(f"📄 比較対象Knowledge: {faq_source} ({len(faq_df)}件)")
    else:
        st.warning("📄 比較対象Knowledge: なし（Phase 3-2をスキップ）")

    with st.expander("📊 設定確認"):
        col1, col2, col3 = st.columns(3)
        col1.metric("Phase 2 閾値", f"{p2_threshold:.2f}")
        col2.metric("Phase 3-1 閾値", f"{p3_1_threshold:.2f}")
        col3.metric("Phase 3-2 閾値", f"{p3_2_threshold:.2f}")

    run_button = st.button(
        f"🚀 処理開始（{len(snow_df)}件）",
        type="primary",
        use_container_width=True,
    )

    if run_button:
        overall_start_time = datetime.now()

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = str(OUTPUT_DIR)
        intermediate_dir = str(INTERMEDIATE_DIR)

        try:
            with st.status("処理中...", expanded=True) as status:
                original_count = len(snow_df)
                # 既存処理へ渡す直前にのみLegacy列名へ変換
                legacy_input_df = to_legacy_schema(snow_df)

                # ===========================
                # Phase 0: 完全一致＆類似度チェック（問い合わせ内容）
                # ===========================
                st.write("🔄 **Phase 0**: 完全一致＆類似度チェック（問い合わせ内容）...")

                from knowledge_distillation.pre_deduplication import (
                    run_phase0,
                    run_phase2,
                )

                (
                    p0_adopted_df,
                    p0_representative_df,
                    p0_group_mapping,
                    p0_records,
                    gid_tracker,
                ) = run_phase0(
                    df=legacy_input_df,
                    output_dir=intermediate_dir,
                )
                after_p0 = len(p0_adopted_df)
                representative_count = len(p0_representative_df)
                st.write(
                    f"  ✅ 完了: {original_count}件 → {after_p0}件（代表: {representative_count}件）"
                )

                # ===========================
                # Phase 1: AIクレンジング（代表のみ）
                # ===========================
                st.write(
                    f"🔄 **Phase 1**: AIクレンジング（代表{representative_count}件のみ）..."
                )

                from knowledge_distillation.SNOW_Cleansing import clean_dataframe

                # 代表データのみクレンジング
                cleaned_representative_df = clean_dataframe(
                    df=p0_representative_df,
                    max_concurrent=max_concurrent,
                )

                # クレンジング結果をグループメンバーに適用
                st.write("  📝 クレンジング結果をグループメンバーに適用中...")

                # 全採用データ用のDataFrameを作成
                cleaned_rows = []
                for rep_idx, member_indices in p0_group_mapping.items():
                    if rep_idx not in cleaned_representative_df.index:
                        continue

                    rep_row = cleaned_representative_df.loc[rep_idx]

                    for member_idx in member_indices:
                        if member_idx not in p0_adopted_df.index:
                            continue

                        original_row = p0_adopted_df.loc[member_idx]

                        # クレンジング結果を適用（代表のQ/Aを使用、生データは各自のもの）
                        new_row = {
                            "質問": rep_row.get("質問", ""),
                            "回答": rep_row.get("回答", ""),
                            "カテゴリ": rep_row.get("カテゴリ", ""),
                            "キーワード": rep_row.get("キーワード", ""),
                            "リンク名": rep_row.get("リンク名", ""),
                            "立場": rep_row.get("立場", ""),
                            "信頼度": rep_row.get("信頼度", ""),
                            "信頼度理由": rep_row.get("信頼度理由", ""),
                            "AIリスクレベル": rep_row.get("AIリスクレベル", ""),
                            "リスク理由": rep_row.get("リスク理由", ""),
                            "件名": original_row.get("件名", ""),
                            "概要": original_row.get("概要", ""),
                            "対応結果": original_row.get("対応結果", ""),
                        }
                        if "リンク名" in original_row.index and not str(
                            new_row.get("リンク名", "")
                        ).strip():
                            new_row["リンク名"] = original_row.get("リンク名", "")

                        cleaned_rows.append((member_idx, new_row))

                # DataFrameに変換（インデックス保持）
                cleaned_df = pd.DataFrame(
                    [row for _, row in cleaned_rows],
                    index=[idx for idx, _ in cleaned_rows],
                )

                after_p1 = len(
                    cleaned_df[cleaned_df["回答"].astype(str) != "-"]
                )
                st.write(f"  ✅ 完了: {after_p0}件（FAQ候補: {after_p1}件）")

                # ===========================
                # Phase 2: 完全一致＆類似度チェック（質問）
                # ===========================
                st.write("🔄 **Phase 2**: 完全一致＆類似度チェック（質問）...")

                p2_df, p2_records, gid_tracker = run_phase2(
                    df=cleaned_df,
                    threshold=p2_threshold,
                    output_dir=intermediate_dir,
                    processing_records=p0_records,
                    gid_tracker=gid_tracker,
                )
                after_p2 = len(p2_df[p2_df["回答"].astype(str) != "-"])
                st.write(f"  ✅ 完了: {after_p1}件 → {after_p2}件")

                # ===========================
                # Phase 3: Embedding重複除去
                # ===========================
                st.write("🔄 **Phase 3**: Embedding照合...")

                from knowledge_distillation.deduplication_system import run_phase3

                final_df, final_records, gid_tracker, final_excel = run_phase3(
                    df=p2_df,
                    faq_df=faq_df,
                    threshold_q=p3_1_threshold,
                    threshold_faq=p3_2_threshold,
                    embedding_batch_size=embedding_batch_size,
                    output_dir=output_dir,
                    processing_records=p2_records,
                    gid_tracker=gid_tracker,
                )
                review_target_count, exact_faq_count = summarize_review_targets(
                    final_records
                )
                st.write(
                    "  ✅ 完了: "
                    f"{after_p2}件 → ナレッジ候補数 {review_target_count}件"
                    f"（FAQほぼ一致除外 {exact_faq_count}件）"
                )

                status.update(label="✅ 全処理完了", state="complete")

            elapsed = datetime.now() - overall_start_time
            processing_time = (
                f"{elapsed.seconds // 60}分 {elapsed.seconds % 60}秒"
            )

            st.success("処理が正常に完了しました")

            st.session_state["processing_done"] = True
            st.session_state["output_dir"] = output_dir
            st.session_state["processing_time"] = processing_time

            # 推定APIコスト計算（代表のみクレンジングするため大幅節約）
            estimated_cost = (
                representative_count * 0.003  # GPT-4.1クレンジング（代表のみ）
                + (after_p2 * 2) * 0.0001  # Embedding（Phase3-1, 3-2）
            ) * 150

            st.session_state["reduction_summary"] = {
                "original": original_count,
                "after_p0": after_p0,
                "representative": representative_count,
                "after_p1": after_p1,
                "after_p2": after_p2,
                "final": review_target_count,
                "exact_faq": exact_faq_count,
                "estimated_cost": estimated_cost,
            }

            st.rerun()

        except Exception as e:
            st.error(f"❌ エラーが発生しました: {e}")
            st.exception(e)
