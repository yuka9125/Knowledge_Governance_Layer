"""
インシデントデータクレンジングツール（高速並列処理版）

【改善点】
- 非同期並列処理で800件を3分以内に処理
- 並列度を調整可能（デフォルト30並列）
- プログレスバーで進捗確認
- エラーハンドリング強化
- ★修正: 元のDataFrameインデックスを保持
- ★修正: FAQ適性判定（依頼事項の除外）
- ★修正: 固有名詞の保持
- ★修正: 回答の厳格化（推測禁止）

【セットアップ】
pip install openai pandas html asyncio tqdm aiohttp
"""

from openai import AzureOpenAI, AsyncAzureOpenAI
import pandas as pd
import html
import re
import json
import asyncio
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm.asyncio import tqdm
import os


def clean_dataframe(
    df: pd.DataFrame, max_concurrent: int = 50
) -> pd.DataFrame:
    """
    DataFrameを直接クレンジング（Streamlit用）

    ★重要: 入力DataFrameのインデックスを保持して返す

    Args:
        df: 生のSNOWインシデントDataFrame（件名、カテゴリ、概要、対応結果）
        max_concurrent: 並列数

    Returns:
        クレンジング済みDataFrame（質問、回答、カテゴリ、キーワード、リンク名、立場、信頼度、理由）
        ※インデックスは入力dfと同一
        ※FAQ除外データは質問=「【FAQ除外】理由」、回答=「-」
    """
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    cleaner = FastIncidentDataCleaner(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        model="gpt-4.1",
        max_concurrent=max_concurrent,
    )

    # 非同期処理を同期的に実行
    results = asyncio.run(cleaner.process_batch_async(df))

    # 結果をインデックス順にソート
    results.sort(
        key=lambda x: list(df.index).index(x[0]) if x[0] in df.index else 0
    )

    # ★修正: 元のインデックスを使用してDataFrameを構築
    # ★修正: FAQ除外データの出力形式を変更
    converted_data = {}
    for original_idx, cleaned_data in results:
        if cleaned_data.is_faq_candidate:
            # FAQ候補の場合は通常通り
            converted_data[original_idx] = {
                "質問": cleaned_data.question,
                "回答": cleaned_data.answer,
                "カテゴリ": cleaned_data.category,
                "キーワード": cleaned_data.keywords,
                "リンク名": cleaned_data.link_names,
                "立場": cleaned_data.user_role,
                "信頼度": cleaned_data.confidence_score,
                "信頼度理由": cleaned_data.confidence_reason,
                "AIリスクレベル": cleaned_data.risk_level,
                "リスク理由": cleaned_data.risk_reason,
            }
        else:
            # FAQ除外の場合は特別な形式
            converted_data[original_idx] = {
                "質問": f"【FAQ除外】{cleaned_data.exclusion_reason}",
                "回答": "-",
                "カテゴリ": cleaned_data.category,
                "キーワード": "",
                "リンク名": "",
                "立場": "",
                "信頼度": "",
                "信頼度理由": "",
                "AIリスクレベル": "",
                "リスク理由": "",
            }

    # 元のインデックス順で行を作成
    rows = []
    indices = []
    for idx in df.index:
        if idx in converted_data:
            rows.append(converted_data[idx])
            indices.append(idx)

    # インデックスを明示的に設定してDataFrameを作成
    result_df = pd.DataFrame(rows, index=indices)

    return result_df


@dataclass
class CleanedIncidentData:
    """クリーンアップされたインシデントデータ"""

    question: str
    answer: str
    category: str
    keywords: str
    link_names: str
    user_role: str
    is_answer_modified: bool
    confidence_score: float
    confidence_reason: str = ""
    risk_level: str = ""
    risk_reason: str = ""
    is_faq_candidate: bool = True  # ★追加: FAQ候補かどうか
    exclusion_reason: str = ""  # ★追加: 除外理由
    processing_error: bool = False  # API処理失敗フラグ
    error_summary: str = ""  # 失敗時の要約（生データは含めない）


class FastIncidentDataCleaner:
    """非同期並列処理によるインシデントデータクリーナー"""

    def __init__(
        self,
        api_key: str = None,
        azure_endpoint: str = None,
        model: str = "gpt-4.1",
        max_concurrent: int = 50,
    ):
        """
        初期化

        Args:
            api_key: OpenAI APIキー
            azure_endpoint: Azure OpenAIエンドポイント
            model: 使用するモデル
            max_concurrent: 最大並列数（デフォルト50）
        """
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_endpoint = azure_endpoint or os.getenv(
            "AZURE_OPENAI_ENDPOINT"
        )

        # 非同期クライアントの作成
        self.async_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version="2025-04-01-preview",
            azure_endpoint=self.azure_endpoint,
        )

        self.model = model
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_error_summary_len = 80

    def decode_html_entities(self, text: str) -> str:
        """HTMLエンティティを正常な文字に変換"""
        if not text or pd.isna(text):
            return ""

        text = str(text)
        decoded = html.unescape(text)

        entity_map = {
            "&#xff08;": "(",
            "&#xff09;": ")",
            "&#xff1a;": ":",
            "&#xff1d;": "=",
            "&#xff0b;": "+",
            "&#xff5e;": "～",
            "&#61;": "=",
        }

        for entity, char in entity_map.items():
            decoded = decoded.replace(entity, char)

        return decoded

    def _sanitize_error_summary(self, error: Exception | str) -> str:
        """例外文字列を安全に短縮する（生データ・機密値の露出抑制）。"""
        text = str(error) if error is not None else "UnknownError"
        text = text.replace("\r", " ").replace("\n", " ")
        text = re.sub(r"\s+", " ", text).strip()

        # APIキー/トークン風の長い文字列をマスク
        text = re.sub(r"sk-[A-Za-z0-9\-_]{8,}", "sk-***", text)
        text = re.sub(
            r"\b[A-Za-z0-9_\-]{24,}\b",
            "***",
            text,
        )

        if len(text) > self.max_error_summary_len:
            text = text[: self.max_error_summary_len] + "..."
        return text or "UnknownError"

    def _build_processing_error_result(
        self, category: str, error: Exception | str
    ) -> CleanedIncidentData:
        """再処理対象として扱う安全な失敗オブジェクトを返す。"""
        summary = self._sanitize_error_summary(error)
        return CleanedIncidentData(
            question="",
            answer="",
            category=category,
            keywords="",
            link_names="",
            user_role="",
            is_answer_modified=False,
            confidence_score=0.0,
            confidence_reason="API処理に失敗したため判定不可",
            risk_level="low",
            risk_reason="FAQ候補として利用しないため公開リスクは低い",
            is_faq_candidate=False,
            exclusion_reason=f"生成処理失敗({summary})",
            processing_error=True,
            error_summary=summary,
        )

    async def clean_incident_with_openai_async(
        self,
        subject: str,
        category: str,
        overview: str,
        response: str,
        original_index,  # ★修正: int以外のインデックスにも対応
        max_retries: int = 5,  # ★追加: リトライ回数
        base_wait_seconds: float = 2.0,  # ★追加: リトライ待機の基準秒
        use_jitter: bool = False,  # ★追加: ジッター有無
    ) -> Tuple[any, CleanedIncidentData]:
        """
        非同期でインシデントデータをクリーンアップ

        Args:
            subject: 件名
            category: カテゴリ
            overview: 概要
            response: 対応結果
            original_index: 元のDataFrameインデックス（順序保持用）
            max_retries: 最大リトライ回数（デフォルト5回）
            base_wait_seconds: リトライ待機の基準秒
            use_jitter: 待機時間にジッターを付与するか

        Returns:
            (original_index, CleanedIncidentData): インデックスとクリーンアップデータのタプル
        """
        # セマフォで同時実行数を制限
        async with self.semaphore:
            # HTMLエンティティをデコード（リトライ前に1回だけ実行）
            overview_decoded = self.decode_html_entities(overview)
            response_decoded = self.decode_html_entities(response)

            last_error = None

            for attempt in range(max_retries):
                try:
                    # ★修正: プロンプト全体を改善版に置き換え
                    prompt = f"""
あなたは社内システムのインシデント対応記録を「ナレッジ候補」向けに整形する専門家です。
入力データの内容に基づいて、適切な質問と回答を生成してください。

## 入力データ
- 件名: {subject}
- カテゴリ: {category}
- 概要（質問の元）: {overview_decoded}
- 対応結果（回答の元）: {response_decoded}

## 出力形式（必ずこのJSON形式で出力）
```json
{{
    "is_faq_candidate": true,
    "exclusion_reason": "",
    "question": "ユーザーが検索しやすい質問文",
    "answer": "解決方法を簡潔にまとめた回答",
    "category": "{category}",
    "keywords": "キーワード1;キーワード2;キーワード3",
    "link_names": "URL名またはマニュアル名",
    "user_role": "申請者",
    "is_answer_modified": false,
    "confidence_score": 0.95,
    "confidence_reason": "質問タイプと回答が対応し、手順が具体的なため",
    "risk_level": "low",
    "risk_reason": "権限・個人情報・会計処理への直接影響が小さいため"
}}
```

## STEP 1: ナレッジ候補適性判定（最初に必ず実施）

このインシデントがナレッジ候補化に適しているか判定してください。

**ナレッジ候補化すべきでないケース（is_faq_candidate: false）:**
- 作業代行依頼（「〜を登録してください」「〜の設定をお願いします」）
- マスタ登録/削除/変更/更新の代行依頼
- 権限付与・アカウント作成の依頼
- 個別データの修正・削除依頼
- 特定ユーザーへの個別対応
- 確認代行依頼（「〜を確認してください」「〜の状況を教えてください」）


**ナレッジ候補化すべきケース（is_faq_candidate: true）:**
- 操作方法・手順の質問
- エラーの対処方法
- 機能や設定の確認方法
- トラブルシューティング
- 「〜するにはどうすればいいですか」という形式の質問

**判定のポイント:**
- 概要が「依頼」なのか「質問」なのかを見極める
- 対応結果が「代行作業の完了報告」のみの場合は依頼(「～事前承認のもと代行」) → ナレッジ候補除外
- 対応結果に「手順説明」「方法案内」「操作説明」が含まれていれば質問 → ナレッジ候補

is_faq_candidate が false の場合:
- exclusion_reason に理由を記載（例:「マスタ登録の代行依頼のため」）
- question, answer は空文字 "" を設定
- 他のフィールドも空文字またはデフォルト値を設定

---

## STEP 2: 質問文（question）の作成

### 形式の統一
- エラー系: 「〜というエラーが表示される場合の対処法は？」
- 操作方法系: 「〜する方法を教えてください」
- トラブル系: 「〜できない場合はどうすればよいですか？」
- 確認系: 「〜はどこで確認できますか？」

### 固有名詞の保持（重要）
以下は**必ず質問文に残す**（汎用化・省略禁止）:
- システム名・機能名（例: エクセルアップ、TMS、SAP、BHE、ProPlus、ServiceNow）
- マスタ名（例: 勘定科目マスタ、取引先マスタ、エクセルアップマスタ）
- 画面名・帳票名・メニュー名
- エラーメッセージ（「」で囲んで原文を残す）

### 削除すべき情報
- 個人名（○○さん、担当者名、メールアドレス）
- 日時情報（2024/1/1、本日、昨日など）
- チケット番号・インシデント番号

---

## STEP 3: 回答文（answer）の作成

### 厳格なルール
- **対応結果に記載されている情報のみ**を使用する
- 推測・補完・一般的な知識からの追加は**禁止**
- 対応結果にない手順や情報を追加しない

### 対応結果が不十分な場合
- 記載されている内容のみで簡潔に整形する
- 情報が不足している場合は「詳細は担当部門にお問い合わせください」で終える
- is_answer_modified は false のまま
- confidence_score は 0.50〜0.65 に設定

### 削除すべき情報
- 「確認しました」「対応完了」などの事後報告
- 「○○さんに連絡」などの内部対応記録
- 日時情報、チケット番号
- メール送信記録・電話対応記録

---

## STEP 4: その他のフィールド

### キーワード（keywords）
- セミコロン区切りで5〜8個
- 概要と対応結果から、検索で使われそうな単語を抽出
- システム名・機能名・マスタ名も含める

### リンク名（link_names）
- https:// で始まるURL
- .pdf, .xlsx, .docx 等の拡張子付きファイル名
- 該当なしの場合は空文字 ""

### 立場（user_role）
- 「管理者」「ユーザー」「承認者」「業務担当者」「その他」から選択
- 判断できない場合は「その他」

### 信頼度（confidence_score）

まず質問タイプを判定し、回答との整合性を評価する:

**質問タイプの判定:**
- 方法質問（どうすれば/どこで/どのように）→ 回答に手順・場所・方法が必須
- 原因質問（なぜ/どうして）→ 回答に理由・原因が必須
- 可否質問（できますか/可能ですか）→ 回答に可否と条件が必須

**スコア基準:**
- 0.90〜0.95: 質問タイプと回答が完全に対応、手順が具体的
- 0.70〜0.85: 質問タイプと回答が対応、情報がやや不足
- 0.50〜0.65: 対応しているが情報不足
- 0.30〜0.45: 質問タイプと回答タイプが不整合（方法を聞いているのに事実報告のみ等）
- 0.20〜0.35: 回答が事実報告のみで、質問への回答になっていない

**不整合の例:**
- 「どこで確認できますか？」→「登録されています」= 不整合（0.30）
- 「どうすればいいですか？」→「対応しました」= 不整合（0.30）

### 信頼度理由（confidence_reason）
- confidence_score の理由を30文字以内で記載
- 元ログ件数ではなく、この1件の概要と対応結果の整合性を説明
- 例:「質問と回答が対応し手順が具体的なため」
- 例:「回答が事後報告中心で手順が不足するため」

### リスクレベル（risk_level）
- 「critical」「high」「low」から選択
- critical: 認証、権限、管理者操作、アクセス制御に関係する
- high: 人事、給与、セキュリティ、パスワード、個人情報、会計処理に関係する
- low: 上記に該当せず、誤案内時の影響が限定的

### リスク理由（risk_reason）
- risk_level の理由を30文字以内で記載
- low の場合は「手戻りが起きる」など強い影響を書かず、直接影響が小さい理由を書く
- 例:「権限や個人情報への直接影響が小さいため」
- 例:「パスワード案内を含み影響が大きいため」

必ず上記のJSON形式のみを出力してください。説明文は不要です。
補足: existing_faq_diff_reason / review_status / source_logs / cluster_id / knowledge_id は後続処理で自動付与されるため、ここでは出力しないでください。
"""

                    response_api = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": "あなたは社内システムのインシデント対応記録をナレッジ候補向けに変換する専門家です。入力データの内容を分析し、ユーザーが検索しやすい質問文と、問題を解決できる回答文を生成します。依頼事項は候補から除外します。",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,
                        max_tokens=1700,
                    )

                    content = response_api.choices[0].message.content.strip()

                    # JSONの抽出
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        content = content[json_start:json_end].strip()
                    elif "```" in content:
                        json_start = content.find("```") + 3
                        json_end = content.rfind("```")
                        content = content[json_start:json_end].strip()

                    result_data = json.loads(content)

                    # リスト形式の場合は最初の要素を取得
                    if isinstance(result_data, list):
                        if len(result_data) > 0:
                            result_data = result_data[0]
                        else:
                            raise ValueError("Empty list returned from API")

                    # 辞書でない場合はエラー
                    if not isinstance(result_data, dict):
                        raise ValueError(
                            f"Invalid response type: {type(result_data)}"
                        )

                    # ★修正: FAQ適性判定を含めてCleanedIncidentDataを作成
                    is_faq_candidate = result_data.get(
                        "is_faq_candidate", True
                    )
                    exclusion_reason = result_data.get("exclusion_reason", "")

                    if is_faq_candidate:
                        cleaned = CleanedIncidentData(
                            question=result_data.get(
                                "question", overview_decoded
                            ),
                            answer=result_data.get("answer", response_decoded),
                            category=result_data.get("category", category),
                            keywords=result_data.get("keywords", ""),
                            link_names=result_data.get("link_names", ""),
                            user_role=result_data.get("user_role", "その他"),
                            is_answer_modified=result_data.get(
                                "is_answer_modified", False
                            ),
                            confidence_score=result_data.get(
                                "confidence_score", 0.0
                            ),
                            confidence_reason=result_data.get(
                                "confidence_reason", ""
                            ),
                            risk_level=result_data.get("risk_level", ""),
                            risk_reason=result_data.get("risk_reason", ""),
                            is_faq_candidate=True,
                            exclusion_reason="",
                            processing_error=False,
                            error_summary="",
                        )
                    else:
                        # FAQ除外の場合
                        cleaned = CleanedIncidentData(
                            question="",
                            answer="",
                            category=category,
                            keywords="",
                            link_names="",
                            user_role="",
                            is_answer_modified=False,
                            confidence_score=0.0,
                            confidence_reason="FAQ候補対象外のため",
                            risk_level="low",
                            risk_reason="公開対象外のため",
                            is_faq_candidate=False,
                            exclusion_reason=exclusion_reason
                            or "FAQ対象外と判定",
                            processing_error=False,
                            error_summary="",
                        )

                    return (original_index, cleaned)

                except Exception as e:
                    last_error = e
                    error_str = str(e)

                    # 429 Rate Limitエラーの場合はリトライ
                    if "429" in error_str or "RateLimitReached" in error_str:
                        wait_time = (attempt + 1) * base_wait_seconds
                        if use_jitter:
                            wait_time = wait_time + random.uniform(0.2, 1.2)
                        if attempt < max_retries - 1:
                            print(
                                f"⚠️ レート制限 (idx={original_index}): {wait_time}秒後にリトライ ({attempt + 1}/{max_retries})"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                    else:
                        # 429以外のエラーは即座にフォールバック
                        print(
                            f"\n❌ エラー (idx={original_index}, 件名: {subject[:20]}...): {e}"
                        )
                        break

            # 全リトライ失敗またはリトライ不可エラー → フォールバック
            if last_error:
                print(
                    f"\n❌ 最終エラー (idx={original_index}, 件名: {subject[:20]}...): {last_error}"
                )
            # 生データを質問/回答へ流さず、失敗状態として返す
            return (original_index, self._build_processing_error_result(category, last_error))

    def _fallback_cleaning(
        self, category: str, error: Exception | str
    ) -> CleanedIncidentData:
        """互換用途のフォールバック（生データ非流入）。"""
        return self._build_processing_error_result(category, error)

    async def process_batch_async(
        self, df: pd.DataFrame
    ) -> List[Tuple[any, CleanedIncidentData]]:
        """
        データフレーム全体を非同期バッチ処理

        ★修正: 元のDataFrameインデックスを保持

        Args:
            df: 入力データフレーム

        Returns:
            List[tuple]: (original_index, CleanedIncidentData)のリスト
        """
        tasks = []

        # ★修正: df.iterrows()のインデックスをそのまま使用
        for original_idx, row in df.iterrows():
            task = self.clean_incident_with_openai_async(
                subject=str(row["件名"]).strip(),
                category=row["カテゴリ"],
                overview=row["概要"],
                response=row["対応結果"],
                original_index=original_idx,  # ★元のインデックスを渡す
            )
            tasks.append(task)

        # プログレスバー付きで並列実行
        results = []
        for coro in tqdm.as_completed(
            tasks, total=len(tasks), desc="🔄 処理中"
        ):
            result = await coro
            results.append(result)

        # 並列処理で失敗した行を直列で再処理
        failed_indices = [
            idx for idx, data in results if getattr(data, "processing_error", False)
        ]
        if failed_indices:
            print(
                f"⚠️ 並列処理失敗: {len(failed_indices)}件。直列リカバリ処理を開始します。"
            )
            recovered = await self._recover_failed_rows_sequentially(df, failed_indices)
            recovered_map = {idx: data for idx, data in recovered}
            merged_results: List[Tuple[any, CleanedIncidentData]] = []
            for idx, data in results:
                if idx in recovered_map:
                    merged_results.append((idx, recovered_map[idx]))
                else:
                    merged_results.append((idx, data))
            results = merged_results

        return results

    async def _recover_failed_rows_sequentially(
        self, df: pd.DataFrame, failed_indices: List[any]
    ) -> List[Tuple[any, CleanedIncidentData]]:
        """並列失敗行のみを直列で再処理する。"""
        recovered: List[Tuple[any, CleanedIncidentData]] = []
        total = len(failed_indices)
        success_count = 0
        failure_count = 0

        for order, idx in enumerate(failed_indices, start=1):
            row = df.loc[idx]
            print(f"🔁 直列リカバリ {order}/{total} (idx={idx})")
            result_idx, cleaned = await self.clean_incident_with_openai_async(
                subject=str(row["件名"]).strip(),
                category=row["カテゴリ"],
                overview=row["概要"],
                response=row["対応結果"],
                original_index=idx,
                max_retries=8,
                base_wait_seconds=3.0,
                use_jitter=True,
            )

            if getattr(cleaned, "processing_error", False):
                # 最終手段: FAQ対象外として確定（生データは質問/回答に入れない）
                summary = cleaned.error_summary or "UnknownError"
                cleaned = CleanedIncidentData(
                    question="",
                    answer="",
                    category=str(row.get("カテゴリ", "") or ""),
                    keywords="",
                    link_names="",
                    user_role="",
                    is_answer_modified=False,
                    confidence_score=0.0,
                    confidence_reason="生成処理失敗のため判定不可",
                    risk_level="low",
                    risk_reason="FAQ候補として公開しないため",
                    is_faq_candidate=False,
                    exclusion_reason=f"生成処理失敗({summary})",
                    processing_error=False,
                    error_summary=summary,
                )
                print(f"⚠️ 直列リカバリ失敗 (idx={idx}) → FAQ対象外で確定")
                failure_count += 1
            else:
                success_count += 1

            recovered.append((result_idx, cleaned))

        print(
            f"📊 直列リカバリ結果: 成功 {success_count}件 / 失敗 {failure_count}件（合計 {total}件）"
        )
        return recovered

    async def convert_incident_csv_async(
        self, input_file: str, output_file: str
    ):
        """
        インシデントCSVを非同期で高速クリーンアップ

        Args:
            input_file: 入力CSVファイルのパス
            output_file: 出力CSVファイルのパス
        """
        try:
            # エンコーディング自動判定
            encodings = ["utf-8-sig", "utf-8", "cp932", "shift-jis"]
            df = None

            for encoding in encodings:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if df is None:
                raise ValueError(f"ファイルの読み込みに失敗: {input_file}")

            print("🧹 インシデントデータクリーニング開始（高速並列処理版）")
            print(f"📊 処理対象: {len(df)}件")
            print(f"🤖 使用モデル: {self.model}")
            print(f"⚡ 並列数: {self.max_concurrent}")
            print("=" * 50)

            # 非同期バッチ処理の実行
            results = await self.process_batch_async(df)

            # 結果をインデックス順にソート
            results.sort(key=lambda x: x[0])

            # データフレームに変換
            converted_data = []
            excluded_count = 0
            for idx, cleaned_data in results:
                if cleaned_data.is_faq_candidate:
                    new_row = {
                        "質問": cleaned_data.question,
                        "回答": cleaned_data.answer,
                        "カテゴリ": cleaned_data.category,
                        "キーワード": cleaned_data.keywords,
                        "リンク名": cleaned_data.link_names,
                        "立場": cleaned_data.user_role,
                        "信頼度": cleaned_data.confidence_score,
                        "信頼度理由": cleaned_data.confidence_reason,
                        "AIリスクレベル": cleaned_data.risk_level,
                        "リスク理由": cleaned_data.risk_reason,
                    }
                else:
                    new_row = {
                        "質問": f"【FAQ除外】{cleaned_data.exclusion_reason}",
                        "回答": "-",
                        "カテゴリ": cleaned_data.category,
                        "キーワード": "",
                        "リンク名": "",
                        "立場": "",
                        "信頼度": "",
                        "信頼度理由": "",
                        "AIリスクレベル": "",
                        "リスク理由": "",
                    }
                    excluded_count += 1
                converted_data.append(new_row)

            # 結果をCSVに保存
            converted_df = pd.DataFrame(converted_data)
            converted_df.to_csv(output_file, index=False, encoding="utf-8-sig")

            print(f"\n✅ クリーニング完了!")
            print(f"📄 入力: {input_file}")
            print(f"📄 出力: {output_file}")
            print(f"📊 変換件数: {len(converted_df)}件")
            print(f"📊 FAQ除外: {excluded_count}件")

            # 結果の概要表示
            self._display_cleaning_summary(converted_df, converted_data)

        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            raise

    def _display_cleaning_summary(
        self, df: pd.DataFrame, raw_data: List[Dict]
    ):
        """クリーニング結果の概要を表示"""
        print("\n" + "=" * 50)
        print("📊 クリーニング結果の概要")
        print("=" * 50)

        # カテゴリ別統計
        print("\n🏷️  カテゴリ別件数:")
        category_counts = df["カテゴリ"].value_counts()
        for category, count in category_counts.head(10).items():
            print(f"  • {category}: {count}件")

        # FAQ除外統計
        excluded_count = sum(
            1
            for item in raw_data
            if isinstance(item, dict) and item.get("回答", "") == "-"
        )
        print(f"🚫 FAQ除外件数: {excluded_count}/{len(df)}件")

        # キーワード統計
        all_keywords = []
        for keywords_str in df["キーワード"]:
            if pd.notna(keywords_str) and keywords_str:
                all_keywords.extend(keywords_str.split(";"))

        print(f"\n🔍 キーワード統計:")
        print(f"  • 総キーワード数: {len(all_keywords)}")
        print(f"  • ユニークキーワード数: {len(set(all_keywords))}")

        # リンク統計
        link_count = sum(
            1 for link in df["リンク名"] if pd.notna(link) and link.strip()
        )
        print(f"\n🔗 リンク名統計:")
        print(f"  • リンクありの件数: {link_count}件")


def main(
    input_file: str = None, output_file: str = None, max_concurrent: int = 50
):
    """
    メイン処理（ファイルベース）

    Args:
        input_file: 入力CSVパス（省略時はデフォルトパス）
        output_file: 出力CSVパス（省略時はデフォルトパス）
        max_concurrent: 並列数
    """
    print("🧹 インシデントデータクリーニングツール")
    print("⚡ 高速並列処理版 (800件/3分)")
    print("=" * 50)

    # デフォルトパス
    if input_file is None:
        input_file = (
            "C:/Users/US0008098/data/input/1.Q_csv/SNOWインシデント.csv"
        )
    if output_file is None:
        output_file = "C:/Users/US0008098/data/output/SNOW_Cleansing.csv"

    # API認証情報を環境変数から取得
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

    if not api_key or not azure_endpoint:
        print(
            "❌ 環境変数 AZURE_OPENAI_API_KEY と AZURE_OPENAI_ENDPOINT を設定してください"
        )
        return

    # クリーナーの初期化
    cleaner = FastIncidentDataCleaner(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        model="gpt-4.1",
        max_concurrent=max_concurrent,
    )
    print("✅ 非同期OpenAI APIクライアント初期化完了")

    # 非同期処理の実行
    import time

    start_time = time.time()

    asyncio.run(
        cleaner.convert_incident_csv_async(
            input_file=input_file, output_file=output_file
        )
    )

    elapsed_time = time.time() - start_time
    print(f"\n⏱️  処理時間: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分)")


if __name__ == "__main__":
    # 固定パスではなく引数で受け取る
    import sys

    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        input_file = (
            "C:/Users/US0008098/data/input/1.Q_csv/SNOWインシデント.csv"
        )
        output_file = "C:/Users/US0008098/data/output/SNOW_Cleansing.csv"

    main(input_file, output_file)
