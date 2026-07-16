#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
インデックス整合性テストスクリプト

各Phase間でインデックスが正しく保持されているかを検証
"""

import pandas as pd
import sys
import os

# テスト用のモックデータを使用してインデックスの追跡をテスト

def test_index_preservation():
    """インデックス保持のテスト"""
    
    print("=" * 60)
    print("🧪 インデックス整合性テスト")
    print("=" * 60)
    
    # テストデータ作成（意図的に完全一致を含める）
    test_data = {
        "件名": [
            "ログインできない",
            "ログインできない",  # 完全一致（削除対象）
            "パスワードを忘れた",
            "画面が表示されない",
            "ログインエラー",
            "申請方法がわからない",
        ],
        "カテゴリ": ["認証", "認証", "認証", "表示", "認証", "申請"],
        "概要": [
            "システムにログインできません",
            "システムにログインできません",  # 完全一致
            "パスワードを忘れてしまいました",
            "画面が真っ白で何も表示されない",
            "ログイン時にエラーが発生する",
            "休暇申請の方法を教えてください",
        ],
        "対応結果": [
            "パスワードリセットを実施",
            "パスワードリセットを実施。再度ログインを試みていただき、正常にログインできることを確認しました。",  # こちらの方が長いので採用される
            "パスワードリセット手順をご案内",
            "キャッシュクリアで解決",
            "ログインエラーはパスワード有効期限切れが原因でした。パスワードリセットを実施し、解決しました。",
            "申請システムから休暇申請を選択してください",
        ],
        "リンク名": [
            "パスワードリセット手順",
            "パスワードリセット手順;ログイン方法",
            "パスワードリセット手順",
            "キャッシュクリア方法",
            "パスワードリセット手順",
            "休暇申請マニュアル",
        ],
    }

    df = pd.DataFrame(test_data)
    
    print("\n📌 元データ:")
    print(f"  行数: {len(df)}")
    print(f"  インデックス: {list(df.index)}")
    print(df[["概要", "対応結果"]].head())
    
    # ===========================
    # Phase 0: 完全一致チェック
    # ===========================
    print("\n" + "=" * 60)
    print("📋 Phase 0: 完全一致チェック")
    print("=" * 60)
    
    from knowledge_distillation.pre_deduplication import run_phase0
    
    p0_df, _, _, p0_records, gid_tracker = run_phase0(
        df=df,
        output_dir="/tmp/test_output",
    )
    
    print(f"\n📌 Phase 0 結果:")
    print(f"  採用件数: {len(p0_df)}")
    print(f"  採用されたインデックス: {list(p0_df.index)}")
    
    # 期待: インデックス 0 は削除（対応結果が短いため）、1 が採用
    # または、完全一致なのでどちらか一方が採用
    
    print(f"\n📌 処理履歴:")
    for idx, record in sorted(p0_records.items()):
        print(f"  idx={idx}: {record.final_result} (P0_GID={record.p0_gid})")
    
    # ===========================
    # Phase 1: クレンジング（モック）
    # ===========================
    print("\n" + "=" * 60)
    print("📋 Phase 1: クレンジング（モック）")
    print("=" * 60)
    
    # 実際のAPIを呼ばずにモックデータを使用
    mock_cleaned_data = {
        "質問": [],
        "回答": [],
        "カテゴリ": [],
        "キーワード": [],
        "リンク名": [],
    }
    
    for idx in p0_df.index:
        row = p0_df.loc[idx]
        mock_cleaned_data["質問"].append(f"【質問】{row['概要']}")
        mock_cleaned_data["回答"].append(f"【回答】{row['対応結果']}")
        mock_cleaned_data["カテゴリ"].append(row["カテゴリ"])
        mock_cleaned_data["キーワード"].append("キーワード1;キーワード2")
        mock_cleaned_data["リンク名"].append(row.get("リンク名", ""))
    
    # ★重要: 元のインデックスを保持
    cleaned_df = pd.DataFrame(mock_cleaned_data, index=p0_df.index)
    
    print(f"\n📌 クレンジング後:")
    print(f"  行数: {len(cleaned_df)}")
    print(f"  インデックス: {list(cleaned_df.index)}")
    
    # インデックス一致確認
    if set(cleaned_df.index) == set(p0_df.index):
        print("  ✅ インデックスが正しく保持されています")
    else:
        print("  ❌ インデックスが不一致です！")
        assert False, "cleaned_df index does not match p0_df index"
    
    # 生データを結合
    cleaned_df["概要"] = p0_df.loc[cleaned_df.index, "概要"]
    cleaned_df["対応結果"] = p0_df.loc[cleaned_df.index, "対応結果"]
    
    print(f"\n📌 生データ結合後:")
    for idx in cleaned_df.index:
        print(f"  idx={idx}:")
        print(f"    質問: {cleaned_df.loc[idx, '質問'][:30]}...")
        print(f"    概要: {cleaned_df.loc[idx, '概要'][:30]}...")
        # 質問と概要が対応しているか確認
        if cleaned_df.loc[idx, '概要'] in cleaned_df.loc[idx, '質問']:
            print(f"    ✅ 対応OK")
        else:
            print(f"    ⚠️ 確認が必要")
    
    # ===========================
    # Phase 2: 文字列類似度チェック
    # ===========================
    print("\n" + "=" * 60)
    print("📋 Phase 2: 文字列類似度チェック")
    print("=" * 60)
    
    from knowledge_distillation.pre_deduplication import run_phase2
    
    p2_df, p2_records, gid_tracker = run_phase2(
        df=cleaned_df,
        threshold=0.75,
        output_dir="/tmp/test_output",
        processing_records=p0_records,
        gid_tracker=gid_tracker,
    )
    
    print(f"\n📌 Phase 2 結果:")
    print(f"  採用件数: {len(p2_df)}")
    print(f"  採用されたインデックス: {list(p2_df.index)}")
    
    print(f"\n📌 処理履歴:")
    for idx, record in sorted(p2_records.items()):
        print(f"  idx={idx}: {record.final_result} (P0_GID={record.p0_gid}, final_GID={record.final_gid})")
        print(f"    概要: {record.raw_overview[:40]}...")
        print(f"    質問: {record.question[:40]}...")
        # 概要と質問が対応しているか確認
        if record.raw_overview in record.question or record.question == "":
            print(f"    ✅ 対応OK")
        elif "【質問】" in record.question and record.raw_overview in record.question:
            print(f"    ✅ 対応OK（モック形式）")
        else:
            print(f"    ⚠️ 確認が必要")
    
    print("\n" + "=" * 60)
    print("✅ テスト完了")
    print("=" * 60)


def test_values_vs_loc():
    """
    .valuesと.locの違いを示すテスト
    """
    print("\n" + "=" * 60)
    print("🔬 .values vs .loc の違いテスト")
    print("=" * 60)
    
    # 飛び飛びのインデックスを持つDataFrame
    df1 = pd.DataFrame({
        "A": ["a0", "a2", "a4"],
        "B": ["b0", "b2", "b4"],
    }, index=[0, 2, 4])
    
    # 連番インデックスのDataFrame
    df2 = pd.DataFrame({
        "C": ["c0", "c1", "c2"],
    }, index=[0, 1, 2])
    
    print("\n📌 df1（飛び飛びインデックス）:")
    print(df1)
    print(f"  インデックス: {list(df1.index)}")
    
    print("\n📌 df2（連番インデックス）:")
    print(df2)
    print(f"  インデックス: {list(df2.index)}")
    
    # .valuesを使った代入（問題のあるパターン）
    print("\n❌ 問題のあるパターン: .values を使用")
    df2_bad = df2.copy()
    df2_bad["A"] = df1["A"].values  # 位置ベースで代入される
    print(df2_bad)
    print("  → df2のインデックス1にdf1のインデックス2のデータが入ってしまう")
    
    # .locを使った代入（正しいパターン）
    print("\n✅ 正しいパターン: .loc を使用（インデックスを合わせる）")
    # まずdf2のインデックスをdf1と同じにする
    df2_good = pd.DataFrame({
        "C": ["c0", "c1", "c2"],
    }, index=[0, 2, 4])  # df1と同じインデックス
    df2_good["A"] = df1.loc[df2_good.index, "A"]
    print(df2_good)
    print("  → インデックスが一致しているので正しく対応")


if __name__ == "__main__":
    # テスト実行
    test_values_vs_loc()
    print("\n")
    test_index_preservation()
