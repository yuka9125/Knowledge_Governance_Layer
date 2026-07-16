# 再デプロイ不要でナレッジ反映（Blob）— 初心者向けセットアップ

「ナレッジ出力を実行するだけで、URL（serving）に自動反映される」状態にするための**一回だけ**の設定です。
**Command Prompt でも PowerShell でもOK**（Azure CLI が動けば同じ）。

> これは“あると便利”な設定です。やらなくても、これまでどおり再デプロイすれば反映はできます。

---

## 0. 準備（最初に1回）

1. **Azure CLI が入っているか確認**（ターミナルで）:
   ```
   az version
   ```
   入っていなければ https://aka.ms/installazurecliwindows からインストール。

2. **ログイン**:
   ```
   az login
   ```
   ブラウザが開くのでサインイン。

3. **自分のアプリとリソースグループ名を確認**:
   ```
   az containerapp list -o table
   ```
   表の中の `Name`（= `kgl-serving` のはず）と `ResourceGroup`（例: `KGL-rg`）を**メモ**。
   以下では `RG = KGL-rg`、`APP = kgl-serving`、地域 `eastus` として書きます（違えば置き換え）。

---

## 1. ストレージアカウントを作る

`<ストレージ名>` は**世界で一意・小文字英数字のみ・3〜24文字**。例：`kglstore0531a`（被ったら数字を変える）。

```
az storage account create -n kglstore0531a -g KGL-rg -l eastus --sku Standard_LRS
```
（1分ほどかかります）

## 2. 入れ物（コンテナ）を作る

```
az storage container create --account-name kglstore0531a -n approved-knowledge
```

## 3. 接続文字列を取得（あとで使うのでコピー）

```
az storage account show-connection-string -g KGL-rg -n kglstore0531a --query connectionString -o tsv
```
出力された**長い1行**（`DefaultEndpointsProtocol=https;...` で始まる）を**コピー**しておく。
※これは秘密情報。Git やスクショに貼らないこと。

## 4. 今のナレッジを1回アップロード

リポジトリのフォルダへ移動してから実行：

```
cd C:\Users\yukai\Desktop\Knowledge_Governance_Layer\Knowledge_Governance_Layer-git
az storage blob upload --account-name kglstore0531a -c approved-knowledge -f data\approved_knowledge.json -n approved_knowledge.json --overwrite
```

## 5. serving（URL）に「Blobを読む」設定をする（この時だけ更新が走る）

接続文字列をまず**シークレット**として登録（手順3でコピーした文字列を二重引用符の中に貼る）：

```
az containerapp secret set -n kgl-serving -g KGL-rg --secrets "storageconn=ここに手順3の接続文字列を貼る"
```

次に環境変数を設定（シークレットを参照させる）：

```
az containerapp update -n kgl-serving -g KGL-rg --set-env-vars APPROVED_KNOWLEDGE_BLOB_CONTAINER=approved-knowledge APPROVED_KNOWLEDGE_BLOB_NAME=approved_knowledge.json AZURE_STORAGE_CONNECTION_STRING=secretref:storageconn
```

これで serving は**ローカルのファイルではなく Blob を読む**ようになります。

## 6. 反映の確認（30秒ほど待ってから）

```
curl https://kgl-serving.yellowisland-ad734fe6.eastus.azurecontainerapps.io/health
```
`{"status":"ok"}` が返ればOK。チャットUIで質問しても確認できます。

---

## 7. これ以降の運用（再デプロイ不要）

distillation（ナレッジ生成アプリ）を動かす**このPCの同じターミナル**で、アプリ起動前に
**同じ Blob 設定**を入れておきます。

**Command Prompt の場合：**
```
set APPROVED_KNOWLEDGE_BLOB_CONTAINER=approved-knowledge
set AZURE_STORAGE_CONNECTION_STRING=ここに手順3の接続文字列を貼る
open_knowledge_distillation.bat
```

**PowerShell の場合：**
```
$env:APPROVED_KNOWLEDGE_BLOB_CONTAINER="approved-knowledge"
$env:AZURE_STORAGE_CONNECTION_STRING="ここに手順3の接続文字列を貼る"
.\open_knowledge_distillation.bat
```

→ アプリで「ナレッジ出力」を実行すると、ローカル出力に加えて**Blobへ自動アップロード**され、
画面に「☁️ Blobへアップロードしました」と出ます。
serving は次のアクセスで自動的に新しいナレッジを読み込みます（**再デプロイ不要**）。

> 毎回ターミナルを開き直すと `set` は消えます。固定したい場合は `.env` に
> `APPROVED_KNOWLEDGE_BLOB_CONTAINER` と `AZURE_STORAGE_CONNECTION_STRING` を書いておけば、
> アプリ起動時に自動で読み込まれます（`.env` はコミットしないこと）。

---

## つまずいたら

| 症状 | 対処 |
|---|---|
| `az` が見つからない | Azure CLI 未インストール → 手順0-1 |
| ストレージ名が作れない | 既に使われている名前 → `kglstore0531b` 等に変更 |
| `ResourceGroup` が違う | 手順0-3 の表で実際の名前を確認して置き換え |
| URLに反映されない | 手順5の update 後に30秒待つ／`az containerapp revision list -n kgl-serving -g KGL-rg -o table` で最新リビジョンを確認 |
| 接続文字列の貼り付けでエラー | 必ず二重引用符 `"..."` の中に貼る |
