# Deploy to Azure Container Apps

This document packages only the serving layer:

- FastAPI Governed Knowledge API
- Semantic Kernel support agent
- Minimal chat UI
- `data/approved_knowledge.json` as the approved-only knowledge source

The app does not read draft data, Excel files, or unapproved knowledge at runtime.

## Prerequisites

Human executes:

```bash
az login
az account set --subscription "Azure subscription 1"
```

Human executes if the providers are not registered:

```bash
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
az provider register --namespace Microsoft.ContainerRegistry
```

## Variables

Human executes:

```bash
RG=KGL-rg
LOC=eastus
ENVNAME=kgl-cae
APP=kgl-serving
```

Use `eastus` so the Container App is co-located with the Azure OpenAI resource.

## Create Container Apps Environment

Human executes:

```bash
az containerapp env create -n $ENVNAME -g $RG -l $LOC
```

## Build and Deploy from Source

Human executes from the repository root:

```bash
az containerapp up -n $APP -g $RG --environment $ENVNAME \
  --source ./serving --ingress external --target-port 8000 --location $LOC
```

`./serving` is a self-contained container build context. Its Dockerfile copies
`serving/data/approved_knowledge.json` into the image as
`/app/data/approved_knowledge.json`, which is the runtime path read by the app.

Before redeploying after a knowledge update, refresh the container build copy:

```bash
cp data/approved_knowledge.json serving/data/approved_knowledge.json
```

On Windows PowerShell:

```powershell
Copy-Item -LiteralPath data\approved_knowledge.json -Destination serving\data\approved_knowledge.json -Force
```

## Configure Secrets and Environment Variables

Human executes. Do not commit the real key.

```bash
az containerapp secret set -n $APP -g $RG --secrets aoai-key=<AZURE_OPENAI_API_KEYの値>
az containerapp update -n $APP -g $RG --set-env-vars \
  AZURE_OPENAI_ENDPOINT=https://kgl-openai.openai.azure.com/ \
  AZURE_OPENAI_API_VERSION=<Playgroundの値> \
  AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1 \
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-large \
  AZURE_OPENAI_API_KEY=secretref:aoai-key
```

`AZURE_OPENAI_CHAT_DEPLOYMENT` and
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT` are Azure OpenAI deployment names.

## Get the Public URL

Human executes:

```bash
az containerapp show -n $APP -g $RG --query properties.configuration.ingress.fqdn -o tsv
```

The reviewer URL is:

```text
https://<fqdn>
```

Verify:

```bash
curl https://<fqdn>/health
```

Open `https://<fqdn>/` in a browser and send a sample question through the chat
UI. `/chat` requires the Azure OpenAI environment variables above.

## Updating approved_knowledge.json

For the MVP, `approved_knowledge.json` is bundled into the image. To update it:

1. Human exports the reviewed Excel result to `data/approved_knowledge.json`.
2. Human copies it to `serving/data/approved_knowledge.json`.
3. Human runs `az containerapp up` again from the repository root.

Future production hardening can move the approved knowledge file to Azure Blob
Storage so knowledge updates do not require rebuilding the image.

## ナレッジ更新を「再デプロイ不要」にする（Blob Storage ホットスワップ）

serving は Blob を ETag で監視して**自動再読込**します。distillation の「ナレッジ出力」も
Blob 設定があれば**自動でアップロード**します。これにより、

```
レビュー → ナレッジ出力 → Blobへアップ → serving が自動反映（再デプロイ不要）
```

### 一回だけの準備

1. ストレージアカウント＋コンテナを作成
   ```bash
   SA=kglstorage12345          # 小文字英数字・世界で一意
   CONTAINER=approved-knowledge
   az storage account create -g $RG -n $SA -l $LOC --sku Standard_LRS
   az storage container create --account-name $SA -n $CONTAINER
   CONN=$(az storage account show-connection-string -g $RG -n $SA --query connectionString -o tsv)
   ```
2. 最新のナレッジを一度アップロード
   ```bash
   az storage blob upload --account-name $SA -c $CONTAINER \
     -f data/approved_knowledge.json -n approved_knowledge.json --overwrite
   ```
3. serving（Container App）に Blob を読ませる環境変数を設定して反映（**この時だけ再デプロイ/更新**）
   ```bash
   az containerapp update -n $APP -g $RG --set-env-vars \
     APPROVED_KNOWLEDGE_BLOB_CONTAINER=$CONTAINER \
     APPROVED_KNOWLEDGE_BLOB_NAME=approved_knowledge.json \
     AZURE_STORAGE_CONNECTION_STRING="$CONN"
   ```
   > 本番ではマネージドID（`AZURE_STORAGE_ACCOUNT_URL` ＋ ロール `Storage Blob Data Reader`）推奨。

### 以降の運用（再デプロイ不要）

- distillation を回すマシンの環境変数に同じ Blob 設定（`APPROVED_KNOWLEDGE_BLOB_CONTAINER` /
  `AZURE_STORAGE_CONNECTION_STRING` など）を入れておく
- レビュー済みExcelから「ナレッジ出力」を実行すると、ローカル出力＋**Blobへ自動アップロード**
- serving は次のリクエストで ETag 変化を検知し、**自動で新ナレッジを読み込む**（再デプロイ・再ビルド不要）

## Local Docker Check

Human or Codex can execute if Docker is available:

```bash
docker build -t kgl-serving ./serving
docker run --rm -p 8000:8000 --env-file .env kgl-serving
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"VPNに接続できません"}'
```

