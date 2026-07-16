# Knowledge Governance Layer

問い合わせログから FAQ 候補を生成し、人間レビューで承認された Knowledge だけを回答 API / エージェント / UI から返す MVP です。

## Structure

```text
knowledge_distillation/   # Batch layer: draft generation, Excel review output, approved_knowledge export
serving/                  # Serving layer: FastAPI, Semantic Kernel agent, minimal chat UI
benchmark/                # Synthetic benchmark and golden-query fixtures
data/approved_knowledge.json
```

`serving/` は `knowledge_distillation/` を import しません。回答側が参照する知識ソースは `data/approved_knowledge.json` だけです。

## Draft / Distillation

```bash
pip install -r knowledge_distillation/requirements.txt
python -m knowledge_distillation ingest --adapter csv --path sample.csv --no-openai
python -m knowledge_distillation normalize --adapter csv --path sample.csv --out data/intermediate/normalized_snow_input.csv
python -m knowledge_distillation export
python -m knowledge_distillation.approved_knowledge_exporter FAQ_final_result.xlsx -o data/approved_knowledge.json
```

For a screen-recording demo, start from these synthetic files:

```text
benchmark/demo/knowledge_distillation_start_inquiries.csv
data/approved_knowledge.json
```

The inquiry starter file intentionally uses `第1対応` / `第2対応` / `最終結果`
columns so the demo can show that multiple response-history columns are merged
into one answer candidate.
Phase 3-2 no longer needs an existing FAQ CSV upload. It compares candidates
against approved Knowledge loaded from `data/approved_knowledge.json`.
`benchmark/demo/knowledge_distillation_start_existing_faq.csv` remains as a
legacy/reference starter for explaining the old FAQ shape, but it is not
uploaded in the current UI.

On Windows, launch the Streamlit UI with:

```powershell
.\open_knowledge_distillation.ps1
```

or double-click:

```text
open_knowledge_distillation.bat
```

## Serving

```bash
pip install -r serving/requirements.txt
python -m uvicorn serving.web_app:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `GET /knowledge/search?q=...`
- `POST /chat`
- `GET /`

## Configuration

Copy `.env.example` to `.env` for local execution. Secrets must stay in environment variables and must not be committed.

Important paths:

- `APPROVED_KNOWLEDGE_PATH`: optional override for the approved Knowledge contract file.
- Default contract file: `data/approved_knowledge.json`

Azure OpenAI values are deployment names, not model names.

## Azure Container Apps

Container deployment instructions are in
[`docs/deploy-containerapps.md`](docs/deploy-containerapps.md).

The Container Apps package uses `serving/Dockerfile` and bundles only approved
knowledge. After updating `data/approved_knowledge.json`, refresh
`serving/data/approved_knowledge.json` and redeploy the container.

## Phase F Preparation

Phase F-0 synthetic benchmark inputs and fixed measurement rules are documented
in [`docs/phase-f-prep.md`](docs/phase-f-prep.md).

```bash
python benchmark/conflict/evaluate_conflicts.py
```

Demo scenario inputs for Before / Governance / After / no-match are in
`benchmark/demo/phase_f_demo_scenarios.json`.

```bash
python benchmark/demo/run_demo_scenarios.py --validate-only
```

Live-flow silent demo video recording instructions are in
[`docs/phase-f-demo-recording.md`](docs/phase-f-demo-recording.md).
