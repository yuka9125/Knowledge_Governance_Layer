# Phase F-0 Preparation

This document fixes the inputs and measurement rules used before the
three-minute Phase F demo video.

## Scope

Phase F demo itself shows three scenes:

1. Before: old approved FAQ returns the old VPN answer.
2. Governance: distillation creates an update candidate, P3-2 review confirms it,
   and `approved_knowledge.json` is updated.
3. After and no-match: approved VPN answer is returned, while an unapproved
   question escalates to human review without guessing.

Phase F-0 prepares the benchmark and operating rules that make the demo
measurable.

## Synthetic Dataset

Input file:

```text
benchmark/conflict/synthetic_conflict_dataset.json
```

The file is explicitly marked as synthetic data and contains four labels:

- `統合すべき`: same topic and compatible answer. Expected conflict is false.
- `新旧矛盾`: same topic but old/new, enabled/disabled, or possible/impossible
  answer conflict. Expected conflict is true.
- `似てるが別物`: related wording but a different issue. Expected conflict is
  false.
- `無関係`: unrelated question. Expected conflict is false.

This dataset is the Phase F input for conflict precision and recall.

## Thresholds

Embedding model assumption is fixed to `text-embedding-3-large`.

Current fixed thresholds:

```text
Serving answer search threshold: 0.80
P3-2 exact match threshold:      0.95
P3-2 similar threshold:          0.85
P3-2 review threshold:           0.75
```

The deployed Azure Container Apps serving endpoint was checked with golden
queries on 2026-05-31 JST:

```text
GET /health -> 200
VPNにつながりません -> answerable=true, knowledge_id=demo-vpn-001
経費精算の申請方法を教えて -> answerable=false, fallback=human_review
承認済みになっていない申請期限を教えて -> answerable=false, fallback=human_review
```

Golden query file:

```text
benchmark/eval/phase_f_golden_queries.json
```

Run against the deployed serving URL:

```bash
python benchmark/eval/run_golden_queries.py \
  --base-url https://kgl-serving.yellowisland-ad734fe6.eastus.azurecontainerapps.io \
  --golden benchmark/eval/phase_f_golden_queries.json \
  --timeout 60
```

Latest Phase F-0 result: 4 / 4 passed, accuracy 100.0%.

For Phase F-0, the serving search threshold remains `0.80` because it supports
the demo's approved-answer and no-match branches on the deployed large-embedding
configuration. The P3-2 thresholds remain `0.95 / 0.85 / 0.75` and are exposed
as constants in `knowledge_distillation/knowledge_output_utils.py` so later
changes are explicit.

After recording benchmark numbers for a submission, do not change these
thresholds without re-running and re-reporting the benchmark.

## Fixed Conflict Rule

Conflict detection for the Phase F-0 benchmark is fixed as:

```text
similarity_large >= 0.75
AND candidate/existing answers contain an opposite-signal pair
```

Opposite-signal examples include:

```text
旧 vs 新
古い vs 最新
旧システム vs 新システム
廃止 vs 利用
使用不可 vs 使用可能
できません vs できます
不要 vs 必要
```

The evaluator is:

```bash
python benchmark/conflict/evaluate_conflicts.py
```

It prints precision, recall, and F1. The dataset is synthetic, so these numbers
must be described as synthetic benchmark results.

Latest local result recorded on 2026-05-31 JST:

```text
total=12 tp=3 fp=0 tn=9 fn=0
precision=1.0000 recall=1.0000 f1=1.0000
```

## Demo Scenario Inputs

Phase F demo inputs are fixed in:

```text
benchmark/demo/phase_f_demo_scenarios.json
```

The file is explicitly marked as synthetic data and contains four scenes:

- `before`: old approved VPN knowledge returns the old restart answer.
- `governance`: synthetic inquiry log becomes a P3-2 existing FAQ update
  candidate and is approved as an existing FAQ update.
- `after`: updated approved VPN knowledge returns the latest-client answer.
- `no_match`: an unapproved question returns `answerable=false` and
  `fallback=human_review`.

The API fixtures are:

```text
benchmark/demo/approved_knowledge_before.json
benchmark/demo/approved_knowledge_after.json
```

Validate the scenario structure without starting the API:

```bash
python benchmark/demo/run_demo_scenarios.py --validate-only
```

Run the Before scene locally:

```bash
APPROVED_KNOWLEDGE_PATH=benchmark/demo/approved_knowledge_before.json \
python -m uvicorn serving.governed_knowledge_api:app --port 8000

python benchmark/demo/run_demo_scenarios.py \
  --base-url http://127.0.0.1:8000 \
  --expected-state before
```

Run the After and no-match scenes locally:

```bash
APPROVED_KNOWLEDGE_PATH=benchmark/demo/approved_knowledge_after.json \
python -m uvicorn serving.governed_knowledge_api:app --port 8000

python benchmark/demo/run_demo_scenarios.py \
  --base-url http://127.0.0.1:8000 \
  --expected-state after
```

On PowerShell, set the fixture path before starting Uvicorn:

```powershell
$env:APPROVED_KNOWLEDGE_PATH = "benchmark/demo/approved_knowledge_after.json"
python -m uvicorn serving.governed_knowledge_api:app --port 8000
```

## approved_knowledge Update Policy

Default policy: merge by upsert.

When distillation is re-run, the default exporter behavior is to merge into the
existing `approved_knowledge.json` by `knowledge_id`:

- existing FAQ update: use `既存FAQ_ID` as `knowledge_id` and overwrite that item
- new approved FAQ: append as a new item
- existing unrelated approved knowledge: keep it

Command:

```bash
python -m knowledge_distillation.approved_knowledge_exporter FAQ_final_result.xlsx -o data/approved_knowledge.json
```

Snapshot overwrite is available only when the measurement or demo needs a fixed
one-run output:

```bash
python -m knowledge_distillation.approved_knowledge_exporter FAQ_final_result.xlsx -o data/approved_knowledge.json --no-merge
```

Use `--no-merge` for isolated benchmark fixtures. Use the default merge policy
for normal distillation updates and for the demo's Governance -> After flow.

## Embedding Consistency Check

Before Phase F recording, verify that no small-model or 1536-dimension fixed
configuration remains:

```bash
rg -n "text-embedding-3-small|1536|dimensions|dimension"
rg -n "text-embedding-3-large|AZURE_OPENAI_EMBEDDING"
```

As of Phase F-0, the serving layer defaults to `text-embedding-3-large`, and no
`1536` fixed vector dimension was found.
