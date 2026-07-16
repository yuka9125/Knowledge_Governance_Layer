# Phase F Demo Recording Guide

This guide prepares a silent video asset for the Phase F demo. It is intended
to show the real operating flow: distillation commands, Excel review,
approved Knowledge merge, and Serving query results. All visible content uses
synthetic/demo data and must not be presented as customer data.

## Shot List

Target length: about 90 seconds.

1. Scope
   - Show the four real demo steps: distillation, Excel review, merge, Serving
     question.
2. knowledge_distillation
   - Run `prepare_live_demo.py`, `normalize`, `ingest --no-openai`, and `export`
     against the synthetic inquiry log.
3. Excel review
   - Open the generated `02_FAQ_final_result_for_review.xlsx`.
   - Show the P3-2 existing FAQ update candidate.
   - Change the review result to `採用`.
4. approved_knowledge merge
   - Run `approved_knowledge_exporter` against the approved workbook.
   - Show that `demo-vpn-001` is updated in place.
5. Serving questions
   - Before: `VPNにつながりません` returns the old restart answer.
   - After: the same question returns the updated latest-client answer.
   - No match: `承認済みになっていない申請期限を教えて` returns
     `answerable=false` and `fallback=human_review`.

## Prepare Live Demo Files

```powershell
python benchmark/demo/prepare_live_demo.py

python -m knowledge_distillation normalize `
  --adapter csv `
  --path _system\data\phase_f_live_demo\01_synthetic_inquiry_log.csv `
  --out _system\data\phase_f_live_demo\01_normalized.csv `
  --question-col question `
  --answer-cols answer `
  --title-col ticket_id `
  --category-col category

python -m knowledge_distillation ingest `
  --adapter csv `
  --path _system\data\phase_f_live_demo\01_synthetic_inquiry_log.csv `
  --question-col question `
  --answer-cols answer `
  --db-path _system\data\phase_f_live_demo\knowledge.db `
  --no-openai

python -m knowledge_distillation export `
  --db-path _system\data\phase_f_live_demo\knowledge.db `
  --out-dir _system\data\phase_f_live_demo\distillation_output
```

Open the generated workbook and approve the row:

```text
_system/data/phase_f_live_demo/02_FAQ_final_result_for_review.xlsx
```

For repeatable rendering, an already approved fixture is also generated:

```text
_system/data/phase_f_live_demo/03_FAQ_final_result_approved.xlsx
```

Merge the approved row into Serving Knowledge:

```powershell
python -m knowledge_distillation.approved_knowledge_exporter `
  _system\data\phase_f_live_demo\03_FAQ_final_result_approved.xlsx `
  -o _system\data\phase_f_live_demo\approved_knowledge_merged.json `
  --base _system\data\phase_f_live_demo\approved_knowledge_before.json
```

Generate the Serving query result fixture from the actual service code:

```powershell
python - <<'PY'
import json
from pathlib import Path
from serving.governed_knowledge_api import GovernedKnowledgeService

root = Path("_system/data/phase_f_live_demo")
queries = [
    ("before", root / "approved_knowledge_before.json", "VPNにつながりません"),
    ("after", root / "approved_knowledge_merged.json", "VPNにつながりません"),
    ("no_match", root / "approved_knowledge_merged.json", "承認済みになっていない申請期限を教えて"),
]
results = []
for scene, path, q in queries:
    svc = GovernedKnowledgeService(approved_path=path)
    results.append({"scene": scene, "query": q, "result": svc.search(q)})
(root / "04_serving_query_results.json").write_text(
    json.dumps({"synthetic_data": True, "results": results}, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
PY
```

## Generate Silent Video Asset

The normal environment does not require Azure OpenAI keys for this recording.
The renderer uses the synthetic fixtures in `benchmark/demo/`.

Install the local rendering dependencies once if needed:

```powershell
& "C:\Users\yukai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m pip install --user imageio imageio-ffmpeg
```

PowerShell command using the bundled Codex runtime:

```powershell
& "C:\Users\yukai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" benchmark/demo/render_live_demo_video.py
```

Optional shorter preview:

```powershell
& "C:\Users\yukai\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" benchmark/demo/render_live_demo_video.py --duration-scale 0.25 --out _system/data/phase_f_live_demo/phase_f_live_demo_preview.mp4
```

Output files:

```text
_system/data/phase_f_live_demo/phase_f_live_demo.mp4
_system/data/phase_f_live_demo/phase_f_live_demo_poster.png
```

The generated `.mp4` is silent.

## Live API Checks

Before recording or publishing, keep the executable checks current:

```bash
python benchmark/conflict/evaluate_conflicts.py
python benchmark/demo/run_demo_scenarios.py --validate-only
```

Before fixture:

```powershell
$env:APPROVED_KNOWLEDGE_PATH = "benchmark/demo/approved_knowledge_before.json"
python -m uvicorn serving.governed_knowledge_api:app --port 8000
python benchmark/demo/run_demo_scenarios.py --base-url http://127.0.0.1:8000 --expected-state before
```

After fixture:

```powershell
$env:APPROVED_KNOWLEDGE_PATH = "benchmark/demo/approved_knowledge_after.json"
python -m uvicorn serving.governed_knowledge_api:app --port 8000
python benchmark/demo/run_demo_scenarios.py --base-url http://127.0.0.1:8000 --expected-state after
```
