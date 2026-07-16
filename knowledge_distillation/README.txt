Knowledge Distillation
======================

This package contains the batch side of the Knowledge Governance Layer:

- inquiry log cleansing
- FAQ candidate generation
- duplicate and existing FAQ checks
- Excel review output
- approved Knowledge export

The serving layer must not import this package. The only handoff contract is:

  data/approved_knowledge.json

Common commands:

  python -m knowledge_distillation --help
  python -m knowledge_distillation.approved_knowledge_exporter FAQ_final_result.xlsx -o data/approved_knowledge.json

Other draft/intermediate outputs may remain under data/outputs or data/intermediate.
