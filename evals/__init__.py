"""Candle-Fire evaluation harness.

Tier-1 deterministic red/green gate. Two suites:

  offline — no runtime data, no LLM, no network. Runs in PR CI on every push.
            (normalization alias collapse, therapy-landscape gold, gold-file schema)

  data    — needs the ChromaDB index + graph pickle present locally. Runs as a
            deploy gate before `git push hf main`, where the real assets exist.
            (KG-expansion recall, retrieval recall@k/precision@k/MRR, citation existence)

See docs/eval-plan.md. Entry point: `python -m evals.runner`.
"""
