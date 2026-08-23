# Agent triage evaluation

Backend: `groq:openai/gpt-oss-120b` · 20 anomalies sampled from 30,000 trades 
(seed 42, stratified by anomaly type). Regenerate with 
`python scripts/evaluate_agents.py --backend groq`.

| Metric | Result | Answer key |
|---|---|---|
| Tool-selection accuracy (first call) | **84.2%** | `tools.expected_first_tool` rubric |
| Structured-output validity | **90.0%** | Pydantic schema validation, first attempt |
| Routing accuracy | **73.7%** | generator's anomaly-injection ground truth |
| Hallucinated tool names | 0 of 19 | name not in the tool registry |
| Wall clock | 345.7s (17.28s per anomaly) | |

Verdicts: {'ESCALATE': 14, 'AUTO_CLOSE': 5}  
Categories: {'INFRASTRUCTURE': 8, 'CLIENT_BEHAVIOUR': 6, 'MARKET_ACTIVITY': 6}

## How to read this

Tool selection is judged on the FIRST call only. Later calls depend on what
earlier ones returned, so grading them against a fixed key would punish the
agent for adapting correctly.

Routing accuracy is bounded above by the detector: a finding the detector
should never have raised can still be routed correctly (auto-closed), but the
agent cannot escalate something that was never surfaced to it.

Run `--backend scripted` for the rule-based control arm. The model has to beat
a handful of if-statements to justify its latency.

## Prior run: a contradictory tool description

The first measured run of this harness scored **36.8%** tool-selection accuracy,
against **100%** for the scripted rule-based control. A model losing that badly to
a handful of if-statements is a bug report, not a result, so it was worth reading
the actual choices rather than reporting the number.

The model was picking `get_trade_detail` for every trade-scoped anomaly. Its
stated reason was consistent and sensible. And it was right to: the tool's own
description read *"The full record for one trade id. Use to inspect a trade-scoped
anomaly."* The description instructed exactly the behaviour the answer key marked
wrong.

The fault was in the tool descriptions and the prompt, not the model. Two changes:

1. Descriptions now state what each tool **establishes** rather than when to reach
   for it, and `get_trade_detail` says outright that the anomaly context already
   contains the trade's own fields.
2. The investigator prompt now states the underlying principle: an anomaly is a
   value that looks extreme, so judging it needs a **baseline**, not a restatement
   of the value.

| Metric | Before | After |
|---|---|---|
| Tool-selection accuracy | 36.8% | **84.2%** |
| Routing accuracy | 57.9% | **73.7%** |
| Structured-output validity | 92.2% | 90.0% |
| Hallucinated tool names | 0/19 | 0/19 |

Validity moved within noise on 19 cases and should not be read as a change.

The transferable lesson: in an agent, **the tool descriptions are program text**,
not documentation. A contradiction between a description and the evaluation's
answer key is indistinguishable from a weak model until the traces are read.
