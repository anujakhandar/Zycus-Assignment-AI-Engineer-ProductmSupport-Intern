# Evaluation report

Generated: `2026-08-27T22:04:13+00:00`  
Model: `gemini-3.5-flash-lite`  
Pass threshold: `0.7`  
LLM judge: `enabled`

## Summary

| Metric | Value |
|---|---|
| Test cases | 13 |
| Passed | 12 |
| Failed | 1 |
| Pass rate | 92% |
| Mean quality score | 0.921 |
| Adversarial cases | 3/3 passed |
| Live API calls | 0 |
| Cache hits | 32 |
| Runtime | 0.1s |

## By task

| Task | Cases | Passed | Mean quality |
|---|---|---|---|
| account_brief | 6 | 6 | 0.975 |
| triage | 7 | 6 | 0.875 |

## Cases

| Case | Task | Result | Quality | Rule | Judge | Notes |
|---|---|---|---|---|---|---|
| `T1` | triage | PASS | 0.850 | 1.000 | 0.50 | all criteria met |
| `T2` | triage | PASS | 0.850 | 1.000 | 0.50 | all criteria met |
| `T3` | triage | PASS | 0.892 | 0.846 | 1.00 | cites the billing doc |
| `T4` | triage | PASS | 1.000 | 1.000 | 1.00 | all criteria met |
| `T5` | triage | FAIL | 0.533 | 0.333 | 1.00 | product area is an access area, cites the SSO troubleshooting doc |
| `T6-adversarial-tone` | triage | PASS (adv) | 1.000 | 1.000 | 1.00 | all criteria met |
| `T7-adversarial-sparse` | triage | PASS (adv) | 1.000 | 1.000 | 1.00 | all criteria met |
| `B1` | account_brief | PASS | 1.000 | 1.000 | 1.00 | all criteria met |
| `B2` | account_brief | PASS | 1.000 | 1.000 | 1.00 | all criteria met |
| `B3` | account_brief | PASS | 0.850 | 1.000 | 0.50 | all criteria met |
| `B4` | account_brief | PASS | 1.000 | 1.000 | - | all criteria met |
| `B5` | account_brief | PASS | 1.000 | 1.000 | - | all criteria met |
| `B6-adversarial-no-tickets` | account_brief | PASS (adv) | 1.000 | 1.000 | 1.00 | all criteria met |

## Failures in detail

### `T5` - SSO access failure for new joiners routes to Security Engineering

- **Critical criterion failed:** cites the SSO troubleshooting doc
- Criterion failed: product area is an access area
- Judge: The response explicitly identifies group-to-role mapping under 'Settings > SSO > Group Mapping' as the concrete cause, directly matching the rubric requirement.


## How to reproduce

```bash
pip install -r requirements.txt
python main.py eval
```

Cached model responses are committed under `fixtures/llm_cache/`, so this runs without an API key and returns the same results.
