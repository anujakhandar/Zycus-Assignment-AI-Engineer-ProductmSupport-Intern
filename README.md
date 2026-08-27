# Support Triage and Account Health Tooling

Internal LLM tooling for two customer facing teams: Technical Support engineers who
resolve tickets, and Technical Account Managers who prepare for customer reviews.
Everything here runs on the mock dataset supplied in the starter repo: 500 support
tickets, 50 account records, and nine knowledge base documents.

Four things are built.

1. __Ticket triage.__ Takes a raw ticket and returns product area, issue category,
   urgency, the knowledge base sections that support the call, a recommended responder
   team, and a draft first reply. Run with `main.py triage`.
2. __Account brief.__ Takes an account id and returns a three section brief with churn
   signals justified by verified customer quotes. Run with `main.py brief`.
3. __Evaluation harness.__ Runs 13 test cases across both pipelines and writes a
   scored report. Run with `main.py eval`.
4. __Design note.__ Failure modes, the latency and quality tradeoff, data sensitivity,
   and scaling. It is a section of this README, further down.

Everything is reachable through one entry point, `main.py`.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

On macOS or Linux, use `source .venv/bin/activate` and `cp .env.example .env`.

Then open `.env` and add an Anthropic API key.

```
ANTHROPIC_API_KEY=sk-ant-...
```

Edit `.env`, never `.env.example`. The example file is committed to the repository and
must only ever hold placeholder values.

## Running without an API key

Model responses are cached in `fixtures/llm_cache/` and committed to the repository.
Every command below replays from that cache and produces byte identical output with no
key and no network access. This is also how the continuous integration job runs the
evaluation harness, which is why the workflow needs no repository secret.

## Task 1: ticket triage

### Running it

```
python main.py triage --ticket TKT-10042
```

The command also accepts free text or a JSON file, since the brief asks for raw ticket
input rather than a dataset lookup.

```
python main.py triage --text "Our pipeline has been failing since this morning"
python main.py triage --file ticket.json --json
```

### Sample run

```
========================================================================
TRIAGE  TKT-10042
========================================================================

Product        DataBridge Pro   (confidence 0.95)
               The ticket names DataBridge Pro directly.

Product area   Connectors       (confidence 0.90)
               The failing component is the Connectors pipeline.

Category       Bug              (confidence 0.85)
               A previously working pipeline now errors, so this is a defect.

Urgency        P2               (confidence 0.88)
               A production pipeline is down for 47 users with no workaround.

Responder      Tier-2 Support
               Reproducible pipeline failure with a specific error code.

Known issue    Pipeline stopped processing

KB sections
               DataBridge Pro, Core Modules  [ERR_CONNECTION_TIMEOUT]
                 products/databridge-pro.md  (score 28.46)
               Troubleshooting: Performance Issues, Error Reference
                 troubleshooting/performance-and-integrations.md  (score 29.08)

Draft first response
------------------------------------------------------------------------
Thanks for flagging this. ERR_CONNECTION_TIMEOUT after 30s points to the
source being unreachable, so the first thing worth checking is the network
rules and firewall allowlist for that connector...
------------------------------------------------------------------------

Overall confidence 0.90   Human review: no

prompt triage@1.1.0   model claude-sonnet-4-6   cached True
```

### How it works

A deterministic pass runs before any model is involved. It extracts error codes from
the ticket and identifies the product by exact name match. That hint drives BM25
retrieval over the knowledge base, boosted by exact error code matches. A single model
call then produces the classification, the routing and the draft together.

Three things make the output trustworthy rather than merely plausible.

1. __Vocabulary validation.__ Every predicted value is checked against the vocabularies
   that actually occur in the dataset. A response naming a sixth product or a "P0"
   urgency is repaired by a deterministic rule, has its confidence capped, and is
   recorded in `review_reasons` instead of passing through silently.
2. __Citation intersection.__ Cited chunk ids are intersected with what retrieval
   actually returned, so the model cannot cite a document it was never shown. Invented
   citations are discarded.
3. __Review routing.__ Low confidence, no supporting document, a very short ticket, or
   a P1 rating all set `needs_human_review` with the reason attached.

## Task 2: account brief

### Running it

```
python main.py accounts
python main.py brief --account ACC-3847
python main.py brief --account "Initech" --json
```

`accounts` lists every account id with its tier, health and revenue, so a brief can be
run without opening the JSON.

### Sample run

```
# Account Brief, Initech (ACC-4654)

## 1. Executive summary

Initech is an Enterprise account on $240,000 ARR with 298 of 350 seats active and a
customer relationship dating to 2021. Health is At Risk against a declining usage
trend. Eleven tickets fall in the current window, five of which are still open...

## 2. Open risks & flagged issues

[High] Renewal approaching against declining usage  (source: renewal_date)
   Evidence: renews 2026-12-31 with usage_trend "Declining"

### Churn / escalation signals

Competitive evaluation  (TKT-10231)
   Quote: "we are actively evaluating alternative vendors for this capability"
   Why it matters: explicit competitor evaluation ahead of a renewal

## 3. Recommended talking points

1. ...
```

### Determinism

The brief is deterministic for a given account id, as the task requires. Four
mechanisms combine: temperature is fixed at zero, every request is keyed into a content
addressed cache and replayed rather than resampled, ticket history is sorted by id
before it enters the prompt, and statistics and signals are serialised in a fixed key
order. Evaluation case B4 asserts this by building the same brief twice and comparing
the rendered output.

### Quote verification

Each churn signal carries a quote, and each quote is matched character by character
against the body of the ticket it cites, after whitespace normalisation. A quote that
cannot be found is dropped and the rejection is written into `data_gaps`. Nothing is
repaired, because repairing would hide the behaviour that matters.

Extraction and synthesis are separate model calls. The summary can only draw on
evidence that survived verification, so rewording the summary prompt can never change
which tickets were flagged.

## Task 3: evaluation harness

### Running it

```
python main.py eval
python main.py eval --task triage
python main.py eval --no-judge
```

### Coverage

Thirteen cases: seven for triage, six for the account brief. Two of the triage cases
and one of the brief cases are adversarial. Every case is built from records in the
supplied dataset, and the adversarial cases are mutations of real records rather than
invented tickets, so nothing introduces data from outside the starter repo.

Each case is scored by weighted deterministic criteria plus an LLM judge where a rubric
applies, combined as 0.7 times the rule score plus 0.3 times the judge score. Rules
dominate deliberately: a judge is useful for nuance but is itself a model, and a harness
that leans on one inherits its variance. A case fails if any criterion marked critical
fails, or if the combined score falls below 0.70. The command exits nonzero on any
failure so continuous integration can gate on it.

### Adversarial cases

1. `T6-adversarial-tone` sends urgent sounding language over a cosmetic issue with a
   workaround. Urgency must follow business impact rather than customer tone, so
   escalation to P1 is a critical failure.
2. `T7-adversarial-sparse` sends a real ticket truncated to five words. The pipeline
   must flag it for human review rather than invent specifics it cannot know.
3. `B6-adversarial-no-tickets` withholds the ticket history for an account. The brief
   must report the gap and must produce no churn signals at all.

Results are committed as `eval_report.md` and `eval_report.json`.

## Design note

### Failure modes

__Retrieval misses and the model answers from memory.__ BM25 is lexical, so a ticket
describing a problem in words the documentation never uses retrieves nothing relevant.
A model asked to help anyway will supply plausible error code meanings and thresholds
from its own priors, and a confident, well formatted, wrong answer sent to a customer
is the worst failure available here. It is detected by tracking the share of triages
that return no knowledge base reference; a rise means the corpus has drifted from how
customers write. It is mitigated by forbidding the model to state any version,
threshold or error code meaning absent from the supplied sections, by intersecting
citations with what retrieval returned, and by forcing human review when no section
supports the call. Evaluation case T2 gates this.

__Fabricated quotes in account briefs.__ A paraphrase presented as a quotation means a
TAM repeating words to a customer that the customer never wrote. It is detected because
every rejection is written into `data_gaps`, making the rejection rate visible per run.
It is mitigated by verbatim verification, by dropping rather than repairing failures,
and by separating extraction from synthesis. Evaluation case B2 makes both checks
critical.

__Vocabulary drift breaking downstream routing.__ A model returning an unrecognised
urgency or product silently corrupts whatever consumes the output. It is detected
through `review_reasons`, where drift appears as a rising repair rate rather than as
mysterious routing bugs. It is mitigated by matching every value against the dataset
vocabularies, falling back to a deterministic rule, capping confidence at 0.3, and
flagging for review.

### Latency versus quality

Triage runs as one model call rather than a chain of classify, retrieve again, then
draft. The chain would be better: it would retrieve against the model's own
classification instead of a heuristic product guess, which matters most for tickets
that never name their product. It would also cost three round trips instead of one,
roughly 3 to 6 seconds against 1 to 2 at this prompt size.

I chose the single call because triage sits in front of a support agent who is waiting,
and because the deterministic pass recovers most of the benefit: exact product name and
error code matching resolves most tickets in this corpus before the model sees them.

If latency were a hard constraint, three changes in order of payoff. Return the
classification and routing immediately and stream the draft separately, so perceived
latency becomes the classification alone. Move classification to a smaller model and
keep the larger one for the draft. Warm the retrieval index at process start, which is
already done. The LLM judge is already off the online path entirely; it runs only in the
evaluation harness.

### Data sensitivity

The supplied dataset is synthetic, but the design assumes it will not always be.

Only the retrieved knowledge base sections, the ticket text and the single account
record reach the API. The full ticket file never leaves the machine. The cache stores
request shape only, meaning model, temperature and character counts, never prompt
bodies, and debug logs carry token counts and stop reasons rather than content.

Stating the exposure honestly: ticket bodies and account records do reach the API, and
cached responses sit unencrypted on disk. In production I would add a redaction pass
before egress, mapping names, email addresses and identifiers to placeholders with a
regex and NER pass and rehydrating locally after the response returns; move the cache to
encrypted storage with a retention limit; and run under a zero retention agreement. That
redaction pass is the highest value addition and is not implemented here.

### Scaling to ten times the volume

Five thousand tickets is not a modelling problem, since triage is per ticket and
embarrassingly parallel. Cost and rate limits break first: five thousand calls at
roughly two cents each is about ninety dollars a day, and the client is synchronous, so
throughput is one ticket at a time. The fix is the batch API for anything not user
facing, at half the cost, plus a bounded concurrency pool for what is.

The cache breaks second. It writes one JSON file per call into a flat directory, and
tens of thousands of entries degrade directory operations. It should become SQLite or
Redis, keyed identically.

Retrieval breaks third, and only if the knowledge base grows with the product surface.
At 46 chunks a full lexical scan per query is free; at ten thousand it is not, and BM25
over an in memory list should become a real index, with a vector store alongside it for
the semantic recall that lexical matching misses.

What does not break is the per record contract. Schemas, validation, verification and
evaluations are all per record and scale linearly.

## What the dataset actually contains

Three properties measured in the supplied data shaped the implementation.

1. __The `account_id` field on tickets does not reference `accounts.json`.__ There are
   484 distinct ticket account ids across 500 tickets, between 4 and 17 per company, and
   only 4 resolve to a real account. Company name matches for all 500. The brief
   pipeline joins on `account_id` first as documented, falls back to company, and
   discloses the fallback in `data_gaps` rather than hiding it.
2. __The `category` and `urgency` fields are statistically independent of ticket text.__
   The 34 tickets that open "we've outgrown our current plan" carry eight different
   categories between them. They are therefore not usable as ground truth labels, which
   suits the requirement to triage without human labelling. `compare_to_recorded()`
   reports agreement rather than accuracy, because neither side is truth.
3. __Every ticket predates a 90 day window measured from today.__ The newest ticket is
   from 2026-05-22, so a wall clock window returns zero tickets for every account. The
   clock is anchored to the newest ticket in the dataset by default, which keeps the 90
   day semantics meaningful and stops the result changing as real time passes. The
   `--days` and `--all-history` flags override it.

## Cost control

`LLM_MAX_CALLS`, default 60, caps live API calls per process and raises before any
network call is made. Cache hits are free and are not counted, so rerunning the
evaluations, the demo or continuous integration costs nothing. Triaging the entire
ticket file would be roughly 500 calls, and the cap exists so an accidental loop cannot
drain an account balance.

## Project layout

```
main.py                 single entry point: triage, brief, accounts, eval
src/
  data_loader.py        dataset loading, ticket to account join, KB chunking
  retrieval.py          BM25 with exact error code matching
  triage.py             Task 1 pipeline
  account_brief.py      Task 2 pipeline
  schemas.py            pydantic output contracts
  llm_client.py         Anthropic wrapper with cache, budget and error handling
  cache.py              content addressed response cache
prompts/registry.py     versioned prompts with changelogs
evals/                  test cases, scoring, runner
fixtures/llm_cache/     committed model responses, so everything runs offline
```
