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

Everything is reachable through one entry point, `main.py`. There is also a Streamlit
interface, `app.py`, for people who do not work in a terminal.

## How I approached this

I started by profiling the dataset rather than writing pipeline code, because I wanted
to know what was actually in it before deciding what to build. That turned out to be the
most useful hour I spent. Three of the properties I found contradicted what I had
assumed from reading the schema document, and each one changed a design decision. Those
findings are written up in a section further down, and they are the reason several parts
of this project look the way they do.

The second thing I decided early was that the system should never present something it
cannot support. A support tool that invents an error code meaning, or a brief that
attributes words to a customer who never wrote them, is worse than no tool at all,
because a human will act on it. Most of the engineering here is about making that
impossible rather than unlikely: citations are filtered against what was actually
retrieved, quotes are checked character by character and dropped when they fail, and
anything the model returns that is outside the known vocabulary is repaired by rule and
flagged for review.

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

## REST interface

The brief allows triage to be exposed as a callable Python function or as an endpoint.
Both exist. `triage_ticket` is importable and is what the CLI, the web interface and the
evaluation harness all call directly. For HTTP callers there is a small FastAPI service.

```
python main.py serve
```

Then open `http://127.0.0.1:8000/docs` for the generated documentation.

1. `GET /health` reports what the process has loaded, which is 500 tickets, 50 accounts
   and 46 knowledge base sections.
2. `POST /triage` takes a subject and body and returns the full triage result.
3. `GET /accounts` lists every account with the fields needed to choose one.
4. `POST /brief` takes an account id and an optional window and returns the brief.

The knowledge base index is built once at startup and shared across requests rather than
rebuilt per call. Model errors return 503 rather than 500, because a missing key, an
exhausted call budget or an uncached request in offline mode are all configuration
problems the caller can act on rather than server faults. An unknown account returns 404.

## Web interface

Both tools are also available through a Streamlit interface aimed at people who do not
use a terminal.

```
streamlit run app.py
```

The triage tab takes a ticket, either picked from the dataset or pasted in, and shows
the classification with its confidence and reasoning, the routing decision, the
knowledge base sections that support it in expandable panels, and an editable draft
reply. Anything flagged for human review is called out at the top before the agent
reads anything else.

The account brief tab takes an account, shows the headline numbers, then builds the
brief. Risks are colour coded by severity, churn signals are shown as the customer's
own quoted words next to the ticket id they came from, and the whole brief can be
downloaded as markdown. Data gaps are collapsed into a panel rather than hidden, so a
TAM can see what the brief could not establish.

Because responses are cached, the interface works with no API key for anything that has
been generated before. A request that has never been made shows a plain explanation
rather than an error trace.

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

## Where I ran into difficulty

### Deciding what to do about a foreign key that does not work

Finding that `account_id` does not join was quick. Deciding what to do about it took much
longer, because both options are defensible and they lead to different products.

Joining strictly on `account_id`, as the schema document says, is the honest reading of
the specification. It also means 496 of 500 tickets have no account context, so almost
every brief would be built from nothing. Falling back to company name recovers all 500
and produces a genuinely useful brief, but it silently invents a relationship the data
does not actually assert, and if the real system ever had two accounts for one company
it would merge them.

What I eventually settled on was refusing to make the choice invisible. The pipeline
tries `account_id` first, falls back to company, and writes a sentence into `data_gaps`
saying exactly which path it took and why. The fallback is also a parameter, defaulting
to off in the loader, so a caller who wants the strict behaviour gets it. The reasoning I
applied was that the danger is not picking the wrong join, it is a TAM reading a brief
without knowing which join produced it.

I hit the same question again with the 90 day window and answered it the same way. Every
ticket predates a 90 day window measured from today, so the literal reading returns zero
rows for every account. I anchored the clock to the newest ticket in the data, which
keeps the requirement meaningful, and said so in the output rather than quietly
returning an empty brief.

### Getting determinism that actually holds

Task 2 asks for deterministic output, and my first assumption was that setting
temperature to zero would be enough. It is not. Temperature zero narrows variation but
does not eliminate it, and even if the model were perfectly stable, my own code was not:
I was passing ticket history in whatever order it came out of the file, and serialising
statistics from a dictionary whose key order could shift. Two runs could differ without
the model doing anything different at all.

Fixing it needed four separate changes rather than one setting. Ticket history is sorted
by id before it enters the prompt. Statistics are counted into a fixed key order. Churn
signals are sorted by ticket id and quote. And every request is hashed and cached, so an
identical input replays a stored response instead of being resampled.

That last one had a consequence I did not anticipate but came to like. Because the cache
makes runs reproducible, it also makes the whole project runnable with no API key at
all, which is what lets continuous integration execute the full evaluation suite on
every push without a secret. A decision I made for correctness solved an infrastructure
problem I had not started thinking about yet.

The part I found genuinely uncomfortable was accepting that I could not measure accuracy.
Because `category` and `urgency` in the dataset are unrelated to ticket text, there is no
ground truth to score against, so an evaluation harness built on comparing predictions to
recorded labels would produce numbers that look rigorous and mean nothing. I ended up
writing acceptance criteria per case instead, checking properties I can actually defend,
and reporting agreement with the recorded fields separately as a measurement of the data
rather than of the model.

## One thing I would add next

If I were continuing this, I would capture what the support agent actually sends after
editing the draft reply, and store it alongside the draft the system produced.

The reason is that this project's hardest constraint was the absence of trustworthy
labels. Every quality judgement I made had to come from hand written criteria, because
the labels in the dataset carry no signal. But a support desk generates the labels it
needs as a side effect of doing its job. If an agent changes the urgency before
responding, that is a correction. If they rewrite the whole draft, that is a strong
negative signal about the response. If they send it unedited, that is an endorsement.

Diffing the sent reply against the generated draft would give a continuously growing set
of real examples at no extra cost to anyone, and it would turn the evaluation harness
from a fixed set of thirteen cases that I wrote into a regression suite that grows with
actual usage. It would also make the confidence scores meaningful, because they could
finally be calibrated against whether a human accepted the output rather than against
the model's own estimate of itself.

I would build it as an append only log of the triage id, the generated output, the sent
output and the edit distance between them, and promote any case where an agent
overrode a high confidence prediction straight into the eval suite for review.

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
