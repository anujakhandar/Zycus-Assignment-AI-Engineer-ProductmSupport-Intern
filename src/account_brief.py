"""Account briefs.

Takes one account from ``data/accounts.json`` plus its recent ticket history
and produces the brief a technical account manager would want before a call:

* **Health summary** - what ``health_status``, ``usage_trend``, seat
  utilisation, NPS and days-since-last-login add up to, in plain language.
* **Ticket picture** - volume over the last 90 days (see
  ``data_loader.get_account_tickets``), the split by category and urgency,
  recurring themes across ticket subjects, and anything still unresolved.
* **Risk signals** - renewal date proximity, open P1s, escalation notes, a
  departed champion, declining usage, low satisfaction scores; ranked, with the
  evidence that supports each one.
* **Talking points and next actions** - what to raise on the next call and what
  to do before it, tied back to specific tickets and account fields.

Where a ticket cannot be resolved to an account (a known gap in this dataset,
reported by ``data_loader.JoinReport``), the brief says so rather than quietly
working from a partial history.

Planned entry points::

    build_account_brief(account, tickets, *, llm=None, days=90) -> AccountBrief
    build_portfolio_briefs(accounts, tickets, ...) -> list[AccountBrief]
"""
