# Autonomy Golf Agent Integration

Use this when an agent is asked to install autonomy golf into an existing project.

Canonical source: [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf)

Before making changes:

- read [autonomy-golf.md](autonomy-golf.md) for the manifesto, public framing, and intended purpose
- read [../CHANGELOG.md](../CHANGELOG.md) for the canonical template and parser-facing shape
- read [autonomy-golf-checklist.md](autonomy-golf-checklist.md) for the maintenance loop

Related files:

- public explainer / portable manifesto: [autonomy-golf.md](autonomy-golf.md)
- maintenance loop: [autonomy-golf-checklist.md](autonomy-golf-checklist.md)
- canonical changelog template and parser input shape: [../CHANGELOG.md](../CHANGELOG.md)
- front-door overview and badge flow: [../README.md](../README.md)

## What To Install

Start from the working bundle in the canonical repository. Copy or adapt:

- `CHANGELOG.md`
- `docs/autonomy-golf.md`
- `docs/autonomy-golf-agent.md`
- `docs/autonomy-golf-checklist.md`
- `tools/changelog_scores.py`
- `tools/render_autonomy_badge.py`

The goal is not just to add a badge. The goal is to install a disciplined loop that helps the project move toward total automation in a way that stays fun to follow and meaningful to trust. Record:

- who drove each change
- how autonomous it really was
- how well it was grounded
- what the change means, why it was needed, and what it is for
- whether the project is becoming more autonomous over time

## Score Model

Use the score ladder and header conventions already defined in [../CHANGELOG.md](../CHANGELOG.md). The essential scale is:

| Tier | Score |
| --- | ---: |
| `Fully human` | `6` |
| `Human-driven` | `5` |
| `Human-directed, AI-shaped` | `4` |
| `AI-identified within brief, human-shaped` | `3` |
| `AI-identified within brief, human-approved` | `2` |
| `Self-initiated, human-approved` | `1` |
| `Fully autonomous` | `0` |

`Grounding` is separate and unscored. Lower `score` is better.
`complexity` should follow the canonical parser rule: sum the top-level provenance weights, then add `+1` for each nested sub-bullet under provenance items scored `3` or higher, excluding `Meaning:`, `Motivation:`, and `Purpose:` narrative lines.

## Hard Rules

- Lean on the existing template and checklist instead of inventing local variants.
- Bias toward under-claiming autonomy.
- Score only top-level provenance bullets.
- Keep directly derivative same-tier details nested.
- Treat meaning, motivation, and purpose as first-class changelog content, not optional narrative garnish.
- Keep `Grounding` separate from provenance.
- Do not let “agent suggested” drift into `Fully autonomous`.
- Keep the changelog readable to humans and stable for parsers at the same time.

## Changelog Contract

Do not restate the changelog shape from memory. Use [../CHANGELOG.md](../CHANGELOG.md) as the canonical template for:

- subsystem-prefixed headers
- provenance section labels
- `score` and optional `complexity`
- `Grounding`
- the lag-by-one commit-ID model under `Latest`
- the rule that a commit must not amend itself just to stamp its own hash; only a later commit may move it into committed history

## Grounding

Grounding is the evidence layer.

Record:

- files changed
- checks run
- measurements, if any

Use the strongest practical grounding the change deserves. If stronger validation is not practical, say so plainly. Meaningful test coverage is one useful grounding dimension where automated tests are the right validation surface.

## Install Loop

1. Start from [../README.md](../README.md) to understand the visible project shape.
2. Copy or adapt [../CHANGELOG.md](../CHANGELOG.md), [autonomy-golf.md](autonomy-golf.md), and [autonomy-golf-checklist.md](autonomy-golf-checklist.md).
3. Copy or adapt [../tools/changelog_scores.py](../tools/changelog_scores.py) and [../tools/render_autonomy_badge.py](../tools/render_autonomy_badge.py).
4. Add a README badge or snapshot driven by the parser output.
5. If the project already has change-management hooks, wire autonomy golf into that path instead of inventing a parallel ritual. In practice, the best place is usually the pre-commit or pre-merge flow.
6. Tell future agents to maintain the system through the checklist, not ad hoc.

## Maintenance Handoff

Once autonomy golf is installed, tell agents to read and follow [autonomy-golf-checklist.md](autonomy-golf-checklist.md).

That checklist should drive the normal loop. Do not duplicate it into a project-specific prompt unless the project genuinely needs extra rules.

## Suggested Prompt

> Integrate autonomy golf into this project. Reuse the changelog template, parser, badge renderer, and docs from https://github.com/Entrpi/autonomy-golf. Follow the canonical `CHANGELOG.md` shape, use the checklist for maintenance rules, keep grounding separate from provenance, and bias toward under-claiming autonomy.

## Success Condition

The integration is successful when the project can answer, with evidence:

- how autonomous recent work actually was
- which subsystems still need human direction
- whether the accounting is strict enough to trust
- whether the project is moving toward fuller autonomy over time
