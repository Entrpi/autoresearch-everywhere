# Autonomy Golf

This is the portable manifesto for autonomy golf: a short explainer you can drop into any repository that wants to make the drive toward total automation fun, legible, and honest.

Canonical source: [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf)

If you are adopting the system in another project, this file should travel with the rest of the autonomy-golf bundle:

- `CHANGELOG.md`
- `docs/autonomy-golf.md`
- `docs/autonomy-golf-agent.md`
- `docs/autonomy-golf-checklist.md`
- `tools/changelog_scores.py`
- `tools/render_autonomy_badge.py`

## What It Is

Autonomy golf is a simple game for software projects:

- record who actually drove each change
- keep that accounting conservative
- publish the result as a score the project tries to drive down over time as it moves toward total automation

The word `golf` is literal. Lower is better.

- `6` means a fully human change
- `0` means a fully autonomous change

The point is not to make the number look good. The point is to make progress toward total automation visible, fun to chase, and honest enough that the project can tell whether it is actually getting there.

That only works when the project can also build explicit consensus around a change's meaning, motivation, and purpose. Autonomy golf is not just about assigning a score after the fact. It is about forcing the project to say what a change means, why it exists, and what it is for, in a form that both humans and tooling can revisit later.

## Why It Exists

Projects that talk about agents, autonomy, or self-improving workflows all face the same failure mode: ordinary human-guided work keeps accumulating while the narrative gets more autonomous than the evidence.

Autonomy golf exists to make that gap visible while giving the project a game worth playing: keep lowering the score, keep improving the grounding, and keep getting closer to total automation without lying to yourself about the distance.

It gives a project a disciplined way to ask:

- are we actually reducing human steering?
- which subsystems still need the most human direction?
- are we getting more autonomous, or just loosening the bookkeeping?

## Core Model

Each top-level provenance bullet in a changelog entry gets one score:

| Tier | Score |
| --- | ---: |
| `Fully human` | `6` |
| `Human-driven` | `5` |
| `Human-directed, AI-shaped` | `4` |
| `AI-identified within brief, human-shaped` | `3` |
| `AI-identified within brief, human-approved` | `2` |
| `Self-initiated, human-approved` | `1` |
| `Fully autonomous` | `0` |

Each commit entry then exposes:

- `score`: the mean of those top-level provenance weights
- `complexity`: the sum of those top-level provenance weights, plus `+1` for each nested sub-bullet under provenance items scored `3` or higher, excluding `Meaning:`, `Motivation:`, and `Purpose:` narrative lines

`score` is the main autonomy signal. `complexity` is a secondary scope signal.

## Golf Language

The repo should talk like a game, not just a rubric. A simple house interpretation is:

| Score | House term | Meaning |
| --- | --- | --- |
| `0` | `hole in one` | fully autonomous |
| `1` | `albatross` | very strongly self-directed with only light human approval |
| `2` | `eagle` | clearly below par; the agent surfaced and largely drove the change |
| `3` | `birdie` | better than par, but still meaningfully mixed |
| `4` | `par` | mixed agency; a reasonable default target for an early autonomy system |
| `5` | `bogey` | still substantially human-directed |
| `6` | `double bogey` | fully human work |

This is intentionally informal language. It gives teams a fun way to talk about progress toward total automation while the precise provenance tiers keep the score honest.

## Grounding Matters

Autonomy golf tracks a second dimension on purpose:

- `score` tells you how autonomous the change was
- `Grounding` tells you how well the change was validated

`Grounding` is intentionally unscored because provenance and validation are different questions.

That separation is what keeps the game useful:

- strong validation should not make a human-driven change look more autonomous
- weak validation should not disappear behind a low score
- data-driven decisions need both provenance and evidence

## What To Add To A Repo

To install autonomy golf in another project, add these files together:

- [../CHANGELOG.md](../CHANGELOG.md): the structured, agent-managed changelog template
- [autonomy-golf-agent.md](autonomy-golf-agent.md): the reusable agent integration brief
- [autonomy-golf-checklist.md](autonomy-golf-checklist.md): the concise maintenance loop for later updates
- [../tools/changelog_scores.py](../tools/changelog_scores.py): the parser and rollup tool
- [../tools/render_autonomy_badge.py](../tools/render_autonomy_badge.py): the badge and README snapshot generator

If you are reading this outside the canonical repository, get those files from [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf).

## How The Loop Works

The healthy loop is:

1. make a change
2. explain its meaning, motivation, and purpose in `CHANGELOG.md`
3. record its provenance conservatively in `CHANGELOG.md`
4. record grounding honestly
5. regenerate the score outputs
6. publish the updated badge or snapshot
7. use the result to decide where autonomy is still weak

This works best when the changelog is agent-managed and kept in a disciplined structure that is both readable to humans and stable for tooling.

## Start Here

If you want to adopt autonomy golf in another project:

1. point your agent at [autonomy-golf-agent.md](autonomy-golf-agent.md)
2. copy the changelog and tooling from the canonical repository at [Entrpi/autonomy-golf](https://github.com/Entrpi/autonomy-golf)
3. once installed, tell future agents to read and follow [autonomy-golf-checklist.md](autonomy-golf-checklist.md)

That is the simplest path to getting a real autonomy-golf badge and a maintainable history instead of a one-off doc experiment.
