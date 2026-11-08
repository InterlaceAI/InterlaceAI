<div align="center">

<img src="assets/interlace-logo.svg" width="88" alt="Interlace AI">

# Interlace AI

**Getting more out of the models people can actually run.**

</div>

---

## Who we are

Interlace AI is an independent research project, based in Spain, working on one
question:

> How much more can a small language model do, if you stop asking it once?

Training frontier models, and increasingly even serving them, is out of reach
for almost everybody. We work on the other side of that problem: getting more
out of models that fit on one consumer graphics card, and publishing the code
and the raw data behind our main result, so it can be checked rather than
believed.

| | |
|---|---|
| **Open by default** | Apache 2.0, free for research and commercial use |
| **Checkable** | Our main result ships with its 25,600 raw trajectories and the script that re-derives it |
| **Accessible** | Techniques that work on one GPU, not on a cluster |

---

## Best-of-N 1.1

> **The 1.0 line is withdrawn.** Its AIME and trained-verifier numbers came from a
> version with defects in answer extraction and are not reproducible. The full
> account is in [best-of-n/README.md](best-of-n/README.md#what-happened-to-10).

Our first release, and the idea behind it is simple enough to state in a
sentence:

> A language model does not give you an answer. It gives you a distribution
> over answers — and one sample is a poor way to read it.

Sample the same **frozen** model N times, extract the final answer of each
sample and return the most frequent one. Every result on this page is majority
voting; no reward model is involved. No training, no fine-tuning, no new
weights. Often the right answer is already among the samples: in the main
experiment below, at least one of the 128 is right on 93.5% of
the problems.

```bash
pip install "bestofn[math]" torch transformers accelerate
```

```python
from bestofn import BestOfN

engine = BestOfN("your/model", n=16)
r = engine.solve(problem)

r.answer             # what the system returns
r.is_correct(gold)   # was it right?
r.covered(gold)      # was it reachable at all?
```

`"your/model"` is a Hugging Face model name or a local path, run with
`transformers` or, when it is installed, vLLM. With any other engine, generate
the N answers yourself and let `bestofn.select` choose among them.

<div align="center">

| GSM8K, 200 problems, temperature 0.7 | one sample | random pick of 128 | **majority vote of 128** |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct, frozen | 45.3% | 46.3% | **66.5%** |

**+21.2 points, and the weights were never touched.**

</div>

Of those 21.2 points, 20.2 are selection, the vote's lead over a
random pick, and 1.0 comes only from avoiding samples without a final
answer. The 95% confidence interval
of the vote is 60.5–73.0%. With 16 samples the vote already
reaches 61.8%, 78% of the gain at 128.

**Across eight open models**, from 0.5B to 27B
parameters and from three families, with 64 samples per
problem on the same 200 problems, every model improves: from
+2.4 to +37.3 points, and the lower a model starts, the
more it gains. The largest gain is SmolLM2-1.7B-Instruct's, from 21.7%
to 59.0%: 22.4 points of selection and
14.9 from avoiding samples without a final answer. The token cap
was 1,024 per sample (4,096 for the 27B). Full table in
[best-of-n/README.md](best-of-n/README.md#does-it-hold-on-other-models); the
per-model summary is in
[`best-of-n/results/models/summary.json`](best-of-n/results/models/summary.json).

**→ [`best-of-n/`](best-of-n/) — code, measurements, the main experiment's raw trajectories and full documentation**

Also on **[PyPI](https://pypi.org/project/bestofn/)** ·
**[Hugging Face](https://huggingface.co/InterlaceAI/best-of-n)** ·
**[results page](https://huggingface.co/spaces/InterlaceAI/best-of-n-demo)** ·
**[technical note on Zenodo](https://doi.org/10.5281/zenodo.22517054)**

[![PyPI](https://img.shields.io/pypi/v/bestofn?color=blue)](https://pypi.org/project/bestofn/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21936832.svg)](https://doi.org/10.5281/zenodo.21936832)
[![License](https://img.shields.io/badge/code-Apache%202.0-green)](LICENSE)

---

## What makes it different

Plenty of projects sample a model N times, and the voting itself is not ours:
it is self-consistency ([Wang et al., 2022](https://arxiv.org/abs/2203.11171)).
Splitting a result into coverage and selection is not new either; it appears in
[Snell et al. (2024)](https://arxiv.org/abs/2408.03314) and in evaluation
harnesses that report `pass@k` beside `maj@n`. What we add is the measurement
discipline around it, built into the library and, in our view, still uncommon:

**Every selector is compared against random.** A selector that cannot beat
picking a trajectory at random is not selecting anything. Our tables carry that
row, and so should everyone's.

**We report how much of the gain is not the method.** A sample with no
recognisable final answer, for instance one cut off by the token cap, abstains
instead of voting. With more samples, one of them usually does not abstain, and
that alone lifts a random pick. Here it is 1.0 of 21.2 points; we
print it beside the headline instead of banking it. On the 27B model, a
1,024-token cap first showed +7.0 points, only
0.2 of them selection, with 11.2% of samples cut off; at
4,096 tokens the gain is +2.4 (1.1 of it selection,
3.7% cut off), and that is the number we publish
([CHANGELOG](best-of-n/CHANGELOG.md)).

**It tells you which problem you have.** `r.is_correct(gold)` says whether the
system returned the right answer; `r.covered(gold)` says whether any sample
found it. Kept apart, they tell you whether to invest in more sampling or in
better selection.

**The same pool always gives the same answer.** Shuffle the samples, get the
same answer. Symbolic equivalence is not transitive, so this takes a transitive
closure over the answer classes and a tie-break that does not depend on arrival
order — and you can see it in our published curve, where the spread at
N = 128, with the whole pool in play, is exactly zero.

**The raw trajectories of the main experiment are published.** Not a summary —
the complete reasoning text of all 25,600 samples, with token counts,
finish reasons and log-probabilities. The analysis script re-extracts every
answer from that text, so it can catch a parsing bug rather than inherit one,
and it re-derives the main experiment's figures in under four minutes on an
8-thread CPU, with no GPU.

---

## What we are working on

**Nem 1** — an open 4B reasoning model with a new control mechanism of our
own, post-trained from an open base model. In development, not yet released. It will be published with a technical report and a DOI on Zenodo,
and measured with the same discipline.

**Best-of-N v2** (planned) — extending inference-time compute to agentic tasks
and code generation, where a verifier can *run* the candidate instead of
estimating whether it works.

---

## Repository layout

```
best-of-n/
  bestofn/          The library: engine, extractors, selectors, verifier adapters
  scripts/          Run the benchmark; re-derive the published numbers from raw trajectories
  results/          Raw trajectories of the main experiment, and the measurement summaries
  tests/            244 checks in three test scripts, no GPU required
  figures/          Charts and the code that produces them
  TR-2026-02.md     The technical note (also on Zenodo, with its DOI)
  CHANGELOG.md      What changed and why

assets/             Brand assets
LICENSE             Apache 2.0
```

---

## Licensing

**Apache License 2.0** — free to use, modify and redistribute, including
commercially.

Evaluation data: GSM8K is distributed by OpenAI under the MIT licence, so it is
freely redistributable. Our published trajectories contain the model's own
generated text alongside the problem statements.

---

## Contact

**Interlace AI** · `interlaceIA@gmail.com`

Questions, reviews and bug reports are welcome as
[issues](https://github.com/InterlaceAI/InterlaceAI/issues). Open to
collaboration.
