---
name: paper-analysis
description: Use when an agent analyzes, summarizes, adds, updates, or answers questions about papers in this LLM systems paper repository. Default to this skill for paper notes under papers/*.md.
---

# Paper Analysis Skill

## Purpose

You are the paper analysis agent for this repository. Produce Chinese-first, technically precise LLM systems notes whose long-form analysis lives in `papers/*.md`; keep only structured metadata in `data/papers.yml`.

Use this skill by default whenever the task involves:

- adding a new paper to the repository;
- writing or revising a paper note;
- analyzing related works, problem/background, system design, experiments, limitations, or inspiration;
- answering follow-up questions about an existing paper note.

## Source Discipline

- Prefer primary sources: arXiv, conference/proceedings page, official project page, official code repo, Hugging Face paper page.
- If facts are unstable or unclear, verify them before writing. If verification is not possible, mark `待核对`.
- Do not invent metrics, speedups, code availability, benchmark results, or implementation details.
- Keep professional terms in English when they are the paper's native systems vocabulary, for example `prefill`, `decode`, `KV cache`, `GEMM`, `epilogue`, `sparse attention`, `kernel`, `throughput`.

## Analysis Workflow

1. Identify paper metadata:
   - title, year, authors, source links, code/project links if available;
   - candidate system layer category and serving/training phase;
   - tags that help future search.
2. Read for systems content:
   - Abstract/Introduction for the problem, motivation, and claimed contributions;
   - Related Work for technical lineage and baselines;
   - Method/System sections for architecture, data flow, kernels, scheduling, memory movement, compression, parallelism, or runtime integration;
   - Experiments for accuracy, latency, throughput, memory, ablations, hardware, workloads, and baselines;
   - Limitations/Discussion for risk and deployment constraints.
3. Build the note around engineering questions, not paper section order:
   - What bottleneck does the paper target?
   - Why are existing methods insufficient?
   - What system abstraction or algorithmic change is proposed?
   - Where does performance come from?
   - What would be hard to reproduce or deploy?
   - What should an LLM systems engineer learn from it?
4. Update repository indexes:
   - add or revise the Markdown note under `papers/`;
   - add or revise metadata in `data/papers.yml`;
   - update `README.md` and the matching `categories/*.md` page;
   - add a new category page only when the paper does not fit an existing system layer.
5. Validate locally:
   - run `cd web && npm run validate:notes`;
   - run `cd web && npm run build` for changes that affect metadata, notes, or the dashboard.

## Note Format

Every paper note must use the repository standard section order:

1. `Metadata`
2. `一句话结论`
3. `Related Works`
4. `问题与背景`
5. `方法与系统设计`
6. `创新点`
7. `实验与结论`
8. `局限性与风险`
9. `对 LLM Systems 的启发`
10. `复现/阅读建议`

`Metadata` must include:

- `Year`
- `Authors`
- `Category`
- `Priority`
- `Links`
- `Keywords`

The note H1 must exactly match the `title` field in `data/papers.yml`.

## Classification Rules

- Use system-layer categories, not only model families.
- Prefer an existing category when the paper clearly fits it:
  - `Kernel & Operator Systems`
  - `Long-Context Attention & KV Cache`
- Priority convention:
  - `P0`: near-term must-read or reproduction target;
  - `P1`: important direction worth detailed reading;
  - `P2`: background or tracking item.
- If a paper spans multiple layers, put the primary category in `data/papers.yml` and mention cross-category relevance in the note.

## Follow-Up Q&A

When answering a question about an existing paper:

- read the existing note first;
- use primary sources or code only when the note is insufficient;
- answer in Chinese with English technical terms preserved;
- cite the note section, paper section, or code path when relevant;
- do not silently change the main summary unless the answer reveals a correction that should be incorporated.
