# Repository Instructions

This repository is a Chinese-first paper library for LLM systems research. Keep professional terms in English when that makes search and comparison with papers easier, for example `prefill`, `decode`, `KV cache`, `GEMM`, `epilogue`, `sparse attention`, `kernel`, and `throughput`.

## Scope

- Maintain the home page as a categorized reading map with priority, reading links, and note links.
- Keep paper notes under `papers/`.
- Keep system-layer category indexes under `categories/`.
- Keep structured metadata in `data/papers.yml` when adding or changing paper entries.

## Paper Note Standard

Each paper note should include:

- Metadata
- 一句话结论
- Related Works
- 问题与背景
- 方法与系统设计
- 创新点
- 实验与结论
- 局限性与风险
- 对 LLM Systems 的启发
- 复现/阅读建议

## Classification

Use system-layer categories. Prefer categories that describe systems bottlenecks or implementation layers, not only model families.

Current categories:

- `Kernel & Operator Systems`
- `Long-Context Attention & KV Cache`

## Priority Convention

- `P0`: near-term must-read or reproduction target.
- `P1`: important direction worth detailed reading.
- `P2`: background or tracking item.

## Editing Style

- Chinese prose, concise but technically precise.
- Keep names, acronyms, benchmark names, and core systems terms in English.
- Prefer tables for homepage indexes.
- Do not invent paper results. If a metric is unclear, mark it as `待核对`.
