# Agent Pick -- 2026-08-28

## [Prime Agent: A Self-Improving RLM Harness](https://arxiv.org/abs/2608.23552v1)
**arXiv:** 2608.23552v1
**Length:** 16 pages

**Prime Agent** is an open-source harness designed for long-horizon coding and autonomous agent workflows that leverages a persistent IPython environment, cross-trajectory memory, and recursive subagents to maximize test-time compute and context management. By standardizing execution, recovery, and verification, it enables models to reliably construct their own problem-solving strategies without being bottlenecked by framework failures. Across diverse evaluations, the system substantially boosts model performance—raising ARC-AGI-3 RHAE Best@1 from 30% to 95.5% while matching or outperforming existing harnesses on complex tasks like GPU-kernel generation, emulator building, and open-ended gameplay.

**Why this was picked:** A comprehensive open-source agent harness implementing the Recursive Language Model abstraction with persistent REPL execution, cross-trajectory memory/skill preservation, recursive subagent orchestration, and standardized verification. Highly actionable and concretely triable.

`arxiv-curator feedback 2608.23552v1 --rating up`

## [StarHarness: Evolving Harnesses with Stratified Search for Enterprise Environments](https://arxiv.org/abs/2608.24804v1)
**arXiv:** 2608.24804v1

**StarHarness** is an optimization framework that evolves an LLM agent's execution environment—including prompts, tool interfaces, and subagent structures—to fix model-environment mismatches without modifying underlying model weights. Guided by a stratified search based on baseline failure patterns, it improves agent performance by 20–35 percentage points across complex IT, ITSM, and finance benchmarks. Notably, these harness improvements generalize to held-out tasks and transfer seamlessly across different model families, including GPT and Qwen.

**Why this was picked:** Proposes an automated search framework to evolve environment-specific agent harnesses (prompts, tool interfaces, MCP providers, subagent structures) with fixed model weights, demonstrating strong benchmark gains and cross-model transferability. Directly relevant to AI builder tooling and harness engineering.

`arxiv-curator feedback 2608.24804v1 --rating up`

## [Which Eviction Policy Should an LLM Cache Use? A Systematic Study Across Workloads, Capacities, and Encoders](https://arxiv.org/abs/2608.20280v1)
**arXiv:** 2608.20280v1
**Length:** 11 pages

This systematic evaluation reveals that no semantic cache eviction policy meaningfully outperforms standard Least Frequently Used (LFU)—improving on it by at most 0.041 percentage points—making LFU the strongest simple default. More critically, the authors discover a severe validity issue: typical similarity thresholds yield answer-substitutable responses only 2.1–3.9% of the time, causing raw hit rates of 51–60% to plunge to quality-adjusted rates of 1.1–2.2%. Ultimately, the paper demonstrates that ensuring answer validity and tuning non-transferable encoder thresholds are far more vital for LLM caching than fine-tuning eviction policies.

**Why this was picked:** A rigorous, practical systems study evaluating semantic cache eviction policies across workloads and embedding models. It establishes LFU as the optimal simple default and reveals critical practitioner insights regarding answer-substitutability thresholds and cross-encoder non-transferability.

`arxiv-curator feedback 2608.20280v1 --rating up`
