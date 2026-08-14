# Agent Pick -- 2026-08-14

## [AI4AI at Test-Time: Strong-to-Weak Capability Transfer via Harnesses](https://arxiv.org/abs/2608.12307v1)
**arXiv:** 2608.12307v1
**Length:** 23 pages

This paper explores "strong-to-weak scaffolding," a method where a stronger AI model automatically designs and refines test-time code harnesses to improve a weaker model's task performance without modifying its parameters. Across four Theory-of-Mind benchmarks, these automated harnesses nearly doubled weaker target-model accuracy (from 0.49 to 0.91) by offloading unstable reasoning steps into deterministic code, routing, and strict formatting. The findings demonstrate that inference-time harness engineering is a highly effective, retraining-free complement to traditional model distillation.

**Why this was picked:** Investigates test-time capability transfer where a stronger builder model automatically designs inference harnesses (deterministic routing, strict formatting, code offloading) for weaker models, nearly doubling performance without retraining. Highly relevant to agent harness design and builder workflows.

`arxiv-curator feedback 2608.12307v1 --rating up`

## [TEPA: Revoking Stale Memories for Conflict-Robust Language Agents](https://arxiv.org/abs/2608.07429v1)
**arXiv:** 2608.07429v1
**Length:** 14 pages

This paper introduces TEPA, a revocable memory framework for language agents designed to eliminate "memory pollution," where outdated facts persist and conflict with fresh evidence. By explicitly tracking validity, TEPA invalidates stale memories when contradictory information arrives under the same key while preserving the revoked history for auditing and potential re-promotion. Experiments demonstrate that TEPA vastly outperforms standard append-only and last-write-wins memory strategies during dynamic environment and preference shifts, making it a compelling read for researchers working on long-term agent state management.

**Why this was picked:** Directly tackles a critical architectural challenge in long-term agent memory: memory pollution from superseded and conflicting observations. Introduces an explicit precedent revocation and audit mechanism that substantially outperforms standard memory schemes under regime drift.

`arxiv-curator feedback 2608.07429v1 --rating up`

## [Argus: A General-Purpose Agentic Runtime for Long-Horizon Reasoning](https://arxiv.org/abs/2608.05144v1)
**arXiv:** 2608.05144v1

**Argus** is a multi-agent runtime designed for complex, long-horizon tasks that enables LLMs to self-correct and evolve without retraining model weights by maintaining persistent project state, specialized role interactions, and strict verification gates. By accumulating verified strategies and learning from rejected paths, the framework dynamically pivots upon failure and becomes increasingly token-efficient over time. Evaluation across diverse domains shows significant performance gains, including a ~78% success rate on SWE-Bench Pro (up from 59% for a direct copilot) alongside real-world successes in GPU kernel optimization, mathematical data synthesis, and automated paper writing.

**Why this was picked:** Presents a persistent, role-based agent runtime architecture (Manager, Planner, Engineer, Reviewer) that achieves verification-gated self-evolution and state retention across long horizons with frozen model weights.

`arxiv-curator feedback 2608.05144v1 --rating up`
