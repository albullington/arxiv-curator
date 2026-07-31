# Agent Pick -- 2026-07-31

## [The Regression Tax: Decomposing Why Skills Help and Hurt LLM Agents](https://arxiv.org/abs/2607.22520v1)
**arXiv:** 2607.22520v1

This paper reveals that adding procedural skills to LLM agents often causes "regressions"—failing previously solvable tasks—because skills can distort an agent's context, input interpretation (grounding), or output checking (verification). Across nearly 6,000 benchmark runs, the authors find that top-performing skills excel primarily by minimizing these regressions rather than generating larger gains. Consequently, they recommend evaluating agent skills by explicitly separating gains from regressions and prioritizing improvements in grounding and verification over additional procedural guidance.

**Why this was picked:** Provides essential insights for agent skill and tool harness engineering. By analyzing ~6,000 runs, the authors show that adding procedural skills often causes task performance regressions due to 'skill description osmosis', grounding displacement, and verification displacement. Its proposed evaluation framework (decomposing net gains vs. regressions) and focus on grounding/verification over raw procedural guidance are directly applicable to building reliable agent toolkits.

`arxiv-curator feedback 2607.22520v1 --rating up`

## [TRACE-ROUTER: Task-Consistent and Adaptive Online Routing for Agentic AI](https://arxiv.org/abs/2607.22465v1)
**arXiv:** 2607.22465v1

Traditional LLM routers operate on a per-call basis, making them ill-suited for multi-step agentic workflows where quality feedback is only received upon overall task completion. TRACE-Router addresses this by using a contextual bandit to assign an entire task to a single model upfront, updating its routing policy based on the workflow's combined terminal accuracy and latency rewards. Across three agentic benchmarks, it significantly improves cost-performance trade-offs, outperforming the strongest single-model baseline on Terminal-Bench by 7.1 accuracy points while reducing latency by 36%.

**Why this was picked:** Addresses a key limitation in enterprise agent orchestration: per-call routers fail to attribute credit across long-horizon multi-step workflows. TRACE-Router uses contextual bandits to make task-level admission routing decisions, pinning the chosen backend and optimizing terminal accuracy-latency rewards. It offers a practical, triable pattern for agent system developers looking to optimize latency and API costs.

`arxiv-curator feedback 2607.22465v1 --rating up`

## [Can AI agents conduct open-ended AI research? Early evidence from two case studies](https://arxiv.org/abs/2607.27191v1)
**arXiv:** 2607.27191v1

To better evaluate AI agents on open-ended tasks, this paper introduces "shadow evaluations," where frontier models attempt the central research questions of unpublished NeurIPS submissions and are graded directly by the original human authors. Across two case studies, agents autonomously executed all the necessary software engineering but failed to make meaningful progress on the core research questions, leading to clear rejections by the human experts. The findings indicate that while current agents can handle the technical execution of AI research, they still lack the creative problem-solving, strategic decision-making, and high-level reasoning required for publishable science.

**Why this was picked:** Introduces 'shadow evaluations' as a novel evaluation methodology for open-ended agentic tasks by having original paper authors evaluate frontier agents on central research questions. The paper uncovers critical failure modes in long-horizon autonomous agents—such as instruction drift, poor resource awareness, and ineffective backtracking—providing actionable benchmarks for open-ended agent harness development.

`arxiv-curator feedback 2607.27191v1 --rating up`
