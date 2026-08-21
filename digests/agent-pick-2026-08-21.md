# Agent Pick -- 2026-08-21

## [StagedWorkspace: A Versioned Workspace for Knowledge-Work Agents](https://arxiv.org/abs/2608.18050v1)
**arXiv:** 2608.18050v1

This paper introduces **StagedWorkspace**, a framework for knowledge-work AI agents that explicitly syncs parsed search records, native file edits, and review diffs to versioned file states across mixed formats like PDFs, spreadsheets, and slides. Experiments on benchmarks like OfficeQA Pro and APEX-Agents show that maintaining this synchronized state and diff visibility significantly outperforms traditional single-view approaches across multiple frontier models. This is a valuable read for researchers designing tool-use frameworks, agent workspaces, or benchmarks for complex office and multimodal knowledge tasks.

**Why this was picked:** Proposes StagedWorkspace, a concrete workspace architecture for knowledge-work agents that explicitly ties parsed records, file edits, and diff reviews to content-hashed state versions. Highly relevant and actionable for agent harness design.

`arxiv-curator feedback 2608.18050v1 --rating up`

## [Break It Down, Pass It On: Cross-Task Skill Transfer in LLM Agents](https://arxiv.org/abs/2608.20274v1)
**arXiv:** 2608.20274v1
**Length:** 34 pages

This paper investigates cross-task skill transfer in LLM agents, finding that subtask-level and text-based skills transfer far more effectively than task-level or code-based alternatives, which often hurt performance. To evaluate learned skills, the authors introduce an execution-free "skill utility score" based on specificity and abstractness that accurately predicts whether stored skills will improve agent success on new tasks.

**Why this was picked:** Provides actionable insights on agent skill memory design, showing why subtask-level text skills outperform code/task-level skills and introducing an execution-free skill utility score to evaluate skill stores.

`arxiv-curator feedback 2608.20274v1 --rating up`

## [Pandora's AI Model Routing Box: Efficient Allocation with Costly Value Estimation](https://arxiv.org/abs/2608.20316v1)
**arXiv:** 2608.20316v1

This paper tackles the trade-off in AI query routing where accurately predicting which specialist model will perform best is itself computationally expensive. By formalizing this as a classical "Pandora's Box" search problem with costly inspection, the authors derive closed-form policies for both centralized routing and decentralized bidding that decide when refining a performance estimate is worth the cost. Experiments across multi-LLM, RAG, and variable-reasoning benchmarks demonstrate that this approach matches the accuracy of exhaustive estimation while significantly reducing expensive evaluation calls.

**Why this was picked:** Applies the classical Pandora's Box search framework to model routing with costly evaluation, offering practical value-of-information policies for LLM orchestration and multi-model routing pipelines.

`arxiv-curator feedback 2608.20316v1 --rating up`
