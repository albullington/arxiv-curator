# Agent Pick -- 2026-07-17

## [Bridge Evidence: Static Retrieval Utility Does Not Predict Causal Utility in Multi-Step Agentic Search](https://arxiv.org/abs/2607.15253v1)
**arXiv:** 2607.15253v1

This paper reveals that traditional, static measures of document relevance are virtually uncorrelated with a document's actual causal usefulness in multi-step, agentic search. Through counterfactual testing on a language model agent, the authors identify "bridge documents"—texts that appear useless to a static evaluator but are actually crucial because they contain key entities that guide the agent’s subsequent queries. These findings demonstrate that optimizing retrieval systems for static relevance fails to improve multi-step agentic performance, highlighting a critical gap in how we evaluate search tools for AI agents.

**Why this was picked:** This paper provides a crucial evaluation methodology insight for anyone building agentic search or RAG tools: static retrieval utility (the traditional way RAG is evaluated) does not predict causal utility in multi-step agentic trajectories. It exposes the concept of 'bridge documents'—information that is causally load-bearing by providing key search terms/entities for the next step, rather than directly answering the final prompt. This is a fundamental system-level insight for designing and evaluating retrievers used by agents.

`arxiv-curator feedback 2607.15253v1 --rating up`

## [Remember When It Matters: Proactive Memory Agent for Long-Horizon Agents](https://arxiv.org/abs/2607.08716v1)
**arXiv:** 2607.08716v1

To address the issue of agents losing track of key information during long-horizon tasks, this paper introduces a plug-and-play "proactive memory agent" that runs alongside any standard action agent. Instead of relying on passive retrieval, this auxiliary agent maintains a structured memory and actively decides when to inject timely reminders into the action agent's prompt or remain silent. Evaluated on long-context benchmarks like Terminal-Bench 2.0 and $\tau^2$-Bench, this selective intervention method significantly improves task success rates over passive or always-on memory baselines.

**Why this was picked:** This paper introduces a proactive memory agent that runs alongside an action agent and selectively decides when to inject reminders instead of relying on passive retrieval. This is a highly practical and concretely triable memory architecture that addresses 'behavioral state decay' in long-context, long-horizon tasks, showing robust improvements across agent benchmarks.

`arxiv-curator feedback 2607.08716v1 --rating up`

## [Do AI Agents Know When a Task Is Simple? Toward Complexity-Aware Reasoning and Execution](https://arxiv.org/abs/2607.13034v1)
**arXiv:** 2607.13034v1
**Length:** 27 pages

Current LLM agents often over-analyze simple tasks, wasting significant compute and context by treating minor code edits like full-system audits. To address this, the authors introduce **E3 (Estimate, Execute, Expand)**, a framework that estimates the minimum sufficient scope for a task and only scales up its context search if verification fails. On both synthetic and real-world programming benchmarks, E3 matches state-of-the-art success rates while slashing token consumption by up to 91% and financial costs by 85%.

**Why this was picked:** This paper tackles execution-scope estimation with the E3 (Estimate, Execute, Expand) framework, allowing agents to avoid over-reading and over-analyzing simple tasks. It is extremely practical, demonstrating dramatic reductions in token usage (91%) and financial costs (85%) while maintaining task success. It is highly triable and offers a concrete design pattern for building cost-efficient, production-grade agents.

`arxiv-curator feedback 2607.13034v1 --rating up`
