# Agent Pick -- 2026-08-07

## [PAST-Bench: Benchmarking the Foundations of Recursive Self-Improvement in Personal Agents](https://arxiv.org/abs/2608.04003v1)
**arXiv:** 2608.04003v1

**PAST-Bench** is a new benchmark featuring 26 scenarios designed to evaluate whether personal AI agents systematically improve over time by leveraging saved context across sessions. Evaluating multiple models and frameworks, the authors find that while retained experience does yield performance gains, these improvements are highly uneven and frequently fail to use the intended save, retrieve, and update mechanisms. To address these gaps, the authors introduce **Hermes+**, a modified agent framework that significantly enhances experience-driven performance, particularly on tasks requiring agents to overwrite outdated information.

**Why this was picked:** Evaluates whether personal AI agents systematically improve across sessions by retrieving and updating saved context, and introduces Hermes+ with five concrete agent-loop interventions for state management and memory updates. Directly applicable to persistent agent memory and harness design.

`arxiv-curator feedback 2608.04003v1 --rating up`

## [TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories](https://arxiv.org/abs/2608.06346v1)
**arXiv:** 2608.06346v1

To address the challenge of pinpointing root-cause errors in long LLM agent trajectories, the authors introduce **TrajDebug**, a framework that traces error lifecycles using history compression and evidence-based analysis to locate and attribute critical failures. They also present **TrajErrBench**, a benchmark of 486 manually annotated failed trajectories across realistic coding and tool-use scenarios. Evaluated across diverse benchmarks, TrajDebug outperforms existing baselines and generates actionable diagnoses that improve downstream agent success rates.

**Why this was picked:** Proposes TrajDebug, an error-lifecycle tracing framework that uses history compression and evidence-based attribution to identify critical failure steps in long-horizon agent execution. Provides actionable diagnostic feedback and a dataset (TrajErrBench) for agent debugging.

`arxiv-curator feedback 2608.06346v1 --rating up`

## [The Bitter Lesson of Tool Calling](https://arxiv.org/abs/2608.06370v1)
**arXiv:** 2608.06370v1

Tool use transforms LLMs into agents that act beyond their training data, and for code-capable models, programmatic tool calling extends this further by replacing rigid JSON calls with scripts that chain and parallelize naturally. However, a systematic evaluation of tools as code on an established benchmark across current and prior model generations under real-world task conditions has not been conducted. In this work, we empirically compare programmatic tool calling (PTC) to native JSON tool calling across 14 language models on BFCL v4. In the programmatic tool calling paradigm, tools are exposed as typed Python stubs that the model invokes through code, with execution and results handled in a single agent turn. Programmatic tool calling matches or exceeds native JSON tool calling in 11 of 14 models on BFCL v4, with the GPT-5.6 family achieving a 10.6% improvement over the JSON tool calling baseline. Further, it matches or outperforms baseline in 13 of 14 models under parallel fan-out, and holds stable under context rot conditions where baseline degrades 2.3% on average. Our results demonstrate that programmatic tool calling is a viable and robust alternative to JSON tool calling, with performance tracking model capability across release generations.

**Why this was picked:** Presents a systematic empirical comparison showing that programmatic tool calling (exposing tools as Python stubs executed in code) outperforms traditional JSON tool calling across 14 LLMs while offering superior robustness to parallel fan-out and context rot. Concretely triable tool-use architecture pattern.

`arxiv-curator feedback 2608.06370v1 --rating up`
