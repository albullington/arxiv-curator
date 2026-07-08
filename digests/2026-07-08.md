# arXiv Digest -- 2026-07-08

## [Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade](https://arxiv.org/abs/2607.06503v1)
**Score:** 0.685  |  **arXiv:** 2607.06503v1

LLM agents often waste significant compute pursuing multi-step tasks that are doomed to fail before the failure becomes observable. This paper demonstrates that these failures are predictable much earlier from the agent's *internal representations* (hidden states) than from its external behavior. They propose a "recall-controlled probe cascade" that monitors these internal states to proactively abort failing episodes, saving 37-47% of inference compute while guaranteeing a user-specified success rate for non-doomed episodes.

**Why this matches:** This paper introduces a "Recall-Controlled Probe Cascade" for LLM Agents, presenting a novel architectural approach to manage agent episodes through "Early Abort," which aligns with your interest in new agent architectures and practical AI tooling. The "Recall-Controlled" mechanism suggests a novel approach to agent memory or control flow, implying an empirical evaluation rather than pure theory.

`arxiv-curator feedback 2607.06503v1 --rating up`

## [DynaKRAG: A Unified Framework for Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2607.06507v1)
**Score:** 0.664  |  **arXiv:** 2607.06507v1

Multi-hop RAG systems often rely on fixed, predefined steps for acquiring evidence, which limits their ability to adapt as new information or needs arise. DynaKRAG introduces a novel framework that learns to dynamically choose among various evidence operations (like retrieval or query reformulation) based on the current state of understanding, treating evidence acquisition as a state-conditioned control problem. This approach significantly outperforms strong baselines on multiple benchmarks, demonstrating the critical benefit of intelligently coordinating evidence-gathering steps with a learned policy.

**Why this matches:** DynaKRAG proposes a "Unified Framework" with "Learnable Evidence Control in Multi-Hop Retrieval-Augmented Generation," representing a genuinely new architectural approach to how AI systems manage and integrate external information. This could be particularly relevant to your interest in advanced agent memory mechanisms and agent architectures, moving beyond simple retrieval.

`arxiv-curator feedback 2607.06507v1 --rating up`

## [RSF-GLLM: Bridging the Semantic Gap in Multi-Hop Knowledge Graph QA via Recurrent Soft-Flow and Decoupled LLM Generation](https://arxiv.org/abs/2607.06527v1)
**Score:** 0.636  |  **arXiv:** 2607.06527v1

This paper introduces RSF-GLLM, a framework designed to overcome the "semantic gap" in multi-hop Knowledge Graph Question Answering where intermediate nodes may not lexically match the query. It achieves this by decoupling a differentiable graph reasoning module (Recurrent Soft-Flow) that finds paths through semantically dissimilar nodes from an LLM-based answer generation module. RSF-GLLM propagates continuous relevance scores guided by structural cues, extracts grounded reasoning paths to fine-tune an LLM, and demonstrates competitive performance with superior inference efficiency compared to other LLM-based approaches.

**Why this matches:** Based on the provided signals, this paper does not directly match your interests, as it shares no overlapping keywords with your profile and is not similar to any papers you previously liked.

`arxiv-curator feedback 2607.06527v1 --rating up`

## [Bridging Physical Reasoning and Task Generalization via Visual Action Outcome Reasoning Alignment](https://arxiv.org/abs/2607.06522v1)
**Score:** 0.634  |  **arXiv:** 2607.06522v1

Vision-language models (VLMs) often fail at interactive physical reasoning due to hallucinated thought processes and a mismatch between their reasoning and actions. We introduce VAORA, a novel reward design that uses two complementary visual alignment rewards to ground VLM reasoning in both the static visual context and the visual outcomes of its actions. This effectively suppresses hallucinated thoughts and aligns reasoning with behavior, leading to improved generalization in novel physical reasoning tasks and unseen environments.

**Why this matches:** Based on the provided signals, this paper does not appear to match your interests, as there are no overlapping keywords with your profile, and it is not similar to any previously liked papers.

`arxiv-curator feedback 2607.06522v1 --rating up`

## [FootsiesGym: A Fighting Game Benchmark for Two-Player Zero-Sum Imperfect-Information Games](https://arxiv.org/abs/2607.06514v1)
**Score:** 0.633  |  **arXiv:** 2607.06514v1

FootsiesGym is a new open-source environment for training and benchmarking AI in complex two-player, zero-sum, imperfect-information games, using a minimalist 2D fighting game to isolate strategic interactions. It offers a vectorized simulator for high-throughput, reproducible reinforcement learning research, with initial benchmarks and open research directions discussed in the paper.

**Why this matches:** FootsiesGym offers a benchmark for two-player imperfect-information games, which aligns with your interest in practical AI builder tooling for running real experiments and evaluating agents in complex environments.

`arxiv-curator feedback 2607.06514v1 --rating up`

## [FreqDepthKV: Frequency-Guided Depth Sharing for Robust KV Cache Compression in Long-Context LLM Inference](https://arxiv.org/abs/2607.06519v1)
**Score:** 0.631  |  **arXiv:** 2607.06519v1

FreqDepthKV introduces an inference-time method to robustly compress KV caches for long-context LLMs, addressing memory and bandwidth bottlenecks without accuracy loss. It works by factorizing KV states into shared low-frequency components across layers and sparse high-frequency residuals, dynamically assigning attention heads to different compression modes based on their real-time importance. This approach achieves a 3.9x effective compression ratio, significantly boosting throughput and reducing peak memory, while preserving high accuracy across diverse long-context tasks like QA, retrieval, and summarization.

**Why this matches:** The paper's "Frequency-Guided Depth Sharing for Robust KV Cache Compression" proposes a novel architectural approach to managing LLM memory, aligning with your interest in genuinely new agent memory architectures and practical AI builder tooling.

`arxiv-curator feedback 2607.06519v1 --rating up`

## [DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression](https://arxiv.org/abs/2607.06523v1)
**Score:** 0.631  |  **arXiv:** 2607.06523v1

DepthWeave-KV addresses the memory bottleneck of long-context LLM KV caches with a novel, token-adaptive compression method. It achieves this by factorizing KV states across transformer layers and dynamically allocating more reconstruction detail to critical tokens using online error tracking (without retraining). This results in near-full-cache quality, improved accuracy, and an 8.3x memory reduction over prior methods.

**Why this matches:** This paper aligns with your interest in `memory` by introducing a genuinely new architectural approach for `KV Cache Compression`, which offers practical `memory` efficiency relevant to LLM-powered agent architectures.

`arxiv-curator feedback 2607.06523v1 --rating up`

## [Rethinking Indic AI from a Lens of Cultural Heritage Preservation](https://arxiv.org/abs/2607.06544v1)
**Score:** 0.625  |  **arXiv:** 2607.06544v1

This paper examines the dual impact of AI on Indic linguistic and cultural heritage, highlighting how AI can offer inclusion but also risk homogenization given the unique complexities of Indian languages. It provides a comprehensive survey of Indic NLP's historical evolution, details specific challenges for building AI foundation models, and analyzes current Indic foundation models. Finally, the authors propose "Culture Sensing" as a new research direction to develop AI systems that are both culturally meaningful and equitable across diverse Indic languages.

**Why this matches:** Based on the provided signals, this paper does not appear to match your interests. There are no overlapping keywords with your profile, nor is it identified as similar to any of your previously liked papers, especially regarding agentic recommender systems or novel agent architectures.

`arxiv-curator feedback 2607.06544v1 --rating up`

## [Pitwall: Faithful Natural-Language Race-Strategy Briefings from a Calibrated Real-Time Monte Carlo Engine](https://arxiv.org/abs/2607.06495v1)
**Score:** 0.624  |  **arXiv:** 2607.06495v1

Pitwall is a novel production system that generates faithful, real-time natural-language Formula 1 race strategy briefings in English, Spanish, and Portuguese. It achieves this faithfulness by rigorously verifying every generated sentence against a probabilistic race state predicted by a sophisticated Monte Carlo simulation engine, extensively calibrated and validated on historical F1 seasons. The system was successfully confirmed end-to-end at live Grands Prix, demonstrating its ability to accurately predict race outcomes, such as identifying the eventual winner ten laps before the flag.

**Why this matches:** Given that there are no overlapping keywords with your interests and the paper is not similar to any papers you previously liked, the provided signals do not indicate a match for your interests.

`arxiv-curator feedback 2607.06495v1 --rating up`

## [Industry Classification of GitHub Repositories Using the North American Industry Classification System (NAICS)](https://arxiv.org/abs/2607.06505v1)
**Score:** 0.613  |  **arXiv:** 2607.06505v1

GitHub lacks standardized industry classifications for its repositories, hindering research on innovation and technology diffusion. This paper introduces NAICS-GH, a publicly released dataset of 6,588 GitHub repositories accurately labeled with 2-digit NAICS codes, generated via an AI-powered pipeline validated at 96.98% precision against human labels. This resource is valuable for researchers studying industry-specific technology trends, open-source production, and for benchmarking new classification models.

**Why this matches:** Based on the given signals, this paper does not appear to match your interests. There are no overlapping keywords with your profile, nor is it similar to your previously liked papers on agent memory or novel architectural approaches.

`arxiv-curator feedback 2607.06505v1 --rating up`

## [GraphBU: MILP Instance Generation with Graph-Native Block Units](https://arxiv.org/abs/2607.06532v1)
**Score:** 0.605  |  **arXiv:** 2607.06532v1

Generating realistic Mixed-Integer Linear Programming (MILP) instances for solver development is challenging, as existing methods often fail to preserve the crucial internal coupling structure of real-world problems. This paper introduces GraphBU, a novel graph-native generator that uses "block units"—local subproblems combined with their interfaces—for compatibility-checked replacement and instance construction. This approach effectively maintains the original instance's graph statistics and feasibility, significantly improving the training of downstream solvers like Predict-and-Search.

**Why this matches:** Based on the provided signals, there are no overlapping keywords and no previously liked papers that are similar. Therefore, the given signals do not indicate a match between this paper and your interests.

`arxiv-curator feedback 2607.06532v1 --rating up`

## [Hierarchical Acoustic-Semantic Modeling: Modality Separation and Semantic Coherence for Full-Duplex SLMs](https://arxiv.org/abs/2607.06540v1)
**Score:** 0.604  |  **arXiv:** 2607.06540v1

Full-duplex Spoken Language Models (SLMs) often suffer from "modality interference," which this paper identifies as inherent gradient conflicts between acoustic and semantic modeling when sharing deep network parameters. To solve this, they introduce Lychee-FD, a hierarchical framework that separates these conflicting modalities in deep layers while preserving semantic coherence. This approach significantly boosts speech intelligence (+7.4% on Spoken QA) and interaction fluidity (+28.5% on FullDuplexBench), providing the first successful solution to this fundamental problem.

**Why this matches:** While there are no direct keyword overlaps or similar previously liked papers, this paper introduces a Hierarchical Acoustic-Semantic Modeling architecture with Modality Separation and Semantic Coherence. This novel approach aligns with your interest in genuinely new architectural designs, even if its specific domain for Full-Duplex SLMs does not directly map to agentic systems.

`arxiv-curator feedback 2607.06540v1 --rating up`

## [RMISC: A Large-scale Real-world Multivariate Corpus for Time Series Foundation Models](https://arxiv.org/abs/2607.06504v1)
**Score:** 0.600  |  **arXiv:** 2607.06504v1

This paper introduces RMISC, a large-scale, openly accessible real-world multivariate time series corpus designed to overcome the limitations of synthetic data in training Time Series Foundation Models (TSFMs). The authors demonstrate that pretraining advanced TSFMs on RMISC significantly improves their zero-shot generalization performance compared to synthetic data, highlighting the critical role of real-world multivariate data for building more robust TSFMs.

**Why this matches:** Based on the provided signals, this paper does not appear to match your interests, as there are no overlapping keywords with your profile and it is not similar to any of your previously liked papers.

`arxiv-curator feedback 2607.06504v1 --rating up`

## [The Large Cancer Assistant (LCA): A Model-Agnostic Orchestration Framework for Scalable Clinical Decision Support in Oncology](https://arxiv.org/abs/2607.06531v1)
**Score:** 0.599  |  **arXiv:** 2607.06531v1

Existing AI models for clinical decision support in oncology are often rigid, tightly coupling data processing with the AI itself. This paper introduces the Large Cancer Assistant (LCA), a novel, model-agnostic orchestration framework designed to make these systems highly flexible, scalable, and robust. The LCA strictly decouples the AI models from data ingestion and routing logic—ensuring compatibility with any underlying AI and standardizing diverse patient data—with proof-of-concept tests validating its efficiency, robust failure-safety, and genuine algorithmic independence.

**Why this matches:** This paper introduces a "Model-Agnostic Orchestration Framework" for scalable clinical decision support, aligning with your interest in practical tooling for building AI systems and genuinely new architectural approaches.

`arxiv-curator feedback 2607.06531v1 --rating up`

## [ELSA3D: Elastic Semantic Anchoring for Unified 3D Understanding and Generation](https://arxiv.org/abs/2607.06565v1)
**Score:** 0.595  |  **arXiv:** 2607.06565v1

Current unified 3D models struggle with precisely linking text and 3D geometry across different abstraction scales, often collapsing all information into an undifferentiated representation. ELSA3D addresses this with "elastic semantic anchoring," using "Anchor Tokens" to selectively route semantic cues to the most relevant 3D scales and retrieve specific geometric evidence, enabling sparse yet precise cross-modal interaction. This approach achieves state-of-the-art performance across 3D generation (image-to-3D, text-to-3D) and captioning tasks, while significantly reducing computational costs and inference latency.

**Why this matches:** Based on the provided signals, there are no overlapping keywords (such as agents, recommender systems, or memory) with your interests, and this paper is not similar to any of your previously liked papers demonstrating new architectural approaches for agent memory.

`arxiv-curator feedback 2607.06565v1 --rating up`

## [Graph Convolutional Attention: A Spectral Perspective on Graph Denoising and Diffusion](https://arxiv.org/abs/2607.06546v1)
**Score:** 0.595  |  **arXiv:** 2607.06546v1

Standard attention mechanisms used in graph denoising (e.g., graph diffusion models) are suboptimal as they fail to adapt to the inherent spectral diversity across different graphs. This paper introduces Graph Convolutional Attention (GCA), a new mechanism that directly leverages the input graph's spectrum through graph-filtered queries and keys to achieve adaptive spectral denoising. GCA provably outperforms linear attention, leading to improved graph denoising and diffusion, and offers practical benefits like matching transformer performance without expensive structural features and enabling faster inference.

**Why this matches:** This paper, presenting 'Graph Convolutional Attention' from a 'Spectral Perspective,' introduces a novel architectural approach to graph processing. This could align with your interest in genuinely new architectures, particularly those leveraging graphs, as exemplified by papers like 'Graph Memory for LLM Agents.'

`arxiv-curator feedback 2607.06546v1 --rating up`

## [Multi-Agent Deep Reinforcement Learning for Multi Objective Battery Management in Dairy Farms](https://arxiv.org/abs/2607.06489v1)
**Score:** 0.585  |  **arXiv:** 2607.06489v1

This paper addresses the largely unexplored potential for renewable energy integration and carbon reduction in Irish dairy farms. It proposes a novel multi-agent Deep Reinforcement Learning (DRL) control system for multi-objective battery management, combining dynamic pricing with a DRL-based lower layer to optimize energy arbitrage and renewable energy use. Simulations show this system significantly increases profits (up to 18% compared to rule-based models) and distributed generation usage while complying with grid code voltage limits.

**Why this matches:** Based on the provided grounded signals, there are no overlapping keywords and no similarity to previously liked papers. Therefore, this paper does not appear to match your stated interests.

`arxiv-curator feedback 2607.06489v1 --rating up`

## [On the feasibility of dependency parsing of non-human sequences without a gold standard. Is evaluation possible in other species?](https://arxiv.org/abs/2607.06542v1)
**Score:** 0.557  |  **arXiv:** 2607.06542v1

This paper addresses the challenge of evaluating unsupervised dependency parsers for non-human sequences (like primate vocalizations or gestures) when no "gold standard" of correct parses exists. By applying network science, the authors show that the unique statistical properties of non-human communication – specifically, a fast decay in sequence length distribution – make it possible to reliably estimate parser accuracy without a gold standard. This capability is notably absent for human language, where such evaluation remains a much harder problem without a gold standard.

**Why this matches:** This paper discusses the feasibility of evaluation, even in contexts lacking a gold standard, which aligns with your interest in LLM evaluation and practical aspects of building AI systems.

`arxiv-curator feedback 2607.06542v1 --rating up`

## [EntroPath: Maximum Entropy Path Ensemble Embedding for Manifold Learning](https://arxiv.org/abs/2607.06497v1)
**Score:** 0.555  |  **arXiv:** 2607.06497v1

EntroPath is a novel manifold learning method that improves upon existing techniques by considering the *entire ensemble* of diffusion paths between points, rather than relying on local walks or single shortest paths. This allows it to generate a dissimilarity measure that accurately captures true geodesic distances, effectively overcoming challenges like non-uniform sampling density and spurious shortcuts in data graphs. EntroPath consistently outperforms diffusion and shortest-path methods on complex data, proving particularly strong for manifolds with non-uniform sampling or branching trajectories while remaining competitive on local structure metrics.

**Why this matches:** Based on the provided signals, this paper does not appear to match your interests, as there are no overlapping keywords detected and no similarity to your previously liked papers.

`arxiv-curator feedback 2607.06497v1 --rating up`

## [Life Style Levels: Neighborhood Delineation using Geospatial Data](https://arxiv.org/abs/2607.06529v1)
**Score:** 0.532  |  **arXiv:** 2607.06529v1

This study tackles the challenge of mapping fine-scale socioeconomic variations in rapidly urbanizing regions like India, where traditional data is scarce. It proposes a scalable, grid-based framework that uses open-source satellite imagery to analyze building morphology and delineate intra-urban affluence levels across 59 Indian cities. The resulting classifications, validated with ground-level observations, offer a cost-effective and interpretable method for granular urban mapping, also demonstrating utility in identifying informal settlements and mapping consumer loan delinquency.

**Why this matches:** Based on the provided signals, including the paper title "Life Style Levels: Neighborhood Delineation using Geospatial Data" and the explicit lack of overlapping keywords, this paper does not appear to align with your interests in agentic recommender systems, agent architectures, or AI building tooling.

`arxiv-curator feedback 2607.06529v1 --rating up`
