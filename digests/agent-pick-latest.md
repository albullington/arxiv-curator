# Agent Pick -- 2026-07-10

## [DepthWeave-KV: Token-Adaptive Cross-Layer Residual Factorization for Long-Context KV Cache Compression](https://arxiv.org/abs/2607.06523v1)
**arXiv:** 2607.06523v1

DepthWeave-KV presents a novel, token-adaptive method to compress key-value caches for long-context LLMs, addressing the limitations of uniform compression by factorizing K/V states across layers and dynamically allocating precision to critical tokens (like instructions) during inference without retraining. This approach achieves near full-cache performance and improved retrieval accuracy with significantly reduced memory—up to 8.3x KV memory reduction—outperforming prior compressed caches on various long-context benchmarks.

**Why this was picked:** An exceptional paper on KV cache compression for long-context LLMs. By combining cross-layer residual factorization with a token-conditional depth router, it selectively allocates higher reconstruction rank to critical tokens during inference without retraining. This is paired with a fused CUDA kernel, leading to a massive 8.3x reduction in memory usage and 72.8 tokens/sec decoding speed. Highly relevant to LLM serving infrastructure and optimization.

`arxiv-curator feedback 2607.06523v1 --rating up`

## [Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents](https://arxiv.org/abs/2606.06036v1)
**arXiv:** 2606.06036v1

Despite recent progress, LLM agents still struggle with reasoning over long interaction histories. While current memory-augmented agents rely on a static retrieve-then-reason paradigm, this rigid pipeline design prevents them from dynamically adapting memory access to intermediate evidence discovered during inference. To bridge this gap, we propose MRAgent, a framework that combines an associative memory graph with an active reconstruction mechanism. We represent memory as a Cue-Tag-Content graph, where associative tags serve as semantic bridges connecting fine-grained cues to memory contents. Operating on this structure, our active reconstruction mechanism integrates LLM reasoning directly into memory access, allowing the agent to iteratively explore and prune retrieval paths based on accumulated evidence. This ensures that memory retrieval is dynamically adapted to the reasoning context while avoiding combinatorial explosion caused by unconstrained expansion. Experiments on the LoCoMo benchmark and LongMemEval benchmark demonstrate significant improvements over strong baselines (up to 23%), while substantially reducing token and runtime cost, highlighting the effectiveness of active and associative reconstruction for long-horizon memory reasoning.

**Why this was picked:** Proposes a dynamic, active reconstruction framework (MRAgent) for agent memory using a Cue-Tag-Content graph instead of a static retrieve-then-read pipeline. This allows agents to dynamically adapt memory access to intermediate reasoning steps. It is a highly practical, concretely triable pattern for anyone building LLM agents with long-term memory requirements.

`arxiv-curator feedback 2606.06036v1 --rating up`
