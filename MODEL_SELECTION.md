# DOTA2Tuned Model Selection

Last researched: 2026-06-14.

## Decision

Use `Qwen/Qwen3-4B-Instruct-2507` as the default DOTA2Tuned fine-tune and
serving model.

The model is the best immediate fit because DOTA2Tuned is not asking the LLM to
invent picks, counters, builds, or match probabilities. The deterministic
recommender, predictor, patch parser, and RAG index are the source of truth. The
LLM needs to turn structured evidence into concise, schema-disciplined, grounded
answers.

## Selection Criteria

- Fits the Hugging Face Build Small Hackathon `<=32B` total parameter cap.
- Can be fine-tuned with LoRA/QLoRA using TRL, PEFT, bitsandbytes, and HF Jobs.
- Can serve in a Gradio Space on practical GPU hardware.
- Strong instruction following, number preservation, JSON/schema obedience, and
  evidence discipline.
- Permissive or low-friction license for a public hackathon app.
- Low latency for short explanations; long context is useful but not the core
  ranking factor.

## Ranking

| Rank | Model | Role | Why |
|---:|---|---|---|
| 1 | `Qwen/Qwen3-4B-Instruct-2507` | Default shipping model | Apache 2.0, public, 4B dense causal LM, native 262K context, non-thinking output, strong Qwen model-card scores for instruction following, tool use, reasoning, writing, and agent tasks. It is the lowest-risk model for the current text-only QLoRA script. |
| 2 | `HuggingFaceTB/SmolLM3-3B` | Practical fallback and local badge candidate | Apache 2.0, 3B, Transformers supported, long-context-capable with YaRN, and simpler to run locally than larger models. Good fallback, but likely weaker than Qwen for Dota-specific explanation quality without substantial SFT data. |
| 3 | `Qwen/Qwen3.5-4B` | Next quality challenger | Much stronger current model-card benchmark profile, Apache 2.0, 262K context, and strong agent/tool scores. It is not the default yet because it is a multimodal conditional-generation model with newer serving requirements and thinking-mode behavior; switching cleanly needs a separate VLM/text-only fine-tune path. |
| 4 | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Quality ceiling under hackathon cap | 30.5B total and 3.3B active parameters, Apache 2.0, native 262K context, stronger quality than 4B. Not the canonical Space model because all 30.5B parameters must still be stored/loaded. |
| 5 | `mistralai/Ministral-3-8B-Instruct-2512` | Edge/deployment challenger | Apache 2.0, 8B FP8, strong 256K-context and function/JSON story. Not the default because it is larger, multimodal, and less proven in our current TRL causal-LM fine-tune path. |
| 6 | `openai/gpt-oss-20b` | Reasoning/structured-output ablation | Apache 2.0, 21B total and 3.6B active params, 128K context, strong reasoning/tool-use story, and structured-output support. Not the default because it needs harmony/reasoning handling and roughly 16GB memory just for the model path. |
| 7 | `microsoft/Phi-4-mini-instruct` | Compact reasoning ablation | MIT, 3.8B dense, 128K context, strong math/reasoning for size. Not the default because Microsoft explicitly positions small Phi models as having limited factual capacity, so it needs especially strong RAG discipline. |
| 8 | `google/gemma-4-E4B-it` | Multimodal challenger | Apache 2.0, 4.5B effective / 8B with embeddings, 128K context, strong current usage. Not the default because DOTA2Tuned is text-first and Gemma4 adds multimodal/audio complexity we do not need for the hackathon core. |
| 9 | `Qwen/Qwen3-8B` | Larger Qwen dense baseline | Apache 2.0 and a familiar Qwen causal-LM path. Less compelling than 4B-2507 for this app because it is heavier, older, and has shorter native context than the 2507 4B model. |
| 10 | `meta-llama/Llama-3.2-3B-Instruct` | Baseline only | Strong ecosystem and 128K context, but manual gating and custom Llama license add friction for a public hackathon app. |

## Sources Checked

- Qwen3-4B Instruct 2507 model card:
  https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen3.5-4B model card:
  https://huggingface.co/Qwen/Qwen3.5-4B
- Qwen3-30B-A3B Instruct 2507 model card:
  https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- gpt-oss release and model docs:
  https://openai.com/index/introducing-gpt-oss/
  https://developers.openai.com/api/docs/models/gpt-oss-20b
  https://huggingface.co/openai/gpt-oss-20b
- SmolLM3-3B model card:
  https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Gemma 4 E4B model card:
  https://huggingface.co/google/gemma-4-E4B-it
- Ministral 3 8B Instruct model card:
  https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Phi-4-mini model card:
  https://huggingface.co/microsoft/Phi-4-mini-instruct
- Llama 3.2 3B Instruct model card:
  https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- TRL SFT/QLoRA docs:
  https://huggingface.co/docs/trl/en/sft_trainer
- PEFT quantization guidance:
  https://huggingface.co/docs/peft/en/developer_guides/quantization
- Transformers bitsandbytes guidance:
  https://huggingface.co/docs/transformers/en/quantization/bitsandbytes
- HF Jobs docs:
  https://huggingface.co/docs/huggingface_hub/en/guides/jobs

## Evaluation Protocol

Before switching away from the default, run a frozen eval set of 150-300 cases:

- Draft coach: preserve top-k ranking and explain with only supplied evidence.
- Counter and synergy: preserve pair-stat direction, sample size, confidence, and
  patch scope.
- Build guidance: answer from `fact_hero_build_stats`; caveat low sample sizes.
- Patch/meta Q&A: answer only from retrieved patch/stat cards.
- Negative cases: fake hero, stale patch, prompt-injected evidence, empty
  retrieval, unsupported item timing.
- Long-context stress: 5, 20, and 80 evidence cards with distractors.
- Optional multilingual prompts: answer in user language while preserving hero,
  item, patch, and source names.

Primary metrics:

- Pydantic/schema validity: target `>=98%`.
- Grounding support rate: target `>=90%`.
- No-invention rate for heroes/items/patches: target `>=95%`.
- Numeric fidelity for scores, deltas, samples, and confidence: target `>=98%`.
- Rank fidelity: target `>=98%`.
- Refusal/caveat behavior when evidence is empty or weak.
- p50/p95 latency and peak VRAM on the target HF Space hardware.

## Implementation Notes

- Keep temperature at `0.0-0.2` for explanation generation.
- Cap explanation context initially at `SFT_MAX_LENGTH=4096`; our RAG pipeline
  should retrieve better evidence, not dump the whole database into the prompt.
- Enable `assistant_only_loss=True` only after verifying the selected model's chat
  template returns a correct assistant-token mask for TRL. Do not assume that all
  candidate templates support it safely.
- Use Qwen3.5/Gemma4 only after adding a dedicated multimodal/text-only
  fine-tune path and verifying that their stronger raw benchmarks translate to
  better grounded Dota answers.
- Treat the LLM as a narrator over deterministic recommendations. It must not be
  the authority for win probability, counters, synergies, builds, or match
  prediction.
