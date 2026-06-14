---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
- dota-2
- qlora
- peft
- trl
- gradio
- build-small-hackathon
datasets:
- build-small-hackathon/dota2tuned-data
---

# DOTA2Tuned Qwen3 4B LoRA

This is the fine-tuned adapter used by DOTA2Tuned, a Hugging Face Build Small Hackathon Gradio app for Dota 2 draft coaching, hero meta lookup, counter/synergy reasoning, build timing summaries, and match prediction explanations.

The adapter is based on `Qwen/Qwen3-4B-Instruct-2507`, keeping the total model size below the hackathon `<=32B` parameter limit. It was trained with TRL SFT and QLoRA on Modal GPU infrastructure, then served through Modal-backed inference from the Hugging Face Space.

## Intended Behavior

DOTA2Tuned does not use the language model as the source of truth for picks, counters, builds, or match probabilities. Deterministic statistics, a draft predictor, and retrieved patch/stat cards provide the evidence. The adapter's job is to turn that evidence into concise Dota 2 explanations.

The model should:

- answer draft, counter, synergy, build, and match-prediction questions from supplied evidence;
- preserve hero names, item names, patch IDs, scores, deltas, sample sizes, and confidence labels;
- caveat weak data and low sample sizes;
- refuse or qualify answers when evidence is missing, generic, stale, or mentions fake heroes/items.

The model should not invent unsupported hero names, item names, scores, timings, patch changes, or match data.

## Training Data

Training examples are generated from the DOTA2Tuned pipeline and published in the companion dataset repo:

https://huggingface.co/datasets/build-small-hackathon/dota2tuned-data

The examples combine normalized Dota 2 stat cards, draft recommendation outputs, patch-change retrieval snippets, and refusal/caveat behaviors.

## Usage

This repo contains a PEFT LoRA adapter. Load it with the base model through PEFT:

```python
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

model_id = "build-small-hackathon/dota2tuned-qwen3-4b-2507-lora"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoPeftModelForCausalLM.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "system",
        "content": (
            "You are DOTA2Tuned. Use only supplied evidence for concrete Dota 2 "
            "claims and caveat weak data."
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Suggest one mid hero against Phantom Assassin and Witch Doctor.\n\n"
            "Evidence:\n"
            "Phantom Assassin current pro stat card: 3 pro picks, 1 win.\n"
            "Witch Doctor current pro stat card: 8 pro picks, 5 wins."
        ),
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)
outputs = model.generate(inputs, max_new_tokens=160, do_sample=False)
print(tokenizer.decode(outputs[0][inputs.shape[-1] :], skip_special_tokens=True))
```

For the shipped app, use the Space:

https://build-small-hackathon-dota2tuned.hf.space

## Evaluation Notes

The deployed adapter was smoke-tested through Modal and the public Space with prompts covering:

- draft recommendations against Phantom Assassin and Witch Doctor;
- counter/synergy explanations;
- Shadow Fiend build timing without concrete evidence;
- fake hero rejection;
- patch/meta caveats;
- low sample-size caveats.

One grounding weakness was found during testing: generic build prompts could produce unsupported item advice. The deployed Modal inference prompt now explicitly forbids invented hero names, item names, scores, timings, patches, and match data, and asks for uncertainty when evidence is missing or generic.

## Limitations

- The adapter is only as reliable as the supplied Dota 2 evidence.
- The current hackathon dataset is compact, so many hero and matchup samples are sparse.
- Match prediction is draft-only and does not model player skill, execution, lane assignments, itemization, or in-game state.
- Dota 2 patches change quickly; refresh the data pipeline before making current-meta claims.
- The adapter is not affiliated with Valve, OpenDota, STRATZ, Steam, Hugging Face, Modal, or Qwen.

## Project Links

- Space: https://build-small-hackathon-dota2tuned.hf.space
- Dataset: https://huggingface.co/datasets/build-small-hackathon/dota2tuned-data
- GitHub: https://github.com/1ncompleteness/DOTA2Tuned
- Modal alternate UI: https://dracufeuer--dota2tuned-ui.modal.run
