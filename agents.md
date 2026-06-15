# DOTA2Tuned Agent Instructions

DOTA2Tuned is a Hugging Face Space for Dota 2 draft help, meta analysis,
counters, synergies, builds, match prediction, and grounded RAG answers.

Space URL: `https://huggingface.co/spaces/build-small-hackathon/dota2tuned`

Agents can fetch the live Space API instructions with:

```bash
curl https://huggingface.co/spaces/build-small-hackathon/dota2tuned/agents.md
```

## Python Client

Use `gradio_client` for calls:

```python
from gradio_client import Client

client = Client("build-small-hackathon/dota2tuned")
answer_html, evidence_json = client.predict(
    "What should I pick with Crystal Maiden against Phantom Assassin?",
    api_name="/tuned_model",
)
```

## Public Endpoints

- `/tuned_model`: free-form Dota 2 question. Returns HTML answer and JSON evidence.
- `/draft_coach`: draft recommendations. Inputs: allied hero IDs, enemy hero IDs,
  banned hero IDs, role, and scope.
- `/hero_meta`: patch/meta lookup. Inputs: query string, optional hero ID, optional item key.
- `/match_predictor`: win-probability estimate. Inputs: Radiant hero IDs and Dire hero IDs.
- `/hero_builds`: item/build page for one hero ID.
- `/draft_lab`: compact draft card. Inputs: enemy hero IDs, role, mode.
- `/data_status`: current local data and artifact status.

## Calling Conventions

Use integer Dota hero IDs for hero dropdown inputs. Common examples:

- Anti-Mage: `1`
- Crystal Maiden: `5`
- Phantom Assassin: `44`
- Queen of Pain: `39`
- Shadow Fiend: `11`

Use these role values: `carry`, `mid`, `offlane`, `soft support`, `hard support`.
Use these scope values: `pro`, `high-rank`, `public`.

Example draft call:

```python
markdown, evidence_json = client.predict(
    [5],
    [44],
    [],
    "hard support",
    "pro",
    api_name="/draft_coach",
)
```

Example prediction call:

```python
prediction_json = client.predict(
    [1, 2, 3, 25, 5],
    [14, 74, 6, 26, 18],
    api_name="/match_predictor",
)
```

## Safety And Grounding

Prefer `/tuned_model` for natural-language questions because it returns evidence
alongside the answer. Treat generated advice as draft-support guidance, not a
guarantee of match outcome. Do not send API tokens, Steam credentials, or private
match data to the public Space.
