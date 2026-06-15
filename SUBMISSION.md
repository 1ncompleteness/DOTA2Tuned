# DOTA2Tuned Submission Handoff

Last checked: 2026-06-15 16:17 PDT.

## Required Links

- Hugging Face Space: https://build-small-hackathon-dota2tuned.hf.space
- GitHub repo: https://github.com/1ncompleteness/DOTA2Tuned
- Tiny fine-tuned adapter: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
- Balanced fine-tuned adapter target: https://huggingface.co/build-small-hackathon/dota2tuned-minicpm4-1-8b-lora
- Quality fine-tuned adapter target: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-30b-a3b-2507-lora
- Dataset artifacts: https://huggingface.co/datasets/build-small-hackathon/dota2tuned-data
- Modal alternate UI: https://dracufeuer--dota2tuned-ui.modal.run

## Hackathon Fit

- Primary track: Backyard AI.
- User problem: Dota 2 players need draft, counter, synergy, build, and match-outcome guidance that is grounded in patch and match evidence instead of generic hero lore.
- Small-model constraint: all selected model profiles stay below the hackathon `<=32B` parameter cap. Tiny uses `Qwen/Qwen3-4B-Instruct-2507`; Balanced uses `openbmb/MiniCPM4.1-8B`; Quality uses `Qwen/Qwen3-30B-A3B-Instruct-2507`.
- Gradio constraint: canonical app is a Gradio Space under the Build Small Hackathon org.
- Well-Tuned badge: app uses a published fine-tuned adapter on Hugging Face.
- Modal award fit: Modal handles GPU training and adapter inference; the Space remains the required submission app.

Do not claim the Off the Grid or Local-first badges. The app uses Modal for the tuned-model path and uses public Dota data sources during data refreshes.

## Final Submission Checklist

- [x] Space link is live.
- [x] Space runtime reached `RUNNING`.
- [x] Space HTTP check returned `200`.
- [x] Modal UI HTTP check returned `200`.
- [x] Modal remote smoke passed.
- [x] Modal adapter inference passed.
- [x] Fine-tuned adapter is published on Hugging Face.
- [x] Compact serving artifacts are uploaded with the Space.
- [x] Expanded sample artifacts include 16,000 OpenDota match details, 1,950 raw STRATZ details, 15,784 normalized matches, 159,998 player-match rows, 15,508 RAG docs, and 942 SFT examples.
- [x] Dataset card is prepared for the Hub dataset.
- [x] Repository license is declared as Apache-2.0.
- [x] GitHub `main` is pushed and clean.
- [ ] Record short demo video.
- [ ] Publish social post.
- [ ] Submit Space link, demo video link, and social post link by June 15, 2026.

## Demo Video Script

Target length: 60-90 seconds.

1. Open the Space and state the app:
   - "DOTA2Tuned is a small-model Dota 2 draft coach. Stats and predictors choose recommendations; the fine-tuned model explains them with evidence."
2. Draft view:
   - Select a realistic enemy draft such as `Phantom Assassin, Witch Doctor`.
   - Search by names or aliases such as `PA`, `CM`, or `AM`.
   - Show selected hero icon previews.
   - Show recommendations with scores, sample sizes, confidence, and caveats.
3. Ask view:
   - Ask: "Suggest one mid hero against Phantom Assassin and Witch Doctor, and include one caveat."
   - Show model selection, retrieved evidence, source links, and caveats for weak context.
4. Meta or Builds view:
   - Select one hero, item, or skill and show patch/stat cards, observed build timing, or skill order.
5. Predictor view:
   - Select two five-hero drafts and show predicted Radiant win probability.
6. Draft view:
   - Generate a draft card from the same enemy draft.
   - Emphasize that the playful mode still uses the local stat-backed recommender.
7. Data view:
   - Show artifact counts and the Space/dataset/model links used for judging.
7. Close with links:
   - "The Space is live, the adapter is published on Hugging Face, and Modal powers the GPU training and inference path."

## Social Post Draft

Built DOTA2Tuned for the Hugging Face Build Small Hackathon: a Gradio Dota 2 draft coach that combines STRATZ/OpenDota match evidence, deterministic draft stats, and fine-tuned sub-32B adapters for grounded explanations.

It suggests heroes, counters, synergies, builds, match predictions, and caveats weak data instead of inventing unsupported meta claims.

Space: https://build-small-hackathon-dota2tuned.hf.space
Model: https://huggingface.co/build-small-hackathon/dota2tuned-qwen3-4b-2507-lora
Repo: https://github.com/1ncompleteness/DOTA2Tuned

## Final Verification Commands

```bash
uv run ruff check app.py src tests
uv run pytest -q
uv run python -c "from app import demo; print(type(demo).__name__, len(demo.blocks), len(demo.fns))"
uv run dota2tuned modal-smoke
uv run dota2tuned modal-ask "Suggest one mid hero against Phantom Assassin and Witch Doctor. Include one caveat." --context "Use only grounded advice. Mention that sample sizes and patch context matter." --max-new-tokens 160
uv run python scripts/check_public_space.py
uv run python scripts/check_submission_ready.py
curl -L -sS -o /dev/null -w '%{http_code}\n' https://build-small-hackathon-dota2tuned.hf.space
curl -sS -o /dev/null -w '%{http_code}\n' https://dracufeuer--dota2tuned-ui.modal.run
git status --short --branch
```

## Sources Checked

- Hackathon page: https://huggingface.co/build-small-hackathon
- Gradio on Modal guide: https://gradio.app/guides/deploying-gradio-with-modal
