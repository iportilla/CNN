# Lab 5 — Computer vision with LLMs

Lecture notes: [Part 4](../../lecture/04-vision-llms.md)

[`vision_llms.ipynb`](vision_llms.ipynb) asks two vision-language models the same questions about [`nyc-street.jpg`](../../images/nyc-street.jpg) and prints the answers side by side.

## Setup

1. Install [Ollama](https://ollama.com/download) and pull two vision-capable models:

   ```bash
   ollama pull gemma3
   ollama pull llama3.2-vision
   ```

   Any model that shows `vision` under `ollama show <model>` will do. To use different ones, set `VISION_MODELS`, for example `VISION_MODELS=gemma3,llava:7b` in `.env`.
2. `pip install ollama pillow python-dotenv openai`
3. Optional, for section 4: an Azure OpenAI resource with a vision-capable chat deployment, and the three `AZURE_OPENAI_*` values in `.env`.

The committed outputs were produced with `gemma4` and `llava:7b`. Yours will differ with other models.

## What to hand in

A short table: for each question, each model's answer, and whether you judge it correct, wrong, or not answerable from the image.
