# Lab 6 — Image generation

Lecture notes: [Part 4 §4.7](../../lecture/04-vision-llms.md)

[`image_generation.ipynb`](image_generation.ipynb) uses the OpenAI Images API to generate an image from a prompt, produce variations, and edit part of an image with a mask.

## Setup

1. Add `OPENAI_API_KEY=...` to `.env` in the repo root.
2. `pip install openai pillow python-dotenv`

Generated files are written to `images/` next to the notebook; that folder is git-ignored. Each call is billed to your OpenAI account.

## Tasks

1. Run the notebook as is.
2. Change the mask so the **left** half is regenerated instead of the bottom half.
3. Do the sneaker design exercise from the lecture notes: three prompt revisions, with a one-line note on what you changed and why.
