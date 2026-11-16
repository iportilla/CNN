# Lab 3 — Azure AI Vision: Image Analysis

Lecture notes: [Part 3 §3.2](../../lecture/03-azure-vision.md)

[`azure_image_analysis.ipynb`](azure_image_analysis.ipynb) sends photos from [`images/`](../../images) to the Image Analysis REST API and reads back categories, tags, captions, colours and landmarks.

## Setup

1. In the [Azure portal](https://portal.azure.com), create a **Computer Vision** resource (the free F0 tier is enough).
2. Open **Keys and Endpoint** and copy one key and the endpoint.
3. Copy `.env.example` in the repo root to `.env` and fill in:

   ```
   AZURE_VISION_KEY=...
   AZURE_VISION_ENDPOINT=https://<region>.api.cognitive.microsoft.com/
   ```

   The endpoint must end with `/`.
4. `pip install requests python-dotenv pillow matplotlib azure-cognitiveservices-vision-computervision`

## Tasks

1. Run the notebook. Find the caption, the top five tags and the dominant colours for the Times Square photo.
2. Analyse `pexels-photo-338515.jpeg`. Which landmark does the service name, and with what confidence?
3. Analyse `nyc-street.jpg`. Save the tags; you will compare them with a vision LLM in [Lab 5](../05-vision-llms).
4. Try three more landmarks from `images/`. Which does the service miss?

No Azure subscription? Use [Vision Studio](https://portal.vision.cognitive.azure.com/gallery/imageanalysis) in the browser.
