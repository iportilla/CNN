# Lab 4 — Azure Custom Vision: apples vs. bananas

Lecture notes: [Part 3 §3.3](../../lecture/03-azure-vision.md)

Train an image classifier on your own labels without writing a model. Custom Vision reuses a pretrained network and retrains only its final layer on your images (transfer learning).

## Images

All in [`images/`](../../images).

| Tag | Train on | Hold back for testing |
|---|---|---|
| `apple` | `apple-3.jpeg`, `apples-041efb2.jpg`, `apples-101-about-1440x810.jpg`, `apples-close-up_screen.jpg`, `apples-royalty-free-image-164084111-1537885595.jpg` | `Red_Apple.jpg` |
| `banana` | `Bunch-of-bananas.jpg`, `all-about-bananas.jpg`, `bananas2.jpeg`, `fresh-ripe-bananas.jpg`, `green-bananas.jpg` | A banana photo of your own |

Five images per tag is the minimum the service accepts. It is enough to see transfer learning work and far too few to trust the metrics.

## Steps

1. Sign in at [customvision.ai](https://www.customvision.ai/) with your Azure account.
2. **New project** → create a Custom Vision resource if prompted → Project type **Classification**, Classification type **Multiclass**, Domain **General**.
3. **Add images** → select the five apple photos → tag `apple` → upload. Repeat for `banana`.
4. **Train** → Quick Training.
5. Open the **Performance** tab. Note precision and recall for each tag. Move the probability threshold and watch them change.
6. **Quick Test** → upload `Red_Apple.jpg`, then your own banana photo.
7. Optional: **Publish** the iteration, then call it from code with [`predict.py`](predict.py).

## Calling the published model

After publishing, open **Prediction URL** in the Performance tab. Add to `.env` in the repo root:

```
CUSTOM_VISION_ENDPOINT=https://<your-resource>.cognitiveservices.azure.com/
CUSTOM_VISION_PREDICTION_KEY=...
CUSTOM_VISION_PROJECT_ID=...
CUSTOM_VISION_ITERATION=Iteration1
```

```bash
python predict.py ../../images/Red_Apple.jpg
```

## Questions

1. What did the model say about `nyc-street.jpg`? A multiclass model must pick one of its tags. How would you make it able to say "neither"?
2. Add a third tag, `green banana` vs. `ripe banana`, with the images you have. What happens to precision and recall?
3. Swap fruit for trees: collect hemlock and Japanese cherry photos and repeat. How many images did you need before Quick Test was reliable?

Reference: [Build a classifier with Custom Vision](https://learn.microsoft.com/azure/ai-services/custom-vision-service/getting-started-build-a-classifier)
