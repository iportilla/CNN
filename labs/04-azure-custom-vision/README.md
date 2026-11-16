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

[`predict.py`](predict.py) uploads one local image to your published iteration and prints each tag with its probability.

### 1. Get the four values

In the Custom Vision portal, open your project → **Performance** tab → select the published iteration → **Prediction URL**. The dialog shows a URL and a key:

```
https://eastus.api.cognitive.microsoft.com/customvision/v3.0/Prediction/1a2b3c4d-0000-0000-0000-000000000000/classify/iterations/Iteration1/image
Prediction-Key: 0123456789abcdef0123456789abcdef
```

| `.env` variable | Where it comes from | In the example above |
|---|---|---|
| `CUSTOM_VISION_ENDPOINT` | The URL up to, but not including, `/customvision/...` | `https://eastus.api.cognitive.microsoft.com/` |
| `CUSTOM_VISION_PROJECT_ID` | The ID after `/Prediction/`. It is also in the browser address bar after `/projects/` | `1a2b3c4d-0000-0000-0000-000000000000` |
| `CUSTOM_VISION_ITERATION` | The name after `/iterations/`: the **published name** you chose when you clicked Publish | `Iteration1` |
| `CUSTOM_VISION_PREDICTION_KEY` | The `Prediction-Key` value shown under the URL | `0123456789abcdef0123456789abcdef` |

Notes:

- The dialog shows two URLs, one ending in `/url` (image on the web) and one ending in `/image` (image file). They share the same endpoint, project ID and iteration; `predict.py` uses `/image`.
- `CUSTOM_VISION_ITERATION` is the published **name**, not the `iterationId` in the browser address bar.
- No **Prediction URL** button? The iteration is not published yet. Click **Publish** first.

### 2. Put them in `.env`

`.env` lives in the **repo root** (the `CNN` folder), next to the main `README.md`. If it does not exist yet, copy `.env.example` to `.env`. Fill in the four lines, with no quotes and no spaces around `=`:

```
CUSTOM_VISION_ENDPOINT=https://eastus.api.cognitive.microsoft.com/
CUSTOM_VISION_PREDICTION_KEY=<your prediction key>
CUSTOM_VISION_PROJECT_ID=<your project id>
CUSTOM_VISION_ITERATION=Iteration1
```

The key is a credential. `.env` is git-ignored; keep it that way, and do not paste the key into a notebook, a chat or a commit.

### 3. Run it

Activate the virtual environment first; your prompt shows `(.venv)`. The image path is relative to the folder you are in, so use one of these two forms.

From the repo root (recommended):

```bash
python labs/04-azure-custom-vision/predict.py images/Red_Apple.jpg
```

Or from this lab folder:

```bash
cd labs/04-azure-custom-vision
python predict.py ../../images/Red_Apple.jpg
```

The same commands work in PowerShell on Windows.

The output is one line per tag in your project:

```
apple        99.2%
banana       0.8%
```

The tags are whatever your project was trained on. Against a cat-vs-dog project, for example:

```
(.venv) CNN % python labs/04-azure-custom-vision/predict.py images/dog1.jpeg
dog          100.0%
cat          0.0%
```

### Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `can't open file '.../predict.py'` | You are not in the repo root | `cd` into the `CNN` folder, or use the lab-folder form above |
| `FileNotFoundError: ... '../../images/...'` | The image path does not match the folder you are in | From the repo root use `images/<file>`; from the lab folder use `../../images/<file>` |
| `MissingSchema: Invalid URL '/customvision/v3.0/Prediction//classify/...'` | `CUSTOM_VISION_ENDPOINT` or `CUSTOM_VISION_PROJECT_ID` is empty in `.env` | Fill in all four values and save the file |
| `KeyError: 'CUSTOM_VISION_...'` | The variable is missing, or `.env` is not in the repo root | Check the file is named exactly `.env` and where it is |
| `401 Client Error` | Wrong key, or a training key instead of the prediction key | Copy `Prediction-Key` from the Prediction URL dialog again |
| `404 Client Error` | Wrong project ID or iteration name, or the iteration is not published | Compare both with the Prediction URL; publish the iteration |
| Every image gets the same tag | The image is not one of the project's classes | A multiclass model must pick one of its tags; see question 1 below |

## Questions

1. What did the model say about `nyc-street.jpg`? A multiclass model must pick one of its tags. How would you make it able to say "neither"?
2. Add a third tag, `green banana` vs. `ripe banana`, with the images you have. What happens to precision and recall?
3. Swap fruit for trees: collect hemlock and Japanese cherry photos and repeat. How many images did you need before Quick Test was reliable?

Reference: [Build a classifier with Custom Vision](https://learn.microsoft.com/azure/ai-services/custom-vision-service/getting-started-build-a-classifier)
