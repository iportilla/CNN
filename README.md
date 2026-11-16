# CNNs and Computer Vision with LLMs and Azure

A lecture with six hands-on labs. It starts with one convolution written by hand and ends with asking a multimodal LLM why the traffic has stopped.

```mermaid
flowchart LR
    A[1 · How a CNN sees<br/>convolution, pooling] --> B[2 · Train one in Keras<br/>MNIST, CIFAR-100, Fashion-MNIST]
    B --> C[3 · Use one on Azure<br/>Image Analysis, Custom Vision]
    C --> D[4 · Ask questions of images<br/>vision LLMs, generation]
```

## Learning objectives

After this lecture you can:

1. Explain convolution, ReLU and pooling, and compute a layer's output size and weight count by hand.
2. Build, train and evaluate a CNN in Keras, and read its training curves for overfitting.
3. Call Azure AI Vision to analyse an image, and train a custom classifier with Azure Custom Vision from a handful of photos.
4. Describe how a vision-language model turns an image into tokens, and query one locally (Ollama) and on Azure OpenAI.
5. Choose between a CNN, a prebuilt API, a custom-trained service and a vision LLM for a given problem.

## Agenda

Sized for one 2.5-hour session. Parts 1–2 and 3–4 also work as two 75-minute sessions.

| Time | Part | Notes | Slides | Lab |
|---|---|---|---|---|
| 0:00 | Recap: layers, loss, optimizer, activation functions | — | [Computer-Vision-101](slides/Computer-Vision-101.pptx) 1–4 | — |
| 0:10 | **1 · How a network learns to see** | [01-cnn-theory](lecture/01-cnn-theory.md) | [CNN-Theory](slides/CNN-Theory.pptx) | [Lab 1 · CNN 101](labs/01-cnn-101) (15 min) |
| 0:50 | **2 · CNNs in Keras** | [02-cnn-in-keras](lecture/02-cnn-in-keras.md) | [Computer-Vision-101](slides/Computer-Vision-101.pptx) 9–19 | [Lab 2 · Keras CNNs](labs/02-keras-cnn) (25 min, finish as homework) |
| 1:25 | Break | | | |
| 1:35 | **3 · Computer vision on Azure** | [03-azure-vision](lecture/03-azure-vision.md) | [Computer-Vision-101](slides/Computer-Vision-101.pptx) 7, 20–21, 30–33 | [Lab 3 · Image Analysis](labs/03-azure-image-analysis) (10 min) · [Lab 4 · Custom Vision](labs/04-azure-custom-vision) (15 min) |
| 2:05 | **4 · Computer vision with LLMs** | [04-vision-llms](lecture/04-vision-llms.md) | [Computer-Vision-101](slides/Computer-Vision-101.pptx) 8, 22–29 | [Lab 5 · Vision LLMs](labs/05-vision-llms) (15 min) · [Lab 6 · Image generation](labs/06-image-generation) (homework) |
| 2:25 | Wrap-up: which tool for which problem | [04 §4.6](lecture/04-vision-llms.md#46-choosing-a-tool) | | |

## Labs

| # | Lab | You will | Needs |
|---|---|---|---|
| 1 | [CNN 101](labs/01-cnn-101) | Code a 3×3 convolution and 2×2 max pooling in plain Python | NumPy, scikit-image |
| 2 | [Keras CNNs](labs/02-keras-cnn) | Train CNNs on MNIST, CIFAR-100 and Fashion-MNIST; classify your own handwriting | TensorFlow |
| 3 | [Azure Image Analysis](labs/03-azure-image-analysis) | Get tags, captions and landmarks from a REST API | Azure Computer Vision key |
| 4 | [Azure Custom Vision](labs/04-azure-custom-vision) | Train an apple-vs-banana classifier in the portal and call it from Python | Azure account |
| 5 | [Vision LLMs](labs/05-vision-llms) | Compare two multimodal models on the same photo; optional Azure OpenAI | Ollama |
| 6 | [Image generation](labs/06-image-generation) | Generate, vary and mask-edit images | OpenAI API key |

## Setup

You need [Git](https://git-scm.com/downloads) and [Python](https://www.python.org/downloads/) 3.10 or newer. Labs 1 and 2 need no keys or accounts. Labs 3–6 read credentials from `.env`; see each lab's README. Do not commit `.env`.

### macOS and Linux

In Terminal:

```bash
git clone https://github.com/iportilla/CNN.git
cd CNN
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
jupyter lab
```

- Check your Python first with `python3 --version`. On macOS, install it from python.org or with `brew install python`. On Ubuntu/Debian, run `sudo apt install python3 python3-venv python3-pip`.
- Edit `.env` with any text editor, for example `nano .env`. The file is hidden in Finder and in `ls`; use `ls -a` to see it.
- Ollama for Lab 5: on macOS, download the app from [ollama.com/download](https://ollama.com/download). On Linux, run `curl -fsSL https://ollama.com/install.sh | sh`.

### Windows

In PowerShell:

```powershell
git clone https://github.com/iportilla/CNN.git
cd CNN
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
jupyter lab
```

- When installing Python from python.org, tick **Add python.exe to PATH**. Check it with `py --version`.
- If `Activate.ps1` is blocked with "running scripts is disabled on this system", run this once in the same window and try again:

  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

- In Command Prompt instead of PowerShell, activate with `.venv\Scripts\activate.bat` and copy with `copy .env.example .env`.
- Edit `.env` with `notepad .env`. If you create the file by hand, make sure Notepad does not save it as `.env.txt`.
- Ollama for Lab 5: run the Windows installer from [ollama.com/download](https://ollama.com/download), then open a new PowerShell window so `ollama` is on the PATH.
- TensorFlow on Windows runs on the CPU only. That is enough for Lab 2 if you lower `EPOCHS`; for a GPU, use [WSL 2](https://learn.microsoft.com/windows/wsl/install) and follow the Linux steps, or use Colab.

### All platforms

- Your prompt shows `(.venv)` when the environment is active. Activate it again in every new terminal window before running `jupyter lab`.
- If `pip install -r requirements.txt` cannot find a TensorFlow version, your Python is newer than TensorFlow supports. Create the environment with an older Python (`python3.12 -m venv .venv`, or `py -3.12 -m venv .venv` on Windows).
- Paths inside the notebooks use forward slashes and work unchanged on every platform.
- No local install? Labs 1 and 2 run as they are in [Google Colab](https://colab.research.google.com/): **File → Open notebook → GitHub**, then enter `iportilla/CNN`.

## Repository layout

```
lecture/   Lecture notes, one file per part
slides/    PowerPoint decks
labs/      Six labs: notebooks, READMEs, solutions
images/    Sample photos used by Labs 3–5
```

## Credits and further reading

- MIT 6.S191 Introduction to Deep Learning, Lecture 3: Deep Computer Vision — [slides](http://introtodeeplearning.com/slides/6S191_MIT_DeepLearning_L3.pdf) · [introtodeeplearning.com](http://introtodeeplearning.com)
- Stanford CS231n — [cs231n.github.io](https://cs231n.github.io/)
- Google codelab — [Build convolutions and perform pooling](https://developers.google.com/codelabs/tensorflow-3-convolutions)
- Microsoft Learn — [Azure AI Vision](https://learn.microsoft.com/azure/ai-services/computer-vision/overview) · [Custom Vision](https://learn.microsoft.com/azure/ai-services/custom-vision-service/overview) · [Azure AI Search](https://learn.microsoft.com/azure/search/search-what-is-azure-search)
- The Lab 6 notebook is adapted from the OpenAI Cookbook. Sample photos in `images/` are third-party and included for classroom use only.

Code and notes: MIT License, see [LICENSE](LICENSE).
