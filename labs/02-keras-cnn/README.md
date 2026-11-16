# Lab 2 — CNNs in Keras

Lecture notes: [Part 2](../../lecture/02-cnn-in-keras.md)

| Order | Notebook | Dataset | Task |
|---|---|---|---|
| 1 | [`CNN-example.ipynb`](CNN-example.ipynb) | MNIST | Read through, run, then classify a digit from [`samples/`](samples) or your own photo |
| 2 | [`cnn-1-lowCode.ipynb`](cnn-1-lowCode.ipynb) | MNIST | Complete the `#TODO` layers |
| 3 | [`CNN-2-lowCode.ipynb`](CNN-2-lowCode.ipynb) | CIFAR-100 | Complete the `#TODO` layers |
| 4 | [`CNN-3-lowCode.ipynb`](CNN-3-lowCode.ipynb) | Fashion-MNIST | Compare dropout on/off, then complete the `#TODO` layers |

Each low-code notebook first trains a complete reference model, then gives you a second model with missing layers. Cells with `model.add(*)` will not run until you fill them in. Worked answers are in [`solutions/`](solutions).

The notebooks train for 20 to 30 epochs. On a laptop CPU, lower `EPOCHS` to 5 while you experiment, or run in [Google Colab](https://colab.research.google.com/) with a GPU.

```bash
pip install tensorflow matplotlib pillow
```
