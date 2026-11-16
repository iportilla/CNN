# Part 2 — CNNs in Keras

Labs: [`labs/02-keras-cnn`](../labs/02-keras-cnn)

Part 1 built one convolution and one pooling step by hand. Keras gives you the same operations as layers and learns the filter values for you.

## 2.1 The recipe

Every lab in this part follows the same five steps.

```python
from tensorflow.keras import layers, models

# 1. Shape the data: (samples, height, width, channels), scaled to 0..1
X_train = X_train.reshape(-1, 28, 28, 1) / 255

# 2. Stack the layers
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax"),
])

# 3. Choose loss and optimizer
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 4. Train
history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

# 5. Evaluate on images the model has never seen
model.evaluate(X_test, y_test)
```

Things students trip over:

- **Four dimensions.** `Conv2D` expects a channel axis even for grayscale, hence the reshape to `(…, 28, 28, 1)`. Colour images have 3 channels.
- **Scaling.** Dividing by 255 puts pixels in 0..1; training converges faster.
- **Labels and loss must match.** Integer labels (`7`) go with `sparse_categorical_crossentropy`. One-hot labels from `to_categorical` (`[0,0,0,0,0,0,0,1,0,0]`) go with `categorical_crossentropy`. The labs use the second, `CNN-example` uses the first.
- **Softmax** turns the last layer into probabilities that sum to 1; the prediction is the `argmax`.

`model.summary()` prints the table from Part 1 §1.7. Check its numbers against the formulas.

## 2.2 The labs

Open the `*-lowCode` notebooks. Each trains a reference model, then gives you a second model with `#TODO` lines to complete. Answers are in [`solutions/`](../labs/02-keras-cnn/solutions).

| Notebook | Dataset | Images | Classes | Reference result in the notebook |
|---|---|---|---|---|
| [`CNN-example.ipynb`](../labs/02-keras-cnn/CNN-example.ipynb) | MNIST digits | 28×28 gray | 10 | 99.0% test accuracy after 5 epochs |
| [`cnn-1-lowCode.ipynb`](../labs/02-keras-cnn/cnn-1-lowCode.ipynb) | MNIST digits | 28×28 gray | 10 | 98.9% with 42,090 weights, 20 epochs |
| [`CNN-2-lowCode.ipynb`](../labs/02-keras-cnn/CNN-2-lowCode.ipynb) | CIFAR-100 | 32×32 colour | 100 | 33.3% with 95,684 weights, 30 epochs |
| [`CNN-3-lowCode.ipynb`](../labs/02-keras-cnn/CNN-3-lowCode.ipynb) | Fashion-MNIST | 28×28 gray | 10 | 84.2% without dropout, 82.2% with |

`CNN-example` also classifies your own handwriting: it loads an image from [`samples/`](../labs/02-keras-cnn/samples), converts it to 28×28 grayscale and predicts the digit. Draw a digit, photograph it and try.

## 2.3 What the results teach

**The same architecture is not equally good everywhere.** The three-conv network that scores 99% on MNIST scores 33% on CIFAR-100. Random guessing on 100 classes is 1%, so 33% is real learning, but small colour photos of 100 kinds of object are a much harder problem than centred digits. More filters, more data augmentation or a pretrained backbone are the usual next steps.

**Dropout is not automatically a win.** In Lab 3 the model with `Dropout(0.3)` after each of the first two convolutions scored about 2 points lower than the model without it after 20 epochs. Dropout slows learning in exchange for a smaller gap between training and test curves. Look at both curves, not only the final number.

**Read the curves.** Each lab plots training and test loss and accuracy per epoch:

| What you see | What it means | Try |
|---|---|---|
| Both curves still improving | Underfitting / not done | More epochs, more filters |
| Training keeps improving, test flattens or worsens | Overfitting | Dropout, augmentation, fewer weights |
| Both flat and low | Model too small or learning rate off | Bigger model, different optimizer |

## 2.4 Discussion

1. `Conv2D(16, (3,3))` on a 28×28×1 input has 160 weights. On a 32×32×3 CIFAR image it has 448. Why?
2. The first dense layer often holds as many weights as all the convolutions combined. Where in the summary table can you see that, and what would shrink it?
3. Your digit photo is misclassified. List three ways your photo differs from an MNIST image.

Next: [Part 3 — Computer vision on Azure](03-azure-vision.md)
