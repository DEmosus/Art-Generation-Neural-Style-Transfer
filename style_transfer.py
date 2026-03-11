import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

CONTENT_LAYERS = ["block5_conv2"]

STYLE_LAYERS = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]

CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2

IMAGE_SIZE = 400

# -------------------------------------------------
# IMAGE UTILITIES
# -------------------------------------------------


def load_image(path):
    img = Image.open(path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img)

    img = img.astype(np.float32) / 255.0
    img = img[None, :]
    return tf.constant(img)


def show(img):
    img = tf.squeeze(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# -------------------------------------------------
# GRAM MATRIX
# -------------------------------------------------


def gram_matrix(x):
    x = tf.reshape(x, (-1, x.shape[-1]))
    gram = tf.matmul(x, x, transpose_a=True)
    return gram / tf.cast(tf.shape(x)[0], tf.float32)


# -------------------------------------------------
# VGG FEATURE EXTRACTOR
# -------------------------------------------------


def build_vgg():

    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")

    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in STYLE_LAYERS + CONTENT_LAYERS]

    model = tf.keras.Model(vgg.input, outputs)

    return model


# -------------------------------------------------
# FEATURE EXTRACTION
# -------------------------------------------------


def extract_features(model, image):

    image = image * 255.0
    image = tf.keras.applications.vgg19.preprocess_input(image)

    outputs = model(image)

    style_outputs = outputs[: len(STYLE_LAYERS)]
    content_outputs = outputs[len(STYLE_LAYERS) :]

    style_features = [gram_matrix(x) for x in style_outputs]

    return style_features, content_outputs


# -------------------------------------------------
# LOSS FUNCTIONS
# -------------------------------------------------


def content_loss(content, generated):

    return tf.reduce_mean((generated - content) ** 2)


def style_loss(style, generated):

    return tf.reduce_mean((generated - style) ** 2)


def total_loss(style_targets, content_targets, style_outputs, content_outputs):

    style_l = 0
    content_l = 0

    for s, g in zip(style_targets, style_outputs):
        style_l += style_loss(s, g)

    for c, g in zip(content_targets, content_outputs):
        content_l += content_loss(c, g)

    style_l *= STYLE_WEIGHT
    content_l *= CONTENT_WEIGHT

    return style_l + content_l


# -------------------------------------------------
# STYLE TRANSFER CLASS
# -------------------------------------------------


class StyleTransfer:

    def __init__(self, content_path, style_path):

        self.model = build_vgg()

        self.content = load_image(content_path)
        self.style = load_image(style_path)

        self.generated = tf.Variable(self.content)

        style_targets, _ = extract_features(self.model, self.style)
        _, content_targets = extract_features(self.model, self.content)

        self.style_targets = style_targets
        self.content_targets = content_targets

        self.optimizer = tf.keras.optimizers.Adam(0.02)

    @tf.function
    def train_step(self):

        with tf.GradientTape() as tape:

            style_outputs, content_outputs = extract_features(
                self.model, self.generated
            )

            loss = total_loss(
                self.style_targets, self.content_targets, style_outputs, content_outputs
            )

        grad = tape.gradient(loss, self.generated)

        self.optimizer.apply_gradients([(grad, self.generated)])

        self.generated.assign(tf.clip_by_value(self.generated, 0.0, 1.0))

        return loss

    def train(self, epochs=2000):

        for i in range(epochs):

            loss = self.train_step()

            if i % 200 == 0:
                print(f"step {i} loss {loss.numpy():.4f}")
                show(self.generated)


# -------------------------------------------------
# RUN
# -------------------------------------------------

transfer = StyleTransfer("content.jpg", "style.jpg")

transfer.train(2000)
