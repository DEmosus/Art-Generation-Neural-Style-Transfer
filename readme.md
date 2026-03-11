# Neural Style Transfer with TensorFlow (VGG19)

## Project Overview

This project implements **Neural Style Transfer (NST)** using **TensorFlow and a pretrained VGG19 network**. The system generates a new image that combines:

- The **content structure** of one image
- The **artistic style** of another image

The technique follows the approach introduced in the paper:

**"A Neural Algorithm of Artistic Style" – Gatys et al., 2015**

Instead of training a neural network to generate stylized images directly, this method **optimizes the pixels of an image** so that it simultaneously matches:

- the **content representation** of the content image
- the **style representation** of the style image

---

# Example Concept

Neural Style Transfer attempts to solve the following problem:

**Given:**

- Content Image → photograph
- Style Image → artwork or painting

**Generate:**

- An image that looks like the **content photo painted in the style artwork**

Mathematically the optimization objective is:

$$
J*{total} = \alpha J*{content} + \beta J\_{style}
$$

Where:

- $(J\_{content})$ = content reconstruction loss
- $(J\_{style})$ = style reconstruction loss
- $(\alpha)$ = content weight
- $(\beta)$ = style weight

---

# Key Features

- Implementation using **TensorFlow 2**
- Uses **VGG19 pretrained on ImageNet**
- Extracts **multi-layer style representations**
- Uses **Gram matrices** for style encoding
- Implements **gradient-based image optimization**
- Structured for **learning and experimentation**

---

# Project Structure

```text
neural-style-transfer/
│
├── content.jpg
├── style.jpg
│
├── Art_Generation_Neural_Style_Transfer.ipynb
├── style_transfer.py
│
├── README.md
│
└── outputs/
├── step_200.png
├── step_400.png
└── final_output.png
```

### Description

| File                                         | Description                                 |
| -------------------------------------------- | ------------------------------------------- |
| `Art_Generation_Neural_Style_Transfer.ipynb` | Jupyter notebook with detailed explanations |
| `style_transfer.py`                          | Standalone implementation                   |
| `content.jpg`                                | Input image providing structure             |
| `style.jpg`                                  | Image providing artistic style              |
| `outputs/`                                   | Generated stylized images                   |

---

# Neural Style Transfer Architecture

The pipeline follows these steps:

1. Load **content image**
2. Load **style image**
3. Pass images through **pretrained VGG19**
4. Extract **feature maps**
5. Compute **Gram matrices for style**
6. Initialize **generated image**
7. Optimize image using **gradient descent**

The generated image gradually evolves to minimize:

- Content difference
- Style difference

---

# Content Representation

Content is extracted from **deep convolutional layers** of VGG19.

Example layer used:
`block5_conv2`

Content loss is defined as:

$$
J\_{content}(C,G) =
\frac{1}{2} \sum (F_G - F_C)^2
$$

Where:

- $(F_C)$ = content image features
- $(F_G)$ = generated image features

Deep layers capture **object structure and layout**, making them ideal for content representation.

---

# Style Representation

Style is captured using **Gram matrices of feature activations**.

Style layers used:

```text
block1_conv1
block2_conv1
block3_conv1
block4_conv1
block5_conv1
```

Gram matrix:

$$
G = F^T F
$$

Where:

- $(F)$ = feature map matrix
- $(G)$ = correlation matrix

The Gram matrix captures:

- texture
- color patterns
- repeated artistic motifs

Style loss compares Gram matrices:

$$
J\_{style} = \sum_l w_l ||G_l^S - G_l^G||^2
$$

---

# Loss Function

The optimization objective combines style and content loss:

$$
J*{total} = \alpha J*{content} + \beta J\_{style}
$$

Hyperparameters used:

| Parameter      | Value       |
| -------------- | ----------- |
| CONTENT_WEIGHT | $(10^4)$    |
| STYLE_WEIGHT   | $(10^{-2})$ |

Higher content weight preserves the original structure more strongly.

---

# Optimization Process

Unlike traditional neural networks, this method **does not train a model**.

Instead:

- The **generated image itself is optimized**.

Gradient descent updates the image pixels:

$$
I\_{t+1} = I_t - \eta \nabla J
$$

Where:

- $(I)$ = generated image
- $(\eta)$ = learning rate

Optimization is performed using the **Adam optimizer**.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/neural-style-transfer.git
cd neural-style-transfer
```

Install dependencies:

```bash
pip install tensorflow numpy pillow matplotlib
```

# Running the Project

Place your images in the project directory:

```bash
content.jpg
style.jpg
```

Run the notebook:

```bash
jupyter notebook style_transfer.ipynb
```

or run the Python script:

```bash
python style_transfer.py
```

# Training Output

During optimization, the generated image is updated every few iterations.

Example console output:

```bash
step 0 loss 12345.231
step 200 loss 4567.321
step 400 loss 2012.433
...
```

Intermediate images show how style gradually appears while preserving structure.

# Customization

Users can experiment with:

#### Changing Style Layers

```bash
STYLE_LAYERS = [...]
```

Different layers emphasize different texture scales.

#### Adjusting Weights

```bash
CONTENT_WEIGHT = 1e4
STYLE_WEIGHT = 1e-2
```

Increasing style weight produces stronger artistic effects.

#### Image Resolution

```bash
IMAGE_SIZE = 400
```

Higher resolution produces better images but increases computation time.

# Results

Neural style transfer can produce images that resemble:

- Oil paintings
- Watercolor art
- Impressionist styles
- Abstract textures

The quality depends heavily on:

- style image complexity
- number of iterations
- hyperparameter tuning

# Limitations

Classic neural style transfer has several limitations:

- Slow optimization (hundreds to thousands of iterations)
- High memory usage
- Not real-time

Modern approaches address this with:

- Feedforward style networks
- GAN-based stylization
- Diffusion-based generative models

# Learning Goals of This Project

This project helps understand several important deep learning concepts:

- Feature extraction using pretrained CNNs
- Representing textures with Gram matrices
- Image optimization using gradients
- Multi-objective loss functions
- Transfer learning in computer vision

# References

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015)  
_A Neural Algorithm of Artistic Style_

Available at:  
https://arxiv.org/abs/1508.06576

# Key Takeaways

- Neural style transfer separates content from style
- CNN feature activations encode rich visual information
- Style can be represented using feature correlations
- Gradient optimization can generate artistic images

This project provides a clear, educational implementation suitable for learning deep learning concepts in computer vision and generative modeling.
