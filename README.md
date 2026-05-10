```markdown
# MNIST Digit Generator using DCGAN

A Deep Convolutional Generative Adversarial Network (DCGAN) built from scratch using TensorFlow that learns to generate realistic handwritten digits by training on the MNIST dataset.

---

## Results

### Generated Digits (After 50 Epochs)
<img width="329" height="328" alt="image" src="https://github.com/user-attachments/assets/d8c56a59-ad58-42c7-9924-c10e164ef958" />


### Loss Curves
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/d14c837f-7ad4-4fa7-aecc-e1311583ba0a" />


---

## FID Score

| Metric | Score |
|--------|-------|
| FID Score | **17.71** |
| Range for good MNIST GANs | 10 - 30 |

> Lower FID = better. Score of 17.71 indicates the generated images closely match the distribution of real MNIST digits.

---

## How it Works

A GAN consists of two neural networks competing against each other:

- **Generator** — Takes 100-dimensional random noise as input and learns to generate realistic 28×28 digit images through a series of Conv2DTranspose (upsampling) layers
- **Discriminator** — Takes a 28×28 image as input and learns to distinguish real MNIST digits from Generator's fakes through Conv2D (downsampling) layers

They are trained simultaneously in a minimax game:
- Discriminator tries to correctly classify real vs fake
- Generator tries to fool the Discriminator into calling its fakes real
- Over time, Generator gets so good that Discriminator can no longer tell the difference

---

## Model Architecture

### Generator
```
Input: Random noise (100,)
→ Dense(7×7×256) + BatchNorm + LeakyReLU
→ Reshape(7, 7, 256)
→ Conv2DTranspose(128, 5×5, stride=2) → (14, 14, 128) + BatchNorm + LeakyReLU
→ Conv2DTranspose(64,  5×5, stride=2) → (28, 28, 64)  + BatchNorm + LeakyReLU
→ Conv2DTranspose(1,   5×5, stride=1) → (28, 28, 1)   + Tanh
Output: Fake image (28, 28, 1)
```

### Discriminator
```
Input: Image (28, 28, 1)
→ Conv2D(64,  5×5, stride=2) + LeakyReLU + Dropout(0.3)
→ Conv2D(128, 5×5, stride=2) + LeakyReLU + Dropout(0.3)
→ Flatten
→ Dense(1, sigmoid)
Output: Probability of being real (0 to 1)
```

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MNIST (60,000 images) |
| Epochs | 50 |
| Batch Size | 256 |
| Optimizer | Adam (lr=1e-4) |
| Noise Dimension | 100 |
| Generator Loss | Binary Crossentropy |
| Discriminator Loss | Binary Crossentropy |

---

## Key Design Choices

- **BatchNormalization in Generator** — Stabilizes training by normalizing layer outputs. `use_bias=False` used because BatchNorm's beta parameter already acts as bias
- **LeakyReLU instead of ReLU** — Prevents dead neurons by allowing small negative gradients
- **Dropout in Discriminator** — Intentionally weakens Discriminator to keep competition balanced with Generator
- **tanh activation in final Generator layer** — Matches the [-1, 1] normalization of real images
- **Two separate GradientTapes** — Generator and Discriminator are updated independently with their own losses

---

## How to Run

### Requirements
```bash
pip install tensorflow matplotlib imageio scipy numpy
```

### Run in Google Colab
1. Open the notebook in Google Colab
2. Change runtime to GPU: `Runtime → Change runtime type → GPU`
3. Run all cells in order
4. Generated images will be saved after each epoch

---

## Project Structure

```
mnist-dcgan/
│
├── dcgan_mnist.ipynb     # Main notebook
├── gan_training.gif      # Training progression GIF
├── loss_plot.png         # Generator vs Discriminator loss curves
├── generated_digits.png  # Final generated digits
└── README.md             # This file
```
