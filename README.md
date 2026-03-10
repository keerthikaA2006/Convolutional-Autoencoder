## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
### Problem Statement:
The objective of this project is to design and implement a convolutional autoencoder using PyTorch for image denoising. The model aims to remove noise from handwritten digit images while preserving important visual details. By learning compressed feature representations, the autoencoder reconstructs clean images from noisy inputs, improving image clarity and quality. This experiment demonstrates the ability of deep learning models to perform unsupervised feature learning and noise reduction effectively, which can be extended to real-world applications such as image restoration, medical imaging, and preprocessing for computer vision tasks.

### Dataset:
The MNIST dataset is used for this project. It contains 70,000 grayscale images of handwritten digits (0–9), each of size 28×28 pixels. The dataset is divided into 60,000 training images and 10,000 test images. Each image is normalized and converted into a tensor before being used. Gaussian noise is artificially added to the images to simulate noisy conditions, allowing the autoencoder to learn how to reconstruct the clean versions. MNIST is widely used for benchmarking image classification and reconstruction tasks due to its simplicity and well-defined structure.

## DESIGN STEPS

### STEP 1:
Problem Definition: Build a convolutional autoencoder to remove noise from MNIST handwritten digit images.

### STEP 2:
Data Preprocessing: Load MNIST dataset, convert images to tensors, and add Gaussian noise.

### STEP 3:
Model Design: Create an encoder–decoder architecture using Conv2D and ConvTranspose2D layers with ReLU and Sigmoid activations.

### STEP 4:
Model Compilation: Move model to device, use MSELoss for reconstruction, and Adam optimizer for weight updates.

### STEP 5:
Training: Train the model with noisy inputs and clean targets to minimize reconstruction loss.

### STEP 6:
Evaluation: Test the model on unseen data and visualize original, noisy, and denoised images.

### STEP 7:
Result Analysis: The autoencoder effectively reduces noise and reconstructs clean digit images.

## PROGRAM
### Name: keerthika A
### Register Number:212224220048

```py
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # [1,28,28] -> [32,14,14]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [32,14,14] -> [64,7,7]
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # [64,7,7] -> [32,14,14]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # [32,14,14] -> [1,28,28]
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```

```py
# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()               # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)
summary(model, input_size=(1, 28, 28))
```

```py
# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)   # Compare denoised output vs original clean image
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
```

```py
# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print("Name: keerthika A")
    print("Register Number: 212224220048 ")
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```


## OUTPUT

### Model Summary

<img width="715" height="501" alt="Screenshot 2026-03-10 155456" src="https://github.com/user-attachments/assets/6033a0c8-36bc-47db-82b5-2b584fdb15f7" />

### Original vs Noisy Vs Reconstructed Image


<img width="1741" height="723" alt="Screenshot 2026-03-10 155535" src="https://github.com/user-attachments/assets/a50907ac-0b81-4446-bdeb-717f3c156f0e" />


## RESULT
The autoencoder successfully denoised the images, accurately reconstructing clean handwritten digits from noisy inputs.

