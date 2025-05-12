# File: KM4/modular-toolkit/PreBuildModel/MNIST_MLP.py

import torch
import torch.nn as nn

# Helper module to flatten the image tensor
class Flatten(nn.Module):
    """
    Flattens the input tensor excluding the batch dimension.
    Used to convert image grids (e.g., 1x28x28) into vectors for MLP input.
    """
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        # output shape: (batch_size, channels * height * width)
        return x.view(x.size(0), -1)

class MNIST_MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) designed for MNIST image classification.
    It flattens the input image and passes it through linear layers with ReLU activations.
    """
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        """
        Initializes the MNIST MLP model.

        Args:
            input_size (int): The total number of pixels in the flattened input image (e.g., 28 * 28 = 784).
            hidden_sizes (list[int]): A list containing the number of neurons in each hidden layer.
            output_size (int): The number of output classes (e.g., 10 for digits 0-9).
            **kwargs: Additional keyword arguments (currently unused but included for potential future flexibility).
        """
        super().__init__(**kwargs) # Pass kwargs to the parent nn.Module constructor

        # Start with the Flatten layer
        layers = [Flatten()]

        # Dynamically create hidden layers
        layer_sizes = [input_size] + hidden_sizes # Combine input size and hidden layer sizes
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            layers.append(nn.ReLU())
            # Optional: Add Dropout here if needed: layers.append(nn.Dropout(p=0.5))

        # Add the final output layer
        layers.append(nn.Linear(layer_sizes[-1], output_size))

        # Note: nn.CrossEntropyLoss (commonly used for classification) includes the Softmax activation,
        # so we don't typically add a Softmax layer here.

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor, expected shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: The output tensor (logits), shape (batch_size, output_size).
        """
        return self.network(x)

# Example of how to instantiate (for testing, not needed for the toolkit itself here)
# if __name__ == '__main__':
#     # MNIST parameters
#     input_dim = 28 * 28  # 784
#     hidden_dims = [128, 64]
#     output_dim = 10
#
#     # Create model instance
#     model = MNIST_MLP(input_size=input_dim, hidden_sizes=hidden_dims, output_size=output_dim)
#     print(model)
#
#     # Create a dummy input batch (e.g., 4 images)
#     dummy_input = torch.randn(4, 1, 28, 28) # (batch_size, channels, height, width)
#     output = model(dummy_input)
#     print("Output shape:", output.shape) # Should be (4, 10)
