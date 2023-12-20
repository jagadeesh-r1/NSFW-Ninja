import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.autograd import Variable

def fgsm_attack(image, epsilon, gradient):
    # Collect the element-wise sign of the data gradient
    perturbed_image = image + epsilon * gradient.sign()
    # Clip the perturbed image to ensure it stays within valid pixel intensity range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adversarial_example(model, original_image, grad_cam_map, target_class, epsilon=0.01):
    model.eval()

    # Convert the original image and Grad-CAM map to PyTorch variables
    original_image = Variable(original_image, requires_grad=True)
    grad_cam_map = Variable(grad_cam_map)

    # Zero out the existing gradients on the original image
    model.zero_grad()

    # Forward pass to obtain predictions
    output = model(original_image)
    # Get the score for the target class
    target_score = output[0, target_class]

    # Backward pass to compute gradients with respect to Grad-CAM map
    target_score.backward(retain_graph=True)

    # Extract gradients from the image
    gradient = original_image.grad.data

    # Use FGSM to generate the adversarial example
    perturbed_image = fgsm_attack(original_image, epsilon, gradient)

    return perturbed_image.data

# Example usage
# Assume you have a PyTorch model named "model"
# Assume grad_cam_map and original_image are your Grad-CAM outputs and original images
# Ensure your model and input images are in the appropriate format and preprocessed accordingly

# Set the target class for the adversarial attack
target_class = 0  # Change this to your target class

# Generate adversarial example
perturbed_image = generate_adversarial_example(model, original_image, grad_cam_map, target_class)

# Evaluate the model on the perturbed image to check if the attack is successful
model.eval()
output_perturbed = model(perturbed_image)
predicted_class_perturbed = torch.argmax(output_perturbed).item()

print(f"Original Prediction: {torch.argmax(output).item()}")
print(f"Perturbed Prediction: {predicted_class_perturbed}")
