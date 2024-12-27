import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import argparse
import os
import logging
import sys
from PIL import ImageFile
import time

# Enable loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging for real-time progress tracking
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set log level to DEBUG for detailed output
logger.addHandler(logging.StreamHandler(sys.stdout))  # Output logs to the console

def test(model, test_loader, criterion, device):
    """
    Evaluate the model on the test dataset and log the performance.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
        criterion (torch.nn.Module): Loss function to evaluate the model.
        device (torch.device): Device to run the model on (CPU or GPU).
    
    Returns:
        total_loss (float): Average loss on the test set.
        total_acc (float): Accuracy on the test set.
    """
    logger.info("Testing started!")  # Log the start of testing
    model.eval()  # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
    
    running_loss = 0  # Variable to accumulate loss
    running_corrects = 0  # Variable to accumulate correct predictions

    with torch.no_grad():  # Disable gradient calculation during testing
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # Compute loss
            
            # Update running metrics
            running_loss += loss.item()  # Accumulate loss
            _, preds = torch.max(outputs, 1)  # Get predicted labels
            running_corrects += torch.sum(preds == labels.data)  # Count correct predictions

    # Compute average loss and accuracy
    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects.double().item() / len(test_loader.dataset)

    # Log the results
    logger.info(f"Testing Loss: {total_loss:.4f}")
    logger.info(f"Testing Accuracy: {total_acc:.4f}")
    
    return total_loss, total_acc

def train(model, train_loader, validation_loader, criterion, optimizer, device, batch_size):
    """
    Train the model on the training dataset and validate it on the validation dataset.
    
    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
        validation_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer to update the model's weights.
        device (torch.device): Device to run the model on (CPU or GPU).
        batch_size (int): The number of samples per batch.
    
    Returns:
        model (torch.nn.Module): The trained model.
    """
    logger.info("Training started!")  # Log the start of training
    epochs = 2  # Set the number of epochs (2 for tuning job, change as needed)
    best_loss = float('inf')  # Initialize best validation loss to a large value
    patience = 3  # Early stopping patience (when validation loss stops improving)
    loss_counter = 0  # Counter for early stopping

    # Dictionary to handle train and validation datasets
    dataset_phases = {'train': train_loader, 'valid': validation_loader}
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)  # Learning rate scheduler

    for epoch in range(1, epochs + 1):
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set the model to training mode
            else:
                model.eval()  # Set the model to evaluation mode

            running_loss = 0  # Variable to accumulate loss
            running_corrects = 0  # Variable to accumulate correct predictions

            for batch_idx, (inputs, labels) in enumerate(dataset_phases[phase]):
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to the appropriate device

                with torch.set_grad_enabled(phase == 'train'):  # Enable gradients only in training phase
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)  # Compute the loss

                    if phase == 'train':
                        optimizer.zero_grad()  # Zero the gradients
                        loss.backward()  # Backpropagate the loss
                        optimizer.step()  # Update the model's weights

                # Update running metrics
                running_loss += loss.item()  # Accumulate loss
                _, preds = torch.max(outputs, 1)  # Get predicted labels
                running_corrects += torch.sum(preds == labels.data)  # Count correct predictions

                # Log progress every batch
                processed_images_count = batch_idx * batch_size + len(inputs)
                logger.info(f"{phase.capitalize()} epoch: {epoch} [{processed_images_count}/{len(dataset_phases[phase].dataset)} ({100.0 * processed_images_count / len(dataset_phases[phase].dataset):.0f}%)] Loss: {loss.item():.6f}")

            # Compute epoch-level statistics
            epoch_loss = running_loss / len(dataset_phases[phase])
            epoch_acc = running_corrects.double().item() / len(dataset_phases[phase].dataset)

            logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

            if phase == 'valid':
                if epoch_loss < best_loss:  # Update best loss if validation loss improves
                    best_loss = epoch_loss
                    loss_counter = 0  # Reset the loss counter
                else:
                    loss_counter += 1  # Increment the loss counter if validation loss worsens

        scheduler.step()  # Update the learning rate
        if loss_counter >= patience:  # Trigger early stopping if the validation loss stops improving
            logger.info("Early stopping triggered!")
            break

    return model

def net():
    """
    Initializes and returns a pre-trained ResNet-50 model for transfer learning.
    
    Returns:
        model (torch.nn.Module): The initialized model.
    """
    output_size = 133  # Number of classes in the output layer (e.g., dog breeds)
    model = models.resnet50(pretrained=True)  # Load the pre-trained ResNet-50 model

    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers except the final fully connected layer

    # Modify the final fully connected layer for the classification task
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, output_size)
    )
    return model

def create_data_loaders(data, batch_size):
    """
    Creates and returns data loaders for training, validation, and test datasets.
    
    Args:
        data (str): Path to the dataset directory.
        batch_size (int): The number of samples per batch.
    
    Returns:
        tuple: Three DataLoader objects for training, validation, and testing.
    """
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path = os.path.join(data, 'valid')

    # Define transformations for training and test datasets
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)

    # Create data loaders for batching
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, validation_loader, test_loader

def main(args):
    """
    Main function to train, validate, test the model, and save it.
    
    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    # Initialize the model
    model = net()

    # Set device for training (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Enable multi-GPU training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)  # Transfer the model to the chosen device

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.module.fc.parameters() if torch.cuda.device_count() > 1 else model.fc.parameters(),
        lr=args.learning_rate
    )

    # Load data
    train_loader, validation_loader, test_loader = create_data_loaders(args.data, args.batch_size)

    # Train the model
    start_time = time.time()
    model = train(model, train_loader, validation_loader, criterion, optimizer, device, args.batch_size)
    logger.info(f"Training completed in {time.time() - start_time:.2f} seconds.")

    # Test the model
    start_time = time.time()
    test_loss, test_acc = test(model, test_loader, criterion, device)
    logger.info(f"Testing completed in {time.time() - start_time:.2f} seconds.")
    logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    model_path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()

    # Run the main function
    main(args)
