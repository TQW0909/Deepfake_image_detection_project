import argparse
import copy
import sys
import time
import itertools
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy

from Utils.dataloader import *

from Models.baseline import BaselineCNN
from Models.model import vgg16

from torch.utils.tensorboard import SummaryWriter

OPTS = None

loss_func = nn.BCEWithLogitsLoss()

hyperparams = {
    'learning_rates': [1e-1, 1e-2, 1e-3],
    'batch_sizes': [16, 32, 64]
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['baseline_cnn', 'cnn'])
    parser.add_argument('--learning-rate', '-r', type=float, default=1e-1)
    parser.add_argument('--batch-size', '-b', type=int, default=32)
    parser.add_argument('--num-epochs', '-T', type=int, default=30) 
    parser.add_argument('--hidden-dim', '-i', type=int, default=200)
    parser.add_argument('--dropout-prob', '-p', type=float, default=0.0)
    parser.add_argument('--cnn-num-channels', '-c', type=int, default=5)
    parser.add_argument('--cnn-kernel-size', '-k', type=int, default=3)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--misclas', action='store_true')
    return parser.parse_args()

def plot_history(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_tuning_results(results):
    # Extract data for plotting
    learning_rates = [r['learning_rate'] for r in results]
    batch_sizes = [r['batch_size'] for r in results]
    accuracies = [r['val_accuracy'] for r in results]

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(learning_rates, batch_sizes, accuracies, c=accuracies, cmap='viridis', depthshade=True)
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Batch Size')
    ax.set_zlabel('Validation Accuracy')
    plt.show()


def train(model, 
          train_loader, 
          dev_loader, 
          num_epochs=10, 
          patience=3):

    writer = SummaryWriter('runs/' + OPTS.model)
    sample_inputs = torch.randn(1, 3, 256, 256)  # Adjust the shape based on your model
    writer.add_graph(model, sample_inputs)

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    start_time = time.time()
    # loss_func = nn.BCEWithLogitsLoss()
    if OPTS.model == 'baseline_cnn':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-2)
    elif OPTS.model == 'cnn':
        base_params = list(map(id, model.features.parameters()))  # Parameters of the pre-trained features
        new_params = filter(lambda p: id(p) not in base_params, model.parameters())  # Newly added parameters

        # Adjusting the optimizer for fine-tuning with different learning rates:
        optimizer = optim.SGD([
            {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': 0.01},
            {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 0.01 * 0.1},
        ], momentum=0.9, weight_decay=1e-2)

    
    # Initialize variables for early stopping
    best_loss = float('inf')
    best_model = None
    epochs_no_improve = 0

    total_train = 0
    correct_train = 0

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs.squeeze(), labels)  # Ensure dimensions match between outputs and labels
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate training accuracy
            predicted_train = torch.sigmoid(outputs).squeeze().round()  # Convert logits to probabilities and then to binary predictions
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

        # Calculate average losses
        train_loss = running_loss / len(train_loader)

        train_accuracy = 100 * correct_train / total_train
        
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Validation loop (after each training epoch)
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients
            for inputs, labels in  tqdm(dev_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
                inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
                outputs = model(inputs)
                loss = loss_func(outputs.squeeze(), labels)  # Ensure dimensions match
                val_running_loss += loss.item()
                predicted = torch.sigmoid(outputs).round()  # For BCELoss, round outputs to 0 or 1
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_loss = val_running_loss / len(dev_loader)
        val_accuracy = 100 * correct / total

        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/validation', val_accuracy, epoch)

        # Store for graph plot
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)


        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
    
    # Load the best model state
    model.load_state_dict(best_model)
    
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds.')

    writer.close()

    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

def evaluate(model, test_loader):
    model.eval()

    # Evaluation loop
    total_loss = 0.0  # Total loss for all batches
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', unit='batch'):

            inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
            outputs = model(inputs)

            # Compute the loss for this batch and add to total loss
            loss = loss_func(outputs.squeeze(), labels)  # Ensure dimensions match between outputs and labels
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size to get absolute batch loss

            # Calculate accuracy
            predicted = torch.sigmoid(outputs).squeeze().round()  # Convert logits to probabilities then to binary predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = 100 * correct / total

    print(f'Evaluation Loss: {avg_loss}, Accuracy: {accuracy}%')
    
    return avg_loss, accuracy

def tune_hyperparameters(model, hyperparams, num_epochs=10, patience=3):
    best_val_accuracy = 0
    best_hyperparams = None
    results = []

    # Iterate over all combinations of hyperparameters
    for lr, batch_size in itertools.product(hyperparams['learning_rates'], hyperparams['batch_sizes']):
        print(f"\nTraining with learning rate: {lr} and batch size: {batch_size}")

        # Create new instances of train and validation dataloaders with the new batch size
        train_loader = create_data_loader(train_data, batch_size, True)  # shuffle=True for training data
        dev_loader = create_data_loader(dev_data, batch_size, False)     # shuffle=False for validation data

        # Define the optimizer with the current hyperparameters
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)

        # Train the model using the current set of hyperparameters
        _, data = train(model, train_loader, dev_loader, num_epochs, patience)

        # Unpack the data for the current training session
        train_losses, val_losses, train_accuracies, val_accuracies = data

        # Determine the best validation accuracy in the current training session
        max_val_accuracy = max(val_accuracies)

        # Store the results
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'val_accuracy': max_val_accuracy
        })

        # Update the best hyperparameters if the current model is better
        if max_val_accuracy > best_val_accuracy:
            best_val_accuracy = max_val_accuracy
            best_hyperparams = {'learning_rate': lr, 'batch_size': batch_size}

    # Plot the results
    plot_tuning_results(results)

    print(f"\nBest hyperparameters: Learning Rate={best_hyperparams['learning_rate']}, Batch Size={best_hyperparams['batch_size']} with Validation Accuracy={best_val_accuracy}")
    return best_hyperparams

def collect_misclassified(model, data_loader):
    model.eval()  # Set the model to evaluation mode.
    misclassified = []  # Store misclassified example data.

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations.
        for inputs, labels in data_loader:
            inputs, labels = inputs.float(), labels.float()  # Ensure data type consistency.
            outputs = model(inputs)  # Get model outputs.

            # Convert outputs to predicted labels
            predicted = torch.sigmoid(outputs).squeeze().round()  # Adjust this based on your output format.

            # Compare predictions to true label
            mismatches = predicted != labels
            if any(mismatches):
                misclassified_examples = inputs[mismatches]
                misclassified_labels = labels[mismatches]
                misclassified_preds = predicted[mismatches]
                
                for example, label, pred in zip(misclassified_examples, misclassified_labels, misclassified_preds):
                    misclassified.append((example, label.item(), pred.item()))  # Store the misclassified example and the true/predicted labels.
                
    
    misclassified_samples = random.sample(misclassified, min(5, len(misclassified)))

    plt.figure(figsize=(10, 10))
    for i, (image, true_label, pred_label) in enumerate(misclassified_samples):
        plt.subplot(5, 5, i + 1)  # Adjust grid dimensions based on sample size.
        image = image.permute(1, 2, 0)  # Rearrange dimensions from (C, H, W) to (H, W, C) if necessary.
        plt.imshow(image)
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')
    plt.show()


    return misclassified


def main():
    # Set random seed, for reproducibility
    torch.manual_seed(0)

    # Train model
    if OPTS.model == 'baseline_cnn':
        model = BaselineCNN()
    elif OPTS.model == 'cnn':
        model = vgg16
    
    if OPTS.tune:
        tune_hyperparameters(model, hyperparams)
    else:
        model, data = train(model, train_loader=train_loader, dev_loader=dev_loader)

        # Evaluate the model
        print('\nEvaluating final model:')
        # train_acc = evaluate(model, X_train, y_train, 'Train')
        # dev_acc = evaluate(model, X_dev, y_dev, 'Dev')
        if OPTS.test:
            test_acc = evaluate(model, test_loader=test_loader)

        # Unpack the history and plot
        if OPTS.plot:
            train_losses, val_losses, train_accuracies, val_accuracies = data
            plot_history(train_losses, val_losses, train_accuracies, val_accuracies)

    if OPTS.misclas:
        misclassified = collect_misclassified(model, dev_loader)


if __name__ == '__main__':
    OPTS = parse_args()

    main()
    