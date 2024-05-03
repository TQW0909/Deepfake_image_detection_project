import argparse
import copy
import sys
import time
import itertools
import random
import pandas as pd
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
import torch.cuda as cuda 

from Utils.dataloader import *

from Models.baseline import BaselineCNN
from Models.model import *

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ExponentialLR

OPTS = None

loss_func = nn.BCEWithLogitsLoss()

# Dict of hyperparameters and the potential values to test
hyperparams = {
    'learning_rates': [1e-3, 1e-4],
    'batch_sizes': [32, 64],
    'weight_decays': [1e-4, 1e-3],
    'optimizers': ['SGD', 'Adam'],
    'schedulers': ['StepLR', 'ExponentialLR'],
}

# hyperparams = {
#     'learning_rates': [1e-2, 1e-1],
#     'batch_sizes': [32, 64],
#     'weight_decays': [1e-4, 1e-3],
#     'optimizers': ['SGD', 'Adam'],
#     'schedulers': ['StepLR', 'ExponentialLR'],
# }

# hyperparams = {
#     'learning_rates': [1e-4],
#     'batch_sizes': [64],
#     'weight_decays': [1e-3],
#     'optimizers': ['SGD'],
#     'schedulers': ['ExponentialLR'],
# }

device = None

# Enables the user to specify different functionaility
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['baseline_cnn', 'cnn'])
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--misclas', action='store_true')
    parser.add_argument('--GPU', action='store_true')
    parser.add_argument('testset', choices=['train', 'test', 'both'])
    return parser.parse_args()

# Plots the train and validation loss and accuracy for each epoch
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
    plt.savefig('plot_history.png')
    # plt.show()

# Plot the results of hyperparameter tuning (Used in midterm report)
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

# Trains the model on test set and tests on the validation set
def train(model, 
          train_loader, 
          dev_loader, 
          num_epochs=10, 
          patience=3,
          optimizer=None,
          scheduler=None):

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    start_time = time.time()

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
        # for inputs, labels in train_loader:
            inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
            inputs, labels = inputs.to(device), labels.to(device) # Enabling GPU
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
        
        # Validation loop (after each training epoch)
        model.eval()  # Set the model to evaluation mode
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():  # No need to track gradients
            for inputs, labels in  tqdm(dev_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
            # for inputs, labels in dev_loader:
                inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
                inputs, labels = inputs.to(device), labels.to(device) # Enabling GPU
                outputs = model(inputs)
                loss = loss_func(outputs.squeeze(), labels)  # Ensure dimensions match
                val_running_loss += loss.item()
                predicted = torch.sigmoid(outputs).round()  # For BCELoss, round outputs to 0 or 1
                total += labels.size(0)
                correct += (predicted.squeeze() == labels).sum().item()

        val_loss = val_running_loss / len(dev_loader)
        val_accuracy = 100 * correct / total

        # Store for graph plot
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        if scheduler:
            scheduler.step()

        print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}%, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}%')
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, OPTS.model + '1_best.pth')  # Save the best model
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
    
    # Load the best model state
    model.load_state_dict(torch.load(OPTS.model + '1_best.pth'))
    
    end_time = time.time()
    print(f'Training completed in {end_time - start_time:.2f} seconds.')

    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluate the model on the testset specified
def evaluate(model, test_loader, test_set):
    model.eval()

    # Evaluation loop
    total_loss = 0.0  # Total loss for all batches
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', unit='batch'):
        # for inputs, labels in test_loader:

            inputs, labels = inputs.float(), labels.float()  # Ensure data is the correct type
            inputs, labels = inputs.to(device), labels.to(device) # Enabling GPU

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

    print(f'Evaluation Loss for {test_set}: {avg_loss}, Accuracy: {accuracy}%')
    
    return avg_loss, accuracy


# Tunes hyperparamerters using the dict of values to find the best combination (simple grid-search approach)
def tune_hyperparameters(model, hyperparams, num_epochs=10, patience=3):
    best_val_accuracy = 0
    best_hyperparams = None
    results = []

    # Iterate over all combinations of hyperparameters
    for lr, batch_size, weight_decay, optimizer_name, scheduler_name in itertools.product(
        hyperparams['learning_rates'], 
        hyperparams['batch_sizes'], 
        hyperparams['weight_decays'],
        hyperparams['optimizers'],
        hyperparams['schedulers'],
    ):  
        # Re-initialize the model
        model = create_vgg16_model()
        nn.DataParallel(model)
        model.to(device)
        torch.cuda.empty_cache()
        
        # Train model
        print(f"\nTraining with Learning Rate: {lr}, Batch Size: {batch_size}, Weight Decay: {weight_decay}, Optimizer: {optimizer_name}, Scheduler: {scheduler_name}")

        # Create new instances of train and validation dataloaders with the new batch size
        train_loader = create_data_loader(train_data, batch_size, True)
        dev_loader = create_data_loader(dev_data, batch_size, False)

        # Initialize the optimizer
        if optimizer_name == 'SGD':
            optimizer = optim.SGD([
                {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr':lr},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr':lr * 0.1},
            ], momentum=0.9, weight_decay=weight_decay)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr':lr},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr':lr * 0.1},
            ], weight_decay=weight_decay)

        # Initialize the scheduler
        scheduler = None
        if scheduler_name == 'StepLR':
            scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
        elif scheduler_name == 'ExponentialLR':
            scheduler = ExponentialLR(optimizer, gamma=0.95)

        # Train and evaluate the model using the current hyperparameters
        _, (train_losses, val_losses, train_accuracies, val_accuracies) = train(model, train_loader=train_loader, dev_loader=dev_loader, optimizer=optimizer, scheduler=scheduler)

        val_accuracy = max(val_accuracies)

        # Store the results
        results.append({
            'learning_rate': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'optimizer': optimizer_name,
            'scheduler': scheduler_name,
            'val_accuracy': val_accuracy,
        })

        # Update the best hyperparameters
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_hyperparams = {
                'learning_rate': lr,
                'batch_size': batch_size,
                'weight_decay': weight_decay,
                'optimizer': optimizer_name,
                'scheduler': scheduler_name,
            }

    # Convert results to a df
    df = pd.DataFrame(results)

    # Output results as a table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Save the figure as an image
    plt.savefig("hyperparameters1.png", bbox_inches='tight')

    print(f"\nBest Hyperparameters: {best_hyperparams} with Validation Accuracy={best_val_accuracy}")
    return best_hyperparams


# Outputs 5 misclassified samples from a given set
def collect_misclassified(model, data_loader, dataset):
    model.eval()  # Set the model to evaluation mode.
    device = next(model.parameters()).device
    misclassified = []  # Store misclassified example data.

    with torch.no_grad():  # Turn off gradients for validation, saves memory and computations.
        for inputs, labels in data_loader:
            inputs, labels = inputs.float(), labels.float()  # Ensure data type consistency.
            inputs, labels = inputs.to(device), labels.to(device) # Enabling GPU

            outputs = model(inputs) 

            # Convert outputs to predicted labels
            predicted = torch.sigmoid(outputs).squeeze().round() 

            mismatches = predicted != labels

            if any(mismatches):
                misclassified_examples = inputs[mismatches]
                misclassified_labels = labels[mismatches]
                misclassified_preds = predicted[mismatches]
                
                for example, label, pred in zip(misclassified_examples, misclassified_labels, misclassified_preds):
                    example = example.cpu().numpy()
                    label = label.item()
                    pred = pred.item()
                    misclassified.append((example, label, pred))                
    
    misclassified_samples = random.sample(misclassified, min(5, len(misclassified))) # Can change to more examples if needed

    # Create figure to display examples
    plt.figure(figsize=(10, 10))
    for i, (image, true_label, pred_label) in enumerate(misclassified_samples):
        plt.subplot(5, 5, i + 1)  
        image = image.transpose(1, 2, 0)
        plt.imshow(image)
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')
    # plt.show()
    plt.savefig('missclass_' + dataset + '.png')


    return misclassified


def main():
    # Set random seed, for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)  # For CUDA

    global device

    if torch.cuda.is_available():
        if OPTS.GPU:
            device = torch.device("cuda")

            # Check how many GPUs are available
            num_gpus = torch.cuda.device_count()

            # Display each GPU's name
            for i in range(num_gpus):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

            print(f"Total GPUs available: {num_gpus}")  
        else:
            device = torch.device("cpu")
            print("GPU is available but not being used by choice.")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU.")


    # Train model
    if OPTS.model == 'baseline_cnn':
        model = BaselineCNN()
    elif OPTS.model == 'cnn':
        model = create_vgg16_model()

    nn.DataParallel(model) # For multi-GPU
    model.to(device)
    
    if OPTS.tune:
        tune_hyperparameters(model, hyperparams)
    else:
        if OPTS.model == 'baseline_cnn':
            optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.001)

            scheduler = None

        elif OPTS.model == 'cnn':

            # Adjusting the optimizer for fine-tuning with different learning rates:
            optimizer = optim.Adam([
                {'params': filter(lambda p: p.requires_grad, model.classifier.parameters()), 'lr': 0.0001},
                {'params': filter(lambda p: p.requires_grad, model.features.parameters()), 'lr': 0.0001 * 0.1},
            ], weight_decay=0.0001)

            scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        model, data = train(model, train_loader=train_loader, dev_loader=dev_loader, optimizer=optimizer, scheduler=scheduler)

        # Evaluate the model
        print('\nEvaluating final model:')

        if OPTS.test:
            if OPTS.testset == 'train':
                test_loss_trainset, test_acc_trainset = evaluate(model, test_loader=test_loader, test_set="deepfake_and_real_images (TRAIN)")
            elif OPTS.testset == 'test':
                test_loss_testset, test_acc_testset = evaluate(model, test_loader=deepfake_faces_test_loader, test_set="deepfake_faces_test (TEST)")
            elif OPTS.testset == 'both':
                test_loss_trainset, test_acc_trainset = evaluate(model, test_loader=test_loader, test_set="deepfake_and_real_images (TRAIN)")
                test_loss_testset, test_acc_testset = evaluate(model, test_loader=deepfake_faces_test_loader, test_set="deepfake_faces_test (TEST)")

        # Plot history
        if OPTS.plot:
            train_losses, val_losses, train_accuracies, val_accuracies = data
            plot_history(train_losses, val_losses, train_accuracies, val_accuracies)

    # Show misclassified examples
    if OPTS.misclas:
        misclassified = collect_misclassified(model, deepfake_faces_test_loader, 'test')
        misclassified = collect_misclassified(model, test_loader, 'train')


if __name__ == '__main__':
    OPTS = parse_args()

    main()
    