"""
Training functions for medical image classification models.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import time
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from .model_builder import get_model
import sys
sys.path.append("..")
from data_processing.augmentation import get_augmentation_pipeline

class MedicalImageDataset(Dataset):
    """PyTorch Dataset for medical images."""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset.
        
        Args:
            image_paths: List of paths to images
            labels: List of labels (class indices)
            transform: PyTorch transforms for data augmentation
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label

def prepare_data_generators(data_dir, batch_size=32, img_size=(224, 224), validation_split=0.2, augmentation=True):
    """
    Prepare TensorFlow data generators for training and validation.
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        img_size: Image size (height, width)
        validation_split: Fraction of data to use for validation
        augmentation: Whether to use data augmentation
        
    Returns:
        Training and validation data generators
    """
    if augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Validation generator
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def train_tensorflow_model(model, train_generator, validation_generator, epochs=50, 
                         callbacks=None, output_dir='data/models'):
    """
    Train a TensorFlow model.
    
    Args:
        model: Compiled Keras model
        train_generator: Training data generator
        validation_generator: Validation data generator
        epochs: Number of epochs to train
        callbacks: List of Keras callbacks
        output_dir: Directory to save model checkpoints
        
    Returns:
        Training history
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up default callbacks if not provided
    if callbacks is None:
        model_checkpoint = ModelCheckpoint(
            os.path.join(output_dir, 'model_best_val_acc.h5'),
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        tensorboard = TensorBoard(
            log_dir=os.path.join(output_dir, 'logs'),
            histogram_freq=1
        )
        
        callbacks = [model_checkpoint, early_stopping, reduce_lr, tensorboard]
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(os.path.join(output_dir, 'model_final.h5'))
    
    return history

def train_pytorch_model(model, train_loader, val_loader, criterion=None, optimizer=None, 
                       epochs=50, device=None, output_dir='data/models'):
    """
    Train a PyTorch model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
        output_dir: Directory to save model checkpoints
        
    Returns:
        Training history as a dictionary
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to device
    model = model.to(device)
    
    # Set default criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Set default optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Initialize history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Initialize best validation accuracy
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        time_elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{epochs} | Time: {time_elapsed:.2f}s')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_best.pth'))
            print(f'Model saved with Val Acc: {val_acc:.4f}')
        
        print('-' * 60)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pth'))
    
    return history

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history (Keras history or dictionary)
        save_path: Path to save the plot
    """
    if isinstance(history, tf.keras.callbacks.History):
        # Convert Keras history to dictionary
        history = history.history
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['loss'], label='Train Loss')
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    if 'accuracy' in history:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    else:
        acc_key = 'train_acc'
        val_acc_key = 'val_acc'
        
    ax2.plot(history[acc_key], label='Train Accuracy')
    if val_acc_key in history:
        ax2.plot(history[val_acc_key], label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def train_model_pipeline(data_dir, model_name='resnet', framework='tensorflow', 
                        batch_size=32, img_size=(224, 224), epochs=50, 
                        output_dir='data/models', num_classes=2):
    """
    Complete pipeline for model training.
    
    Args:
        data_dir: Directory containing the dataset
        model_name: Name of the model to train
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        batch_size: Batch size for training
        img_size: Image size (height, width)
        epochs: Number of epochs to train
        output_dir: Directory to save model checkpoints
        num_classes: Number of output classes
        
    Returns:
        Trained model and training history
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    if framework == 'tensorflow':
        # Prepare data generators
        train_generator, validation_generator = prepare_data_generators(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            validation_split=0.2,
            augmentation=True
        )
        
        # Get model
        model = get_model(
            model_name=model_name,
            framework=framework,
            input_shape=(*img_size, 3),
            num_classes=num_classes
        )
        
        # Train model
        history = train_tensorflow_model(
            model=model,
            train_generator=train_generator,
            validation_generator=validation_generator,
            epochs=epochs,
            output_dir=output_dir
        )
        
        # Plot training history
        plot_training_history(
            history=history,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
        
        return model, history
    
    elif framework == 'pytorch':
        # Define transformations
        train_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Prepare dataset
        data_dir = Path(data_dir)
        all_classes = [d.name for d in data_dir.iterdir() if d.is_dir()]
        class_to_idx = {cls_name: i for i, cls_name in enumerate(all_classes)}
        
        image_paths = []
        labels = []
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                class_idx = class_to_idx[class_dir.name]
                for img_path in class_dir.glob('*.jpg') + class_dir.glob('*.png'):
                    image_paths.append(str(img_path))
                    labels.append(class_idx)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Create datasets
        train_dataset = MedicalImageDataset(X_train, y_train, transform=train_transform)
        val_dataset = MedicalImageDataset(X_val, y_val, transform=val_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Get model
        model = get_model(
            model_name=model_name,
            framework=framework,
            num_classes=num_classes
        )
        
        # Train model
        history = train_pytorch_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            output_dir=output_dir
        )
        
        # Plot training history
        plot_training_history(
            history=history,
            save_path=os.path.join(output_dir, 'training_history.png')
        )
        
        return model, history
    
    else:
        raise ValueError(f"Unknown framework: {framework}")