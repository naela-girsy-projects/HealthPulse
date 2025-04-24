"""
Model architectures for medical image classification.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, applications, regularizers
import torch
import torch.nn as nn
import torchvision.models as torch_models

def create_custom_cnn(input_shape=(224, 224, 3), num_classes=2):
    """
    Create a custom CNN model.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_resnet_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet'):
    """
    Create a ResNet50 model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        
    Returns:
        Compiled Keras model
    """
    base_model = applications.ResNet50(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet'):
    """
    Create an EfficientNetB3 model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        
    Returns:
        Compiled Keras model
    """
    base_model = applications.EfficientNetB3(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.BatchNormalization(),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_vgg_model(input_shape=(224, 224, 3), num_classes=2, weights='imagenet'):
    """
    Create a VGG16 model with transfer learning.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        weights: Pre-trained weights ('imagenet' or None)
        
    Returns:
        Compiled Keras model
    """
    base_model = applications.VGG16(
        weights=weights,
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

# PyTorch Models
class ResNetModel(nn.Module):
    """ResNet model using PyTorch."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNetModel, self).__init__()
        self.model = torch_models.resnet50(pretrained=pretrained)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class EfficientNetModel(nn.Module):
    """EfficientNet model using PyTorch."""
    
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = torch_models.efficientnet_b3(pretrained=pretrained)
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def get_model(model_name, framework='tensorflow', **kwargs):
    """
    Factory function to get a model based on name.
    
    Args:
        model_name: Name of the model ('custom_cnn', 'resnet', 'efficientnet', 'vgg')
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        **kwargs: Additional arguments for model creation
        
    Returns:
        Model instance
    """
    if framework == 'tensorflow':
        if model_name == 'custom_cnn':
            return create_custom_cnn(**kwargs)
        elif model_name == 'resnet':
            return create_resnet_model(**kwargs)
        elif model_name == 'efficientnet':
            return create_efficientnet_model(**kwargs)
        elif model_name == 'vgg':
            return create_vgg_model(**kwargs)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    elif framework == 'pytorch':
        if model_name == 'resnet':
            return ResNetModel(**kwargs)
        elif model_name == 'efficientnet':
            return EfficientNetModel(**kwargs)
        else:
            raise ValueError(f"Unknown model name for PyTorch: {model_name}")
    
    else:
        raise ValueError(f"Unknown framework: {framework}")