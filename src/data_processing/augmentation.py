"""
Data augmentation functions for medical images.
"""
import albumentations as A
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def get_augmentation_pipeline(severity='medium'):
    """
    Create an augmentation pipeline with albumentations.
    
    Args:
        severity: Severity of augmentations ('light', 'medium', 'heavy')
        
    Returns:
        Albumentations transformation pipeline
    """
    if severity == 'light':
        return A.Compose([
            A.RandomRotate90(p=0.2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
        ])
    
    elif severity == 'medium':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        ])
    
    elif severity == 'heavy':
        return A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=20, max_width=20, p=0.3),
        ])
    
    else:
        raise ValueError(f"Unknown severity: {severity}")

def get_tf_data_generator(augmentation_severity='medium'):
    """
    Create a TensorFlow/Keras ImageDataGenerator for augmentation.
    
    Args:
        augmentation_severity: Severity of augmentations
        
    Returns:
        ImageDataGenerator instance
    """
    if augmentation_severity == 'light':
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            rescale=1./255
        )
    
    elif augmentation_severity == 'medium':
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            rescale=1./255,
            fill_mode='nearest'
        )
    
    elif augmentation_severity == 'heavy':
        return ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1,
            rescale=1./255,
            fill_mode='nearest'
        )
    
    else:
        raise ValueError(f"Unknown severity: {augmentation_severity}")

def visualize_augmentations(image, severity='medium', num_samples=5):
    """
    Visualize augmentations applied to a sample image.
    
    Args:
        image: Input image (numpy array)
        severity: Severity of augmentations
        num_samples: Number of augmented samples to visualize
    """
    aug_pipeline = get_augmentation_pipeline(severity)
    
    plt.figure(figsize=(20, 4))
    
    # Display original image
    plt.subplot(1, num_samples+1, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    # Display augmented images
    for i in range(num_samples):
        augmented = aug_pipeline(image=image)['image']
        plt.subplot(1, num_samples+1, i+2)
        plt.imshow(augmented)
        plt.title(f"Augmented {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def generate_augmented_samples(image_path, output_dir, severity='medium', num_samples=5):
    """
    Generate and save augmented samples for a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save augmented images
        severity: Severity of augmentations
        num_samples: Number of augmented samples to generate
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create augmentation pipeline
    aug_pipeline = get_augmentation_pipeline(severity)
    
    # Generate and save augmented samples
    stem = Path(image_path).stem
    for i in range(num_samples):
        augmented = aug_pipeline(image=image)['image']
        output_path = output_dir / f"{stem}_aug_{i+1}.png"
        
        # Save image
        cv2.imwrite(
            str(output_path), 
            cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        )
    
    print(f"Generated {num_samples} augmented samples from {image_path}")

def augment_dataset(input_dir, output_dir, severity='medium', samples_per_image=5):
    """
    Augment an entire dataset of images.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save augmented images
        severity: Severity of augmentations
        samples_per_image: Number of augmented samples per input image
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_paths = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    
    for image_path in image_paths:
        # Get relative path to maintain directory structure
        rel_path = image_path.relative_to(input_dir)
        image_output_dir = output_dir / rel_path.parent
        
        # Generate augmented samples
        generate_augmented_samples(
            image_path=image_path,
            output_dir=image_output_dir,
            severity=severity,
            num_samples=samples_per_image
        )
    
    print(f"Augmentation complete. Processed {len(image_paths)} images.")

if __name__ == "__main__":
    # Example usage
    augment_dataset(
        input_dir="data/processed",
        output_dir="data/augmented",
        severity='medium',
        samples_per_image=5
    )