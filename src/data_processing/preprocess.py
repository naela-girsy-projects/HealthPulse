"""
Image preprocessing functions for medical image classification.
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def resize_image(image, target_size=(224, 224)):
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: Target size (height, width)
        
    Returns:
        Resized image
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def normalize_image(image):
    """
    Normalize image to [0, 1] range.
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    return image.astype(np.float32) / 255.0

def clahe_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization.
    
    Args:
        image: Input grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced image
    """
    # Convert to grayscale if the image has 3 channels
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_image = clahe.apply(image)
    
    # Convert back to 3 channels if needed for model input
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    
    return enhanced_image

def preprocess_image(image_path, target_size=(224, 224), enhance=True):
    """
    Complete preprocessing pipeline for a single image.
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        enhance: Whether to apply CLAHE enhancement
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply CLAHE enhancement if specified
    if enhance:
        image = clahe_enhancement(image)
    
    # Resize image
    image = resize_image(image, target_size)
    
    # Normalize image
    image = normalize_image(image)
    
    return image

def _process_image(args):
    """
    Process a single image (for parallel processing).
    
    Args:
        args: Tuple of (image_path, input_dir, output_dir, target_size, enhance)
        
    Returns:
        True if processing was successful, False otherwise
    """
    image_path, input_dir, output_dir, target_size, enhance = args
    try:
        # Get relative path to maintain directory structure
        rel_path = image_path.relative_to(input_dir)
        output_path = output_dir / rel_path
        
        # Create output subdirectory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preprocess image
        processed_image = preprocess_image(image_path, target_size, enhance)
        
        # Save processed image
        cv2.imwrite(str(output_path), cv2.cvtColor(
            (processed_image * 255).astype(np.uint8), 
            cv2.COLOR_RGB2BGR
        ))
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False

def preprocess_dataset(input_dir, output_dir, target_size=(224, 224), enhance=True, n_workers=4):
    """
    Preprocess an entire dataset of images.
    
    Args:
        input_dir: Directory containing raw images
        output_dir: Directory to save processed images
        target_size: Target size for resizing
        enhance: Whether to apply CLAHE enhancement
        n_workers: Number of worker processes for parallel processing
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of images
    image_paths = list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.png'))
    
    # Prepare arguments for parallel processing
    process_args = [(image_path, input_dir, output_dir, target_size, enhance) 
                   for image_path in image_paths]
    
    # Process images in parallel or sequentially based on n_workers
    if n_workers > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm(
                executor.map(_process_image, process_args),
                total=len(image_paths),
                desc="Preprocessing images"
            ))
    else:
        # Process sequentially if n_workers is 1
        results = []
        for args in tqdm(process_args, desc="Preprocessing images"):
            results.append(_process_image(args))
    
    # Report results
    success_count = sum(results)
    print(f"Successfully preprocessed {success_count}/{len(image_paths)} images.")

if __name__ == "__main__":
    # Example usage
    preprocess_dataset(
        input_dir="data/raw",
        output_dir="data/processed",
        target_size=(224, 224),
        enhance=True,
        n_workers=4
    )