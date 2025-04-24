"""
Visualization tools for medical image classification models.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
import torch
from torch.nn import functional as F
from PIL import Image

def generate_gradcam_tensorflow(model, img_array, layer_name=None, pred_index=None):
    """
    Generate Grad-CAM visualization for TensorFlow model.
    
    Args:
        model: TensorFlow model
        img_array: Input image as numpy array (1, height, width, channels)
        layer_name: Name of the target layer for Grad-CAM
        pred_index: Index of the target class (None for max prediction)
        
    Returns:
        Heatmap as numpy array
    """
    # Get the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            # Check if layer is a Conv2D layer - this is more reliable
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
            # Alternative method for other layer types that might have 4D output
            try:
                if hasattr(layer, 'output_shape') and len(layer.output_shape) == 4:
                    layer_name = layer.name
                    break
            except:
                continue
    
    # If no suitable layer was found, use the first layer as fallback
    if layer_name is None and len(model.layers) > 0:
        layer_name = model.layers[0].name
        print(f"Warning: No convolutional layer found. Using {layer_name} as fallback.")
    
    # Create a model that maps the input image to the activations of the target layer
    # and to the final predictions
    try:
        grad_model = Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Gradient tape
        with tf.GradientTape() as tape:
            # Get the conv output and model output
            conv_output, predictions = grad_model(img_array)
            
            # Get the predicted class if not specified
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            
            # Get the target class score
            class_channel = predictions[:, pred_index]
        
        # Gradient of the target class with respect to the output feature map
        grads = tape.gradient(class_channel, conv_output)
        
        # Vector of shape (batch_size, channels) with mean gradients per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by the mean gradients
        conv_output = conv_output[0]
        for i in range(pooled_grads.shape[0]):
            conv_output[:, :, i] *= pooled_grads[i]
        
        # Mean over all weighted feature maps
        heatmap = tf.reduce_mean(conv_output, axis=-1)
        
        # ReLU to only keep positive influence
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap
    
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        # Return a blank heatmap as fallback
        return np.zeros((7, 7))  # Small blank heatmap that will be resized later

def generate_gradcam_pytorch(model, img_tensor, target_layer=None, target_class=None):
    """
    Generate Grad-CAM visualization for PyTorch model.
    
    Args:
        model: PyTorch model
        img_tensor: Input image as PyTorch tensor (1, channels, height, width)
        target_layer: Target layer for Grad-CAM
        target_class: Index of the target class (None for max prediction)
        
    Returns:
        Heatmap as numpy array
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get the output of the model
    img_tensor.requires_grad_()
    output = model(img_tensor)
    
    # Get the predicted class if not specified
    if target_class is None:
        target_class = output.argmax().item()
    
    # Get the target layer's output
    if target_layer is None:
        # Find the last convolutional layer
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                target_layer = module
                break
    
    # Grad-CAM algorithm
    # 1. Get the gradient of the target class with respect to the feature maps
    model.zero_grad()
    output[:, target_class].backward()
    
    # 2. Get the gradients and activations
    gradients = target_layer.weight.grad.data
    activations = target_layer.output.data
    
    # 3. Global average pooling of the gradients
    weights = gradients.mean(dim=(0, 2, 3))
    
    # 4. Weight the activations by the gradients
    cam = torch.zeros(activations.shape[2:]).to(img_tensor.device)
    for i, w in enumerate(weights):
        cam += w * activations[0, i]
    
    # 5. Apply ReLU and normalize
    cam = F.relu(cam)
    cam = cam / (cam.max() + 1e-10)
    
    # Convert to numpy array
    cam = cam.cpu().numpy()
    
    return cam

def apply_heatmap(original_img, heatmap, alpha=0.6):
    """
    Apply heatmap to original image.
    
    Args:
        original_img: Original image as numpy array (height, width, channels)
        heatmap: Heatmap as numpy array (height, width)
        alpha: Alpha blending factor
        
    Returns:
        Blended image as numpy array
    """
    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    # Convert heatmap to RGB with Jet colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to BGR if needed
    if len(original_img.shape) == 3 and original_img.shape[2] == 3:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
    else:
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_GRAY2BGR)
    
    # Convert to float32 for alpha blending
    heatmap = heatmap.astype(np.float32) / 255.0
    original_bgr = original_bgr.astype(np.float32) / 255.0
    
    # Apply heatmap to original image
    cam_img = heatmap * alpha + original_bgr * (1 - alpha)
    cam_img = np.uint8(255 * cam_img)
    
    # Convert back to RGB
    cam_img = cv2.cvtColor(cam_img, cv2.COLOR_BGR2RGB)
    
    return cam_img

def visualize_gradcam(model, image_path, framework='tensorflow', target_class=None, 
                    layer_name=None, alpha=0.6, save_path=None):
    """
    Generate and visualize Grad-CAM for an image.
    
    Args:
        model: Model (TensorFlow or PyTorch)
        image_path: Path to input image
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        target_class: Index of the target class (None for max prediction)
        layer_name: Name of the target layer for Grad-CAM
        alpha: Alpha blending factor for heatmap overlay
        save_path: Path to save the visualization
        
    Returns:
        Tuple of (original_image, heatmap, cam_image)
    """
    # Load and preprocess image
    if framework == 'tensorflow':
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original image for visualization
        original_img = img.copy()
        
        # Preprocess for model
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Generate Grad-CAM
        heatmap = generate_gradcam_tensorflow(
            model=model,
            img_array=img,
            layer_name=layer_name,
            pred_index=target_class
        )
        
    elif framework == 'pytorch':
        # Read image
        img = Image.open(image_path).convert('RGB')
        
        # Store original image for visualization
        original_img = np.array(img)
        
        # Preprocess for model
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = preprocess(img).unsqueeze(0)
        
        # Generate Grad-CAM
        heatmap = generate_gradcam_pytorch(
            model=model,
            img_tensor=img_tensor,
            target_layer=layer_name,
            target_class=target_class
        )
        
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Apply heatmap to original image
    cam_img = apply_heatmap(original_img, heatmap, alpha)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(cam_img)
    axes[2].set_title('Grad-CAM')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    return original_img, heatmap, cam_img

def visualize_multiple_models(image_path, models, model_names, framework='tensorflow', 
                            target_class=None, save_path=None):
    """
    Compare Grad-CAM visualizations from multiple models for the same image.
    
    Args:
        image_path: Path to input image
        models: List of models
        model_names: List of model names
        framework: Deep learning framework ('tensorflow' or 'pytorch')
        target_class: Index of the target class (None for max prediction)
        save_path: Path to save the visualization
    """
    # Load and preprocess image
    if framework == 'tensorflow':
        # Read image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Store original image for visualization
        original_img = img.copy()
        
        # Preprocess for model
        preprocessed_img = cv2.resize(img, (224, 224))
        preprocessed_img = preprocessed_img / 255.0
        preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
        
    elif framework == 'pytorch':
        # Read image
        img = Image.open(image_path).convert('RGB')
        
        # Store original image for visualization
        original_img = np.array(img)
        
        # Preprocess for model
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        preprocessed_img = preprocess(img).unsqueeze(0)
        
    else:
        raise ValueError(f"Unknown framework: {framework}")
    
    # Generate Grad-CAM for each model
    heatmaps = []
    cam_images = []
    
    for model in models:
        if framework == 'tensorflow':
            heatmap = generate_gradcam_tensorflow(
                model=model,
                img_array=preprocessed_img,
                pred_index=target_class
            )
        else:
            heatmap = generate_gradcam_pytorch(
                model=model,
                img_tensor=preprocessed_img,
                target_class=target_class
            )
        
        cam_img = apply_heatmap(original_img, heatmap)
        
        heatmaps.append(heatmap)
        cam_images.append(cam_img)
    
    # Visualize
    n_models = len(models)
    fig, axes = plt.subplots(2, n_models+1, figsize=(5*(n_models+1), 10))
    
    # Display original image
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].axis('off')
    
    # Display heatmaps and Grad-CAMs for each model
    for i in range(n_models):
        axes[0, i+1].imshow(heatmaps[i], cmap='jet')
        axes[0, i+1].set_title(f'{model_names[i]} Heatmap')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(cam_images[i])
        axes[1, i+1].set_title(f'{model_names[i]} Grad-CAM')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    
    plt.show()