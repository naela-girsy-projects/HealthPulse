"""
Evaluation metrics for medical image classification models.
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report,
    precision_recall_curve, average_precision_score
)
import pandas as pd
import seaborn as sns
from pathlib import Path
import os

def evaluate_tensorflow_model(model, test_dir, batch_size=32, img_size=(224, 224)):
    """
    Evaluate a TensorFlow model on test data.
    
    Args:
        model: Trained Keras model
        test_dir: Directory containing test images
        batch_size: Batch size for evaluation
        img_size: Image size (height, width)
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create test data generator
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Evaluate model
    print("Evaluating model...")
    results = model.evaluate(test_generator, verbose=1)
    metrics = dict(zip(model.metrics_names, results))
    
    # Get predictions
    print("Getting predictions...")
    y_pred_prob = model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = test_generator.classes
    
    # Calculate additional metrics
    print("Calculating metrics...")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # For binary classification
    if len(class_names) == 2:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:, 1])
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_prob[:, 1])
        
        metrics.update({
            'auc': auc,
            'avg_precision': avg_precision
        })
    else:
        # For multiclass, calculate macro average AUC
        auc = roc_auc_score(tf.keras.utils.to_categorical(y_true), y_pred_prob, multi_class='ovr', average='macro')
        metrics.update({
            'auc': auc
        })
    
    # Update metrics dictionary
    metrics.update({
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'class_names': class_names
    })
    
    return metrics

def evaluate_pytorch_model(model, test_loader, device=None, num_classes=2):
    """
    Evaluate a PyTorch model on test data.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to evaluate on ('cuda' or 'cpu')
        num_classes: Number of output classes
        
    Returns:
        Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set model to evaluation mode
    model.eval()
    
    all_predictions = []
    all_probs = []
    all_targets = []
    
    # Disable gradient calculation
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            
            # Store predictions and targets
            all_predictions.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Convert to numpy arrays
    y_pred = np.array(all_predictions)
    y_pred_prob = np.array(all_probs)
    y_true = np.array(all_targets)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Get class names
    class_names = [str(i) for i in range(num_classes)]
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Calculate AUC for binary and multiclass
    if num_classes == 2:
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        
        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
        avg_precision = average_precision_score(y_true, y_pred_prob[:, 1])
    else:
        # For multiclass, calculate macro average AUC
        auc = roc_auc_score(
            np.eye(num_classes)[y_true], 
            y_pred_prob, 
            multi_class='ovr', 
            average='macro'
        )
        avg_precision = None
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'avg_precision': avg_precision,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob,
        'class_names': class_names
    }
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 8), save_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Class names
        figsize: Figure size
        save_path: Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_prob, class_names, figsize=(10, 8), save_path=None):
    """
    Plot ROC curve for binary or multiclass classification.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        class_names: Class names
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Binary classification
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
        auc = roc_auc_score(y_true, y_pred_prob[:, 1])
        
        plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        
    # Multiclass classification
    else:
        # Convert to one-hot encoding
        y_true_onehot = np.eye(len(class_names))[y_true]
        
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
            auc = roc_auc_score(y_true_onehot[:, i], y_pred_prob[:, i])
            
            plt.plot(fpr, tpr, label=f'{class_names[i]}: AUC = {auc:.3f}')
        
        plt.plot([0, 1], [0, 1], 'k--')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_prob, class_names, figsize=(10, 8), save_path=None):
    """
    Plot precision-recall curve for binary or multiclass classification.
    
    Args:
        y_true: True labels
        y_pred_prob: Predicted probabilities
        class_names: Class names
        figsize: Figure size
        save_path: Path to save the plot
    """
    plt.figure(figsize=figsize)
    
    # Binary classification
    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob[:, 1])
        ap = average_precision_score(y_true, y_pred_prob[:, 1])
        
        plt.plot(recall, precision, label=f'AP = {ap:.3f}')
        
    # Multiclass classification
    else:
        # Convert to one-hot encoding
        y_true_onehot = np.eye(len(class_names))[y_true]
        
        for i in range(len(class_names)):
            precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_pred_prob[:, i])
            ap = average_precision_score(y_true_onehot[:, i], y_pred_prob[:, i])
            
            plt.plot(recall, precision, label=f'{class_names[i]}: AP = {ap:.3f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-recall curve saved to {save_path}")
    
    plt.show()

def generate_evaluation_report(metrics, output_dir='data/evaluation', model_name='model'):
    """
    Generate comprehensive evaluation report with visualizations.
    
    Args:
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save evaluation results
        model_name: Name of the model for file naming
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    y_pred_prob = metrics['y_pred_prob']
    class_names = metrics['class_names']
    report = metrics['classification_report']
    
    # Save metrics to CSV
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(output_dir, f'{model_name}_classification_report.csv'))
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        save_path=os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    )
    
    # Plot ROC curve
    plot_roc_curve(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        class_names=class_names,
        save_path=os.path.join(output_dir, f'{model_name}_roc_curve.png')
    )
    
    # Plot precision-recall curve
    plot_precision_recall_curve(
        y_true=y_true,
        y_pred_prob=y_pred_prob,
        class_names=class_names,
        save_path=os.path.join(output_dir, f'{model_name}_precision_recall_curve.png')
    )
    
    # Generate summary statistics
    summary = {
        'accuracy': metrics.get('accuracy', 0),
        'precision': metrics.get('precision', 0),
        'recall': metrics.get('recall', 0),
        'f1': metrics.get('f1', 0),
        'auc': metrics.get('auc', 0)
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(output_dir, f'{model_name}_summary.csv'), index=False)
    
    print(f"Evaluation report generated in {output_dir}")
    
    return summary_df

def compare_models(model_metrics, model_names, output_dir='data/evaluation'):
    """
    Compare multiple models based on their performance metrics.
    
    Args:
        model_metrics: List of metric dictionaries for each model
        model_names: List of model names
        output_dir: Directory to save comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract summary metrics for each model
    summaries = []
    
    for metrics, name in zip(model_metrics, model_names):
        summary = {
            'model': name,
            'accuracy': metrics.get('accuracy', 0),
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1': metrics.get('f1', 0),
            'auc': metrics.get('auc', 0)
        }
        summaries.append(summary)
    
    # Create DataFrame
    df = pd.DataFrame(summaries)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    
    # Plot comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    plt.figure(figsize=(15, 10))
    
    x = np.arange(len(model_names))
    width = 0.15
    multiplier = 0
    
    for metric in metrics_to_plot:
        offset = width * multiplier
        plt.bar(x + offset, df[metric], width, label=metric)
        multiplier += 1
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Comparison')
    plt.xticks(x + width * 2, model_names)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
    plt.show()
    
    return df