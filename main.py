#!/usr/bin/env python
"""
HealthPulse: Medical Image Classification System
Main entry point for the application
"""
import os
import argparse
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.data_processing.preprocess import preprocess_dataset
from src.data_processing.augmentation import augment_dataset
from src.models.training import train_model_pipeline
from src.models.model_builder import get_model
from src.evaluation.metrics import evaluate_tensorflow_model, generate_evaluation_report
from src.evaluation.visualization import visualize_gradcam
from src.web_app.app import app as flask_app


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        'data/raw',
        'data/processed',
        'data/augmented',
        'data/models',
        'data/evaluation',
        'data/uploads'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("Project directories created successfully.")


def preprocess_data(args):
    """Run the preprocessing pipeline."""
    print(f"Preprocessing data from {args.input_dir} to {args.output_dir}...")
    preprocess_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_size=(args.image_size, args.image_size),
        enhance=args.enhance,
        n_workers=args.workers
    )
    print("Preprocessing completed!")


def augment_data(args):
    """Run the data augmentation pipeline."""
    print(f"Augmenting data from {args.input_dir} to {args.output_dir}...")
    augment_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        severity=args.severity,
        samples_per_image=args.samples
    )
    print("Data augmentation completed!")


def train_model(args):
    """Train a model using the specified parameters."""
    print(f"Training {args.model} model using {args.framework}...")
    model, history = train_model_pipeline(
        data_dir=args.data_dir,
        model_name=args.model,
        framework=args.framework,
        batch_size=args.batch_size,
        img_size=(args.image_size, args.image_size),
        epochs=args.epochs,
        output_dir=args.output_dir,
        num_classes=args.num_classes
    )
    print(f"Model training completed! Model saved to {args.output_dir}")


def evaluate_model(args):
    """Evaluate a trained model."""
    print(f"Evaluating model {args.model_path}...")
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model_path)
    
    # Evaluate
    metrics = evaluate_tensorflow_model(
        model=model,
        test_dir=args.test_dir,
        batch_size=args.batch_size,
        img_size=(args.image_size, args.image_size)
    )
    
    # Generate report
    summary = generate_evaluation_report(
        metrics=metrics,
        output_dir=args.output_dir,
        model_name=args.model_name
    )
    
    print(f"Evaluation completed! Reports saved to {args.output_dir}")
    print(f"Summary: {summary}")


def visualize_model(args):
    """Generate visualizations for model interpretability."""
    print(f"Generating Grad-CAM visualization for {args.image_path}...")
    
    # Load model
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model_path)
    
    # Generate visualization
    visualize_gradcam(
        model=model,
        image_path=args.image_path,
        framework='tensorflow',
        target_class=args.target_class,
        save_path=args.output_path
    )
    
    print(f"Visualization saved to {args.output_path}")


def run_web_app(args):
    """Run the Flask web application."""
    print("Starting HealthPulse web application...")
    flask_app.config['MODEL_PATH'] = args.model_path
    flask_app.config['UPLOAD_FOLDER'] = args.upload_folder
    flask_app.run(host=args.host, port=args.port, debug=args.debug)


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='HealthPulse: Medical Image Classification System')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Setup directories command
    setup_parser = subparsers.add_parser('setup', help='Create necessary directories')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess raw medical images')
    preprocess_parser.add_argument('--input_dir', type=str, required=True, help='Directory with raw images')
    preprocess_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed images')
    preprocess_parser.add_argument('--image_size', type=int, default=224, help='Target image size')
    preprocess_parser.add_argument('--enhance', action='store_true', help='Apply CLAHE enhancement')
    preprocess_parser.add_argument('--workers', type=int, default=4, help='Number of worker processes')
    
    # Augment command
    augment_parser = subparsers.add_parser('augment', help='Augment processed images')
    augment_parser.add_argument('--input_dir', type=str, required=True, help='Directory with processed images')
    augment_parser.add_argument('--output_dir', type=str, required=True, help='Directory to save augmented images')
    augment_parser.add_argument('--severity', type=str, default='medium', choices=['light', 'medium', 'heavy'], help='Augmentation severity')
    augment_parser.add_argument('--samples', type=int, default=5, help='Samples per image')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--data_dir', type=str, required=True, help='Directory with training data')
    train_parser.add_argument('--model', type=str, default='resnet', choices=['custom_cnn', 'resnet', 'efficientnet', 'vgg'], help='Model architecture')
    train_parser.add_argument('--framework', type=str, default='tensorflow', choices=['tensorflow', 'pytorch'], help='Deep learning framework')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    train_parser.add_argument('--image_size', type=int, default=224, help='Image size')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    train_parser.add_argument('--output_dir', type=str, default='data/models', help='Directory to save model')
    train_parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    evaluate_parser.add_argument('--test_dir', type=str, required=True, help='Directory with test data')
    evaluate_parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    evaluate_parser.add_argument('--image_size', type=int, default=224, help='Image size')
    evaluate_parser.add_argument('--output_dir', type=str, default='data/evaluation', help='Directory to save evaluation results')
    evaluate_parser.add_argument('--model_name', type=str, default='model', help='Model name for reports')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate model visualizations')
    visualize_parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    visualize_parser.add_argument('--image_path', type=str, required=True, help='Path to image for visualization')
    visualize_parser.add_argument('--output_path', type=str, required=True, help='Path to save visualization')
    visualize_parser.add_argument('--target_class', type=int, help='Target class for visualization')
    
    # Web app command
    webapp_parser = subparsers.add_parser('webapp', help='Run the web application')
    webapp_parser.add_argument('--model_path', type=str, default='data/models/model_best_val_acc.h5', help='Path to trained model')
    webapp_parser.add_argument('--upload_folder', type=str, default='data/uploads', help='Folder for uploaded images')
    webapp_parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    webapp_parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    webapp_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_directories()
    elif args.command == 'preprocess':
        preprocess_data(args)
    elif args.command == 'augment':
        augment_data(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'visualize':
        visualize_model(args)
    elif args.command == 'webapp':
        run_web_app(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()