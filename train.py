#!/usr/bin/env python3
"""
Concrete Crack Detection - Training Script
ML Zoomcamp 2025 Capstone Project

This script trains a MobileNetV2-based model for binary classification
of concrete surface images (crack/no-crack).
"""

import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import json


# Configuration
CONFIG = {
    "data_dir": "data",
    "model_dir": "models",
    "image_size": (224, 224),
    "batch_size": 32,
    "epochs": 5,
    "learning_rate": 0.0001,
    "validation_split": 0.2,
    "random_seed": 42,
}


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seeds set to {seed}")


def load_dataset(data_dir: str, image_size: tuple, batch_size: int, validation_split: float):
    """
    Load and prepare the dataset using tf.keras.utils.image_dataset_from_directory.

    Returns:
        train_ds, val_ds, class_names: Training and validation datasets with class names
    """
    print(f"\nLoading dataset from {data_dir}...")

    # Load training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=CONFIG["random_seed"],
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Load validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=CONFIG["random_seed"],
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary"
    )

    # Get class names
    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")
    print(f"Training batches: {len(train_ds)}")
    print(f"Validation batches: {len(val_ds)}")

    return train_ds, val_ds, class_names


def create_data_augmentation():
    """Create data augmentation pipeline."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def create_model(image_size: tuple, learning_rate: float):
    """
    Create MobileNetV2-based model with custom classification head.

    Architecture:
    - MobileNetV2 backbone (frozen, ImageNet weights)
    - Global Average Pooling
    - Dropout (0.2)
    - Dense output (sigmoid for binary classification)
    """
    print("\nCreating MobileNetV2-based model...")

    # Input layer
    inputs = keras.Input(shape=(*image_size, 3))

    # Data augmentation (only during training)
    x = create_data_augmentation()(inputs)

    # Preprocessing for MobileNetV2
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    # Base model (frozen)
    base_model = MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    print(f"Loaded MobileNetV2 base model (frozen)")

    # Feature extraction
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    # Create model
    model = keras.Model(inputs, outputs, name="crack_detector")

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ]
    )

    return model


def create_callbacks(model_dir: str):
    """Create training callbacks."""
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        # Model checkpoint
        ModelCheckpoint(
            filepath=os.path.join(model_dir, "crack_detector_best.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        # Learning rate reduction
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
    ]

    return callbacks


def convert_to_tflite(model, save_path: str):
    """Convert Keras model to TFLite for optimized edge and cloud deployment."""
    print(f"\nConverting model to TFLite format...")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimize for size and latency
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    # Report size
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"TFLite model saved to {save_path} ({size_mb:.2f} MB)")

    return tflite_model


def main(args):
    """Main training pipeline."""
    print("="*60)
    print("  Concrete Crack Detection - Training Pipeline")
    print("  ML Zoomcamp 2025 Capstone Project")
    print("="*60)

    # Set seeds for reproducibility
    set_seeds(CONFIG["random_seed"])

    # Create model directory
    os.makedirs(CONFIG["model_dir"], exist_ok=True)

    # Load dataset
    train_ds, val_ds, class_names = load_dataset(
        data_dir=args.data_dir or CONFIG["data_dir"],
        image_size=CONFIG["image_size"],
        batch_size=CONFIG["batch_size"],
        validation_split=CONFIG["validation_split"]
    )

    # Optimize dataset performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    print("Dataset optimized with caching and prefetching")

    # Create model
    model = create_model(
        image_size=CONFIG["image_size"],
        learning_rate=CONFIG["learning_rate"]
    )

    # Model summary
    model.summary()

    # Create callbacks
    callbacks = create_callbacks(CONFIG["model_dir"])

    # Training
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on validation set
    print("\n" + "="*60)
    print("Evaluating model on validation set...")
    print("="*60)

    results = model.evaluate(val_ds, verbose=1)

    metrics = {
        "loss": results[0],
        "accuracy": results[1],
        "precision": results[2],
        "recall": results[3],
        "auc": results[4],
    }

    # Calculate F1 score
    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1_score"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])
    else:
        metrics["f1_score"] = 0

    print("\nValidation Results:")
    print("-" * 30)
    for metric, value in metrics.items():
        print(f"  {metric.capitalize():12s}: {value:.4f}")

    # Save final model
    model_path = os.path.join(CONFIG["model_dir"], "crack_detector.h5")
    model.save(model_path)
    keras_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"\nKeras model saved to {model_path} ({keras_size:.2f} MB)")

    # Convert to TFLite
    tflite_path = os.path.join(CONFIG["model_dir"], "crack_detector.tflite")
    convert_to_tflite(model, tflite_path)

    # Save training config and metrics
    results_data = {
        "config": CONFIG,
        "class_names": class_names,
        "metrics": metrics,
        "final_epoch": len(history.history["accuracy"]),
    }

    results_path = os.path.join(CONFIG["model_dir"], "training_results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"Training results saved to {results_path}")

    print("\n" + "="*60)
    print("  Training completed successfully!")
    print("="*60)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train concrete crack detection model")
    parser.add_argument("--data-dir", type=str, help="Path to dataset directory", default=None)
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=None)

    args = parser.parse_args()

    # Override config if args provided
    if args.epochs:
        CONFIG["epochs"] = args.epochs

    main(args)
