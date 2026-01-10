#!/usr/bin/env python3
"""
Test script for Concrete Crack Detection API
ML Zoomcamp 2025 Capstone Project - Concrete Crack Detector

Tests the predict endpoint with sample images.
"""

import requests
import argparse
from pathlib import Path


def test_predict(url: str, image_path: str, threshold: float = 0.5):
    """
    Test the /predict endpoint with an image.

    Args:
        url: API endpoint URL
        image_path: Path to image file
        threshold: Classification threshold (default: 0.5)
    """
    # Prepare the request
    files = {"image": open(image_path, "rb")}
    data = {"threshold": threshold}

    print(f"Testing with image: {image_path}")
    print(f"API URL: {url}")
    print(f"Threshold: {threshold}")
    print("-" * 50)

    try:
        # Make request
        response = requests.post(url, files=files, data=data)

        # Check response
        if response.status_code == 200:
            result = response.json()
            print("\nPrediction successful!")
            print(f"  Prediction:  {result['prediction']}")
            print(f"  Probability: {result['probability']:.4f}")
            print(f"  Confidence:  {result['confidence']:.4f}")
            print(f"  Model type:  {result.get('model_type', 'Unknown')}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"\nException: {str(e)}")


def test_health(url: str):
    """Test the /health endpoint."""
    health_url = url.replace("/predict", "/health")
    print(f"Testing health endpoint: {health_url}")
    print("-" * 50)

    try:
        response = requests.get(health_url)

        if response.status_code == 200:
            print("\nAPI is healthy!")
            print(response.json())
        else:
            print(f"\nError: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"\nException: {str(e)}")


def test_info(url: str):
    """Test the root endpoint for API info."""
    info_url = url.replace("/predict", "")
    print(f"Testing info endpoint: {info_url}")
    print("-" * 50)

    try:
        response = requests.get(info_url)

        if response.status_code == 200:
            info = response.json()
            print("\nAPI Info:")
            print(f"  Service:  {info.get('service', 'Unknown')}")
            print(f"  Version:  {info.get('version', 'Unknown')}")
            print(f"  Model:    {info.get('model', 'Unknown')}")
            print(f"  Status:   {info.get('status', 'Unknown')}")
            if 'config' in info:
                print(f"  Config:   {info['config']}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.json())

    except Exception as e:
        print(f"\nException: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Test Concrete Crack Detection API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080/predict",
        help="API endpoint URL (default: http://localhost:8080/predict)"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to test image"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (default: 0.5)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Test health endpoint only"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Get API info only"
    )

    args = parser.parse_args()

    print("="*60)
    print("  Concrete Crack Detection API - Test Script")
    print("="*60)
    print()

    if args.info:
        test_info(args.url)
    elif args.health:
        test_health(args.url)
    elif args.image:
        # Check if image exists
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            return

        test_predict(args.url, args.image, args.threshold)
    else:
        print("No test specified. Use --image, --health, or --info")
        print("\nExamples:")
        print("  python test_api.py --health")
        print("  python test_api.py --info")
        print("  python test_api.py --image data/Positive/00001.jpg")
        print("  python test_api.py --image data/Negative/00001.jpg --threshold 0.7")

    print()


if __name__ == "__main__":
    main()
