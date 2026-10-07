#!/usr/bin/env python
"""CLI entrypoint: run the full training pipeline.

Usage: python scripts/train_model.py [--skip-shap]
"""
from fraud_detection.models.train import main

if __name__ == "__main__":
    main()
