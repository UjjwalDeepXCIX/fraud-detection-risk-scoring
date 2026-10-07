#!/usr/bin/env python
"""CLI entrypoint: generate the synthetic transaction dataset.

Usage: python scripts/generate_data.py [--n-transactions 60000] ...
"""
from fraud_detection.data.generate import main

if __name__ == "__main__":
    main()
