#!/usr/bin/env python
"""CLI entrypoint: build and persist the transaction entity graph.

Usage: python scripts/build_graph.py
"""
from fraud_detection.graph.build import main

if __name__ == "__main__":
    main()
