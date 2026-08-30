#!/usr/bin/env python3
"""Retired compatibility entry point; ET-derived scalar features are excluded."""

from train_classifier import main


if __name__ == "__main__":
    raise SystemExit(main())
