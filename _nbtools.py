"""Minimal notebook builder shared by all example generators."""
import json


def md(source):
    """Create a Markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }


def code(source):
    """Create a Code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source if isinstance(source, list) else [source],
    }


def write_notebook(cells, path):
    """Write a list of cells as a .ipynb file."""
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "cells": cells,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {path}")
