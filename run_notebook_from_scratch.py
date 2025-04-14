#!/usr/bin/env python
"""
Generate a Jupyter notebook for running SwitchableTokenizer experiments from scratch.

This script creates a notebook with the necessary code to run the experiments
with models trained from scratch instead of fine-tuning.
"""

import json
import os

# Create the notebook structure
notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# SwitchableTokenizer Experiments (Training from Scratch)\n",
                "\n",
                "This notebook runs the SwitchableTokenizer experiments with models trained from scratch instead of fine-tuning.\n",
                "This allows evaluating how well the switchable tokenizer approach works with freshly initialized models.\n",
                "\n",
                "## Overview of Experiments\n",
                "\n",
                "1. **Experiment 1: Feasibility and Performance** - Compares the switchable tokenizer model with monolingual models\n",
                "2. **Experiment 2: Comparison vs. Concatenated Vocab** - Compares the switchable tokenizer with a 128k concatenated vocabulary\n",
                "3. **Experiment 3: Multilingual Baseline** - Compares against a standard multilingual tokenizer"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Environment Setup\n",
                "\n",
                "First, let's set up our environment by cloning the repository and installing dependencies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Clone the repository\n",
                "!git clone https://github.com/hardesttype/switch-tokenizer.git\n",
                "!cd switch-tokenizer"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install required packages\n",
                "# For colab you need only `datasets`\n",
                "!pip install -q datasets # torch transformers datasets tokenizers matplotlib seaborn tqdm numpy pandas"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Add the repository to the Python path\n",
                "import sys\n",
                "sys.path.append('/content/switch-tokenizer')\n",
                "\n",
                "# Set environment variables\n",
                "import os\n",
                "os.environ[\"TOKENIZERS_PARALLELISM\"] = \"false\"\n",
                "\n",
                "# Import common libraries\n",
                "import torch\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from transformers import set_seed\n",
                "\n",
                "# Check for GPU availability\n",
                "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                "print(f\"Using device: {device}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Experiment 1: Feasibility and Performance\n",
                "\n",
                "This experiment compares the performance of a model using the switchable tokenizer against individual monolingual models of similar size. It evaluates perplexity on held-out test data for each language.\n",
                "\n",
                "We'll train the models from scratch instead of fine-tuning pre-trained models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import necessary modules for Experiment 1\n",
                "from experiments.experiment1_feasibility import main as experiment1_main\n",
                "import argparse\n",
                "\n",
                "# Set up argument parser for Experiment 1\n",
                "def run_experiment1(data_limit=500, epochs=1, batch_size=4, seed=42):\n",
                "    # Save original sys.argv\n",
                "    orig_argv = sys.argv.copy()\n",
                "    \n",
                "    # Set new sys.argv with from_scratch flag\n",
                "    sys.argv = ['experiment1_feasibility.py', \n",
                "                '--from_scratch',\n",
                "                f'--data_limit={data_limit}',\n",
                "                f'--epochs={epochs}',\n",
                "                f'--batch_size={batch_size}',\n",
                "                f'--seed={seed}',\n",
                "                '--output_dir=./experiment1_from_scratch_output']\n",
                "    \n",
                "    print(f\"Running with command: {' '.join(sys.argv)}\\n\")\n",
                "    \n",
                "    # Run the experiment\n",
                "    try:\n",
                "        experiment1_main()\n",
                "    finally:\n",
                "        # Restore original sys.argv\n",
                "        sys.argv = orig_argv"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Experiment 1 with small data for quicker execution in Colab\n",
                "# Adjust parameters as needed based on your computational resources\n",
                "run_experiment1(data_limit=100, epochs=1, batch_size=4)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Experiment 2: Comparison vs. Concatenated Vocab\n",
                "\n",
                "This experiment compares the performance of the switchable tokenizer model (with shared 64k vocabulary) against a model using a 128k concatenated vocabulary. It evaluates both perplexity and parameter efficiency.\n",
                "\n",
                "We'll train both models from scratch."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up function to run Experiment 2\n",
                "from experiments.experiment2_concatenated_vocab import main as experiment2_main\n",
                "\n",
                "def run_experiment2(data_limit=500, epochs=1, batch_size=4, seed=42):\n",
                "    # Save original sys.argv\n",
                "    orig_argv = sys.argv.copy()\n",
                "    \n",
                "    # Set new sys.argv with from_scratch flag\n",
                "    sys.argv = ['experiment2_concatenated_vocab.py', \n",
                "                '--from_scratch',\n",
                "                f'--data_limit={data_limit}',\n",
                "                f'--epochs={epochs}',\n",
                "                f'--batch_size={batch_size}',\n",
                "                f'--seed={seed}',\n",
                "                '--output_dir=./experiment2_from_scratch_output']\n",
                "    \n",
                "    print(f\"Running with command: {' '.join(sys.argv)}\\n\")\n",
                "    \n",
                "    # Run the experiment\n",
                "    try:\n",
                "        experiment2_main()\n",
                "    finally:\n",
                "        # Restore original sys.argv\n",
                "        sys.argv = orig_argv"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Experiment 2\n",
                "run_experiment2(data_limit=100, epochs=1, batch_size=4)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Experiment 3: Multilingual Baseline\n",
                "\n",
                "This experiment compares the switchable tokenizer against a standard multilingual tokenizer baseline. It evaluates tokenization efficiency and model perplexity.\n",
                "\n",
                "We'll train both models from scratch."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Set up function to run Experiment 3\n",
                "from experiments.experiment3_multilingual_baseline import main as experiment3_main\n",
                "\n",
                "def run_experiment3(data_limit=500, epochs=1, batch_size=4, seed=42):\n",
                "    # Save original sys.argv\n",
                "    orig_argv = sys.argv.copy()\n",
                "    \n",
                "    # Set new sys.argv with from_scratch flag\n",
                "    sys.argv = ['experiment3_multilingual_baseline.py', \n",
                "                '--from_scratch',\n",
                "                f'--data_limit={data_limit}',\n",
                "                f'--epochs={epochs}',\n",
                "                f'--batch_size={batch_size}',\n",
                "                f'--seed={seed}',\n",
                "                '--output_dir=./experiment3_from_scratch_output']\n",
                "    \n",
                "    print(f\"Running with command: {' '.join(sys.argv)}\\n\")\n",
                "    \n",
                "    # Run the experiment\n",
                "    try:\n",
                "        experiment3_main()\n",
                "    finally:\n",
                "        # Restore original sys.argv\n",
                "        sys.argv = orig_argv"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run Experiment 3\n",
                "run_experiment3(data_limit=100, epochs=1, batch_size=4)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Conclusion\n",
                "\n",
                "This notebook demonstrates how to run the SwitchableTokenizer experiments with models trained from scratch instead of fine-tuning pre-trained models.\n",
                "\n",
                "Training from scratch allows us to evaluate how well the switchable tokenizer approach works without relying on knowledge already embedded in pre-trained models. This is particularly important for:\n",
                "\n",
                "1. Understanding the inherent capabilities of the switchable tokenizer architecture\n",
                "2. Evaluating tokenization efficiency with a clean model\n",
                "3. Comparing parameter efficiency without pre-training bias\n",
                "\n",
                "Note that these experiments can be computationally intensive. In Google Colab, we use reduced dataset sizes and epochs to complete the experiments within the available GPU time limits."
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook
with open('switch_tokenizer_from_scratch.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"Notebook created: switch_tokenizer_from_scratch.ipynb") 