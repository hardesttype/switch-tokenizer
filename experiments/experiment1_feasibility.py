#!/usr/bin/env python
"""
Experiment 1: Feasibility and Performance

Trains the shared 64k model and monolingual models of similar size,
then evaluates and compares their performance on held-out test sets.
"""

import os
import sys
import argparse
import torch
import numpy as np
import re
from pathlib import Path
from dotenv import load_dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed, AutoConfig,
    TrainerCallback
)
from datasets import load_dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, upload_folder

from src.switch_tokenizer import SwitchableTokenizer
from src.model_utils import create_model_with_switchable_tokenizer
from src.data_utils import (
    prepare_multilingual_datasets,
    SwitchableDataCollator
)
from src.evaluation import calculate_perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: Feasibility and Performance")
    
    # Dataset paths
    parser.add_argument("--en_dataset", type=str, default='wikimedia/wikipedia',
                        help="Path to English dataset")
    parser.add_argument("--en_subset", type=str, default='20231101.en',
                        help="English dataset subset")
    parser.add_argument("--ru_dataset", type=str, default="wikimedia/wikipedia", 
                        help="Path to Russian dataset")
    parser.add_argument("--ru_subset", type=str, default="20231101.ru", 
                        help="Russian dataset subset")
    parser.add_argument("--data_limit", type=int, default=10000, 
                        help="Limit number of examples per language (for faster experiments)")
    parser.add_argument("--first_shard_only", action="store_true", 
                        help="Use only the first shard (train-00000) of each dataset for faster testing")
    
    # Tokenizer paths
    parser.add_argument("--en_tokenizer", type=str, default="gpt2", 
                        help="English tokenizer")
    parser.add_argument("--ru_tokenizer", type=str, default="ai-forever/ruGPT-3.5-13B", 
                        help="Russian tokenizer")
    
    # Base model
    parser.add_argument("--base_model", type=str, default="gpt2-medium", 
                        help="Base model architecture")
    parser.add_argument("--from_scratch", action="store_true", 
                        help="Train models from scratch instead of fine-tuning")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./experiment1_output", 
                        help="Directory to save outputs")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=3, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=128, 
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    
    # HuggingFace upload options
    parser.add_argument("--upload_to_hub", action="store_true", 
                        help="Upload results to Hugging Face Hub")
    parser.add_argument("--hub_repo_id", type=str, default=None,
                        help="Hugging Face Hub repository ID (format: username/repo-name)")
    parser.add_argument("--hub_private", action="store_true",
                        help="Make the HuggingFace repository private")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="Hugging Face Hub API token (if not provided, will use HUGGING_FACE_HUB_TOKEN env variable)")
    
    return parser.parse_args()

class LossCallback(TrainerCallback):
    """Callback to collect and store training losses."""
    
    def __init__(self):
        self.training_losses = []
        self.eval_losses = []
        self.current_step = 0
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Collect training loss
        if "loss" in logs:
            self.training_losses.append((self.current_step, logs["loss"]))
            self.current_step += 1
        
        # Collect eval loss
        if "eval_loss" in logs:
            self.eval_losses.append((self.current_step, logs["eval_loss"]))

def train_switchable_model(args):
    """Train the shared 64k model with switchable tokenizer."""
    print("\n=== Training Switchable Model ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    switchable_output_dir = os.path.join(args.output_dir, "switchable_model")
    os.makedirs(switchable_output_dir, exist_ok=True)
    
    # Initialize switchable tokenizer
    tokenizer = SwitchableTokenizer(
        en_tokenizer_path=args.en_tokenizer,
        ru_tokenizer_path=args.ru_tokenizer,
        shared_vocab_size=None,  # Automatically determine the maximum safe shared vocab size
    )
    
    # Prepare datasets
    print("Preparing datasets...")
    dataset_configs = {
        "EN": {
            "path": args.en_dataset, 
            "name": args.en_subset,
        },
        "RU": {
            "path": args.ru_dataset, 
            "name": args.ru_subset,
        },
    }
    
    # Add appropriate configuration based on whether to use first shard only
    if args.first_shard_only:
        dataset_configs["EN"]["data_files"] = "train-00000-of-*.parquet"
        dataset_configs["RU"]["data_files"] = "train-00000-of-*.parquet"
        # Add verification_mode to avoid mismatches
        dataset_configs["EN"]["verification_mode"] = "no_checks"
        dataset_configs["RU"]["verification_mode"] = "no_checks"
        # Pass the data limit parameter for limiting examples - use half for each language
        dataset_configs["EN"]["apply_limit"] = True
        dataset_configs["RU"]["apply_limit"] = True
        dataset_configs["EN"]["limit"] = args.data_limit // 2
        dataset_configs["RU"]["limit"] = args.data_limit // 2
    else:
        dataset_configs["EN"]["split"] = f"train[:{args.data_limit // 2}]"
        dataset_configs["RU"]["split"] = f"train[:{args.data_limit // 2}]"
    
    datasets = prepare_multilingual_datasets(
        tokenizer=tokenizer,
        dataset_configs=dataset_configs,
        max_length=args.max_seq_length,
        train_test_split=0.1,
        seed=args.seed,
    )
    
    # Create model
    print("Creating model...")
    model = create_model_with_switchable_tokenizer(
        model_name_or_path=args.base_model,
        tokenizer=tokenizer,
        from_scratch=getattr(args, 'from_scratch', False),  # Use a default value if attribute doesn't exist
    )
    model.to(args.device)
    
    # Create data collator
    collator = SwitchableDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling
    )
    
    # Initialize loss callback
    loss_callback = LossCallback()
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=switchable_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(switchable_output_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=1,
        do_eval=True,  # Enable evaluation during training
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
        callbacks=[loss_callback],  # Add our callback
    )
    
    # Train the model
    print("Training model...")
    trainer.train()
    
    # Save final model and tokenizer
    model.save_pretrained(os.path.join(switchable_output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(switchable_output_dir, "final_tokenizer"))
    
    return model, tokenizer, datasets, loss_callback.training_losses

def train_monolingual_models(args):
    """Train separate monolingual models for English and Russian."""
    print("\n=== Training Monolingual Models ===")
    
    monolingual_results = {}
    
    # Train a model for each language
    for lang, lang_name in [("EN", "english"), ("RU", "russian")]:
        print(f"\nTraining {lang_name} monolingual model...")
        
        # Create output directory
        lang_output_dir = os.path.join(args.output_dir, f"{lang_name}_model")
        os.makedirs(lang_output_dir, exist_ok=True)
        
        # Load tokenizer
        if lang == "EN":
            tokenizer = AutoTokenizer.from_pretrained(args.en_tokenizer)
        else:  # RU
            tokenizer = AutoTokenizer.from_pretrained(args.ru_tokenizer)
            
        # Set padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load dataset - use data_files for first_shard_only
        if lang == "EN":
            if args.first_shard_only:
                dataset = load_dataset(
                    args.en_dataset, 
                    name=args.en_subset, 
                    data_files=f"{args.en_subset}/train-00000-of-*.parquet",
                    split="train",
                    verification_mode="no_checks"  # Skip verification to avoid mismatch errors
                )
                # Apply the data limit
                if args.data_limit and args.data_limit < len(dataset):
                    dataset = dataset.select(range(args.data_limit))
            else:
                dataset = load_dataset(
                    args.en_dataset, 
                    name=args.en_subset, 
                    split=f"train[:{args.data_limit}]"
                )
        else:  # RU
            if args.first_shard_only:
                dataset = load_dataset(
                    args.ru_dataset, 
                    name=args.ru_subset, 
                    data_files=f"{args.ru_subset}/train-00000-of-*.parquet",
                    split="train",
                    verification_mode="no_checks"  # Skip verification to avoid mismatch errors
                )
                # Apply the data limit
                if args.data_limit and args.data_limit < len(dataset):
                    dataset = dataset.select(range(args.data_limit))
            else:
                dataset = load_dataset(
                    args.ru_dataset, 
                    name=args.ru_subset, 
                    split=f"train[:{args.data_limit}]"
                )
        
        # Split dataset - use manual splitting for first_shard_only to avoid verification errors
        if args.first_shard_only:
            # Use manual train/test split to avoid caching issues
            total_examples = len(dataset)
            test_size = int(total_examples * 0.1)  # Use 10% for testing
            train_size = total_examples - test_size
            
            # Use simple slicing for train/test split
            indices = list(range(total_examples))
            train_indices = indices[:train_size]
            test_indices = indices[train_size:]
            
            train_dataset = dataset.select(train_indices)
            test_dataset = dataset.select(test_indices)
        else:
            # Use standard train_test_split for non-shard data
            dataset_split = dataset.train_test_split(test_size=0.1, seed=args.seed)
            train_dataset, test_dataset = dataset_split["train"], dataset_split["test"]
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding="max_length",
                return_special_tokens_mask=True,
            )
        
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )
        
        # Initialize model
        if getattr(args, 'from_scratch', False):
            config = AutoConfig.from_pretrained(args.base_model)
            config.vocab_size = len(tokenizer)
            model = AutoModelForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.base_model)
            # Resize token embeddings if needed
            if len(tokenizer) != model.config.vocab_size:
                model.resize_token_embeddings(len(tokenizer))
        
        model.to(args.device)
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Initialize loss callback
        loss_callback = LossCallback()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=lang_output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            logging_dir=os.path.join(lang_output_dir, "logs"),
            logging_steps=100,
            save_steps=1000,
            eval_steps=1000,
            save_total_limit=1,
            do_eval=True,  # Enable evaluation during training
            seed=args.seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            callbacks=[loss_callback],  # Add our callback
        )
        
        # Train the model
        trainer.train()
        
        # Save model and tokenizer
        model.save_pretrained(os.path.join(lang_output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(lang_output_dir, "final_tokenizer"))
        
        # Store results
        monolingual_results[lang] = {
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": tokenized_train,
            "test_dataset": tokenized_test,
            "losses": loss_callback.training_losses,  # Store the losses
        }
    
    return monolingual_results

def calculate_tokenization_efficiency(tokenizer, texts, language=None):
    """
    Calculate tokenization efficiency (tokens per word) for a set of texts.
    
    Args:
        tokenizer: The tokenizer to evaluate
        texts: List of text strings to tokenize
        language: Optional language code for switchable tokenizer
    
    Returns:
        float: Average number of tokens per word
    """
    total_tokens = 0
    total_words = 0
    
    for text in texts:
        # Skip empty texts
        if not text or not isinstance(text, str):
            continue
            
        # Count words (simple whitespace-based splitting)
        # This is a simplification; a more sophisticated word tokenizer could be used
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        
        if word_count == 0:
            continue
        
        # Tokenize the text
        try:
            if isinstance(tokenizer, SwitchableTokenizer) and language is not None:
                # For switchable tokenizer, use the language parameter in encode method
                token_ids = tokenizer.encode(text, language=language, add_language_token=False)
                token_count = len(token_ids)  # No special tokens to subtract since add_language_token=False
            else:
                # For regular tokenizer, use encode method
                token_ids = tokenizer.encode(text)
                # Subtract special tokens (typically includes BOS/EOS)
                if hasattr(tokenizer, "num_special_tokens_to_add"):
                    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
                    token_count = len(token_ids) - special_tokens
                else:
                    # Default assumption: 1 special token added
                    token_count = len(token_ids) - 1
            
            # Ensure we don't count negative tokens (in case of errors)
            token_count = max(0, token_count)
            
            # Add to totals
            total_tokens += token_count
            total_words += word_count
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            continue
    
    # Calculate average tokens per word
    if total_words == 0:
        return float('inf')  # Avoid division by zero
    
    return total_tokens / total_words

def evaluate_models(args, switchable_model, switchable_tokenizer, switchable_datasets, monolingual_results):
    """Evaluate both switchable and monolingual models."""
    print("\n=== Evaluating Models ===")
    
    results = {
        "switchable": {},
        "monolingual": {},
        "monolingual_combined": {},  # Add a new section for monolingual models on combined data
        "tokenization_efficiency": {},  # Add section for tokenization efficiency
    }
    
    # Set up SimpleDataset class for all datasets
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, examples):
            self.examples = examples
        
        def __getitem__(self, idx):
            return {k: v.squeeze() for k, v in self.examples[idx].items()}
        
        def __len__(self):
            return len(self.examples)
    
    # Ensure all tokenizers have padding tokens
    print("Setting up tokenizers...")
    # Check/set padding token for switchable tokenizer's underlying tokenizers
    for lang, info in switchable_tokenizer.lang_map.items():
        tokenizer = info["tokenizer"]
        if tokenizer.pad_token is None:
            print(f"Setting pad token for switchable tokenizer {lang}")
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Check/set padding tokens for monolingual tokenizers
    for lang, data in monolingual_results.items():
        tokenizer = data["tokenizer"]
        if tokenizer.pad_token is None:
            print(f"Setting pad token for {lang} monolingual tokenizer")
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Create shared evaluation samples for both tokenization efficiency and perplexity
    print("Preparing shared evaluation samples...")
    shared_evaluation_samples = {
        "EN": [],
        "RU": []
    }
    
    # Extract text samples and create language-specific evaluation datasets
    lang_evaluation_datasets = {}
    
    try:
        # Access the original examples directly from the switchable test dataset
        raw_examples = switchable_datasets["test"].examples
        
        # First collect all samples by language
        for example in raw_examples:
            if "text" in example and "language" in example and example["text"]:
                lang = example["language"]
                if lang in shared_evaluation_samples:
                    # Truncate very long texts to avoid indexing errors
                    text = example["text"]
                    if len(text) > 500:  # Limit text length to avoid tokenization issues
                        text = text[:500]
                    shared_evaluation_samples[lang].append(text)
        
        # If we have too many samples, take a subset to keep evaluation manageable
        for lang in shared_evaluation_samples:
            if len(shared_evaluation_samples[lang]) > 100:
                # Randomly sample 100 examples (using fixed seed for reproducibility)
                np.random.seed(args.seed)
                indices = np.random.choice(
                    len(shared_evaluation_samples[lang]), 
                    100, 
                    replace=False
                )
                shared_evaluation_samples[lang] = [shared_evaluation_samples[lang][i] for i in indices]
            
            print(f"  {lang}: {len(shared_evaluation_samples[lang])} shared evaluation samples")
        
        # Now create tokenized datasets for each language for each model type
        for lang in ["EN", "RU"]:
            if not shared_evaluation_samples[lang]:
                continue
                
            # Prepare switchable tokenizer format for this language
            tokenized_examples = []
            for text in shared_evaluation_samples[lang]:
                try:
                    # Use tokenizer's tokenize/encode methods directly with language parameter
                    if hasattr(switchable_tokenizer, "encode"):
                        # Get tokenizer's model max length
                        max_length = min(args.max_seq_length, 512)  # Use a safer default
                        
                        # Find the appropriate underlying tokenizer
                        lang_info = switchable_tokenizer.lang_map[lang]
                        underlying_tokenizer = lang_info["tokenizer"]
                        
                        # First tokenize with the underlying tokenizer
                        token_ids = underlying_tokenizer.encode(
                            text,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt"
                        )
                        
                        # Create tensors with appropriate shapes
                        input_ids = token_ids
                        labels = input_ids.clone()
                        attention_mask = torch.ones_like(input_ids)
                        
                        # Add language token if needed (prepend to input_ids and update other tensors)
                        lang_token_id = lang_info["lang_token_id"]
                        if lang_token_id is not None:
                            # Add language token to the beginning
                            lang_token_tensor = torch.tensor([[lang_token_id]], device=input_ids.device)
                            input_ids = torch.cat([lang_token_tensor, input_ids], dim=1)
                            labels = torch.cat([lang_token_tensor, labels], dim=1)
                            attention_mask = torch.cat([torch.ones_like(lang_token_tensor), attention_mask], dim=1)
                        
                        tokenized_examples.append({
                            "input_ids": input_ids,
                            "labels": labels,
                            "attention_mask": attention_mask
                        })
                    else:
                        print(f"Warning: switchable_tokenizer doesn't have encode method for {lang}")
                except Exception as e:
                    print(f"Error tokenizing {lang} text: {e}")
                    continue
            
            # Store the dataset for later use if we have any examples
            if tokenized_examples:
                lang_evaluation_datasets[f"switchable_{lang}"] = SimpleDataset(tokenized_examples)
                print(f"Created switchable dataset for {lang} with {len(tokenized_examples)} examples")
            else:
                print(f"Warning: No valid tokenized examples for switchable_{lang}")
            
            # Also create monolingual format datasets if we have the tokenizer
            if lang in monolingual_results:
                mono_tokenizer = monolingual_results[lang]["tokenizer"]
                mono_tokenized = []
                
                for text in shared_evaluation_samples[lang]:
                    try:
                        # Get tokenizer's model max length
                        max_length = min(args.max_seq_length, 512)  # Use a safer default
                        
                        encodings = mono_tokenizer(
                            text,
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        # Add labels for causal LM
                        encodings["labels"] = encodings["input_ids"].clone()
                        mono_tokenized.append(encodings)
                    except Exception as e:
                        print(f"Error tokenizing {lang} text with monolingual tokenizer: {e}")
                        continue
                
                # Create dataset if we have any examples
                if mono_tokenized:
                    lang_evaluation_datasets[f"monolingual_{lang}"] = SimpleDataset(mono_tokenized)
                    print(f"Created monolingual dataset for {lang} with {len(mono_tokenized)} examples")
                else:
                    print(f"Warning: No valid tokenized examples for monolingual_{lang}")
        
        # Create a combined dataset for switchable model evaluation
        combined_texts = []
        for lang in ["EN", "RU"]:
            combined_texts.extend([(text, lang) for text in shared_evaluation_samples[lang]])
        
        # Shuffle the combined texts
        np.random.seed(args.seed)
        np.random.shuffle(combined_texts)
        
        # Tokenize the combined dataset for switchable model
        combined_tokenized = []
        for text, lang in combined_texts:
            try:
                # Get tokenizer's model max length
                max_length = min(args.max_seq_length, 512)  # Use a safer default
                
                # Find the appropriate underlying tokenizer
                lang_info = switchable_tokenizer.lang_map[lang]
                underlying_tokenizer = lang_info["tokenizer"]
                
                # First tokenize with the underlying tokenizer
                token_ids = underlying_tokenizer.encode(
                    text,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                
                # Create tensors with appropriate shapes
                input_ids = token_ids
                labels = input_ids.clone()
                attention_mask = torch.ones_like(input_ids)
                
                # Add language token if needed (prepend to input_ids and update other tensors)
                lang_token_id = lang_info["lang_token_id"]
                if lang_token_id is not None:
                    # Add language token to the beginning
                    lang_token_tensor = torch.tensor([[lang_token_id]], device=input_ids.device)
                    input_ids = torch.cat([lang_token_tensor, input_ids], dim=1)
                    labels = torch.cat([lang_token_tensor, labels], dim=1)
                    attention_mask = torch.cat([torch.ones_like(lang_token_tensor), attention_mask], dim=1)
                
                combined_tokenized.append({
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                })
            except Exception as e:
                print(f"Error tokenizing combined text: {e}")
                continue
        
        # Create combined dataset if we have any examples
        if combined_tokenized:
            lang_evaluation_datasets["switchable_combined"] = SimpleDataset(combined_tokenized)
            print(f"Created combined dataset with {len(combined_tokenized)} examples")
        else:
            print("Warning: No valid tokenized examples for combined dataset")
        
    except Exception as e:
        print(f"Error preparing shared evaluation datasets: {e}")
        import traceback
        traceback.print_exc()
    
    # Now evaluate all models using the shared datasets
    print("\nEvaluating models on shared evaluation datasets:")
    
    # First evaluate switchable model on each language-specific dataset and combined
    if switchable_model is not None:
        for dataset_key in ["switchable_EN", "switchable_RU", "switchable_combined"]:
            if dataset_key in lang_evaluation_datasets:
                dataset = lang_evaluation_datasets[dataset_key]
                lang = dataset_key.split("_")[1]  # Extract language part
                
                if len(dataset) > 0:
                    try:
                        # Create data collator and loader
                        data_collator = SwitchableDataCollator(tokenizer=switchable_tokenizer, mlm=False)
                        loader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            collate_fn=data_collator
                        )
                        
                        # Calculate perplexity
                        print(f"Evaluating switchable model on shared {lang} dataset...")
                        ppl = calculate_perplexity(switchable_model, loader, args.device)
                        results["switchable"][lang] = ppl
                        print(f"Switchable model {lang} perplexity on shared dataset: {ppl:.2f}")
                    except Exception as e:
                        print(f"Error evaluating switchable model on {lang}: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Then evaluate monolingual models on their language-specific datasets
    for lang, lang_name in [("EN", "english"), ("RU", "russian")]:
        if lang in monolingual_results:
            model = monolingual_results[lang]["model"]
            tokenizer = monolingual_results[lang]["tokenizer"]
            dataset_key = f"monolingual_{lang}"
            
            if dataset_key in lang_evaluation_datasets:
                dataset = lang_evaluation_datasets[dataset_key]
                
                if len(dataset) > 0:
                    try:
                        # Create data collator and loader
                        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                        loader = DataLoader(
                            dataset,
                            batch_size=args.batch_size,
                            collate_fn=data_collator
                        )
                        
                        # Calculate perplexity
                        print(f"Evaluating {lang_name} model on shared {lang} dataset...")
                        ppl = calculate_perplexity(model, loader, args.device)
                        results["monolingual"][lang] = ppl
                        print(f"{lang_name.capitalize()} model perplexity on shared dataset: {ppl:.2f}")
                    except Exception as e:
                        print(f"Error evaluating {lang_name} model: {e}")
                        import traceback
                        traceback.print_exc()
    
    # Now evaluate monolingual models on the other language's dataset
    for lang, lang_name in [("EN", "english"), ("RU", "russian")]:
        if lang in monolingual_results:
            model = monolingual_results[lang]["model"]
            tokenizer = monolingual_results[lang]["tokenizer"]
            
            # Determine the "other" language
            other_lang = "RU" if lang == "EN" else "EN"
            other_texts = shared_evaluation_samples[other_lang]
            
            if other_texts:
                try:
                    # Tokenize with this model's tokenizer
                    tokenized_texts = []
                    for text in other_texts:
                        try:
                            # Get tokenizer's model max length
                            max_length = min(args.max_seq_length, 512)  # Use a safer default
                            
                            encodings = tokenizer(
                                text,
                                truncation=True,
                                max_length=max_length,
                                padding="max_length",
                                return_tensors="pt"
                            )
                            encodings["labels"] = encodings["input_ids"].clone()
                            tokenized_texts.append(encodings)
                        except Exception as e:
                            print(f"Error tokenizing other language text: {e}")
                            continue
                    
                    # Create dataset if we have any examples
                    if tokenized_texts:
                        cross_dataset = SimpleDataset(tokenized_texts)
                        
                        # Create loader
                        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
                        loader = DataLoader(
                            cross_dataset,
                            batch_size=args.batch_size,
                            collate_fn=data_collator
                        )
                        
                        # Calculate perplexity
                        print(f"Evaluating {lang_name} model on {other_lang} dataset...")
                        ppl = calculate_perplexity(model, loader, args.device)
                        results["monolingual_combined"][lang] = ppl
                        print(f"{lang_name.capitalize()} model perplexity on {other_lang} dataset: {ppl:.2f}")
                    else:
                        print(f"Warning: No valid tokenized examples for {lang_name} model on {other_lang} data")
                except Exception as e:
                    print(f"Error evaluating {lang_name} model on {other_lang} data: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Calculate tokenization efficiency using the shared evaluation samples
    for lang in ["EN", "RU"]:
        if shared_evaluation_samples[lang]:
            # Calculate for switchable tokenizer
            print(f"Calculating tokenization efficiency for switchable tokenizer on {lang}...")
            try:
                tokens_per_word = calculate_tokenization_efficiency(
                    switchable_tokenizer, 
                    shared_evaluation_samples[lang], 
                    language=lang
                )
                if tokens_per_word > 0 and tokens_per_word != float('inf'):
                    results["tokenization_efficiency"][f"switchable_{lang}"] = tokens_per_word
                    print(f"Switchable tokenizer {lang} tokens per word: {tokens_per_word:.3f}")
            except Exception as e:
                print(f"Error calculating tokenization efficiency for switchable tokenizer on {lang}: {e}")
            
            # Calculate for monolingual tokenizer
            if lang in monolingual_results:
                lang_name = "english" if lang == "EN" else "russian"
                tokenizer = monolingual_results[lang]["tokenizer"]
                print(f"Calculating tokenization efficiency for {lang_name} tokenizer...")
                try:
                    tokens_per_word = calculate_tokenization_efficiency(tokenizer, shared_evaluation_samples[lang])
                    results["tokenization_efficiency"][f"monolingual_{lang}"] = tokens_per_word
                    print(f"{lang_name.capitalize()} tokenizer tokens per word: {tokens_per_word:.3f}")
                except Exception as e:
                    print(f"Error calculating tokenization efficiency for {lang_name} tokenizer: {e}")
    
    # After other evaluation code, add cross-language tokenization efficiency measurements
    print("\nMeasuring cross-language tokenization efficiency...")
    cross_efficiency = {}
    
    # Measure EN tokenizer on RU text
    if "EN" in monolingual_results and shared_evaluation_samples.get("RU"):
        try:
            en_tokenizer = monolingual_results["EN"]["tokenizer"]
            ru_samples = shared_evaluation_samples["RU"]
            
            print("Measuring EN tokenizer efficiency on RU text...")
            en_on_ru_efficiency = calculate_tokenization_efficiency(en_tokenizer, ru_samples)
            cross_efficiency["EN_on_RU"] = en_on_ru_efficiency
            print(f"EN tokenizer on RU text: {en_on_ru_efficiency:.3f} tokens/word")
        except Exception as e:
            print(f"Error measuring EN tokenizer on RU text: {e}")
    
    # Measure RU tokenizer on EN text
    if "RU" in monolingual_results and shared_evaluation_samples.get("EN"):
        try:
            ru_tokenizer = monolingual_results["RU"]["tokenizer"]
            en_samples = shared_evaluation_samples["EN"]
            
            print("Measuring RU tokenizer efficiency on EN text...")
            ru_on_en_efficiency = calculate_tokenization_efficiency(ru_tokenizer, en_samples)
            cross_efficiency["RU_on_EN"] = ru_on_en_efficiency
            print(f"RU tokenizer on EN text: {ru_on_en_efficiency:.3f} tokens/word")
        except Exception as e:
            print(f"Error measuring RU tokenizer on EN text: {e}")
    
    # Store cross-language measurements
    results["cross_tokenization_efficiency"] = cross_efficiency
    
    return results

def plot_results(results, output_dir):
    """Plot the perplexity comparison results."""
    print("\n=== Plotting Results ===")
    
    # Create perplexity figure
    plt.figure(figsize=(10, 6))
    
    # Plot perplexity data
    labels = ['EN', 'RU', 'Combined']
    switchable_values = [
        results["switchable"].get("EN", 0),
        results["switchable"].get("RU", 0),
        results["switchable"].get("combined", 0)
    ]
    
    monolingual_values = [
        results["monolingual"].get("EN", 0),
        results["monolingual"].get("RU", 0),
        min(results["monolingual_combined"].get("EN", float('inf')), 
            results["monolingual_combined"].get("RU", float('inf')))  # Use the better of the two models
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, switchable_values, width, label='Switchable')
    plt.bar(x + width/2, monolingual_values, width, label='Monolingual')
    
    plt.xlabel('Language')
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Model Perplexity Comparison')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(switchable_values):
        if v > 0:
            plt.text(i - width/2, v + 5, f'{v:.1f}', ha='center')
    
    for i, v in enumerate(monolingual_values):
        if v > 0 and v != float('inf'):
            plt.text(i + width/2, v + 5, f'{v:.1f}', ha='center')
    
    # Save the perplexity plot
    plot_path = os.path.join(output_dir, "perplexity_comparison.png")
    plt.savefig(plot_path)
    print(f"Perplexity plot saved to {plot_path}")
    
    # Create tokenization efficiency figure
    plt.figure(figsize=(10, 6))
    
    # Plot tokenization efficiency data
    tokenization_labels = ['EN', 'RU']
    switchable_efficiency = [
        results["tokenization_efficiency"].get("switchable_EN", 0),
        results["tokenization_efficiency"].get("switchable_RU", 0)
    ]
    
    monolingual_efficiency = [
        results["tokenization_efficiency"].get("monolingual_EN", 0),
        results["tokenization_efficiency"].get("monolingual_RU", 0)
    ]
    
    # Print debug info
    print("Tokenization efficiency values for plotting:")
    print(f"Switchable EN: {switchable_efficiency[0]}")
    print(f"Switchable RU: {switchable_efficiency[1]}")
    print(f"Monolingual EN: {monolingual_efficiency[0]}")
    print(f"Monolingual RU: {monolingual_efficiency[1]}")
    
    x = np.arange(len(tokenization_labels))
    width = 0.35
    
    # Ensure we have valid values before plotting
    valid_switchable = [v if v > 0 and v != float('inf') else 0 for v in switchable_efficiency]
    valid_monolingual = [v if v > 0 and v != float('inf') else 0 for v in monolingual_efficiency]
    
    # If all values for either model are 0, set a small value to make the bar visible with a label
    if all(v == 0 for v in valid_switchable) and not all(v == 0 for v in valid_monolingual):
        print("Warning: No valid switchable tokenizer efficiency values, using placeholders for visualization")
        # Use a small fraction of the monolingual value to make visible bars with "N/A" labels
        for i in range(len(valid_switchable)):
            if valid_monolingual[i] > 0:
                valid_switchable[i] = valid_monolingual[i] * 0.1  # Small visible bar
    
    plt.bar(x - width/2, valid_switchable, width, label='Switchable')
    plt.bar(x + width/2, valid_monolingual, width, label='Monolingual')
    
    plt.xlabel('Language')
    plt.ylabel('Tokens per Word (lower is better)')
    plt.title('Tokenization Efficiency Comparison')
    plt.xticks(x, tokenization_labels)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(switchable_efficiency):
        if v > 0 and v != float('inf'):
            plt.text(i - width/2, valid_switchable[i] + 0.05, f'{v:.2f}', ha='center')
        elif valid_switchable[i] > 0:  # It's a placeholder
            plt.text(i - width/2, valid_switchable[i] + 0.05, "N/A", ha='center')
    
    for i, v in enumerate(monolingual_efficiency):
        if v > 0 and v != float('inf'):
            plt.text(i + width/2, v + 0.05, f'{v:.2f}', ha='center')
    
    # Save the tokenization efficiency plot
    efficiency_plot_path = os.path.join(output_dir, "tokenization_efficiency_comparison.png")
    plt.savefig(efficiency_plot_path)
    print(f"Tokenization efficiency plot saved to {efficiency_plot_path}")
    
    # Create detailed tokenization efficiency figure (showing all three tokenizers)
    plt.figure(figsize=(12, 7))
    
    # We need to compare:
    # 1. EN tokenizer on EN text
    # 2. EN tokenizer on RU text
    # 3. RU tokenizer on EN text
    # 4. RU tokenizer on RU text
    # 5. Switchable tokenizer on EN text
    # 6. Switchable tokenizer on RU text
    
    # Create extended keys to look for in results
    extended_keys = {
        "EN_tokenizer_EN": results["tokenization_efficiency"].get("monolingual_EN", 0),
        "RU_tokenizer_RU": results["tokenization_efficiency"].get("monolingual_RU", 0),
        "Switchable_EN": results["tokenization_efficiency"].get("switchable_EN", 0),
        "Switchable_RU": results["tokenization_efficiency"].get("switchable_RU", 0),
    }
    
    # If we have cross-language measurements, add them
    if "cross_tokenization_efficiency" in results:
        extended_keys["EN_tokenizer_RU"] = results["cross_tokenization_efficiency"].get("EN_on_RU", 0)
        extended_keys["RU_tokenizer_EN"] = results["cross_tokenization_efficiency"].get("RU_on_EN", 0)
    else:
        # Set to 0 if not available
        extended_keys["EN_tokenizer_RU"] = 0
        extended_keys["RU_tokenizer_EN"] = 0
    
    # Print debug info for detailed plot
    print("\nDetailed tokenization efficiency values:")
    for key, value in extended_keys.items():
        print(f"{key}: {value}")
    
    # Group data for plotting
    labels = ['EN Text', 'RU Text']
    en_tokenizer_values = [extended_keys["EN_tokenizer_EN"], extended_keys["EN_tokenizer_RU"]]
    ru_tokenizer_values = [extended_keys["RU_tokenizer_EN"], extended_keys["RU_tokenizer_RU"]]
    switchable_values = [extended_keys["Switchable_EN"], extended_keys["Switchable_RU"]]
    
    # Ensure all values are valid
    en_tokenizer_values = [v if v > 0 and v != float('inf') else 0 for v in en_tokenizer_values]
    ru_tokenizer_values = [v if v > 0 and v != float('inf') else 0 for v in ru_tokenizer_values]
    switchable_values = [v if v > 0 and v != float('inf') else 0 for v in switchable_values]
    
    x = np.arange(len(labels))
    width = 0.25
    
    # Plot the bars
    plt.bar(x - width, en_tokenizer_values, width, label='EN Tokenizer')
    plt.bar(x, ru_tokenizer_values, width, label='RU Tokenizer')
    plt.bar(x + width, switchable_values, width, label='Switchable Tokenizer')
    
    plt.xlabel('Text Language')
    plt.ylabel('Tokens per Word (lower is better)')
    plt.title('Detailed Tokenization Efficiency by Tokenizer and Language')
    plt.xticks(x, labels)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(en_tokenizer_values):
        if v > 0:
            plt.text(i - width, v + 0.05, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(ru_tokenizer_values):
        if v > 0:
            plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(switchable_values):
        if v > 0:
            plt.text(i + width, v + 0.05, f'{v:.2f}', ha='center')
    
    # Save the detailed tokenization efficiency plot
    detailed_efficiency_path = os.path.join(output_dir, "detailed_tokenization_efficiency.png")
    plt.savefig(detailed_efficiency_path)
    print(f"Detailed tokenization efficiency plot saved to {detailed_efficiency_path}")
    
    # Plot cross-language tokenization efficiency
    if "cross_tokenization_efficiency" in results:
        plt.figure(figsize=(8, 6))
        
        # Extract cross-tokenization data
        cross_data = results["cross_tokenization_efficiency"]
        
        # Create bar chart data
        cross_labels = ['EN→RU', 'RU→EN']
        cross_values = [
            cross_data.get('EN_on_RU', 0),
            cross_data.get('RU_on_EN', 0)
        ]
        
        # Ensure values are valid
        cross_values = [v if v > 0 and v != float('inf') else 0 for v in cross_values]
        
        # Create bar chart with gradient colors
        plt.bar(cross_labels, cross_values, color=['#ff9999', '#99ccff'])
        
        plt.ylabel('Tokens per Word')
        plt.title('Cross-Language Tokenization Efficiency')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add values on top of bars
        for i, v in enumerate(cross_values):
            if v > 0:
                plt.text(i, v + 0.1, f'{v:.2f}', ha='center')
        
        # Save the plot
        cross_plot_path = os.path.join(output_dir, "cross_tokenization_efficiency.png")
        plt.savefig(cross_plot_path)
        print(f"Cross-tokenization efficiency plot saved to {cross_plot_path}")
    
    # Show the plots if in a GUI environment
    plt.show()

def plot_losses(switchable_losses, monolingual_results, output_dir):
    """Plot training losses for all models."""
    print("\n=== Plotting Training Losses ===")
    
    plt.figure(figsize=(10, 6))
    
    # Plot switchable model losses
    if switchable_losses:
        steps, losses = zip(*switchable_losses)
        plt.plot(steps, losses, label='Switchable Model', linewidth=2)
    
    # Plot monolingual model losses
    colors = {'EN': 'red', 'RU': 'green'}
    for lang, lang_name in [("EN", "English"), ("RU", "Russian")]:
        if lang in monolingual_results and "losses" in monolingual_results[lang]:
            lang_losses = monolingual_results[lang]["losses"]
            if lang_losses:
                steps, losses = zip(*lang_losses)
                plt.plot(steps, losses, label=f'{lang_name} Model', color=colors[lang], linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot
    losses_plot_path = os.path.join(output_dir, "training_losses.png")
    plt.savefig(losses_plot_path)
    print(f"Training losses plot saved to {losses_plot_path}")
    
    # Show the plot if in a GUI environment
    plt.show()

def get_hf_token(args):
    """
    Get Hugging Face token from secure sources in this priority order:
    1. From HF_TOKEN environment variable (set by Colab secrets)
    2. From .env file
    3. From HUGGING_FACE_HUB_TOKEN environment variable
    4. From command line argument
    """
    # First check for HF_TOKEN (likely from Colab secrets)
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
        
    # Try loading from .env file next
    env_path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
        
    # Check other sources in priority order
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    
    # Command line argument as fallback
    if args.hub_token:
        print("Warning: Using token from command line is less secure than using secrets or .env file")
        return args.hub_token
        
    return None

def upload_to_huggingface(args, output_dir):
    """Upload experiment results to Hugging Face Hub."""
    print("\n=== Uploading Results to Hugging Face Hub ===")
    
    if args.hub_repo_id is None:
        print("Error: --hub_repo_id is required for uploading to Hugging Face Hub")
        return False
    
    # Get token securely
    token = get_hf_token(args)
    
    if token is None:
        print("Error: No token found. Please set HF_TOKEN in .env file or environment variable")
        print("Create a .env file in the project root with: HF_TOKEN=your_token_here")
        return False
    
    try:
        # Initialize the Hugging Face API
        api = HfApi(token=token)
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=args.hub_repo_id,
                exist_ok=True,
                private=args.hub_private,
            )
            print(f"Repository {args.hub_repo_id} created or already exists")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return False
        
        # Create a temporary directory with the content to upload
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create directories in the temporary folder
            final_tmp_dir = os.path.join(tmp_dir, "models_and_results")
            os.makedirs(final_tmp_dir, exist_ok=True)
            
            # Copy only final models and tokenizers (not checkpoints)
            print("Copying final models and tokenizers...")
            
            # Switchable model
            switchable_final_model = os.path.join(output_dir, "switchable_model", "final_model")
            switchable_final_tokenizer = os.path.join(output_dir, "switchable_model", "final_tokenizer")
            
            if os.path.exists(switchable_final_model):
                switchable_model_dest = os.path.join(final_tmp_dir, "switchable_model")
                os.makedirs(switchable_model_dest, exist_ok=True)
                shutil.copytree(switchable_final_model, os.path.join(switchable_model_dest, "model"), dirs_exist_ok=True)
                shutil.copytree(switchable_final_tokenizer, os.path.join(switchable_model_dest, "tokenizer"), dirs_exist_ok=True)
            
            # English model
            english_final_model = os.path.join(output_dir, "english_model", "final_model")
            english_final_tokenizer = os.path.join(output_dir, "english_model", "final_tokenizer")
            
            if os.path.exists(english_final_model):
                english_model_dest = os.path.join(final_tmp_dir, "english_model")
                os.makedirs(english_model_dest, exist_ok=True)
                shutil.copytree(english_final_model, os.path.join(english_model_dest, "model"), dirs_exist_ok=True)
                shutil.copytree(english_final_tokenizer, os.path.join(english_model_dest, "tokenizer"), dirs_exist_ok=True)
            
            # Russian model
            russian_final_model = os.path.join(output_dir, "russian_model", "final_model")
            russian_final_tokenizer = os.path.join(output_dir, "russian_model", "final_tokenizer")
            
            if os.path.exists(russian_final_model):
                russian_model_dest = os.path.join(final_tmp_dir, "russian_model")
                os.makedirs(russian_model_dest, exist_ok=True)
                shutil.copytree(russian_final_model, os.path.join(russian_model_dest, "model"), dirs_exist_ok=True)
                shutil.copytree(russian_final_tokenizer, os.path.join(russian_model_dest, "tokenizer"), dirs_exist_ok=True)
            
            # Copy result files and plots
            print("Copying result files and plots...")
            results_dir = os.path.join(final_tmp_dir, "results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Copy all PNG plot files
            for plot_file in ["perplexity_comparison.png", "tokenization_efficiency_comparison.png", 
                             "detailed_tokenization_efficiency.png", "cross_tokenization_efficiency.png",
                             "training_losses.png"]:
                plot_path = os.path.join(output_dir, plot_file)
                if os.path.exists(plot_path):
                    shutil.copy2(plot_path, os.path.join(results_dir, plot_file))
            
            # Copy results text file
            results_txt = os.path.join(output_dir, "experiment1_results.txt")
            if os.path.exists(results_txt):
                shutil.copy2(results_txt, os.path.join(results_dir, "experiment1_results.txt"))
            
            # Create a README.md file with experiment information
            readme_path = os.path.join(tmp_dir, "README.md")
            with open(readme_path, "w") as f:
                f.write(f"# Switchable Tokenizer Experiment Results\n\n")
                f.write(f"## Experiment Settings\n\n")
                f.write(f"- Base model: {args.base_model}\n")
                f.write(f"- English tokenizer: {args.en_tokenizer}\n")
                f.write(f"- Russian tokenizer: {args.ru_tokenizer}\n")
                f.write(f"- Training from scratch: {getattr(args, 'from_scratch', False)}\n")
                f.write(f"- Data limit: {args.data_limit} samples\n")
                f.write(f"- Epochs: {args.epochs}\n\n")
                f.write(f"## Results Overview\n\n")
                f.write(f"This repository contains:\n\n")
                f.write(f"- Final trained models (no checkpoints)\n")
                f.write(f"- Tokenizers\n")
                f.write(f"- Performance plots (perplexity, tokenization efficiency, training losses)\n")
                f.write(f"- Detailed results in experiment1_results.txt\n\n")
                f.write(f"For more details on the implementation, visit the [Switch-Tokenizer project repository](https://github.com/yourusername/switch-tokenizer).\n")
            
            # Upload the content to Hugging Face Hub
            print(f"Uploading only final models and results to {args.hub_repo_id}...")
            upload_folder(
                folder_path=tmp_dir,
                repo_id=args.hub_repo_id,
                token=token,
                repo_type="model",
                ignore_patterns=["*.pyc", "__pycache__", "*.git*"],
            )
            
            print(f"Results successfully uploaded to https://huggingface.co/{args.hub_repo_id}")
            return True
            
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    args = parse_args()
    set_seed(args.seed)
    
    print("Starting Experiment 1...")
    print(f"Using device: {args.device}")
    print(f"Training from scratch: {getattr(args, 'from_scratch', False)}")
    print(f"Sample sizes are balanced: the switchable model uses {args.data_limit // 2} samples per language (total: {args.data_limit})")
    print(f"Each monolingual model uses {args.data_limit} samples of its language")
    
    # Step 1: Train the switchable tokenizer model
    print("\nStep 1: Training switchable tokenizer model...")
    switchable_model, switchable_tokenizer, switchable_datasets, switchable_losses = train_switchable_model(args)
    
    # Step 2: Train monolingual models
    print("\nStep 2: Training monolingual models...")
    monolingual_results = train_monolingual_models(args)
    
    # Step 3: Evaluate the models
    print("\nStep 3: Evaluating models...")
    results = evaluate_models(
        args,
        switchable_model,
        switchable_tokenizer,
        switchable_datasets,
        monolingual_results
    )
    
    # Step 4: Plot the results
    print("\nStep 4: Plotting results...")
    plot_results(results, args.output_dir)
    
    # Step 5: Plot the training losses
    print("\nPlotting training losses...")
    plot_losses(switchable_losses, monolingual_results, args.output_dir)
    
    # Step 6: Save the results
    results_file = os.path.join(args.output_dir, "experiment1_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Experiment 1: Feasibility and Performance ===\n\n")
        
        # Write training approach
        training_type = "from scratch" if getattr(args, 'from_scratch', False) else "fine-tuned"
        if args.first_shard_only:
            f.write(f"Models were trained {training_type} on the first shard of each language dataset for {args.epochs} epochs.\n")
        else:
            f.write(f"Models were trained {training_type} for {args.epochs} epochs.\n")
            f.write(f"Sample sizes were balanced:\n")
            f.write(f"  - Switchable model: {args.data_limit // 2} examples per language (total: {args.data_limit})\n")
            f.write(f"  - Each monolingual model: {args.data_limit} examples of its language\n")
        f.write(f"Base model: {args.base_model}\n")
        f.write(f"English tokenizer: {args.en_tokenizer}\n")
        f.write(f"Russian tokenizer: {args.ru_tokenizer}\n\n")
        
        # Write switchable model results
        f.write("Switchable Model Perplexity:\n")
        for key, value in results["switchable"].items():
            f.write(f"  {key}: {value:.2f}\n")
        
        # Write monolingual model results
        f.write("\nMonolingual Models Perplexity:\n")
        for key, value in results["monolingual"].items():
            f.write(f"  {key}: {value:.2f}\n")
        
        # Write monolingual combined results
        f.write("\nMonolingual Models on Combined Data Perplexity:\n")
        for key, value in results["monolingual_combined"].items():
            if value != float('inf'):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: Failed to evaluate\n")
        
        # Calculate best monolingual on combined
        best_mono_combined = min(
            results["monolingual_combined"].get("EN", float('inf')),
            results["monolingual_combined"].get("RU", float('inf'))
        )
        if best_mono_combined != float('inf'):
            f.write(f"  Best: {best_mono_combined:.2f}\n")
        else:
            f.write(f"  Best: Failed to evaluate\n")
        
        # Write tokenization efficiency results
        f.write("\nTokenization Efficiency (Tokens per Word, lower is better):\n")
        for lang in ["EN", "RU"]:
            switchable_key = f"switchable_{lang}"
            monolingual_key = f"monolingual_{lang}"
            
            if switchable_key in results["tokenization_efficiency"] and monolingual_key in results["tokenization_efficiency"]:
                switchable_eff = results["tokenization_efficiency"][switchable_key]
                monolingual_eff = results["tokenization_efficiency"][monolingual_key]
                
                f.write(f"  {lang}:\n")
                f.write(f"    Switchable: {switchable_eff:.3f} tokens/word\n")
                f.write(f"    Monolingual: {monolingual_eff:.3f} tokens/word\n")
                
                # Calculate efficiency difference
                if switchable_eff > 0 and monolingual_eff > 0 and switchable_eff != float('inf') and monolingual_eff != float('inf'):
                    diff = ((switchable_eff - monolingual_eff) / monolingual_eff) * 100
                    if diff > 0:
                        f.write(f"    → Switchable uses {diff:.1f}% more tokens per word\n")
                    else:
                        f.write(f"    → Switchable uses {abs(diff):.1f}% fewer tokens per word\n")
        
        # Write comparison
        f.write("\nPerformance Comparison:\n")
        for lang in ["EN", "RU"]:
            if lang in results["switchable"] and lang in results["monolingual"]:
                switchable_ppl = results["switchable"][lang]
                monolingual_ppl = results["monolingual"][lang]
                diff = monolingual_ppl - switchable_ppl
                perc_diff = (diff / monolingual_ppl) * 100
                
                if diff > 0:
                    f.write(f"  {lang}: Switchable model is {abs(perc_diff):.2f}% better\n")
                else:
                    f.write(f"  {lang}: Monolingual model is {abs(perc_diff):.2f}% better\n")
        
        # Add comparison for combined data
        if results["switchable"].get("combined", float('inf')) != float('inf') and best_mono_combined != float('inf'):
            switchable_combined = results["switchable"]["combined"]
            diff = best_mono_combined - switchable_combined
            perc_diff = (diff / best_mono_combined) * 100
            
            if diff > 0:
                f.write(f"  Combined: Switchable model is {abs(perc_diff):.2f}% better\n")
            else:
                f.write(f"  Combined: Best monolingual model is {abs(perc_diff):.2f}% better\n")
    
    # Add training losses to results file
    with open(results_file, "a") as f:
        f.write("\nTraining Losses:\n")
        f.write("  See the 'training_losses.png' file for a visualization of training losses.\n")
    
    print(f"\nResults saved to {results_file}")
    
    # Check if huggingface_hub and dotenv are installed when upload_to_hub is set
    if args.upload_to_hub:
        try:
            import huggingface_hub
        except ImportError:
            print("\nERROR: To upload to HuggingFace Hub, you need to install the huggingface_hub package:")
            print("pip install huggingface_hub")
            sys.exit(1)
            
        try:
            import dotenv
        except ImportError:
            print("\nWARNING: python-dotenv package not found. For secure token handling:")
            print("pip install python-dotenv")
            
    # Add hubupload after all other steps are done
    if args.upload_to_hub:
        print("\nUploading results to Hugging Face Hub...")
        success = upload_to_huggingface(args, args.output_dir)
        if success:
            print(f"Results uploaded successfully to Hugging Face Hub: {args.hub_repo_id}")
        else:
            print("Failed to upload results to Hugging Face Hub")
    
    print("\nExperiment 1 completed!")

if __name__ == "__main__":
    main() 