#!/usr/bin/env python
"""
Experiment 1: Feasibility and Performance

Trains the shared 64k model and monolingual models of similar size,
then evaluates and compares their performance on held-out test sets.
"""

import os
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed
)
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.switch_tokenizer import SwitchableTokenizer
from src.model_utils import create_model_with_switchable_tokenizer
from src.data_utils import (
    prepare_multilingual_datasets,
    SwitchableDataCollator,
    create_data_loaders
)
from src.evaluation import calculate_perplexity

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: Feasibility and Performance")
    
    # Dataset paths
    parser.add_argument("--en_dataset", type=str, default='wikimedia/wikipedia' # "Salesforce/wikitext", 
                        help="Path to English dataset")
    parser.add_argument("--en_subset", type=str, default='20231101.en', # "wikitext-103-v1", 
                        help="English dataset subset")
    parser.add_argument("--ru_dataset", type=str, default="wikimedia/wikipedia", 
                        help="Path to Russian dataset")
    parser.add_argument("--ru_subset", type=str, default="20231101.ru", 
                        help="Russian dataset subset")
    parser.add_argument("--data_limit", type=int, default=10000, 
                        help="Limit number of examples per language (for faster experiments)")
    
    # Tokenizer paths
    parser.add_argument("--en_tokenizer", type=str, default="gpt2", 
                        help="English tokenizer")
    parser.add_argument("--ru_tokenizer", type=str, default="DeepPavlov/rubert-base-cased", 
                        help="Russian tokenizer")
    
    # Base model
    parser.add_argument("--base_model", type=str, default="gpt2-medium", 
                        help="Base model architecture")
    
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
    
    return parser.parse_args()

def train_switchable_model(args):
    """Train the shared 64k model with switchable tokenizer."""
    print("\n=== Training Switchable Tokenizer Model ===")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    switchable_output_dir = os.path.join(args.output_dir, "switchable_model")
    os.makedirs(switchable_output_dir, exist_ok=True)
    
    # Initialize switchable tokenizer
    tokenizer = SwitchableTokenizer(
        en_tokenizer_path=args.en_tokenizer,
        ru_tokenizer_path=args.ru_tokenizer,
    )
    
    # Prepare datasets
    print("Preparing datasets...")
    dataset_configs = {
        "EN": {
            "path": args.en_dataset, 
            "name": args.en_subset,
            "split": f"train[:{args.data_limit}]"
        },
        "RU": {
            "path": args.ru_dataset, 
            "name": args.ru_subset, 
            "split": f"train[:{args.data_limit}]"
        },
    }
    
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
    )
    model.to(args.device)
    
    # Create data collator
    collator = SwitchableDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling
    )
    
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
        evaluation_strategy="steps",
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
    )
    
    # Train the model
    print("Training model...")
    trainer.train()
    
    # Save final model and tokenizer
    model.save_pretrained(os.path.join(switchable_output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(switchable_output_dir, "final_tokenizer"))
    
    return model, tokenizer, datasets

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
        
        # Load dataset
        if lang == "EN":
            dataset = load_dataset(args.en_dataset, split=f"train[:{args.data_limit}]")
        else:  # RU
            dataset = load_dataset(args.ru_dataset, name=args.ru_subset, split=f"train[:{args.data_limit}]")
        
        # Split dataset
        dataset_split = dataset.train_test_split(test_size=0.1, seed=args.seed)
        train_dataset, test_dataset = dataset_split["train"], dataset_split["test"]
        
        # Tokenize dataset
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
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
            evaluation_strategy="steps",
            seed=args.seed,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
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
        }
    
    return monolingual_results

def evaluate_models(args, switchable_model, switchable_tokenizer, switchable_datasets, monolingual_results):
    """Evaluate both switchable and monolingual models."""
    print("\n=== Evaluating Models ===")
    
    results = {
        "switchable": {},
        "monolingual": {},
    }
    
    # Create data loaders for switchable model
    switchable_loaders = create_data_loaders(
        datasets={"test": switchable_datasets["test"]},
        collator=SwitchableDataCollator(tokenizer=switchable_tokenizer, mlm=False),
        batch_size=args.batch_size,
        shuffle_train=False,
    )
    
    # Evaluate switchable model on combined data
    print("Evaluating switchable model on combined test set...")
    switchable_ppl = calculate_perplexity(switchable_model, switchable_loaders["test"], args.device)
    results["switchable"]["combined"] = switchable_ppl
    print(f"Switchable model combined perplexity: {switchable_ppl:.2f}")
    
    # Evaluate switchable model on each language
    for lang in ["EN", "RU"]:
        # Create dataset with just this language
        dataset_config = {
            lang: {"path": args.en_dataset if lang == "EN" else args.ru_dataset,
                  "name": args.ru_subset if lang == "RU" else None,
                  "split": f"test[:{args.data_limit // 10}]"}
        }
        
        lang_datasets = prepare_multilingual_datasets(
            tokenizer=switchable_tokenizer,
            dataset_configs=dataset_config,
            max_length=args.max_seq_length,
            train_test_split=None,
        )
        
        # Create data loader
        lang_loaders = create_data_loaders(
            datasets={"eval": lang_datasets["train"]},
            collator=SwitchableDataCollator(tokenizer=switchable_tokenizer, mlm=False),
            batch_size=args.batch_size,
            shuffle_train=False,
        )
        
        # Calculate perplexity
        print(f"Evaluating switchable model on {lang} test set...")
        lang_ppl = calculate_perplexity(switchable_model, lang_loaders["eval"], args.device)
        results["switchable"][lang] = lang_ppl
        print(f"Switchable model {lang} perplexity: {lang_ppl:.2f}")
    
    # Evaluate monolingual models
    for lang, lang_name in [("EN", "english"), ("RU", "russian")]:
        if lang in monolingual_results:
            print(f"Evaluating {lang_name} monolingual model...")
            
            model = monolingual_results[lang]["model"]
            tokenizer = monolingual_results[lang]["tokenizer"]
            test_dataset = monolingual_results[lang]["test_dataset"]
            
            # Calculate perplexity
            def calculate_mono_perplexity(model, dataset, tokenizer, device):
                model.eval()
                total_loss = 0
                total_length = 0
                
                data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer,
                    mlm=False,
                )
                
                # Create batches
                batch_size = args.batch_size
                batches = []
                for i in range(0, len(dataset), batch_size):
                    batch = dataset[i:i+batch_size]
                    batches.append(data_collator([{k: v for k, v in zip(batch.keys(), values)} 
                                                 for values in zip(*batch.values())]))
                
                with torch.no_grad():
                    for batch in tqdm(batches, desc=f"Calculating {lang} perplexity"):
                        # Move batch to device
                        input_ids = batch["input_ids"].to(device)
                        labels = batch["labels"].to(device)
                        attention_mask = None
                        if "attention_mask" in batch:
                            attention_mask = batch["attention_mask"].to(device)
                        
                        # Forward pass
                        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                        
                        # Get loss
                        loss = outputs.loss.item()
                        length = (labels != -100).sum().item()
                        
                        total_loss += loss * length
                        total_length += length
                
                # Calculate perplexity
                avg_loss = total_loss / total_length if total_length > 0 else float('inf')
                perplexity = np.exp(avg_loss)
                
                return perplexity
            
            monolingual_ppl = calculate_mono_perplexity(model, test_dataset, tokenizer, args.device)
            results["monolingual"][lang] = monolingual_ppl
            print(f"{lang_name.capitalize()} monolingual model perplexity: {monolingual_ppl:.2f}")
    
    return results

def plot_results(results, output_dir):
    """Plot the perplexity comparison results."""
    print("\n=== Plotting Results ===")
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot data
    labels = ['EN', 'RU', 'Combined']
    switchable_values = [
        results["switchable"].get("EN", 0),
        results["switchable"].get("RU", 0),
        results["switchable"].get("combined", 0)
    ]
    
    monolingual_values = [
        results["monolingual"].get("EN", 0),
        results["monolingual"].get("RU", 0),
        0  # No combined value for monolingual models
    ]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, switchable_values, width, label='Switchable 64k')
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
        if v > 0:
            plt.text(i + width/2, v + 5, f'{v:.1f}', ha='center')
    
    # Save the plot
    plot_path = os.path.join(output_dir, "perplexity_comparison.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Show the plot if in a GUI environment
    plt.show()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Step 1: Train the switchable tokenizer model
    switchable_model, switchable_tokenizer, switchable_datasets = train_switchable_model(args)
    
    # Step 2: Train monolingual models
    monolingual_results = train_monolingual_models(args)
    
    # Step 3: Evaluate the models
    results = evaluate_models(
        args,
        switchable_model,
        switchable_tokenizer,
        switchable_datasets,
        monolingual_results
    )
    
    # Step 4: Plot the results
    plot_results(results, args.output_dir)
    
    # Step 5: Save the results
    results_file = os.path.join(args.output_dir, "experiment1_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Experiment 1: Feasibility and Performance ===\n\n")
        
        # Write switchable model results
        f.write("Switchable Model Perplexity:\n")
        for key, value in results["switchable"].items():
            f.write(f"  {key}: {value:.2f}\n")
        
        # Write monolingual model results
        f.write("\nMonolingual Models Perplexity:\n")
        for key, value in results["monolingual"].items():
            f.write(f"  {key}: {value:.2f}\n")
        
        # Write comparison
        f.write("\nComparison:\n")
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
    
    print(f"\nResults saved to {results_file}")
    print("\nExperiment 1 completed!")

if __name__ == "__main__":
    main() 