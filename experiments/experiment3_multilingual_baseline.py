#!/usr/bin/env python
"""
Experiment 3: Comparison vs. Standard Multilingual Baseline

Compares the switchable tokenizer model with a standard multilingual model
using a single tokenizer trained on combined data with a 64k vocabulary.
"""

import os
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed
)
from datasets import Dataset
import matplotlib.pyplot as plt

from src.switch_tokenizer import SwitchableTokenizer
from src.model_utils import create_model_with_switchable_tokenizer
from src.data_utils import (
    SwitchableDataCollator,
    create_data_loaders
)
from src.evaluation import calculate_perplexity
from src.tokenizer_utils import (
    load_multilingual_corpus,
    train_concatenated_tokenizer
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Comparison vs. Standard Multilingual Baseline"
    )
    
    # Dataset paths
    parser.add_argument("--en_dataset", type=str, default="wikimedia/wikipedia", 
                        help="Path to English dataset")
    parser.add_argument("--en_subset", type=str, default="20231101.en", 
                        help="English dataset subset")
    parser.add_argument("--ru_dataset", type=str, default="wikimedia/wikipedia", 
                        help="Path to Russian dataset")
    parser.add_argument("--ru_subset", type=str, default="20231101.ru", 
                        help="Russian dataset subset")
    parser.add_argument("--data_limit", type=int, default=10000, 
                        help="Limit number of examples per language")
    parser.add_argument("--first_shard_only", action="store_true", 
                        help="Use only the first shard (train-00000) of each dataset for faster testing")
    
    # Switchable tokenizer paths (if already trained)
    parser.add_argument("--switchable_tokenizer", type=str, default=None,
                        help="Path to pretrained switchable tokenizer (optional)")
    parser.add_argument("--en_tokenizer", type=str, default="gpt2",
                        help="English tokenizer path/name (used if switchable_tokenizer not provided)")
    parser.add_argument("--ru_tokenizer", type=str, default="ai-forever/ruGPT-3.5-13B",
                        help="Russian tokenizer")
    
    # Base model
    parser.add_argument("--base_model", type=str, default="gpt2-medium", 
                        help="Base model architecture")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Initialize model from scratch instead of using pretrained weights")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./experiment3_output", 
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

def load_and_prepare_data(args):
    """Load and prepare the datasets."""
    print("\n=== Loading Data ===")
    
    # Load multilingual corpus
    corpus = load_multilingual_corpus(
        en_dataset_name=args.en_dataset,
        ru_dataset_name=args.ru_dataset,
        en_subset=args.en_subset,
        ru_subset=args.ru_subset,
        limit=args.data_limit,
        first_shard_only=args.first_shard_only,
    )
    
    # Reserve 10% of the data for testing
    test_corpus = {
        "en": corpus["en"][-args.data_limit//10:],
        "ru": corpus["ru"][-args.data_limit//10:],
    }
    
    # The rest for training
    train_corpus = {
        "en": corpus["en"][:-args.data_limit//10],
        "ru": corpus["ru"][:-args.data_limit//10],
    }
    
    return train_corpus, test_corpus

def prepare_switchable_tokenizer(args, train_corpus):
    """Prepare the switchable tokenizer."""
    print("\n=== Preparing Switchable Tokenizer ===")
    
    # If a switchable tokenizer path is provided, load it
    if args.switchable_tokenizer:
        print(f"Loading switchable tokenizer from {args.switchable_tokenizer}")
        tokenizer = SwitchableTokenizer.from_pretrained(args.switchable_tokenizer)
    else:
        # Otherwise, create a new one from the specified tokenizers
        print(f"Creating switchable tokenizer from {args.en_tokenizer} and {args.ru_tokenizer}")
        tokenizer = SwitchableTokenizer(
            en_tokenizer_path=args.en_tokenizer,
            ru_tokenizer_path=args.ru_tokenizer,
            shared_vocab_size=None,  # Automatically determine the maximum safe shared vocab size
        )
    
    return tokenizer

def train_standard_multilingual_tokenizer(args, train_corpus):
    """Train a standard multilingual tokenizer with 64k vocabulary."""
    print("\n=== Training Standard Multilingual Tokenizer ===")
    
    # Create output directory
    tokenizer_dir = os.path.join(args.output_dir, "tokenizers")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Train a standard multilingual tokenizer
    standard_dir = train_concatenated_tokenizer(
        corpus=train_corpus,
        output_dir=tokenizer_dir,
        vocab_size=64000,  # 64k vocab for combined languages (same size as switchable)
    )
    
    return AutoTokenizer.from_pretrained(standard_dir)

def prepare_switchable_datasets(tokenizer, train_corpus, test_corpus, args):
    """Prepare datasets for the switchable tokenizer model."""
    print("\n=== Preparing Switchable Datasets ===")
    
    # Prepare datasets
    train_data = []
    for text in train_corpus["en"]:
        train_data.append({"text": text, "language": "EN"})
    for text in train_corpus["ru"]:
        train_data.append({"text": text, "language": "RU"})
    
    test_data = []
    for text in test_corpus["en"]:
        test_data.append({"text": text, "language": "EN"})
    for text in test_corpus["ru"]:
        test_data.append({"text": text, "language": "RU"})
    
    # Create dataset objects
    train_dataset = Dataset.from_dict({
        "text": [item["text"] for item in train_data],
        "language": [item["language"] for item in train_data],
    })
    
    test_dataset = Dataset.from_dict({
        "text": [item["text"] for item in test_data],
        "language": [item["language"] for item in test_data],
    })
    
    # Tokenize datasets
    def tokenize_function(examples):
        tokenized_inputs = {"input_ids": [], "attention_mask": []}
        
        for text, lang in zip(examples["text"], examples["language"]):
            encoding = tokenizer.encode(
                text, 
                language=lang,
                add_language_token=True,
                max_length=args.max_seq_length,
                truncation=True,
            )
            
            # Convert to tensors and add padding
            input_ids = torch.tensor(encoding, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            tokenized_inputs["input_ids"].append(input_ids)
            tokenized_inputs["attention_mask"].append(attention_mask)
        
        return tokenized_inputs
    
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "language"],
    )
    
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "language"],
    )
    
    return tokenized_train, tokenized_test

def prepare_standard_datasets(tokenizer, train_corpus, test_corpus, args):
    """Prepare datasets for the standard multilingual model."""
    print("\n=== Preparing Standard Multilingual Datasets ===")
    
    # Prepare datasets
    train_texts = []
    for text in train_corpus["en"]:
        train_texts.append("<en> " + text)
    for text in train_corpus["ru"]:
        train_texts.append("<ru> " + text)
    
    test_texts = []
    for text in test_corpus["en"]:
        test_texts.append("<en> " + text)
    for text in test_corpus["ru"]:
        test_texts.append("<ru> " + text)
    
    # Create dataset objects
    train_dataset = Dataset.from_dict({"text": train_texts})
    test_dataset = Dataset.from_dict({"text": test_texts})
    
    # Tokenize datasets
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
    
    return tokenized_train, tokenized_test

def train_switchable_model(args, tokenizer, train_dataset, test_dataset):
    """Train a model with the switchable tokenizer."""
    print("\n=== Training Switchable Tokenizer Model ===")
    
    # Create output directory
    switchable_output_dir = os.path.join(args.output_dir, "switchable_model")
    os.makedirs(switchable_output_dir, exist_ok=True)
    
    # Create model
    if getattr(args, 'from_scratch', False):
        print("Initializing switchable model from scratch...")
        model = create_model_with_switchable_tokenizer(
            model_name_or_path=args.base_model,
            tokenizer=tokenizer,
            from_scratch=True,
        )
    else:
        print(f"Initializing switchable model from pretrained {args.base_model}...")
        model = create_model_with_switchable_tokenizer(
            model_name_or_path=args.base_model,
            tokenizer=tokenizer,
            from_scratch=False,
        )
    
    model.to(args.device)
    
    # Set up data collator
    data_collator = SwitchableDataCollator(
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
        do_eval=True,  # Enable evaluation during training
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(switchable_output_dir, "final_model"))
    
    return model

def train_standard_model(args, tokenizer, train_dataset, test_dataset):
    """Train a model with the standard multilingual tokenizer."""
    print("\n=== Training Standard Multilingual Model ===")
    
    # Create output directory
    standard_output_dir = os.path.join(args.output_dir, "standard_model")
    os.makedirs(standard_output_dir, exist_ok=True)
    
    # Initialize model
    if getattr(args, 'from_scratch', False):
        print("Initializing standard model from scratch...")
        config = AutoConfig.from_pretrained(args.base_model)
        model = AutoModelForCausalLM.from_config(config)
    else:
        print(f"Initializing standard model from pretrained {args.base_model}...")
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
    
    # Resize token embeddings to match the multilingual tokenizer
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=standard_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(standard_output_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        do_eval=True,  # Enable evaluation during training
        seed=args.seed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(standard_output_dir, "final_model"))
    
    return model

def evaluate_tokenization_efficiency(switchable_tokenizer, standard_tokenizer, test_corpus):
    """Evaluate and compare tokenization efficiency."""
    print("\n=== Evaluating Tokenization Efficiency ===")
    
    results = {
        "switchable": {
            "en": {},
            "ru": {},
        },
        "standard": {
            "en": {},
            "ru": {},
        },
    }
    
    # Analyze English texts
    for lang, texts in test_corpus.items():
        print(f"Analyzing {lang} texts...")
        
        total_words = 0
        switchable_tokens = 0
        standard_tokens = 0
        
        for text in texts:
            words = text.split()
            total_words += len(words)
            
            # Tokenize with switchable tokenizer
            if lang == "en":
                switch_ids = switchable_tokenizer.encode(text, language="EN")
            else:
                switch_ids = switchable_tokenizer.encode(text, language="RU")
            
            # Tokenize with standard tokenizer
            standard_ids = standard_tokenizer.encode(text)
            
            switchable_tokens += len(switch_ids)
            standard_tokens += len(standard_ids)
        
        # Calculate tokens per word
        switchable_tokens_per_word = switchable_tokens / total_words if total_words > 0 else 0
        standard_tokens_per_word = standard_tokens / total_words if total_words > 0 else 0
        
        # Store results
        results["switchable"][lang] = {
            "total_tokens": switchable_tokens,
            "tokens_per_word": switchable_tokens_per_word,
        }
        
        results["standard"][lang] = {
            "total_tokens": standard_tokens,
            "tokens_per_word": standard_tokens_per_word,
        }
        
        print(f"  Switchable tokenizer: {switchable_tokens_per_word:.2f} tokens per word")
        print(f"  Standard tokenizer: {standard_tokens_per_word:.2f} tokens per word")
    
    return results

def evaluate_models(args, switchable_model, standard_model, switchable_tokenizer, 
                  standard_tokenizer, switchable_test, standard_test, test_corpus):
    """Evaluate both models and compare them."""
    print("\n=== Evaluating Models ===")
    
    results = {
        "switchable": {},
        "standard": {},
    }
    
    # Evaluate switchable model
    print("Evaluating switchable model...")
    
    # Use our custom perplexity calculation for switchable model
    data_loader = create_data_loaders(
        datasets={"test": switchable_test},
        collator=SwitchableDataCollator(tokenizer=switchable_tokenizer, mlm=False),
        batch_size=args.batch_size,
        shuffle_train=False,
    )
    
    switchable_ppl = calculate_perplexity(switchable_model, data_loader["test"], args.device)
    results["switchable"]["perplexity"] = switchable_ppl
    print(f"Switchable model perplexity: {switchable_ppl:.2f}")
    
    # Evaluate standard model
    print("Evaluating standard multilingual model...")
    
    # Calculate perplexity for the standard model
    def calculate_std_perplexity(model, dataset, tokenizer, device):
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
            for batch in batches:
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
    
    standard_ppl = calculate_std_perplexity(
        standard_model, 
        standard_test, 
        standard_tokenizer, 
        args.device
    )
    results["standard"]["perplexity"] = standard_ppl
    print(f"Standard model perplexity: {standard_ppl:.2f}")
    
    # Evaluate tokenization efficiency
    tokenization_results = evaluate_tokenization_efficiency(
        switchable_tokenizer, standard_tokenizer, test_corpus
    )
    
    # Combine results
    results["tokenization_efficiency"] = tokenization_results
    
    # Calculate parameter counts
    switchable_params = sum(p.numel() for p in switchable_model.parameters())
    standard_params = sum(p.numel() for p in standard_model.parameters())
    
    results["switchable"]["params"] = switchable_params
    results["standard"]["params"] = standard_params
    
    print(f"\nParameter counts:")
    print(f"Switchable model: {switchable_params:,} parameters")
    print(f"Standard model: {standard_params:,} parameters")
    
    # Compare perplexity
    if switchable_ppl < standard_ppl:
        ppl_improvement = (standard_ppl - switchable_ppl) / standard_ppl * 100
        print(f"\nSwitchable model has {ppl_improvement:.2f}% better perplexity")
        results["perplexity_comparison"] = {
            "better_model": "switchable",
            "improvement_percentage": ppl_improvement,
        }
    else:
        ppl_degradation = (switchable_ppl - standard_ppl) / standard_ppl * 100
        print(f"\nStandard model has {ppl_degradation:.2f}% better perplexity")
        results["perplexity_comparison"] = {
            "better_model": "standard",
            "improvement_percentage": ppl_degradation,
        }
    
    return results

def plot_results(results, output_dir):
    """Plot the comparison results."""
    print("\n=== Plotting Results ===")
    
    # Create directory for plots
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    # Plot perplexity comparison
    plt.figure(figsize=(10, 6))
    
    models = ['Switchable Tokenizer', 'Standard Multilingual']
    perplexities = [
        results["switchable"]["perplexity"],
        results["standard"]["perplexity"]
    ]
    
    plt.bar(models, perplexities, color=['blue', 'green'])
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Model Perplexity Comparison')
    
    # Add values on top of bars
    for i, v in enumerate(perplexities):
        plt.text(i, v + 2, f'{v:.2f}', ha='center')
    
    # Save plot
    perplexity_path = os.path.join(output_dir, "plots", "perplexity_comparison.png")
    plt.savefig(perplexity_path)
    plt.close()
    
    # Plot tokenization efficiency
    plt.figure(figsize=(12, 6))
    
    languages = ["English", "Russian"]
    
    switchable_efficiency = [
        results["tokenization_efficiency"]["switchable"]["en"]["tokens_per_word"],
        results["tokenization_efficiency"]["switchable"]["ru"]["tokens_per_word"]
    ]
    
    standard_efficiency = [
        results["tokenization_efficiency"]["standard"]["en"]["tokens_per_word"],
        results["tokenization_efficiency"]["standard"]["ru"]["tokens_per_word"]
    ]
    
    x = np.arange(len(languages))
    width = 0.35
    
    plt.bar(x - width/2, switchable_efficiency, width, label='Switchable Tokenizer')
    plt.bar(x + width/2, standard_efficiency, width, label='Standard Multilingual')
    
    plt.ylabel('Tokens per Word (lower is better)')
    plt.title('Tokenization Efficiency Comparison')
    plt.xticks(x, languages)
    plt.legend()
    
    # Add values on top of bars
    for i, v in enumerate(switchable_efficiency):
        plt.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(standard_efficiency):
        plt.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center')
    
    # Save plot
    efficiency_path = os.path.join(output_dir, "plots", "tokenization_efficiency.png")
    plt.savefig(efficiency_path)
    plt.close()
    
    print(f"Plots saved to: {os.path.join(output_dir, 'plots')}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and prepare data
    train_corpus, test_corpus = load_and_prepare_data(args)
    
    # Step 2: Prepare tokenizers
    switchable_tokenizer = prepare_switchable_tokenizer(args, train_corpus)
    standard_tokenizer = train_standard_multilingual_tokenizer(args, train_corpus)
    
    # Step 3: Prepare datasets
    switchable_train, switchable_test = prepare_switchable_datasets(
        switchable_tokenizer, train_corpus, test_corpus, args
    )
    
    standard_train, standard_test = prepare_standard_datasets(
        standard_tokenizer, train_corpus, test_corpus, args
    )
    
    # Step 4: Train models
    switchable_model = train_switchable_model(
        args, switchable_tokenizer, switchable_train, switchable_test
    )
    
    standard_model = train_standard_model(
        args, standard_tokenizer, standard_train, standard_test
    )
    
    # Step 5: Evaluate models
    evaluation_results = evaluate_models(
        args, switchable_model, standard_model, switchable_tokenizer,
        standard_tokenizer, switchable_test, standard_test, test_corpus
    )
    
    # Step 6: Plot results
    plot_results(evaluation_results, args.output_dir)
    
    # Step 7: Save results
    results_file = os.path.join(args.output_dir, "experiment3_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Experiment 3: Comparison vs. Standard Multilingual Baseline ===\n\n")
        
        # Describe the data used
        training_type = "from scratch" if getattr(args, 'from_scratch', False) else "fine-tuned"
        f.write(f"Models were trained {training_type}\n")
        if args.first_shard_only:
            f.write(f"Using the first shard of each language dataset\n\n")
        else:
            f.write(f"Using {args.data_limit} examples per language\n\n")
        
        # Write perplexity results
        f.write("Model Perplexity:\n")
        f.write(f"  Switchable Tokenizer: {evaluation_results['switchable']['perplexity']:.2f}\n")
        f.write(f"  Standard Multilingual: {evaluation_results['standard']['perplexity']:.2f}\n")
        
        # Write perplexity comparison
        f.write("\nPerplexity Comparison:\n")
        if evaluation_results['perplexity_comparison']['better_model'] == "switchable":
            f.write(f"  Switchable model is {evaluation_results['perplexity_comparison']['improvement_percentage']:.2f}% better\n")
        else:
            f.write(f"  Standard model is {evaluation_results['perplexity_comparison']['improvement_percentage']:.2f}% better\n")
        
        # Write tokenization efficiency results
        f.write("\nTokenization Efficiency (tokens per word):\n")
        f.write("  English:\n")
        f.write(f"    Switchable: {evaluation_results['tokenization_efficiency']['switchable']['en']['tokens_per_word']:.2f}\n")
        f.write(f"    Standard: {evaluation_results['tokenization_efficiency']['standard']['en']['tokens_per_word']:.2f}\n")
        f.write("  Russian:\n")
        f.write(f"    Switchable: {evaluation_results['tokenization_efficiency']['switchable']['ru']['tokens_per_word']:.2f}\n")
        f.write(f"    Standard: {evaluation_results['tokenization_efficiency']['standard']['ru']['tokens_per_word']:.2f}\n")
        
        # Write parameter counts
        f.write("\nParameter Counts:\n")
        f.write(f"  Switchable model: {evaluation_results['switchable']['params']:,}\n")
        f.write(f"  Standard model: {evaluation_results['standard']['params']:,}\n")
    
    print(f"\nResults saved to {results_file}")
    print("\nExperiment 3 completed!")

if __name__ == "__main__":
    main() 