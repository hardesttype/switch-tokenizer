#!/usr/bin/env python
"""
Experiment 2: Comparison vs. Concatenated Vocab

Compares the performance of the shared 64k model (switchable tokenizer)
against a 128k concatenated vocabulary model.
"""

import os
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling, set_seed, AutoConfig
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
    train_separate_tokenizers,
    train_concatenated_tokenizer,
    create_switchable_tokenizer_from_trained,
    analyze_tokenizer_overlap
)

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 2: Comparison vs. Concatenated Vocab")
    
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
    
    # Base model
    parser.add_argument("--base_model", type=str, default="gpt2-medium", 
                        help="Base model architecture")
    parser.add_argument("--from_scratch", action="store_true", 
                        help="Train models from scratch instead of fine-tuning")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./experiment2_output", 
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
    """Load and prepare the datasets for tokenizer training and model training."""
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

def train_tokenizers(args, train_corpus):
    """Train the separate and concatenated tokenizers."""
    print("\n=== Training Tokenizers ===")
    
    tokenizer_dir = os.path.join(args.output_dir, "tokenizers")
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Train separate tokenizers for English and Russian
    separate_paths = train_separate_tokenizers(
        corpus=train_corpus,
        output_dir=tokenizer_dir,
        vocab_size=64000,  # 64k vocab for each language
    )
    
    # Create a switchable tokenizer from the separate tokenizers
    switchable_dir = create_switchable_tokenizer_from_trained(
        en_tokenizer_path=separate_paths["en"],
        ru_tokenizer_path=separate_paths["ru"],
        output_dir=tokenizer_dir,
    )
    
    # Train a concatenated tokenizer
    concat_dir = train_concatenated_tokenizer(
        corpus=train_corpus,
        output_dir=tokenizer_dir,
        vocab_size=128000,  # 128k vocab for combined languages
    )
    
    # Analyze tokenizer overlap
    overlap_file = os.path.join(tokenizer_dir, "tokenizer_overlap_analysis.txt")
    analysis = analyze_tokenizer_overlap(
        en_tokenizer_path=separate_paths["en"],
        ru_tokenizer_path=separate_paths["ru"],
        sample_texts=train_corpus,
        output_file=overlap_file,
    )
    
    return {
        "switchable": switchable_dir,
        "concatenated": concat_dir,
        "en": separate_paths["en"],
        "ru": separate_paths["ru"],
        "analysis": analysis,
    }

def train_switchable_model(args, switchable_tokenizer_dir, train_corpus, test_corpus):
    """Train a model with the switchable tokenizer."""
    print("\n=== Training Switchable Tokenizer Model ===")
    
    # Create output directory
    switchable_output_dir = os.path.join(args.output_dir, "switchable_model")
    os.makedirs(switchable_output_dir, exist_ok=True)
    
    # Load the switchable tokenizer
    tokenizer = SwitchableTokenizer.from_pretrained(switchable_tokenizer_dir)
    
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
    
    # Create model
    model = create_model_with_switchable_tokenizer(
        model_name_or_path=args.base_model,
        tokenizer=tokenizer,
        from_scratch=getattr(args, 'from_scratch', False),
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(switchable_output_dir, "final_model"))
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": tokenized_train,
        "test_dataset": tokenized_test,
    }

def train_concatenated_model(args, concat_tokenizer_dir, train_corpus, test_corpus):
    """Train a model with the concatenated tokenizer."""
    print("\n=== Training Concatenated Vocabulary Model ===")
    
    # Create output directory
    concat_output_dir = os.path.join(args.output_dir, "concatenated_model")
    os.makedirs(concat_output_dir, exist_ok=True)
    
    # Load the concatenated tokenizer
    tokenizer = AutoTokenizer.from_pretrained(concat_tokenizer_dir)
    
    # Prepare datasets
    train_texts = []
    for text in train_corpus["en"]:
        train_texts.append("<LANG_EN> " + text)
    for text in train_corpus["ru"]:
        train_texts.append("<LANG_RU> " + text)
    
    test_texts = []
    for text in test_corpus["en"]:
        test_texts.append("<LANG_EN> " + text)
    for text in test_corpus["ru"]:
        test_texts.append("<LANG_RU> " + text)
    
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
    
    # Initialize model
    if getattr(args, 'from_scratch', False):
        config = AutoConfig.from_pretrained(args.base_model)
        config.vocab_size = len(tokenizer)
        model = AutoModelForCausalLM.from_config(config)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.base_model)
        # Resize token embeddings to match the concatenated tokenizer
        model.resize_token_embeddings(len(tokenizer))
    
    model.to(args.device)
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=concat_output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=os.path.join(concat_output_dir, "logs"),
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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(concat_output_dir, "final_model"))
    
    return {
        "model": model,
        "tokenizer": tokenizer,
        "train_dataset": tokenized_train,
        "test_dataset": tokenized_test,
    }

def evaluate_models(args, switchable_results, concat_results):
    """Evaluate both models and compare them."""
    print("\n=== Evaluating Models ===")
    
    results = {
        "switchable": {},
        "concatenated": {},
    }
    
    # Evaluate switchable model
    print("Evaluating switchable model...")
    switchable_model = switchable_results["model"]
    switchable_test = switchable_results["test_dataset"]
    
    # Use our custom perplexity calculation for switchable model
    data_loader = create_data_loaders(
        datasets={"test": switchable_test},
        collator=SwitchableDataCollator(tokenizer=switchable_results["tokenizer"], mlm=False),
        batch_size=args.batch_size,
        shuffle_train=False,
    )
    
    switchable_ppl = calculate_perplexity(switchable_model, data_loader["test"], args.device)
    results["switchable"]["perplexity"] = switchable_ppl
    print(f"Switchable model perplexity: {switchable_ppl:.2f}")
    
    # Evaluate concatenated model
    print("Evaluating concatenated model...")
    concat_model = concat_results["model"]
    concat_test = concat_results["test_dataset"]
    
    # Calculate perplexity for the concatenated model
    def calculate_concat_perplexity(model, dataset, tokenizer, device):
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
    
    concat_ppl = calculate_concat_perplexity(
        concat_model, 
        concat_test, 
        concat_results["tokenizer"], 
        args.device
    )
    results["concatenated"]["perplexity"] = concat_ppl
    print(f"Concatenated model perplexity: {concat_ppl:.2f}")
    
    # Calculate parameter counts for comparison
    switchable_params = sum(p.numel() for p in switchable_model.parameters())
    concat_params = sum(p.numel() for p in concat_model.parameters())
    
    results["switchable"]["params"] = switchable_params
    results["concatenated"]["params"] = concat_params
    
    param_diff = concat_params - switchable_params
    param_saving_percentage = (param_diff / concat_params) * 100
    
    print(f"\nParameter counts:")
    print(f"Switchable model: {switchable_params:,} parameters")
    print(f"Concatenated model: {concat_params:,} parameters")
    print(f"Difference: {param_diff:,} parameters ({param_saving_percentage:.2f}% saving with switchable model)")
    
    results["parameter_saving"] = {
        "absolute": param_diff,
        "percentage": param_saving_percentage,
    }
    
    # Compare perplexity
    if switchable_ppl < concat_ppl:
        ppl_improvement = (concat_ppl - switchable_ppl) / concat_ppl * 100
        print(f"\nSwitchable model has {ppl_improvement:.2f}% better perplexity")
        results["perplexity_comparison"] = {
            "better_model": "switchable",
            "improvement_percentage": ppl_improvement,
        }
    else:
        ppl_degradation = (switchable_ppl - concat_ppl) / concat_ppl * 100
        print(f"\nConcatenated model has {ppl_degradation:.2f}% better perplexity")
        results["perplexity_comparison"] = {
            "better_model": "concatenated",
            "improvement_percentage": ppl_degradation,
        }
    
    return results

def plot_results(results, output_dir):
    """Plot the comparison results."""
    print("\n=== Plotting Results ===")
    
    # Create figure for perplexity comparison
    plt.figure(figsize=(10, 6))
    
    models = ['Switchable 64k', 'Concatenated 128k']
    perplexities = [
        results["switchable"]["perplexity"],
        results["concatenated"]["perplexity"]
    ]
    
    plt.bar(models, perplexities, color=['blue', 'orange'])
    plt.ylabel('Perplexity (lower is better)')
    plt.title('Model Perplexity Comparison')
    
    # Add values on top of bars
    for i, v in enumerate(perplexities):
        plt.text(i, v + 2, f'{v:.2f}', ha='center')
    
    # Save the plot
    perplexity_plot_path = os.path.join(output_dir, "perplexity_comparison.png")
    plt.savefig(perplexity_plot_path)
    
    # Create figure for parameter counts
    plt.figure(figsize=(10, 6))
    
    params = [
        results["switchable"]["params"] / 1_000_000,  # Convert to millions
        results["concatenated"]["params"] / 1_000_000  # Convert to millions
    ]
    
    plt.bar(models, params, color=['blue', 'orange'])
    plt.ylabel('Parameters (millions)')
    plt.title('Model Parameter Count Comparison')
    
    # Add values on top of bars
    for i, v in enumerate(params):
        plt.text(i, v + 1, f'{v:.2f}M', ha='center')
    
    # Save the plot
    params_plot_path = os.path.join(output_dir, "parameter_comparison.png")
    plt.savefig(params_plot_path)
    
    print(f"Plots saved to:")
    print(f"  - {perplexity_plot_path}")
    print(f"  - {params_plot_path}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and prepare data
    train_corpus, test_corpus = load_and_prepare_data(args)
    
    # Step 2: Train tokenizers
    tokenizer_paths = train_tokenizers(args, train_corpus)
    
    # Step 3: Train the switchable model
    switchable_results = train_switchable_model(
        args, tokenizer_paths["switchable"], train_corpus, test_corpus
    )
    
    # Step 4: Train the concatenated model
    concat_results = train_concatenated_model(
        args, tokenizer_paths["concatenated"], train_corpus, test_corpus
    )
    
    # Step 5: Evaluate the models
    evaluation_results = evaluate_models(args, switchable_results, concat_results)
    
    # Step 6: Plot the results
    plot_results(evaluation_results, args.output_dir)
    
    # Step 7: Save the results
    results_file = os.path.join(args.output_dir, "experiment2_results.txt")
    with open(results_file, "w") as f:
        f.write("=== Experiment 2: Comparison vs. Concatenated Vocab ===\n\n")
        
        # Write training approach
        training_type = "from scratch" if getattr(args, 'from_scratch', False) else "fine-tuned"
        f.write(f"Models were trained {training_type}\n\n")
        
        # Describe the data used
        if args.first_shard_only:
            f.write(f"Using the first shard of each language dataset\n\n")
        else:
            f.write(f"Using {args.data_limit} examples per language\n\n")
        
        # Write switchable model results
        f.write("Switchable Model (64k vocabulary):\n")
        f.write(f"  Perplexity: {evaluation_results['switchable']['perplexity']:.2f}\n")
        f.write(f"  Parameters: {evaluation_results['switchable']['params']:,}\n")
        
        # Write concatenated model results
        f.write("\nConcatenated Model (128k vocabulary):\n")
        f.write(f"  Perplexity: {evaluation_results['concatenated']['perplexity']:.2f}\n")
        f.write(f"  Parameters: {evaluation_results['concatenated']['params']:,}\n")
        
        # Write comparison
        f.write("\nComparison:\n")
        f.write(f"  Parameter Savings: {evaluation_results['parameter_saving']['absolute']:,} ")
        f.write(f"({evaluation_results['parameter_saving']['percentage']:.2f}%)\n")
        
        # Write performance comparison
        if evaluation_results['perplexity_comparison']['better_model'] == "switchable":
            f.write(f"  Performance: Switchable model is {evaluation_results['perplexity_comparison']['improvement_percentage']:.2f}% better\n")
        else:
            f.write(f"  Performance: Concatenated model is {evaluation_results['perplexity_comparison']['improvement_percentage']:.2f}% better\n")
        
        # Write tokenizer analysis
        f.write("\nTokenizer Analysis:\n")
        f.write(f"  English Tokenizer Vocab Size: {tokenizer_paths['analysis']['en_vocab_size']}\n")
        f.write(f"  Russian Tokenizer Vocab Size: {tokenizer_paths['analysis']['ru_vocab_size']}\n")
        f.write(f"  Vocabulary Overlap: {tokenizer_paths['analysis']['overlap_size']} tokens ")
        f.write(f"({tokenizer_paths['analysis']['overlap_percentage_en']:.2f}% of English, ")
        f.write(f"{tokenizer_paths['analysis']['overlap_percentage_ru']:.2f}% of Russian)\n")
        f.write(f"  Avg English Tokens per Word: {tokenizer_paths['analysis']['avg_en_tokens_per_word']:.2f}\n")
        f.write(f"  Avg Russian Tokens per Word: {tokenizer_paths['analysis']['avg_ru_tokens_per_word']:.2f}\n")
    
    print(f"\nResults saved to {results_file}")
    print("\nExperiment 2 completed!")

if __name__ == "__main__":
    main() 