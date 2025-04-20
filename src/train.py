import os
import argparse
from transformers import Trainer, TrainingArguments, set_seed

from .switch_tokenizer import SwitchableTokenizer
from .model_utils import create_model_with_switchable_tokenizer
from .data_utils import (
    prepare_multilingual_datasets,
    SwitchableDataCollator,
    create_data_loaders,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with the switchable tokenizer")
    
    # Tokenizer and model paths
    parser.add_argument("--en_tokenizer_path", type=str, required=True, help="Path to English tokenizer")
    parser.add_argument("--ru_tokenizer_path", type=str, required=True, help="Path to Russian tokenizer")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2", help="Base model to use")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save models and tokenizers")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    
    # Dataset parameters
    parser.add_argument("--en_dataset", type=str, default="wikitext/wikitext-2-raw-v1", help="English dataset path")
    parser.add_argument("--ru_dataset", type=str, default="wikitext/wikitext-2-raw-v1", help="Russian dataset path")
    parser.add_argument("--test_split", type=float, default=0.1, help="Fraction of data to use for testing")
    
    # Mixed precision training
    parser.add_argument("--fp16", action="store_true", help="Whether to use mixed precision training")
    
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the switchable tokenizer
    print("Initializing switchable tokenizer...")
    tokenizer = SwitchableTokenizer(
        en_tokenizer_path=args.en_tokenizer_path,
        ru_tokenizer_path=args.ru_tokenizer_path,
    )
    
    # Prepare datasets
    print("Preparing datasets...")
    dataset_configs = {
        "EN": {"path": args.en_dataset},
        "RU": {"path": args.ru_dataset},
    }
    
    datasets = prepare_multilingual_datasets(
        tokenizer=tokenizer,
        dataset_configs=dataset_configs,
        max_length=args.max_seq_length,
        train_test_split=args.test_split,
        seed=args.seed,
    )
    
    # Create model
    print("Initializing model...")
    model = create_model_with_switchable_tokenizer(
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        from_scratch=False,  # Default to using pretrained weights
    )
    
    # Create data collator
    collator = SwitchableDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        datasets=datasets,
        collator=collator,
        batch_size=args.batch_size,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=100,
        save_steps=1000,
        eval_steps=1000,
        do_eval=True,  # Enable evaluation during training
        fp16=args.fp16,
        seed=args.seed,
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model and tokenizer
    print("Saving model and tokenizer...")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_tokenizer"))
    
    print("Training complete!")

if __name__ == "__main__":
    main() 