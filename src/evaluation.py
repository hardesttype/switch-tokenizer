#!/usr/bin/env python
"""
Evaluation script for models trained with the SwitchableTokenizer.
Measures perplexity and other metrics on multilingual data.
"""

import argparse
import torch
import numpy as np
from tqdm import tqdm
from transformers import set_seed
import torch.nn.functional as F

from .switch_tokenizer import SwitchableTokenizer
from .model_utils import create_model_with_switchable_tokenizer
from .data_utils import prepare_multilingual_datasets, create_data_loaders, SwitchableDataCollator

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model trained with the switchable tokenizer")
    
    # Model and tokenizer paths
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with the trained model")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory with the switchable tokenizer")
    
    # Evaluation datasets
    parser.add_argument("--en_dataset", type=str, required=True, help="Path to English evaluation dataset")
    parser.add_argument("--ru_dataset", type=str, required=True, help="Path to Russian evaluation dataset")
    
    # Evaluation settings
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

def calculate_perplexity(model, data_loader, device):
    """
    Calculate perplexity on a dataset.
    
    Args:
        model: The language model
        data_loader: DataLoader with evaluation data
        device: Device to use
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating perplexity"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            
            # Extract loss
            loss = outputs.loss
            
            # Count non-padding tokens (where labels != -100)
            num_tokens = (labels != -100).sum().item()
            
            # Accumulate loss and token count
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity

def analyze_token_probabilities(model, tokenizer, device, lang="EN"):
    """
    Analyze token probability patterns based on language context.
    
    Args:
        model: The language model
        tokenizer: SwitchableTokenizer instance
        device: Device to use
        lang: Language to analyze ("EN" or "RU")
        
    Returns:
        Dictionary with analysis results
    """
    model.eval()
    
    # Get the language token ID
    lang_token_id = tokenizer.lang_map[lang]["lang_token_id"]
    
    # Start with just the language token as input
    input_ids = torch.tensor([[lang_token_id]], device=device)
    
    with torch.no_grad():
        # Get the model's output
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Get the next token probabilities from the last position
        next_token_logits = logits[0, -1, :]
        next_token_probs = F.softmax(next_token_logits, dim=-1)
        
        # Get the top tokens for this language
        top_k = 10
        top_probs, top_indices = torch.topk(next_token_probs, top_k)
        
        # Get tokens from the appropriate language tokenizer
        tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist(), language=lang)
        
        # Check probability concentration in each language's token space
        # For this analysis, we'll consider tokens within the shared vocab space
        # that are used in either language's tokenizer
        
        # For simplicity, we'll analyze the first 1000 token IDs
        en_tokens = tokenizer.tokenizer_en.convert_ids_to_tokens(list(range(1000)))
        ru_tokens = tokenizer.tokenizer_ru.convert_ids_to_tokens(list(range(1000)))
        
        # Calculate total probability mass for each language's tokens
        en_prob_mass = 0.0
        ru_prob_mass = 0.0
        
        for i in range(1000):
            en_token = en_tokens[i]
            ru_token = ru_tokens[i]
            
            # Skip special tokens and padding
            if en_token.startswith("[") or en_token.startswith("<") or en_token == "[PAD]":
                continue
            if ru_token.startswith("[") or ru_token.startswith("<") or ru_token == "[PAD]":
                continue
            
            # Get probability for this token ID
            token_prob = next_token_probs[i].item()
            
            # Check if it's a meaningful token in each language
            is_en_meaningful = not en_token.startswith("##") and len(en_token.strip()) > 0
            is_ru_meaningful = not ru_token.startswith("##") and len(ru_token.strip()) > 0
            
            if is_en_meaningful:
                en_prob_mass += token_prob
            if is_ru_meaningful:
                ru_prob_mass += token_prob
    
    results = {
        "language": lang,
        "top_tokens": list(zip(tokens, top_probs.tolist())),
        "en_probability_mass": en_prob_mass,
        "ru_probability_mass": ru_prob_mass,
    }
    
    return results

def evaluate_context_switching(model, tokenizer, device):
    """
    Evaluate how well the model handles context switching between languages.
    
    Args:
        model: The language model
        tokenizer: SwitchableTokenizer instance
        device: Device to use
        
    Returns:
        Dictionary with evaluation results
    """
    model.eval()
    
    # Prepare prompts with explicit language switching
    prompts = [
        # English to Russian
        (
            "English: Hello, how are you today?",
            "Russian: Привет, как ты сегодня?",
            "EN", "RU"
        ),
        # Russian to English
        (
            "Russian: Что нового?",
            "English: What's new?",
            "RU", "EN"
        ),
        # Alternate phrases
        (
            "English: I like apples. Russian: Я люблю яблоки. English: And oranges too.",
            "Russian: И апельсины тоже.",
            "EN", "RU"
        ),
    ]
    
    results = []
    
    for prompt_text, expected_completion, start_lang, target_lang in prompts:
        # Tokenize prompt with the starting language
        prompt_tokens = tokenizer.encode(prompt_text, language=start_lang)
        input_ids = torch.tensor([prompt_tokens], device=device)
        
        # Generate completion
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=30,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
            )
        
        # Get only the newly generated tokens
        generated_ids = output_ids[0][len(prompt_tokens):]
        generated_text = tokenizer.decode(generated_ids.tolist())
        
        # Check if the generated text starts with a language token
        starts_with_lang_token = False
        first_token = tokenizer.convert_ids_to_tokens([generated_ids[0]])[0] if len(generated_ids) > 0 else ""
        if first_token in ["<LANG_EN>", "<LANG_RU>"]:
            starts_with_lang_token = True
            # If it does, remove it from the generated text for readability
            generated_text = tokenizer.decode(generated_ids[1:].tolist())
        
        # Detect the language of the generated text
        # This is a simple heuristic - in a real application, you'd use a more robust method
        detected_lang = "unknown"
        ru_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
        en_chars = set('abcdefghijklmnopqrstuvwxyz')
        
        ru_char_count = sum(1 for c in generated_text.lower() if c in ru_chars)
        en_char_count = sum(1 for c in generated_text.lower() if c in en_chars)
        
        if ru_char_count > en_char_count:
            detected_lang = "RU"
        elif en_char_count > ru_char_count:
            detected_lang = "EN"
        
        results.append({
            "prompt": prompt_text,
            "expected_completion": expected_completion,
            "generated_text": generated_text,
            "starts_with_lang_token": starts_with_lang_token,
            "detected_language": detected_lang,
            "target_language": target_lang,
            "language_match": detected_lang == target_lang,
        })
    
    return results

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = SwitchableTokenizer.from_pretrained(args.tokenizer_dir)
    
    # Load model
    print("Loading model...")
    model = create_model_with_switchable_tokenizer(
        model_name_or_path=args.model_dir,
        tokenizer=tokenizer,
        from_scratch=False,  # Default to using pretrained weights
    )
    model.to(args.device)
    
    # Prepare evaluation datasets
    print("Preparing evaluation datasets...")
    dataset_configs = {
        "EN": {"path": args.en_dataset},
        "RU": {"path": args.ru_dataset},
    }
    
    eval_datasets = prepare_multilingual_datasets(
        tokenizer=tokenizer,
        dataset_configs=dataset_configs,
        max_length=args.max_seq_length,
        train_test_split=None,  # Use the entire dataset for evaluation
    )
    
    # Create data collator
    collator = SwitchableDataCollator(
        tokenizer=tokenizer,
        mlm=False,  # For causal language modeling
    )
    
    # Create data loaders
    data_loaders = create_data_loaders(
        datasets={"eval": eval_datasets["train"]},  # Use the combined dataset
        collator=collator,
        batch_size=args.batch_size,
        shuffle_train=False,  # No need to shuffle for evaluation
    )
    
    # Calculate perplexity
    print("Calculating perplexity on the combined dataset...")
    perplexity = calculate_perplexity(model, data_loaders["eval"], args.device)
    print(f"Combined perplexity: {perplexity:.2f}")
    
    # Calculate language-specific perplexity
    for lang in ["EN", "RU"]:
        # Create a dataset with just this language
        lang_dataset_config = {lang: dataset_configs[lang]}
        lang_datasets = prepare_multilingual_datasets(
            tokenizer=tokenizer,
            dataset_configs=lang_dataset_config,
            max_length=args.max_seq_length,
        )
        
        # Create a data loader
        lang_data_loaders = create_data_loaders(
            datasets={"eval": lang_datasets["train"]},
            collator=collator,
            batch_size=args.batch_size,
            shuffle_train=False,
        )
        
        # Calculate perplexity
        print(f"Calculating perplexity for {lang}...")
        lang_perplexity = calculate_perplexity(model, lang_data_loaders["eval"], args.device)
        print(f"{lang} perplexity: {lang_perplexity:.2f}")
    
    # Analyze token probabilities
    print("\nAnalyzing token probabilities based on language context...")
    en_analysis = analyze_token_probabilities(model, tokenizer, args.device, lang="EN")
    ru_analysis = analyze_token_probabilities(model, tokenizer, args.device, lang="RU")
    
    print("\nToken probability analysis for English context:")
    print(f"Top tokens: {en_analysis['top_tokens']}")
    print(f"English token probability mass: {en_analysis['en_probability_mass']:.4f}")
    print(f"Russian token probability mass: {en_analysis['ru_probability_mass']:.4f}")
    
    print("\nToken probability analysis for Russian context:")
    print(f"Top tokens: {ru_analysis['top_tokens']}")
    print(f"English token probability mass: {ru_analysis['en_probability_mass']:.4f}")
    print(f"Russian token probability mass: {ru_analysis['ru_probability_mass']:.4f}")
    
    # Evaluate context switching
    print("\nEvaluating language context switching...")
    switch_results = evaluate_context_switching(model, tokenizer, args.device)
    
    for i, result in enumerate(switch_results):
        print(f"\nContext switch example {i+1}:")
        print(f"Prompt: {result['prompt']}")
        print(f"Generated: {result['generated_text']}")
        print(f"Expected language: {result['target_language']}")
        print(f"Detected language: {result['detected_language']}")
        print(f"Language match: {result['language_match']}")
        print(f"Generated with explicit language token: {result['starts_with_lang_token']}")
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 