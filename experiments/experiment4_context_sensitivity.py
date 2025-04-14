#!/usr/bin/env python
"""
Experiment 4: Context Sensitivity Analysis

Tests how the model learns to interpret token IDs differently based on language context.
This experiment specifically analyzes how token probabilities shift based on language context.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import set_seed
import pandas as pd
import seaborn as sns

from src.switch_tokenizer import SwitchableTokenizer
from src.model_utils import create_model_with_switchable_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 4: Context Sensitivity Analysis")
    
    # Model and tokenizer paths
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory with the trained model")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Directory with the switchable tokenizer")
    
    # Output directory
    parser.add_argument("--output_dir", type=str, default="./experiment4_output",
                        help="Directory to save outputs")
    
    # Test parameters
    parser.add_argument("--num_test_tokens", type=int, default=100,
                        help="Number of tokens to test for context sensitivity")
    parser.add_argument("--num_prompts", type=int, default=5,
                        help="Number of prompts to test for each token")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

def find_shared_token_ids(tokenizer, num_tokens=100):
    """Find token IDs that have different meanings in different languages."""
    print("\n=== Finding Shared Token IDs ===")
    
    # Track token IDs and their meanings in each language
    shared_tokens = []
    
    # Skip special tokens and very low IDs (often special tokens)
    start_id = 100
    max_id = min(tokenizer.shared_vocab_size, 10000)  # Don't go too high
    
    # Find token IDs that have non-empty meanings in both languages
    for token_id in range(start_id, max_id):
        en_token = tokenizer.convert_ids_to_tokens([token_id], language="EN")[0]
        ru_token = tokenizer.convert_ids_to_tokens([token_id], language="RU")[0]
        
        # Check if the tokens are real words or subwords (not special tokens or padding)
        en_is_valid = not (en_token.startswith("<") or en_token.startswith("[") or 
                          en_token.isspace() or not en_token.strip())
        ru_is_valid = not (ru_token.startswith("<") or ru_token.startswith("[") or 
                          ru_token.isspace() or not ru_token.strip())
        
        # Only consider IDs that have valid, different representations in both languages
        if en_is_valid and ru_is_valid and en_token != ru_token:
            # Try to avoid byte-level tokens that are hard to interpret
            if len(en_token) > 1 and len(ru_token) > 1:
                shared_tokens.append({
                    "id": token_id,
                    "en_token": en_token,
                    "ru_token": ru_token,
                })
        
        # Stop once we have enough shared tokens
        if len(shared_tokens) >= num_tokens:
            break
    
    print(f"Found {len(shared_tokens)} shared token IDs with different meanings")
    
    if len(shared_tokens) < 10:
        # If we can't find many real shared tokens, relax constraints and find some tokens anyway
        print("Not enough shared tokens found. Relaxing constraints to find more tokens...")
        for token_id in range(start_id, max_id):
            en_token = tokenizer.convert_ids_to_tokens([token_id], language="EN")[0]
            ru_token = tokenizer.convert_ids_to_tokens([token_id], language="RU")[0]
            
            if en_token != ru_token and len(shared_tokens) < num_tokens:
                shared_tokens.append({
                    "id": token_id,
                    "en_token": en_token,
                    "ru_token": ru_token,
                })
    
    return shared_tokens

def generate_test_prompts(tokenizer, token_info, num_prompts=5):
    """Generate test prompts that could be completed with the target tokens."""
    # English prompt templates
    en_templates = [
        "I would like to",
        "The best way to",
        "Yesterday I saw a",
        "Let me tell you about the",
        "My favorite thing is",
        "The most important",
        "I need to buy a new",
        "When I was younger, I used to",
        "In the morning, I usually",
        "The last time I went to the"
    ]
    
    # Russian prompt templates
    ru_templates = [
        "Я хотел бы",
        "Лучший способ",
        "Вчера я видел",
        "Позвольте мне рассказать о",
        "Моя любимая вещь это",
        "Самое важное",
        "Мне нужно купить новый",
        "Когда я был моложе, я",
        "Утром я обычно",
        "В последний раз, когда я ходил"
    ]
    
    # Select random templates
    np.random.seed(42)  # For reproducibility
    en_selected = np.random.choice(en_templates, num_prompts, replace=False)
    ru_selected = np.random.choice(ru_templates, num_prompts, replace=False)
    
    # Package prompts with language info
    prompts = []
    for i in range(num_prompts):
        prompts.append({
            "en_prompt": en_selected[i],
            "ru_prompt": ru_selected[i],
            "token_id": token_info["id"],
            "en_token": token_info["en_token"],
            "ru_token": token_info["ru_token"],
        })
    
    return prompts

def analyze_token_probabilities(model, tokenizer, token_prompts, device):
    """
    Analyze how token probabilities shift based on language context.
    
    This tests whether the model correctly favors token k only in the appropriate language context.
    """
    print("\n=== Analyzing Token Probabilities Based on Language Context ===")
    
    model.eval()
    results = []
    
    for prompt_set in tqdm(token_prompts, desc="Analyzing token prompts"):
        token_id = prompt_set["token_id"]
        en_token = prompt_set["en_token"]
        ru_token = prompt_set["ru_token"]
        
        for prompt in prompt_set["prompts"]:
            en_prompt = prompt["en_prompt"]
            ru_prompt = prompt["ru_prompt"]
            
            # Prepare input for English context
            en_input_ids = tokenizer.encode(en_prompt, language="EN")
            en_input = torch.tensor([en_input_ids], device=device)
            
            # Prepare input for Russian context
            ru_input_ids = tokenizer.encode(ru_prompt, language="RU")
            ru_input = torch.tensor([ru_input_ids], device=device)
            
            # Get model predictions for English prompt
            with torch.no_grad():
                en_outputs = model(en_input)
                en_logits = en_outputs.logits[0, -1, :]  # Last token position
                en_probs = F.softmax(en_logits, dim=-1)
                en_token_prob = en_probs[token_id].item()
                
                # Get rank of the token in English context
                en_sorted_probs, en_sorted_indices = torch.sort(en_probs, descending=True)
                en_rank = (en_sorted_indices == token_id).nonzero().item() + 1
            
            # Get model predictions for Russian prompt
            with torch.no_grad():
                ru_outputs = model(ru_input)
                ru_logits = ru_outputs.logits[0, -1, :]  # Last token position
                ru_probs = F.softmax(ru_logits, dim=-1)
                ru_token_prob = ru_probs[token_id].item()
                
                # Get rank of the token in Russian context
                ru_sorted_probs, ru_sorted_indices = torch.sort(ru_probs, descending=True)
                ru_rank = (ru_sorted_indices == token_id).nonzero().item() + 1
            
            # Calculate probability ratio
            if en_token_prob > 0 and ru_token_prob > 0:
                en_ru_ratio = en_token_prob / ru_token_prob
                ru_en_ratio = ru_token_prob / en_token_prob
            else:
                en_ru_ratio = 0 if en_token_prob == 0 else float('inf')
                ru_en_ratio = 0 if ru_token_prob == 0 else float('inf')
            
            # Store results
            result = {
                "token_id": token_id,
                "en_token": en_token,
                "ru_token": ru_token,
                "en_prompt": en_prompt,
                "ru_prompt": ru_prompt,
                "en_token_prob": en_token_prob,
                "ru_token_prob": ru_token_prob,
                "en_rank": en_rank,
                "ru_rank": ru_rank,
                "en_ru_ratio": en_ru_ratio,
                "ru_en_ratio": ru_en_ratio,
            }
            
            results.append(result)
    
    return results

def test_language_specific_generation(model, tokenizer, device):
    """
    Test generation with language-specific prompts.
    
    Tests if the model generates language-specific completions based on the language context.
    """
    print("\n=== Testing Language-Specific Generation ===")
    
    # Set of prompt pairs in English and Russian
    prompt_pairs = [
        {
            "en": "The capital of France is",
            "ru": "Столица Франции это",
            "expected_en": "Paris",
            "expected_ru": "Париж",
        },
        {
            "en": "Today the weather is",
            "ru": "Сегодня погода",
            "expected_en": "sunny",
            "expected_ru": "солнечная",
        },
        {
            "en": "I would like to eat",
            "ru": "Я хотел бы поесть",
            "expected_en": "dinner",
            "expected_ru": "ужин",
        },
        {
            "en": "The best book I've read is",
            "ru": "Лучшая книга, которую я читал, это",
            "expected_en": "The",
            "expected_ru": "Война",
        },
        {
            "en": "My favorite movie is",
            "ru": "Мой любимый фильм это",
            "expected_en": "The",
            "expected_ru": "Москва",
        },
    ]
    
    results = []
    
    for prompt_pair in prompt_pairs:
        en_prompt = prompt_pair["en"]
        ru_prompt = prompt_pair["ru"]
        
        # Generate with English prompt
        en_input_ids = tokenizer.encode(en_prompt, language="EN")
        en_input = torch.tensor([en_input_ids], device=device)
        
        with torch.no_grad():
            en_output_ids = model.generate(
                en_input,
                max_new_tokens=20,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
            )
        
        en_output_text = tokenizer.decode(en_output_ids[0][len(en_input_ids):].tolist())
        
        # Generate with Russian prompt
        ru_input_ids = tokenizer.encode(ru_prompt, language="RU")
        ru_input = torch.tensor([ru_input_ids], device=device)
        
        with torch.no_grad():
            ru_output_ids = model.generate(
                ru_input,
                max_new_tokens=20,
                do_sample=True,
                top_p=0.92,
                temperature=0.8,
            )
        
        ru_output_text = tokenizer.decode(ru_output_ids[0][len(ru_input_ids):].tolist())
        
        # Detect language of generated text
        def detect_language(text):
            # Simple heuristic: count Cyrillic characters
            cyrillic_chars = set('абвгдеёжзийклмнопрстуфхцчшщъыьэюя')
            cyrillic_count = sum(1 for c in text.lower() if c in cyrillic_chars)
            
            if cyrillic_count > len(text) * 0.3:  # If more than 30% characters are Cyrillic
                return "RU"
            else:
                return "EN"
        
        en_output_lang = detect_language(en_output_text)
        ru_output_lang = detect_language(ru_output_text)
        
        # Check if generation is in the correct language
        en_correct = en_output_lang == "EN"
        ru_correct = ru_output_lang == "RU"
        
        result = {
            "en_prompt": en_prompt,
            "ru_prompt": ru_prompt,
            "en_expected": prompt_pair["expected_en"],
            "ru_expected": prompt_pair["expected_ru"],
            "en_generated": en_output_text,
            "ru_generated": ru_output_text,
            "en_language": en_output_lang,
            "ru_language": ru_output_lang,
            "en_correct": en_correct,
            "ru_correct": ru_correct,
        }
        
        results.append(result)
        
        # Print the results
        print(f"\nEnglish prompt: {en_prompt}")
        print(f"Generated (EN): {en_output_text}")
        print(f"Detected language: {en_output_lang} (Correct: {en_correct})")
        
        print(f"\nRussian prompt: {ru_prompt}")
        print(f"Generated (RU): {ru_output_text}")
        print(f"Detected language: {ru_output_lang} (Correct: {ru_correct})")
    
    return results

def plot_probability_distributions(results, output_dir):
    """Plot probability distributions for shared token IDs in different language contexts."""
    print("\n=== Plotting Probability Distributions ===")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Plot probability ratios
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="en_ru_ratio", bins=30, kde=True, color="blue")
    plt.title("Distribution of English/Russian Probability Ratios")
    plt.xlabel("EN/RU Probability Ratio (log scale)")
    plt.ylabel("Count")
    plt.xscale("log")
    plt.axvline(x=1, color="red", linestyle="--")
    plt.savefig(os.path.join(plots_dir, "en_ru_ratio_distribution.png"))
    plt.close()
    
    # Plot token probabilities by language
    plt.figure(figsize=(12, 6))
    
    # Use logarithmic scale for better visualization
    df["en_token_prob_log"] = np.log10(df["en_token_prob"] + 1e-10)
    df["ru_token_prob_log"] = np.log10(df["ru_token_prob"] + 1e-10)
    
    sns.scatterplot(data=df, x="en_token_prob_log", y="ru_token_prob_log", alpha=0.7)
    plt.title("Token Probabilities by Language Context")
    plt.xlabel("English Context Probability (log10)")
    plt.ylabel("Russian Context Probability (log10)")
    
    # Add a diagonal line for reference
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    lims = [max(x_min, y_min), min(x_max, y_max)]
    plt.plot(lims, lims, 'r--', alpha=0.5, label="Equal probability")
    plt.legend()
    
    plt.savefig(os.path.join(plots_dir, "token_probabilities_by_language.png"))
    plt.close()
    
    # Plot token ranks in each language
    plt.figure(figsize=(12, 6))
    plt.scatter(df["en_rank"], df["ru_rank"], alpha=0.7)
    plt.title("Token Ranks by Language Context")
    plt.xlabel("English Context Rank")
    plt.ylabel("Russian Context Rank")
    plt.xscale("log")
    plt.yscale("log")
    plt.savefig(os.path.join(plots_dir, "token_ranks_by_language.png"))
    plt.close()
    
    # Create a heatmap of high-probability tokens
    top_tokens = df.groupby(["token_id", "en_token", "ru_token"])[["en_token_prob", "ru_token_prob"]].mean()
    top_tokens["prob_diff"] = abs(top_tokens["en_token_prob"] - top_tokens["ru_token_prob"])
    top_tokens = top_tokens.reset_index().sort_values("prob_diff", ascending=False).head(15)
    
    plt.figure(figsize=(14, 8))
    token_labels = [f"{row.en_token}/{row.ru_token} (ID: {row.token_id})" for row in top_tokens.itertuples()]
    
    heatmap_data = pd.DataFrame({
        "English": top_tokens["en_token_prob"],
        "Russian": top_tokens["ru_token_prob"]
    }, index=token_labels)
    
    sns.heatmap(heatmap_data, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title("Probability by Language Context for Top Context-Sensitive Tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "token_probability_heatmap.png"))
    plt.close()
    
    print(f"Plots saved to {plots_dir}")

def summarize_results(token_results, generation_results, output_dir):
    """Summarize the results of the context sensitivity analysis."""
    print("\n=== Summarizing Results ===")
    
    # Create a DataFrame from token results
    df = pd.DataFrame(token_results)
    
    # Calculate metrics
    mean_en_ru_ratio = df["en_ru_ratio"].replace([float('inf'), float('-inf'), float('nan')], 0).mean()
    mean_ru_en_ratio = df["ru_en_ratio"].replace([float('inf'), float('-inf'), float('nan')], 0).mean()
    
    # Calculate how often the token is more probable in its "native" language
    en_token_in_en = df["en_token_prob"] > df["ru_token_prob"]
    ru_token_in_ru = df["ru_token_prob"] > df["en_token_prob"]
    
    correct_context_percentage = (en_token_in_en.sum() + ru_token_in_ru.sum()) / (2 * len(df)) * 100
    
    # Summarize language generation results
    gen_df = pd.DataFrame(generation_results)
    en_correct_percentage = gen_df["en_correct"].mean() * 100
    ru_correct_percentage = gen_df["ru_correct"].mean() * 100
    
    # Write summary to file
    summary_file = os.path.join(output_dir, "context_sensitivity_summary.txt")
    with open(summary_file, "w") as f:
        f.write("=== Context Sensitivity Analysis Summary ===\n\n")
        
        f.write("Token Probability Analysis:\n")
        f.write(f"Total token-prompt pairs analyzed: {len(df)}\n")
        f.write(f"Mean EN/RU probability ratio: {mean_en_ru_ratio:.2f}\n")
        f.write(f"Mean RU/EN probability ratio: {mean_ru_en_ratio:.2f}\n")
        f.write(f"Tokens more probable in their 'native' language: {correct_context_percentage:.2f}%\n")
        
        f.write("\nLanguage Generation Analysis:\n")
        f.write(f"English prompts generating English text: {en_correct_percentage:.2f}%\n")
        f.write(f"Russian prompts generating Russian text: {ru_correct_percentage:.2f}%\n")
        
        f.write("\nTop 10 Most Context-Sensitive Tokens:\n")
        
        # Get the most context-sensitive tokens (largest difference in probability between languages)
        top_tokens = df.groupby(["token_id", "en_token", "ru_token"])[["en_token_prob", "ru_token_prob"]].mean()
        top_tokens["prob_diff"] = abs(top_tokens["en_token_prob"] - top_tokens["ru_token_prob"])
        top_tokens = top_tokens.reset_index().sort_values("prob_diff", ascending=False).head(10)
        
        for i, row in enumerate(top_tokens.itertuples(), 1):
            f.write(f"{i}. Token ID {row.token_id}: '{row.en_token}' (EN) / '{row.ru_token}' (RU)\n")
            f.write(f"   EN probability: {row.en_token_prob:.6f}, RU probability: {row.ru_token_prob:.6f}\n")
            f.write(f"   Probability difference: {row.prob_diff:.6f}\n")
    
    print(f"Summary saved to {summary_file}")
    
    # Also save detailed results as JSON
    token_results_file = os.path.join(output_dir, "token_probability_results.json")
    with open(token_results_file, "w") as f:
        json.dump(token_results, f, indent=2)
    
    generation_results_file = os.path.join(output_dir, "generation_results.json")
    with open(generation_results_file, "w") as f:
        json.dump(generation_results, f, indent=2)
    
    print(f"Detailed results saved to:")
    print(f"  - {token_results_file}")
    print(f"  - {generation_results_file}")

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer and model
    print("Loading tokenizer and model...")
    tokenizer = SwitchableTokenizer.from_pretrained(args.tokenizer_dir)
    model = create_model_with_switchable_tokenizer(
        model_name_or_path=args.model_dir,
        tokenizer=tokenizer,
        from_scratch=False,  # Default to using pretrained weights
    )
    model.to(args.device)
    
    # Step 1: Find token IDs with different meanings in different languages
    shared_tokens = find_shared_token_ids(tokenizer, args.num_test_tokens)
    
    # Step 2: Generate test prompts for each token
    token_prompts = []
    for token in shared_tokens:
        prompts = generate_test_prompts(tokenizer, token, args.num_prompts)
        token_prompts.append({
            "token_id": token["id"],
            "en_token": token["en_token"],
            "ru_token": token["ru_token"],
            "prompts": prompts,
        })
    
    # Step 3: Analyze token probabilities in different language contexts
    token_results = analyze_token_probabilities(model, tokenizer, token_prompts, args.device)
    
    # Step 4: Test language-specific generation
    generation_results = test_language_specific_generation(model, tokenizer, args.device)
    
    # Step 5: Plot the results
    plot_probability_distributions(token_results, args.output_dir)
    
    # Step 6: Summarize the results
    summarize_results(token_results, generation_results, args.output_dir)
    
    print("\nExperiment 4 completed!")

if __name__ == "__main__":
    main() 