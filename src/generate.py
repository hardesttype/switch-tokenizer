#!/usr/bin/env python
"""
Generate text using a model trained with the SwitchableTokenizer.
"""

import argparse
import torch
from transformers import set_seed

from .switch_tokenizer import SwitchableTokenizer
from .model_utils import create_model_with_switchable_tokenizer, generate_text

def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using a model trained with the switchable tokenizer")
    
    # Model and tokenizer paths
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with the trained model")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory with the switchable tokenizer")
    
    # Generation settings
    parser.add_argument("--prompt", type=str, default="", help="Prompt for text generation")
    parser.add_argument("--language", type=str, choices=["EN", "RU"], required=True, help="Language for generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.92, help="Top-p sampling parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                         help="Device to use (cuda or cpu)")
    
    return parser.parse_args()

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
    
    # Generate text
    print(f"Generating {args.language} text...")
    
    # Set up generation config
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
    }
    
    # Generate
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        language=args.language,
        max_new_tokens=args.max_length,
        **generation_kwargs
    )
    
    # Print results
    print("\nGeneration Results:")
    print(f"Prompt: {args.prompt}")
    print(f"Generated Text ({args.language}): {generated_text}")
    
    # Also generate in the other language for comparison
    other_lang = "RU" if args.language == "EN" else "EN"
    
    print(f"\nGenerating with the same prompt in {other_lang} for comparison...")
    other_generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        language=other_lang,
        max_new_tokens=args.max_length,
        **generation_kwargs
    )
    
    print(f"Generated Text ({other_lang}): {other_generated}")
    
    # Generate with language switching
    if args.prompt:
        print("\nGenerating with explicit language switching...")
        
        # Prepare a prompt that switches languages
        switch_prompt = f"{args.language}: {args.prompt} {other_lang}:"
        
        # Tokenize with the initial language
        tokens = tokenizer.encode(switch_prompt, language=args.language)
        input_ids = torch.tensor([tokens]).to(args.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_length,
                do_sample=True,
                top_p=args.top_p,
                temperature=args.temperature,
            )
        
        # Decode the full sequence
        full_text = tokenizer.decode(output_ids[0].tolist())
        
        # Extract just the newly generated part
        new_text = tokenizer.decode(output_ids[0][len(tokens):].tolist())
        
        print(f"Switch prompt: {switch_prompt}")
        print(f"Generated continuation: {new_text}")
    
    print("\nGeneration complete!")

if __name__ == "__main__":
    main() 