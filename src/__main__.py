#!/usr/bin/env python
"""
Command-line interface for switch-tokenizer package.
This allows the package to be run as a module: python -m src
"""

import argparse
import sys

from .switch_tokenizer import SwitchableTokenizer
from .model_utils import create_model_with_switchable_tokenizer, generate_text

def main():
    parser = argparse.ArgumentParser(description="Switch-Tokenizer CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Tokenize command
    tokenize_parser = subparsers.add_parser("tokenize", help="Tokenize text with a switchable tokenizer")
    tokenize_parser.add_argument("--en_tokenizer", type=str, default="gpt2", help="English tokenizer")
    tokenize_parser.add_argument("--ru_tokenizer", type=str, default="ai-forever/ruGPT-3.5-13B", help="Russian tokenizer")
    tokenize_parser.add_argument("--text", type=str, required=True, help="Text to tokenize")
    tokenize_parser.add_argument("--language", type=str, choices=["EN", "RU"], required=True, help="Language of the text")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate text with a switchable tokenizer model")
    generate_parser.add_argument("--model_dir", type=str, help="Directory with the model")
    generate_parser.add_argument("--tokenizer_dir", type=str, help="Directory with the tokenizer")
    generate_parser.add_argument("--prompt", type=str, required=True, help="Prompt for text generation")
    generate_parser.add_argument("--language", type=str, choices=["EN", "RU"], required=True, help="Language for generation")
    generate_parser.add_argument("--max_length", type=int, default=50, help="Maximum length to generate")
    
    # Quick demo command
    demo_parser = subparsers.add_parser("demo", help="Run a quick demo of the switchable tokenizer")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "tokenize":
        # Initialize tokenizer
        tokenizer = SwitchableTokenizer(
            en_tokenizer_path=args.en_tokenizer,
            ru_tokenizer_path=args.ru_tokenizer,
        )
        
        # Tokenize text
        token_ids = tokenizer.encode(args.text, language=args.language)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        
        # Print results
        print(f"Text: {args.text}")
        print(f"Language: {args.language}")
        print(f"Token IDs: {token_ids}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {tokenizer.decode(token_ids)}")
        
    elif args.command == "generate":
        # Check if model and tokenizer directories are provided
        if not args.model_dir or not args.tokenizer_dir:
            print("Error: --model_dir and --tokenizer_dir are required for generation.")
            return 1
        
        # Load tokenizer and model
        tokenizer = SwitchableTokenizer.from_pretrained(args.tokenizer_dir)
        model = create_model_with_switchable_tokenizer(
            model_name_or_path=args.model_dir,
            tokenizer=tokenizer,
            from_scratch=False,  # Default to using pretrained weights
        )
        
        # Generate text
        generated_text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            language=args.language,
            max_new_tokens=args.max_length,
        )
        
        # Print results
        print(f"Prompt: {args.prompt}")
        print(f"Generated ({args.language}): {generated_text}")
        
    elif args.command == "demo":
        print("Running Switch-Tokenizer quick demo...")
        
        # Initialize tokenizer with default models
        print("Initializing tokenizer...")
        tokenizer = SwitchableTokenizer(
            en_tokenizer_path="gpt2",
            ru_tokenizer_path="ai-forever/ruGPT-3.5-13B",
        )
        
        # English example
        en_text = "Hello, world! How are you today?"
        print(f"\nEnglish text: {en_text}")
        en_tokens = tokenizer.encode(en_text, language="EN")
        print(f"Token IDs: {en_tokens}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(en_tokens)}")
        print(f"Decoded: {tokenizer.decode(en_tokens)}")
        
        # Russian example
        ru_text = "Привет, мир! Как дела сегодня?"
        print(f"\nRussian text: {ru_text}")
        ru_tokens = tokenizer.encode(ru_text, language="RU")
        print(f"Token IDs: {ru_tokens}")
        print(f"Tokens: {tokenizer.convert_ids_to_tokens(ru_tokens)}")
        print(f"Decoded: {tokenizer.decode(ru_tokens)}")
        
        # Mixed example with language switching
        print("\nDemonstrating language switching...")
        mixed_tokens = [tokenizer.lang_map["EN"]["lang_token_id"]]  # Start with English
        mixed_tokens.extend(tokenizer.encode("Hello! ", language="EN", add_language_token=False))
        mixed_tokens.append(tokenizer.lang_map["RU"]["lang_token_id"])  # Switch to Russian
        mixed_tokens.extend(tokenizer.encode("Привет! ", language="RU", add_language_token=False))
        mixed_tokens.append(tokenizer.lang_map["EN"]["lang_token_id"])  # Back to English
        mixed_tokens.extend(tokenizer.encode("How are you?", language="EN", add_language_token=False))
        
        print(f"Mixed tokens: {mixed_tokens}")
        print(f"Decoded mixed text: {tokenizer.decode(mixed_tokens)}")
        
        # Compare token meanings
        print("\nAnalyzing token ID overlap...")
        for test_id in range(100, 1001, 300):
            en_token = tokenizer.tokenizer_en.convert_ids_to_tokens([test_id])[0]
            ru_token = tokenizer.tokenizer_ru.convert_ids_to_tokens([test_id])[0]
            print(f"ID {test_id}:")
            print(f"  - English token: '{en_token}'")
            print(f"  - Russian token: '{ru_token}'")
        
        print("\nDemo complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 