#!/usr/bin/env python
"""
Example script to demonstrate tokenization with the SwitchableTokenizer.
"""

import argparse
from .switch_tokenizer import SwitchableTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Tokenize text with the switchable tokenizer")
    
    # Tokenizer paths
    parser.add_argument("--en_tokenizer", type=str, help="Path to English tokenizer or HF model name")
    parser.add_argument("--ru_tokenizer", type=str, help="Path to Russian tokenizer or HF model name")
    parser.add_argument("--tokenizer_dir", type=str, help="Path to saved switchable tokenizer (alternative to separate tokenizers)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer
    if args.tokenizer_dir:
        print(f"Loading switchable tokenizer from {args.tokenizer_dir}")
        tokenizer = SwitchableTokenizer.from_pretrained(args.tokenizer_dir)
    else:
        if not args.en_tokenizer or not args.ru_tokenizer:
            print("Error: Must provide either tokenizer_dir or both en_tokenizer and ru_tokenizer")
            return
        
        print(f"Creating switchable tokenizer from {args.en_tokenizer} and {args.ru_tokenizer}")
        tokenizer = SwitchableTokenizer(
            en_tokenizer_path=args.en_tokenizer,
            ru_tokenizer_path=args.ru_tokenizer,
        )
    
    # Example texts
    en_text = "Hello world! This is an example of English text for tokenization."
    ru_text = "Привет мир! Это пример русского текста для токенизации."
    mixed_text = "Hello! Привет! This is a mix of English and русский язык."
    
    # Tokenize English text
    en_tokens = tokenizer.encode(en_text, language="EN")
    print("\nEnglish tokenization:")
    print(f"Text: {en_text}")
    print(f"Token IDs: {en_tokens}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(en_tokens)}")
    print(f"Decoded: {tokenizer.decode(en_tokens)}")
    
    # Tokenize Russian text
    ru_tokens = tokenizer.encode(ru_text, language="RU")
    print("\nRussian tokenization:")
    print(f"Text: {ru_text}")
    print(f"Token IDs: {ru_tokens}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(ru_tokens)}")
    print(f"Decoded: {tokenizer.decode(ru_tokens)}")
    
    # Demonstrate mixed-language tokenization (with manual language switching)
    print("\nMixed language tokenization with manual switching:")
    print(f"Text: {mixed_text}")
    
    # Tokenize parts separately and combine
    parts = [
        (tokenizer.encode("Hello! ", language="EN"), "EN"),
        (tokenizer.encode("Привет! ", language="RU"), "RU"),
        (tokenizer.encode("This is a mix of English and ", language="EN"), "EN"),
        (tokenizer.encode("русский язык.", language="RU"), "RU"),
    ]
    
    # Combine the parts, preserving language tokens
    combined_tokens = []
    for tokens, _ in parts:
        combined_tokens.extend(tokens)
    
    print(f"Combined token IDs: {combined_tokens}")
    print(f"Tokens: {tokenizer.convert_ids_to_tokens(combined_tokens)}")
    print(f"Decoded: {tokenizer.decode(combined_tokens)}")
    
    # Demonstrate ID overlap analysis
    print("\nID overlap analysis:")
    
    # Take a sample ID in the shared space (let's say ID 1000)
    sample_id = 1000
    en_token = tokenizer.convert_ids_to_tokens([sample_id], language="EN")[0]
    ru_token = tokenizer.convert_ids_to_tokens([sample_id], language="RU")[0]
    
    print(f"ID {sample_id} represents:")
    print(f"  - In English tokenizer: '{en_token}'")
    print(f"  - In Russian tokenizer: '{ru_token}'")
    
    # Find another ID with interesting tokens in both languages
    for test_id in range(100, 1000, 100):
        en_token = tokenizer.convert_ids_to_tokens([test_id], language="EN")[0]
        ru_token = tokenizer.convert_ids_to_tokens([test_id], language="RU")[0]
        print(f"ID {test_id} represents:")
        print(f"  - In English tokenizer: '{en_token}'")
        print(f"  - In Russian tokenizer: '{ru_token}'")

if __name__ == "__main__":
    main() 