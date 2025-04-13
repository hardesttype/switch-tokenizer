from typing import Dict, List, Optional, Union, Tuple
import os
from transformers import PreTrainedTokenizer, AutoTokenizer

class SwitchableTokenizer:
    """
    A tokenizer that switchs between different language-specific tokenizers
    while maintaining a shared 64k token ID space.
    
    This tokenizer uses context tokens to indicate which language-specific tokenizer
    should be used for interpreting the token IDs.
    """
    
    def __init__(
        self,
        en_tokenizer_path: str,
        ru_tokenizer_path: str,
        shared_vocab_size: int = 64000,
        special_tokens: Dict[str, int] = None,
    ):
        """
        Initialize the switchable tokenizer with language-specific tokenizers.
        
        Args:
            en_tokenizer_path: Path to the English tokenizer or its name from HF Hub
            ru_tokenizer_path: Path to the Russian tokenizer or its name from HF Hub
            shared_vocab_size: Size of the shared vocabulary space (default: 64000)
            special_tokens: Optional mapping of special token names to IDs
        """
        # Load the individual tokenizers
        self.tokenizer_en = AutoTokenizer.from_pretrained(en_tokenizer_path)
        self.tokenizer_ru = AutoTokenizer.from_pretrained(ru_tokenizer_path)
        
        # Set up the shared vocabulary size
        self.shared_vocab_size = shared_vocab_size
        
        # Define special tokens outside the shared vocab space
        self.special_tokens = {
            "<LANG_EN>": shared_vocab_size,
            "<LANG_RU>": shared_vocab_size + 1,
        }
        
        if special_tokens:
            # Add any additional special tokens
            next_id = shared_vocab_size + 2
            for token_name, token_id in special_tokens.items():
                if token_name not in self.special_tokens:
                    self.special_tokens[token_name] = next_id
                    next_id += 1
        
        # Store the total vocabulary size (shared vocab + special tokens)
        self.vocab_size = shared_vocab_size + len(self.special_tokens)
        
        # Map language codes to tokenizers and language tokens
        self.lang_map = {
            "EN": {
                "tokenizer": self.tokenizer_en,
                "lang_token_id": self.special_tokens["<LANG_EN>"],
                "lang_token": "<LANG_EN>",
            },
            "RU": {
                "tokenizer": self.tokenizer_ru,
                "lang_token_id": self.special_tokens["<LANG_RU>"],
                "lang_token": "<LANG_RU>",
            },
        }
        
        # Register special tokens with the underlying tokenizers
        for tokenizer in [self.tokenizer_en, self.tokenizer_ru]:
            special_tokens_dict = {
                "additional_special_tokens": list(self.special_tokens.keys())
            }
            tokenizer.add_special_tokens(special_tokens_dict)
    
    def encode(
        self, 
        text: str, 
        language: str,
        add_language_token: bool = True,
        **kwargs
    ) -> List[int]:
        """
        Encode the text using the appropriate language-specific tokenizer
        and add the language token at the beginning.
        
        Args:
            text: The text to encode
            language: Language code ("EN" or "RU")
            add_language_token: Whether to add the language token at the beginning
            **kwargs: Additional arguments passed to the tokenizer
            
        Returns:
            List of token IDs
        """
        if language not in self.lang_map:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.lang_map.keys())}")
        
        # Select the appropriate tokenizer based on language
        lang_info = self.lang_map[language]
        tokenizer = lang_info["tokenizer"]
        
        # Tokenize the text
        # We force the IDs to be within the shared vocab space
        token_ids = tokenizer.encode(text, add_special_tokens=False, **kwargs)
        token_ids = [tid for tid in token_ids if tid < self.shared_vocab_size]
        
        # Add language token if requested
        if add_language_token:
            return [lang_info["lang_token_id"]] + token_ids
        return token_ids
    
    def decode(
        self, 
        token_ids: List[int],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode the token IDs, handling language switching based on language tokens.
        
        Args:
            token_ids: List of token IDs to decode
            skip_special_tokens: Whether to skip special tokens in the output
            **kwargs: Additional arguments passed to the tokenizer
            
        Returns:
            Decoded text
        """
        if not token_ids:
            return ""
        
        # Initialize with default language (English)
        current_lang = "EN"
        
        # Buffers to accumulate decoded text and to collect IDs by language
        result = []
        current_segment = []
        
        for token_id in token_ids:
            # Check if this is a language token
            is_lang_token = False
            for lang, info in self.lang_map.items():
                if token_id == info["lang_token_id"]:
                    # Process any accumulated tokens with the current language
                    if current_segment:
                        segment_text = self._decode_segment(current_segment, current_lang, skip_special_tokens, **kwargs)
                        result.append(segment_text)
                        current_segment = []
                    
                    # Switch to the new language
                    current_lang = lang
                    is_lang_token = True
                    break
            
            if not is_lang_token:
                # If it's not a language token, add it to the current segment
                current_segment.append(token_id)
        
        # Process any remaining tokens
        if current_segment:
            segment_text = self._decode_segment(current_segment, current_lang, skip_special_tokens, **kwargs)
            result.append(segment_text)
        
        # Join all decoded segments
        return "".join(result)
    
    def _decode_segment(
        self, 
        token_ids: List[int],
        language: str,
        skip_special_tokens: bool = True,
        **kwargs
    ) -> str:
        """
        Decode a segment of token IDs with a specific language tokenizer.
        
        Args:
            token_ids: List of token IDs to decode
            language: Language code for the tokenizer to use
            skip_special_tokens: Whether to skip special tokens
            **kwargs: Additional arguments passed to the tokenizer
            
        Returns:
            Decoded text for the segment
        """
        if not token_ids:
            return ""
        
        tokenizer = self.lang_map[language]["tokenizer"]
        
        # Filter out any token IDs outside the shared vocab range
        valid_ids = [tid for tid in token_ids if tid < self.shared_vocab_size]
        
        # Handle empty sequence after filtering
        if not valid_ids:
            return ""
        
        return tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens, **kwargs)
    
    def convert_tokens_to_ids(self, tokens: List[str], language: str = None) -> List[int]:
        """
        Convert a list of tokens to token IDs using the appropriate tokenizer.
        
        Args:
            tokens: List of tokens to convert
            language: Language code ("EN" or "RU"). If None, tries to determine from tokens.
            
        Returns:
            List of token IDs
        """
        # If language is not provided, try to infer from special tokens
        if language is None:
            for token in tokens:
                if token == "<LANG_EN>":
                    language = "EN"
                    break
                elif token == "<LANG_RU>":
                    language = "RU"
                    break
            
            if language is None:
                # Default to English if still not determined
                language = "EN"
        
        result = []
        current_lang = None
        current_tokens = []
        
        for token in tokens:
            # Check if it's a special language token
            if token in self.special_tokens:
                # Process any accumulated tokens with current language
                if current_tokens and current_lang:
                    tokenizer = self.lang_map[current_lang]["tokenizer"]
                    ids = tokenizer.convert_tokens_to_ids(current_tokens)
                    result.extend([id for id in ids if id < self.shared_vocab_size])
                    current_tokens = []
                
                # Add the special token ID
                result.append(self.special_tokens[token])
                
                # Update current language if it's a language token
                if token == "<LANG_EN>":
                    current_lang = "EN"
                elif token == "<LANG_RU>":
                    current_lang = "RU"
            else:
                # If no language has been set yet, use the provided language
                if current_lang is None:
                    current_lang = language
                
                # Add to current token buffer
                current_tokens.append(token)
        
        # Process any remaining tokens
        if current_tokens and current_lang:
            tokenizer = self.lang_map[current_lang]["tokenizer"]
            ids = tokenizer.convert_tokens_to_ids(current_tokens)
            result.extend([id for id in ids if id < self.shared_vocab_size])
        
        return result
    
    def convert_ids_to_tokens(self, ids: List[int], language: str = None) -> List[str]:
        """
        Convert a list of token IDs to tokens, handling language switches.
        
        Args:
            ids: List of token IDs to convert
            language: Default language to use for the first tokens
            
        Returns:
            List of tokens
        """
        if language is None:
            # Default to English if not specified
            language = "EN"
        
        result = []
        current_lang = language
        
        for token_id in ids:
            # Check if it's a special token
            for token, id_value in self.special_tokens.items():
                if token_id == id_value:
                    result.append(token)
                    
                    # Update language if it's a language token
                    if token == "<LANG_EN>":
                        current_lang = "EN"
                    elif token == "<LANG_RU>":
                        current_lang = "RU"
                    
                    token_id = None  # Mark as processed
                    break
            
            # If not a special token and within shared vocab range
            if token_id is not None and token_id < self.shared_vocab_size:
                tokenizer = self.lang_map[current_lang]["tokenizer"]
                token = tokenizer.convert_ids_to_tokens([token_id])[0]
                result.append(token)
        
        return result
    
    @property
    def all_special_tokens(self) -> List[str]:
        """
        Get all special tokens used by the tokenizer.
        
        Returns:
            List of all special token strings
        """
        return list(self.special_tokens.keys())
    
    @property
    def all_special_ids(self) -> List[int]:
        """
        Get all special token IDs used by the tokenizer.
        
        Returns:
            List of special token IDs
        """
        return list(self.special_tokens.values())
    
    def save_pretrained(self, save_directory: str) -> Tuple[str, str]:
        """
        Save the switchable tokenizer configuration and component tokenizers.
        
        Args:
            save_directory: Directory to save the tokenizer
            
        Returns:
            Tuple of paths where the component tokenizers were saved
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save component tokenizers
        en_save_dir = os.path.join(save_directory, "en_tokenizer")
        ru_save_dir = os.path.join(save_directory, "ru_tokenizer")
        
        self.tokenizer_en.save_pretrained(en_save_dir)
        self.tokenizer_ru.save_pretrained(ru_save_dir)
        
        # Save config
        import json
        config = {
            "shared_vocab_size": self.shared_vocab_size,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        return en_save_dir, ru_save_dir
    
    @classmethod
    def from_pretrained(cls, directory: str) -> "SwitchableTokenizer":
        """
        Load a switchable tokenizer from a directory.
        
        Args:
            directory: Directory containing the saved tokenizer
            
        Returns:
            SwitchableTokenizer instance
        """
        import json
        
        config_file = os.path.join(directory, "config.json")
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        en_tokenizer_path = os.path.join(directory, "en_tokenizer")
        ru_tokenizer_path = os.path.join(directory, "ru_tokenizer")
        
        return cls(
            en_tokenizer_path=en_tokenizer_path,
            ru_tokenizer_path=ru_tokenizer_path,
            shared_vocab_size=config.get("shared_vocab_size", 64000),
            special_tokens=config.get("special_tokens", None),
        ) 