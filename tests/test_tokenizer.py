import unittest
import warnings
from transformers import AutoTokenizer

from src.switch_tokenizer import SwitchableTokenizer

class TestSwitchableTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize tokenizers for testing
        # We use small, fast-loading tokenizers for tests
        cls.en_tokenizer_name = "gpt2"
        cls.ru_tokenizer_name = "ai-forever/ruGPT-3.5-13B"  # "DeepPavlov/rubert-base-cased"
        
        cls.tokenizer = SwitchableTokenizer(
            en_tokenizer_path=cls.en_tokenizer_name,
            ru_tokenizer_path=cls.ru_tokenizer_name,
            shared_vocab_size=64000,  # Explicitly set for backward compatibility with existing tests
        )
    
    def test_initialization(self):
        """Test that the tokenizer initializes correctly."""
        self.assertIsNotNone(self.tokenizer)
        # No longer test the exact shared_vocab_size as it's automatically adjusted
        self.assertGreaterEqual(self.tokenizer.shared_vocab_size, 64000)
        # Total vocab size should be shared vocab size + number of special tokens
        self.assertEqual(self.tokenizer.vocab_size, 
                        self.tokenizer.shared_vocab_size + len(self.tokenizer.special_tokens))
        
        # Check that language tokens were created
        self.assertIn("<LANG_EN>", self.tokenizer.special_tokens)
        self.assertIn("<LANG_RU>", self.tokenizer.special_tokens)
        self.assertEqual(self.tokenizer.special_tokens["<LANG_EN>"], self.tokenizer.shared_vocab_size)
        self.assertEqual(self.tokenizer.special_tokens["<LANG_RU>"], self.tokenizer.shared_vocab_size + 1)
    
    def test_auto_vocab_size(self):
        """Test automatic determination of shared vocabulary size."""
        # Create a tokenizer with automatic vocab size detection
        auto_tokenizer = SwitchableTokenizer(
            en_tokenizer_path=self.en_tokenizer_name,
            ru_tokenizer_path=self.ru_tokenizer_name,
            shared_vocab_size=None,  # Auto-detect
        )
        
        # Check that shared_vocab_size is close to the maximum of both tokenizers
        self.assertGreaterEqual(auto_tokenizer.shared_vocab_size, 50000)  # Reasonable for GPT2
        
        # Verify total vocab size includes special tokens
        self.assertEqual(auto_tokenizer.vocab_size, 
                         auto_tokenizer.shared_vocab_size + len(auto_tokenizer.special_tokens))
    
    def test_vocab_size_adjustment(self):
        """Test auto-adjustment of small vocabulary size."""
        # Find which tokenizer has the larger vocabulary
        en_tokenizer = AutoTokenizer.from_pretrained(self.en_tokenizer_name)
        ru_tokenizer = AutoTokenizer.from_pretrained(self.ru_tokenizer_name)
        
        en_size = len(en_tokenizer)
        ru_size = len(ru_tokenizer)
        max_size = max(en_size, ru_size)
        
        # Create a tokenizer with a small shared vocab size
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            small_vocab_tokenizer = SwitchableTokenizer(
                en_tokenizer_path=self.en_tokenizer_name,
                ru_tokenizer_path=self.ru_tokenizer_name,
                shared_vocab_size=100,  # Artificially small
            )
            
            # Check that warning was raised
            self.assertTrue(any("Automatically increasing" in str(warning.message) for warning in w))
        
        # Check that shared_vocab_size was automatically increased
        self.assertEqual(small_vocab_tokenizer.shared_vocab_size, max_size)
    
    def test_encode_english(self):
        """Test encoding English text."""
        text = "Hello world"
        token_ids = self.tokenizer.encode(text, language="EN")
        
        # First token should be the language token
        self.assertEqual(token_ids[0], self.tokenizer.special_tokens["<LANG_EN>"])
        
        # Rest of the tokens should be from the English tokenizer
        en_ids = self.tokenizer.tokenizer_en.encode(text, add_special_tokens=False)
        self.assertEqual(token_ids[1:], en_ids)
    
    def test_encode_russian(self):
        """Test encoding Russian text."""
        text = "Привет мир"
        token_ids = self.tokenizer.encode(text, language="RU")
        
        # First token should be the language token
        self.assertEqual(token_ids[0], self.tokenizer.special_tokens["<LANG_RU>"])
        
        # Rest of the tokens should be from the Russian tokenizer
        ru_ids = self.tokenizer.tokenizer_ru.encode(text, add_special_tokens=False)
        self.assertEqual(token_ids[1:], ru_ids)
    
    def test_decode(self):
        """Test decoding token IDs."""
        # English
        en_text = "Hello world"
        en_tokens = self.tokenizer.encode(en_text, language="EN")
        decoded_en = self.tokenizer.decode(en_tokens)
        
        # The decoded text should be exactly the same as the original text
        self.assertEqual(decoded_en, en_text)
        
        # Russian
        ru_text = "Привет мир"
        ru_tokens = self.tokenizer.encode(ru_text, language="RU")
        decoded_ru = self.tokenizer.decode(ru_tokens)
        
        # The decoded text should be exactly the same as the original text
        self.assertEqual(decoded_ru, ru_text)
    
    def test_language_switching(self):
        """Test handling language switching in token sequences."""
        # Create a mixed sequence with language switches
        en_lang_id = self.tokenizer.special_tokens["<LANG_EN>"]
        ru_lang_id = self.tokenizer.special_tokens["<LANG_RU>"]
        
        en_hello = self.tokenizer.encode("Hello", language="EN", add_language_token=False)
        ru_privet = self.tokenizer.encode("Привет", language="RU", add_language_token=False)
        
        # Sequence with explicit language switches
        mixed_ids = [en_lang_id] + en_hello + [ru_lang_id] + ru_privet + [en_lang_id] + en_hello
        
        decoded = self.tokenizer.decode(mixed_ids)
        
        # The decoded text should be exactly "HelloПриветHello"
        expected = "HelloПриветHello"
        self.assertEqual(decoded, expected)
    
    def test_convert_tokens_to_ids(self):
        """Test converting tokens to IDs."""
        # English tokens
        en_tokens = ["Hello", "world"]
        en_ids = self.tokenizer.convert_tokens_to_ids(en_tokens, language="EN")
        
        # Check that IDs match what the English tokenizer would produce
        expected_en_ids = self.tokenizer.tokenizer_en.convert_tokens_to_ids(en_tokens)
        self.assertEqual(en_ids, expected_en_ids)
        
        # Test with language tokens
        tokens_with_lang = ["<LANG_EN>", "Hello", "<LANG_RU>", "мир"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens_with_lang)
        
        # Check first token is the EN language ID
        self.assertEqual(ids[0], self.tokenizer.special_tokens["<LANG_EN>"])
        
        # Check third token is the RU language ID
        self.assertEqual(ids[2], self.tokenizer.special_tokens["<LANG_RU>"])
    
    def test_convert_ids_to_tokens(self):
        """Test converting IDs to tokens."""
        # English
        en_text = "Hello world"
        en_ids = self.tokenizer.encode(en_text, language="EN")
        tokens = self.tokenizer.convert_ids_to_tokens(en_ids)
        
        # First token should be the language token
        self.assertEqual(tokens[0], "<LANG_EN>")
        
        # Russian
        ru_text = "Привет мир"
        ru_ids = self.tokenizer.encode(ru_text, language="RU")
        tokens = self.tokenizer.convert_ids_to_tokens(ru_ids)
        
        # First token should be the language token
        self.assertEqual(tokens[0], "<LANG_RU>")
    
    def test_save_and_load(self):
        """Test saving and loading the tokenizer."""
        import tempfile
        import shutil
        import os
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Save the tokenizer
            save_dirs = self.tokenizer.save_pretrained(tmpdirname)
            
            # Check that component tokenizers were saved
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "en_tokenizer")))
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "ru_tokenizer")))
            self.assertTrue(os.path.exists(os.path.join(tmpdirname, "config.json")))
            
            # Load the tokenizer
            loaded_tokenizer = SwitchableTokenizer.from_pretrained(tmpdirname)
            
            # Check that it loaded correctly
            self.assertEqual(loaded_tokenizer.shared_vocab_size, 
                           self.tokenizer.shared_vocab_size)
            self.assertEqual(loaded_tokenizer.vocab_size, 
                           loaded_tokenizer.shared_vocab_size + len(loaded_tokenizer.special_tokens))
            
            # Check that special tokens exist and have their expected positions
            self.assertEqual(set(loaded_tokenizer.special_tokens.keys()), 
                           set(self.tokenizer.special_tokens.keys()))
            self.assertEqual(loaded_tokenizer.special_tokens["<LANG_EN>"], 
                           loaded_tokenizer.shared_vocab_size)
            self.assertEqual(loaded_tokenizer.special_tokens["<LANG_RU>"], 
                           loaded_tokenizer.shared_vocab_size + 1)
            
            # Test tokenization with loaded tokenizer
            text = "Hello world"
            original_ids = self.tokenizer.encode(text, language="EN")
            loaded_ids = loaded_tokenizer.encode(text, language="EN")
            
            # We can't directly compare IDs because the language token IDs may be different
            # Instead, check that the first token is a language token and the rest match
            self.assertEqual(original_ids[0], self.tokenizer.special_tokens["<LANG_EN>"])
            self.assertEqual(loaded_ids[0], loaded_tokenizer.special_tokens["<LANG_EN>"])
            self.assertEqual(original_ids[1:], loaded_ids[1:])

    def test_cross_vocab_token_handling(self):
        """Test handling of token IDs that are outside the vocabulary range of the current language."""
        # Get vocabulary sizes of both tokenizers
        en_vocab_size = len(self.tokenizer.tokenizer_en)
        ru_vocab_size = len(self.tokenizer.tokenizer_ru)
        
        # Create a sequence mixing English-valid and Russian-only token IDs
        # English tokens first (excluding language token for this test)
        en_hello = self.tokenizer.encode("Hello", language="EN", add_language_token=False)
        
        # Manually create token IDs that would be valid only for Russian
        # (assuming Russian vocab is larger than English)
        if ru_vocab_size > en_vocab_size:
            # Pick some IDs between English vocab size and Russian vocab size
            ru_only_ids = [en_vocab_size + i for i in range(5)]
            
            # Combined sequence with both types of tokens
            mixed_ids = en_hello + ru_only_ids
            
            # Decode with English context - Russian-only tokens should be filtered
            decoded = self.tokenizer._decode_segment(mixed_ids, "EN", skip_special_tokens=True)
            
            # Should only include the English tokens
            expected = self.tokenizer.tokenizer_en.decode(en_hello, skip_special_tokens=True)
            self.assertEqual(decoded, expected)
            
            # Decode with Russian context - all tokens should be included
            if all(tid < ru_vocab_size for tid in mixed_ids):
                decoded_ru = self.tokenizer._decode_segment(mixed_ids, "RU", skip_special_tokens=True)
                # Should be longer than the English-only decoded text
                self.assertNotEqual(decoded, decoded_ru)
                self.assertGreater(len(decoded_ru), len(decoded))

if __name__ == "__main__":
    unittest.main() 