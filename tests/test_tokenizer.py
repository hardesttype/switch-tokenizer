import unittest
import torch
from transformers import AutoTokenizer

from src.switch_tokenizer import SwitchableTokenizer

class TestSwitchableTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize tokenizers for testing
        # We use small, fast-loading tokenizers for tests
        cls.en_tokenizer_name = "gpt2"
        cls.ru_tokenizer_name = "cointegrated/rubert-tiny"
        
        cls.tokenizer = SwitchableTokenizer(
            en_tokenizer_path=cls.en_tokenizer_name,
            ru_tokenizer_path=cls.ru_tokenizer_name,
        )
    
    def test_initialization(self):
        """Test that the tokenizer initializes correctly."""
        self.assertIsNotNone(self.tokenizer)
        self.assertEqual(self.tokenizer.shared_vocab_size, 64000)
        self.assertEqual(self.tokenizer.vocab_size, 64002)  # 64000 + 2 language tokens
        
        # Check that language tokens were created
        self.assertIn("<LANG_EN>", self.tokenizer.special_tokens)
        self.assertIn("<LANG_RU>", self.tokenizer.special_tokens)
        self.assertEqual(self.tokenizer.special_tokens["<LANG_EN>"], 64000)
        self.assertEqual(self.tokenizer.special_tokens["<LANG_RU>"], 64001)
    
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
        # Filter out any IDs that might be beyond our shared vocab space
        ru_ids = [id for id in ru_ids if id < self.tokenizer.shared_vocab_size]
        self.assertEqual(token_ids[1:], ru_ids)
    
    def test_decode(self):
        """Test decoding token IDs."""
        # English
        en_text = "Hello world"
        en_tokens = self.tokenizer.encode(en_text, language="EN")
        decoded_en = self.tokenizer.decode(en_tokens)
        
        # The decoded text might not be exactly the same due to tokenizer specifics
        # but should contain the original text
        self.assertIn("Hello world", decoded_en)
        
        # Russian
        ru_text = "Привет мир"
        ru_tokens = self.tokenizer.encode(ru_text, language="RU")
        decoded_ru = self.tokenizer.decode(ru_tokens)
        
        # The decoded text might not be exactly the same due to tokenizer specifics
        # but should contain the original text
        self.assertIn("Привет", decoded_ru)
    
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
        
        # The decoded text should contain both languages
        self.assertIn("Hello", decoded)
        self.assertIn("Привет", decoded)
        
        # Should appear twice due to the switch back
        self.assertEqual(decoded.count("Hello"), 2)
    
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
            self.assertEqual(loaded_tokenizer.shared_vocab_size, self.tokenizer.shared_vocab_size)
            self.assertEqual(loaded_tokenizer.vocab_size, self.tokenizer.vocab_size)
            self.assertEqual(loaded_tokenizer.special_tokens, self.tokenizer.special_tokens)
            
            # Test tokenization with loaded tokenizer
            text = "Hello world"
            original_ids = self.tokenizer.encode(text, language="EN")
            loaded_ids = loaded_tokenizer.encode(text, language="EN")
            
            self.assertEqual(original_ids, loaded_ids)

if __name__ == "__main__":
    unittest.main() 