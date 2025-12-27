"""
📝 BagleyTokenizer - Custom Tokenizer with Multilingual Support
"""

import os
import json
import regex as re
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BagleyTokenizer:
    """
    Custom BPE Tokenizer for BagleyMoE
    
    Features:
    - Multilingual support (English, Dutch, 100+ languages)
    - Special tokens for personality modes
    - Efficient encoding/decoding
    - Streaming support
    """
    
    # Special tokens
    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|begin|>"
    EOS_TOKEN = "<|end|>"
    UNK_TOKEN = "<|unk|>"
    
    # Personality tokens
    PERSONALITY_TOKENS = [
        "<|chaos|>",
        "<|chill|>", 
        "<|focus|>",
        "<|custom|>",
    ]
    
    # Role tokens
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    
    # Thinking mode tokens (DeepSeek-R1 inspired)
    THINK_START = "<|think|>"
    THINK_END = "<|/think|>"
    
    # Special content tokens
    CODE_START = "<|code|>"
    CODE_END = "<|/code|>"
    IMAGE_TOKEN = "<|image|>"
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        vocab_size: int = 151936,
        model_max_length: int = 131072,
    ):
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length
        
        # Initialize vocab
        self.encoder: Dict[str, int] = {}
        self.decoder: Dict[int, str] = {}
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        
        # Special tokens
        self.special_tokens = self._build_special_tokens()
        self.special_token_ids = {}
        
        # Add special tokens to vocab first
        for i, token in enumerate(self.special_tokens):
            self.encoder[token] = i
            self.decoder[i] = token
            self.special_token_ids[token] = i
        
        # Load vocab if provided
        if vocab_file and os.path.exists(vocab_file):
            self._load_vocab(vocab_file)
        
        if merges_file and os.path.exists(merges_file):
            self._load_merges(merges_file)
        
        # Regex pattern for tokenization
        self.pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.IGNORECASE
        )
        
        # Cache for BPE
        self.cache: Dict[str, str] = {}
        
        logger.info(f"Initialized BagleyTokenizer with vocab size {len(self.encoder)}")
    
    def _build_special_tokens(self) -> List[str]:
        """Build list of all special tokens"""
        tokens = [
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN,
            self.UNK_TOKEN,
            self.SYSTEM_TOKEN,
            self.USER_TOKEN,
            self.ASSISTANT_TOKEN,
            self.THINK_START,
            self.THINK_END,
            self.CODE_START,
            self.CODE_END,
            self.IMAGE_TOKEN,
        ]
        tokens.extend(self.PERSONALITY_TOKENS)
        return tokens
    
    def _load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        
        # Add to encoder/decoder, preserving special tokens
        offset = len(self.special_tokens)
        for token, idx in vocab.items():
            if token not in self.encoder:
                self.encoder[token] = idx + offset
                self.decoder[idx + offset] = token
    
    def _load_merges(self, path: str):
        """Load BPE merges from file"""
        with open(path, 'r', encoding='utf-8') as f:
            merges = f.read().split('\n')
        
        for i, merge in enumerate(merges):
            if merge:
                parts = merge.split()
                if len(parts) == 2:
                    self.bpe_ranks[tuple(parts)] = i
    
    def _bpe(self, token: str) -> str:
        """Apply BPE to a token"""
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return token
        
        while True:
            # Find most frequent pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                new_word.extend(word[i:j])
                i = j
                
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            
            if len(word) == 1:
                break
            
            pairs = self._get_pairs(word)
        
        result = ' '.join(word)
        self.cache[token] = result
        return result
    
    def _get_pairs(self, word: Tuple[str, ...]) -> set:
        """Get all adjacent pairs in a word"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode text to token IDs
        
        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS
            max_length: Maximum sequence length
            truncation: Whether to truncate
            padding: Whether to pad to max_length
            
        Returns:
            List of token IDs
        """
        max_length = max_length or self.model_max_length
        
        # Handle special tokens in text
        tokens = []
        
        if add_special_tokens:
            tokens.append(self.special_token_ids[self.BOS_TOKEN])
        
        # Tokenize
        for match in self.pat.findall(text):
            # Check for special tokens
            if match in self.special_token_ids:
                tokens.append(self.special_token_ids[match])
            else:
                # Apply BPE
                bpe_tokens = self._bpe(match).split(' ')
                for bpe_token in bpe_tokens:
                    if bpe_token in self.encoder:
                        tokens.append(self.encoder[bpe_token])
                    else:
                        tokens.append(self.special_token_ids[self.UNK_TOKEN])
        
        if add_special_tokens:
            tokens.append(self.special_token_ids[self.EOS_TOKEN])
        
        # Truncate
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Pad
        if padding and len(tokens) < max_length:
            pad_id = self.special_token_ids[self.PAD_TOKEN]
            tokens.extend([pad_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.decoder:
                token = self.decoder[token_id]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        text = ''.join(tokens)
        return text
    
    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        personality_mode: str = "chaos",
    ) -> List[int]:
        """
        Encode a chat conversation
        
        Args:
            messages: List of {"role": str, "content": str}
            system_prompt: Optional system prompt
            personality_mode: Personality mode (chaos/chill/focus/custom)
            
        Returns:
            Encoded token IDs
        """
        tokens = [self.special_token_ids[self.BOS_TOKEN]]
        
        # Add personality token
        personality_token = f"<|{personality_mode}|>"
        if personality_token in self.special_token_ids:
            tokens.append(self.special_token_ids[personality_token])
        
        # Add system prompt
        if system_prompt:
            tokens.append(self.special_token_ids[self.SYSTEM_TOKEN])
            tokens.extend(self.encode(system_prompt, add_special_tokens=False))
        
        # Add messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "user":
                tokens.append(self.special_token_ids[self.USER_TOKEN])
            elif role == "assistant":
                tokens.append(self.special_token_ids[self.ASSISTANT_TOKEN])
            elif role == "system":
                tokens.append(self.special_token_ids[self.SYSTEM_TOKEN])
            
            tokens.extend(self.encode(content, add_special_tokens=False))
        
        return tokens
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary"""
        return self.encoder.copy()
    
    def __len__(self) -> int:
        return len(self.encoder)
    
    @property
    def pad_token_id(self) -> int:
        return self.special_token_ids[self.PAD_TOKEN]
    
    @property
    def bos_token_id(self) -> int:
        return self.special_token_ids[self.BOS_TOKEN]
    
    @property
    def eos_token_id(self) -> int:
        return self.special_token_ids[self.EOS_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.special_token_ids[self.UNK_TOKEN]
    
    def save_pretrained(self, path: str):
        """Save tokenizer to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save vocab
        with open(os.path.join(path, "vocab.json"), 'w', encoding='utf-8') as f:
            json.dump(self.encoder, f, ensure_ascii=False, indent=2)
        
        # Save merges
        merges = sorted(self.bpe_ranks.items(), key=lambda x: x[1])
        with open(os.path.join(path, "merges.txt"), 'w', encoding='utf-8') as f:
            for (a, b), _ in merges:
                f.write(f"{a} {b}\n")
        
        # Save config
        config = {
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "special_tokens": self.special_tokens,
        }
        with open(os.path.join(path, "tokenizer_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Tokenizer saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str) -> "BagleyTokenizer":
        """Load tokenizer from disk"""
        vocab_file = os.path.join(path, "vocab.json")
        merges_file = os.path.join(path, "merges.txt")
        config_file = os.path.join(path, "tokenizer_config.json")
        
        # Load config
        vocab_size = 151936
        model_max_length = 131072
        
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
            vocab_size = config.get("vocab_size", vocab_size)
            model_max_length = config.get("model_max_length", model_max_length)
        
        return cls(
            vocab_file=vocab_file if os.path.exists(vocab_file) else None,
            merges_file=merges_file if os.path.exists(merges_file) else None,
            vocab_size=vocab_size,
            model_max_length=model_max_length,
        )
