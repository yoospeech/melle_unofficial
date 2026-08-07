import json
import os

class CharacterTokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<eoa>"]
        
    def train(self, manifest_path, save_path="tokenizer.json"):
        print(f"Loading text from {manifest_path}...")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Collect all unique characters
        chars = set()
        for item in data:
            chars.update(item['text'])
            
        # Sort characters for deterministic ordering
        sorted_chars = sorted(list(chars))
        print(f"Found {len(sorted_chars)} unique characters.")
        
        # Build vocab
        # 0: <pad>, 1: <s>, 2: </s>, 3: <unk>
        for i, token in enumerate(self.special_tokens):
            self.char2id[token] = i
            self.id2char[i] = token
            
        start_idx = len(self.special_tokens)
        for i, char in enumerate(sorted_chars):
            idx = start_idx + i
            self.char2id[char] = idx
            self.id2char[idx] = char
            
        print(f"Total vocab size: {len(self.char2id)}")
        
        # Save
        self.save(save_path)
        
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "char2id": self.char2id,
                "id2char": {k: v for k, v in self.id2char.items()} # keys must be str for json
            }, f, ensure_ascii=False, indent=2)
            
    def load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2id = data["char2id"]
            self.id2char = {int(k): v for k, v in data["id2char"].items()}
            
    def encode(self, text):
        # Add start/end tokens? Usually handled by dataset, but let's just return ids
        ids = []
        for char in text:
            ids.append(self.char2id.get(char, self.char2id["<unk>"]))
        return ids
        
    def decode(self, ids):
        chars = []
        for idx in ids:
            if isinstance(idx, list): # Handle batch decoding if needed, but here simple
                continue
            if hasattr(idx, 'item'): idx = idx.item()
            
            if idx in self.id2char:
                token = self.id2char[idx]
                if token not in self.special_tokens:
                    chars.append(token)
        return "".join(chars)
        
    @property
    def vocab_size(self):
        return len(self.char2id)
    
    @property
    def unk_token_id(self):
        return self.char2id["<unk>"]
    
    @property
    def pad_token_id(self):
        return self.char2id["<pad>"]
        
    @property
    def bos_token_id(self):
        return self.char2id["<s>"]
        
    @property
    def eos_token_id(self):
        return self.char2id["</s>"]
        
    @property
    def eoa_token_id(self):
        return self.char2id["<eoa>"]

if __name__ == "__main__":
    MANIFEST_PATH = os.environ.get("MANIFEST_PATH", "manifest.json")
    SAVE_PATH = os.environ.get("TOKENIZER_PATH", "tokenizer.json")
    
    tokenizer = CharacterTokenizer()
    if os.path.exists(MANIFEST_PATH):
        tokenizer.train(MANIFEST_PATH, save_path=SAVE_PATH)
        
        # Test
        sample_text = "안녕하세요"
        encoded = tokenizer.encode(sample_text)
        print(f"\nTest encoding '{sample_text}':")
        print(f"IDs: {encoded}")
        
        decoded = tokenizer.decode(encoded)
        print(f"Decoded: {decoded}")
    else:
        print(f"Manifest not found at {MANIFEST_PATH}")
