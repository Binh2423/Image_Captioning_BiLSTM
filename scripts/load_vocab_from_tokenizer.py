#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility to load vocabulary from tokenizer.pkl and export to vocab.json
Hỗ trợ robust loading với fallback cho missing classes
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any

try:
    import joblib
except ImportError:
    joblib = None
    warnings.warn("joblib not installed. Install with: pip install joblib")


class SafeVocab:
    """Placeholder class for safe unpickling"""
    def __init__(self):
        self.str_2_idx = {}
        self.idx_2_str = {}
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3


class SafeTokenizer:
    """Placeholder class for safe unpickling"""
    def __init__(self):
        self.vocab = SafeVocab()


class SafeUnpickler(pickle.Unpickler):
    """
    Safe unpickler that handles missing modules/classes
    """
    def find_class(self, module, name):
        # Try to import the actual class first
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            # If not found, use placeholder classes
            if name == 'Vocab':
                return SafeVocab
            elif name == 'Tokenizer':
                return SafeTokenizer
            else:
                # Generic placeholder
                warnings.warn(f"Could not find {module}.{name}, using placeholder")
                return type(name, (), {})


def load_vocab_with_joblib(vocab_path: Path) -> Dict[str, Any]:
    """Load vocab using joblib"""
    if joblib is None:
        raise RuntimeError("joblib not available")
    
    try:
        tokenizer = joblib.load(vocab_path)
        
        if hasattr(tokenizer, 'vocab'):
            vocab_obj = tokenizer.vocab
            
            vocab_dict = {
                'token2idx': dict(vocab_obj.str_2_idx) if hasattr(vocab_obj, 'str_2_idx') else {},
                'idx2token': {str(k): v for k, v in vocab_obj.idx_2_str.items()} if hasattr(vocab_obj, 'idx_2_str') else {},
                'vocab_size': len(vocab_obj.str_2_idx) if hasattr(vocab_obj, 'str_2_idx') else 0,
                'pad_idx': int(vocab_obj.pad_idx) if hasattr(vocab_obj, 'pad_idx') else 0,
                'sos_idx': int(vocab_obj.sos_idx) if hasattr(vocab_obj, 'sos_idx') else 1,
                'eos_idx': int(vocab_obj.eos_idx) if hasattr(vocab_obj, 'eos_idx') else 2,
                'unk_idx': int(vocab_obj.unk_idx) if hasattr(vocab_obj, 'unk_idx') else 3,
            }
            
            return vocab_dict
        else:
            raise ValueError("Loaded object doesn't have 'vocab' attribute")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load with joblib: {e}")


def load_vocab_with_pickle(vocab_path: Path) -> Dict[str, Any]:
    """Load vocab using standard pickle with safe unpickler"""
    try:
        with open(vocab_path, 'rb') as f:
            tokenizer = SafeUnpickler(f).load()
        
        if hasattr(tokenizer, 'vocab'):
            vocab_obj = tokenizer.vocab
            
            vocab_dict = {
                'token2idx': dict(vocab_obj.str_2_idx) if hasattr(vocab_obj, 'str_2_idx') else {},
                'idx2token': {str(k): v for k, v in vocab_obj.idx_2_str.items()} if hasattr(vocab_obj, 'idx_2_str') else {},
                'vocab_size': len(vocab_obj.str_2_idx) if hasattr(vocab_obj, 'str_2_idx') else 0,
                'pad_idx': int(vocab_obj.pad_idx) if hasattr(vocab_obj, 'pad_idx') else 0,
                'sos_idx': int(vocab_obj.sos_idx) if hasattr(vocab_obj, 'sos_idx') else 1,
                'eos_idx': int(vocab_obj.eos_idx) if hasattr(vocab_obj, 'eos_idx') else 2,
                'unk_idx': int(vocab_obj.unk_idx) if hasattr(vocab_obj, 'unk_idx') else 3,
            }
            
            return vocab_dict
        else:
            raise ValueError("Loaded object doesn't have 'vocab' attribute")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load with pickle: {e}")


def load_vocab_from_tokenizer(vocab_path: str) -> Dict[str, Any]:
    """
    Robust vocab loader - tries multiple methods
    
    Args:
        vocab_path: Path to vocab.pkl file
        
    Returns:
        Dictionary with vocab information
    """
    vocab_path = Path(vocab_path)
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    # If already JSON, just load it
    if vocab_path.suffix == '.json':
        with open(vocab_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Try joblib first
    if joblib is not None:
        try:
            print("Trying to load with joblib...")
            vocab_dict = load_vocab_with_joblib(vocab_path)
            print("✓ Successfully loaded with joblib")
            return vocab_dict
        except Exception as e:
            print(f"✗ joblib failed: {e}")
    
    # Try safe pickle unpickler
    try:
        print("Trying to load with safe pickle unpickler...")
        vocab_dict = load_vocab_with_pickle(vocab_path)
        print("✓ Successfully loaded with safe pickle unpickler")
        return vocab_dict
    except Exception as e:
        print(f"✗ Safe pickle unpickler failed: {e}")
        raise RuntimeError(f"All loading methods failed for {vocab_path}")


def save_vocab_json(vocab_dict: Dict[str, Any], output_path: str):
    """Save vocab dictionary to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Saved vocabulary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Load vocabulary from tokenizer.pkl and export to vocab.json'
    )
    parser.add_argument('--vocab-pkl', type=str, default='vocab.pkl',
                       help='Path to input vocab.pkl file')
    parser.add_argument('--output', type=str, default='data/vocab.json',
                       help='Path to output vocab.json file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print vocabulary statistics')
    
    args = parser.parse_args()
    
    print(f"Loading vocabulary from {args.vocab_pkl}...")
    
    try:
        vocab_dict = load_vocab_from_tokenizer(args.vocab_pkl)
        
        # Print statistics
        print("\nVocabulary statistics:")
        print(f"  Vocab size: {vocab_dict['vocab_size']}")
        print(f"  PAD index: {vocab_dict['pad_idx']}")
        print(f"  SOS index: {vocab_dict['sos_idx']}")
        print(f"  EOS index: {vocab_dict['eos_idx']}")
        print(f"  UNK index: {vocab_dict['unk_idx']}")
        
        if args.verbose and vocab_dict.get('token2idx'):
            print(f"\n  Sample tokens:")
            token2idx = vocab_dict['token2idx']
            for i, (token, idx) in enumerate(list(token2idx.items())[:10]):
                print(f"    {token}: {idx}")
            if len(token2idx) > 10:
                print(f"    ... and {len(token2idx) - 10} more tokens")
        
        # Save to JSON
        save_vocab_json(vocab_dict, args.output)
        
        print("\n✓ Success!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
