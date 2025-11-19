#!/usr/bin/env python3
"""
Autoregressive Image Captioning with DeiT and Attention-based Decoder

This script provides a complete training and inference pipeline for image captioning
with the following features:
- DeiT feature extractor with optional freezing
- Linear projection of DeiT features before the encoder
- Autoregressive (uni-directional) decoder with attention
- Configurable dropout and number of layers
- Weight tying between decoder embedding and output projection
- Feature pre-extraction mode to save/load DeiT features for faster training
- CLI flags for training vs inference modes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import timm
from typing import Optional, Tuple, Dict, List
import json
import pickle
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from PIL import Image
from torchvision import transforms
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeiTFeatureExtractor(nn.Module):
    """DeiT-based feature extractor with optional freezing and projection."""
    
    def __init__(
        self,
        model_name: str = 'deit_base_patch16_224',
        pretrained: bool = True,
        freeze: bool = False,
        projection_dim: Optional[int] = None,
    ):
        super().__init__()
        
        # Load DeiT model (PyTorch weights only via timm)
        self.deit = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Keep patch tokens
        )
        
        # Get feature dimension from the model
        self.feature_dim = self.deit.num_features
        
        # Optionally freeze DeiT parameters
        if freeze:
            for param in self.deit.parameters():
                param.requires_grad = False
            logger.info("DeiT backbone frozen")
        
        # Linear projection layer
        self.projection = None
        if projection_dim is not None:
            self.projection = nn.Linear(self.feature_dim, projection_dim)
            self.output_dim = projection_dim
        else:
            self.output_dim = self.feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor of shape (batch_size, 3, H, W)
        Returns:
            features: Tensor of shape (batch_size, num_patches, output_dim)
        """
        # Extract features using DeiT
        features = self.deit.forward_features(x)
        
        # Remove CLS token (keep only patch tokens)
        features = features[:, 1:, :]  # (batch_size, num_patches, feature_dim)
        
        # Apply projection if available
        if self.projection is not None:
            features = self.projection(features)
        
        return features


class AutoregressiveDecoder(nn.Module):
    """Autoregressive decoder with attention mechanism and weight tying."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.pad_idx = pad_idx
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection (will be tied with embedding)
        self.output_projection = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying: share weights between embedding and output projection
        self.output_projection.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.pos_encoding, std=0.02)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence (batch_size, seq_len)
            memory: Encoded image features (batch_size, num_patches, embed_dim)
            tgt_mask: Causal mask for target sequence
            tgt_key_padding_mask: Padding mask for target sequence
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        seq_len = tgt.size(1)
        
        # Embed tokens and add positional encoding
        tgt_emb = self.embedding(tgt)  # (batch_size, seq_len, embed_dim)
        tgt_emb = tgt_emb + self.pos_encoding[:, :seq_len, :]
        tgt_emb = self.dropout(tgt_emb)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(seq_len, tgt.device)
        
        # Apply transformer decoder
        output = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits


class ImageCaptioningModel(nn.Module):
    """Complete image captioning model with DeiT encoder and autoregressive decoder."""
    
    def __init__(
        self,
        vocab_size: int,
        model_name: str = 'deit_base_patch16_224',
        pretrained: bool = True,
        freeze_deit: bool = False,
        projection_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_len: int = 128,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = DeiTFeatureExtractor(
            model_name=model_name,
            pretrained=pretrained,
            freeze=freeze_deit,
            projection_dim=projection_dim,
        )
        
        # Decoder
        self.decoder = AutoregressiveDecoder(
            vocab_size=vocab_size,
            embed_dim=projection_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
        )
        
        self.max_len = max_len
        self.pad_idx = pad_idx
    
    def forward(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        caption_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            images: Image tensor (batch_size, 3, H, W)
            captions: Caption tokens (batch_size, seq_len)
            caption_mask: Padding mask for captions
        Returns:
            logits: Output logits (batch_size, seq_len, vocab_size)
        """
        # Extract image features
        image_features = self.feature_extractor(images)
        
        # Decode captions
        logits = self.decoder(
            tgt=captions,
            memory=image_features,
            tgt_key_padding_mask=caption_mask,
        )
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        start_token: int,
        end_token: int,
        max_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate captions autoregressively.
        
        Args:
            images: Image tensor (batch_size, 3, H, W)
            start_token: Start-of-sequence token ID
            end_token: End-of-sequence token ID
            max_len: Maximum generation length
        Returns:
            generated: Generated token sequences (batch_size, seq_len)
        """
        batch_size = images.size(0)
        device = images.device
        max_len = max_len or self.max_len
        
        # Extract image features once
        image_features = self.feature_extractor(images)
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_len - 1):
            # Get logits for current sequence
            logits = self.decoder(tgt=generated, memory=image_features)
            
            # Get next token (greedy decoding)
            next_token = logits[:, -1, :].argmax(dim=-1)
            
            # Mark finished sequences
            finished = finished | (next_token == end_token)
            
            # Append next token
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return generated


class PreExtractedFeaturesDataset(Dataset):
    """Dataset for pre-extracted DeiT features."""
    
    def __init__(self, features_dir: str, captions_file: str, tokenizer):
        self.features_dir = Path(features_dir)
        self.captions_df = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        image_id = row['image']
        caption = row['caption']
        
        # Load pre-extracted features
        feature_path = self.features_dir / f"{image_id}.pt"
        features = torch.load(feature_path)
        
        # Tokenize caption
        tokens = self.tokenizer.encode(caption)
        
        return features, tokens


class ImageCaptionDataset(Dataset):
    """Dataset for image-caption pairs with on-the-fly feature extraction."""
    
    def __init__(
        self,
        images_dir: str,
        captions_file: str,
        tokenizer,
        transform=None,
    ):
        self.images_dir = Path(images_dir)
        self.captions_df = pd.read_csv(captions_file)
        self.tokenizer = tokenizer
        self.transform = transform
    
    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        image_id = row['image']
        caption = row['caption']
        
        # Load and transform image
        image_path = self.images_dir / image_id
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Tokenize caption
        tokens = self.tokenizer.encode(caption)
        
        return image, tokens


class SimpleTokenizer:
    """Simple tokenizer for captions."""
    
    def __init__(self, vocab: Dict[str, int], max_len: int = 128):
        self.vocab = vocab
        self.max_len = max_len
        self.idx_to_word = {v: k for k, v in vocab.items()}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.start_token = '<start>'
        self.end_token = '<end>'
        self.unk_token = '<unk>'
        
        self.pad_idx = vocab[self.pad_token]
        self.start_idx = vocab[self.start_token]
        self.end_idx = vocab[self.end_token]
        self.unk_idx = vocab[self.unk_token]
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        words = text.lower().split()
        tokens = [self.start_idx]
        tokens.extend([self.vocab.get(word, self.unk_idx) for word in words])
        tokens.append(self.end_idx)
        
        # Pad or truncate
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens.extend([self.pad_idx] * (self.max_len - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in tokens:
            token_id = token_id.item()
            if token_id == self.end_idx:
                break
            if token_id not in [self.pad_idx, self.start_idx]:
                words.append(self.idx_to_word.get(token_id, self.unk_token))
        return ' '.join(words)


def collate_fn(batch, pad_idx: int):
    """Collate function for DataLoader."""
    if isinstance(batch[0][0], torch.Tensor) and batch[0][0].dim() == 2:
        # Pre-extracted features
        features = torch.stack([item[0] for item in batch])
        captions = torch.stack([item[1] for item in batch])
    else:
        # Images
        images = torch.stack([item[0] for item in batch])
        captions = torch.stack([item[1] for item in batch])
        features = images
    
    # Create padding mask
    caption_mask = (captions == pad_idx)
    
    # Split into input and target
    caption_input = captions[:, :-1]
    caption_target = captions[:, 1:]
    caption_mask = caption_mask[:, :-1]
    
    return features, caption_input, caption_target, caption_mask


def pre_extract_features(args):
    """Pre-extract and save DeiT features to disk."""
    logger.info("Pre-extracting features...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load feature extractor
    feature_extractor = DeiTFeatureExtractor(
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze=False,
        projection_dim=args.projection_dim,
    ).to(device).eval()
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Create output directory
    features_dir = Path(args.features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images
    images_dir = Path(args.data_dir) / 'images'
    captions_df = pd.read_csv(args.train_captions)
    
    unique_images = captions_df['image'].unique()
    logger.info(f"Extracting features for {len(unique_images)} images...")
    
    with torch.no_grad():
        for image_id in tqdm(unique_images):
            image_path = images_dir / image_id
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Extract features
            features = feature_extractor(image_tensor)
            
            # Save features
            feature_path = features_dir / f"{image_id}.pt"
            torch.save(features.cpu().squeeze(0), feature_path)
    
    logger.info(f"Features saved to {features_dir}")


def build_vocab(captions_file: str, min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from captions."""
    df = pd.read_csv(captions_file)
    
    # Count word frequencies
    word_freq = {}
    for caption in df['caption']:
        for word in caption.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Filter by frequency
    vocab = {'<pad>': 0, '<start>': 1, '<end>': 2, '<unk>': 3}
    idx = 4
    for word, freq in sorted(word_freq.items()):
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    return vocab


def train(args):
    """Training loop."""
    logger.info("Starting training...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build or load vocabulary
    vocab_path = Path(args.output_dir) / 'vocab.json'
    if vocab_path.exists():
        with open(vocab_path, 'r') as f:
            vocab = json.load(f)
    else:
        vocab = build_vocab(args.train_captions, min_freq=args.min_freq)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)
    
    tokenizer = SimpleTokenizer(vocab, max_len=args.max_len)
    
    # Create datasets
    if args.use_preextracted:
        train_dataset = PreExtractedFeaturesDataset(
            features_dir=args.features_dir,
            captions_file=args.train_captions,
            tokenizer=tokenizer,
        )
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = ImageCaptionDataset(
            images_dir=Path(args.data_dir) / 'images',
            captions_file=args.train_captions,
            tokenizer=tokenizer,
            transform=transform,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_idx),
    )
    
    # Create model
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        model_name=args.model_name,
        pretrained=args.pretrained,
        freeze_deit=args.freeze_deit,
        projection_dim=args.projection_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (features, caption_input, caption_target, caption_mask) in enumerate(progress_bar):
            features = features.to(device)
            caption_input = caption_input.to(device)
            caption_target = caption_target.to(device)
            caption_mask = caption_mask.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if args.use_preextracted:
                # Use pre-extracted features directly
                logits = model.decoder(
                    tgt=caption_input,
                    memory=features,
                    tgt_key_padding_mask=caption_mask,
                )
            else:
                logits = model(features, caption_input, caption_mask)
            
            # Compute loss
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                caption_target.reshape(-1),
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = Path(args.output_dir) / 'final_model.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")


def inference(args):
    """Inference mode for generating captions."""
    logger.info("Starting inference...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary
    vocab_path = Path(args.output_dir) / 'vocab.json'
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    tokenizer = SimpleTokenizer(vocab, max_len=args.max_len)
    
    # Create model
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        model_name=args.model_name,
        pretrained=False,  # Load weights from checkpoint
        freeze_deit=False,
        projection_dim=args.projection_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_idx=tokenizer.pad_idx,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Process input
    if args.image_path:
        # Single image
        image = Image.open(args.image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            generated = model.generate(
                image_tensor,
                start_token=tokenizer.start_idx,
                end_token=tokenizer.end_idx,
                max_len=args.max_len,
            )
        
        caption = tokenizer.decode(generated[0])
        print(f"Generated caption: {caption}")
    
    elif args.images_dir:
        # Directory of images
        results = []
        images_path = Path(args.images_dir)
        
        for image_file in tqdm(list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))):
            image = Image.open(image_file).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                generated = model.generate(
                    image_tensor,
                    start_token=tokenizer.start_idx,
                    end_token=tokenizer.end_idx,
                    max_len=args.max_len,
                )
            
            caption = tokenizer.decode(generated[0])
            results.append({'image': image_file.name, 'caption': caption})
        
        # Save results
        df = pd.DataFrame(results)
        output_path = Path(args.output_dir) / 'generated_captions.csv'
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Autoregressive Image Captioning')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'inference', 'preextract'],
                       required=True, help='Operating mode')
    
    # Model architecture
    parser.add_argument('--model-name', type=str, default='deit_base_patch16_224',
                       help='DeiT model name from timm')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained DeiT weights (PyTorch only)')
    parser.add_argument('--freeze-deit', action='store_true',
                       help='Freeze DeiT backbone parameters')
    parser.add_argument('--projection-dim', type=int, default=512,
                       help='Dimension of linear projection layer')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num-layers', type=int, default=6,
                       help='Number of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--max-len', type=int, default=128,
                       help='Maximum sequence length')
    
    # Feature pre-extraction
    parser.add_argument('--preextract-features', dest='mode', action='store_const',
                       const='preextract', help='Pre-extract and save DeiT features')
    parser.add_argument('--use-preextracted', action='store_true',
                       help='Use pre-extracted features for training')
    parser.add_argument('--features-dir', type=str, default='data/features',
                       help='Directory for pre-extracted features')
    
    # Data paths
    parser.add_argument('--data-dir', type=str, default='data/flickr30k',
                       help='Data directory')
    parser.add_argument('--train-captions', type=str, default='data/flickr30k/train_captions.csv',
                       help='Training captions CSV file')
    parser.add_argument('--val-captions', type=str, default='data/flickr30k/test_captions.csv',
                       help='Validation captions CSV file')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for models and results')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--min-freq', type=int, default=2,
                       help='Minimum word frequency for vocabulary')
    
    # Inference parameters
    parser.add_argument('--checkpoint', type=str,
                       help='Path to model checkpoint for inference')
    parser.add_argument('--image-path', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--images-dir', type=str,
                       help='Directory of images for batch inference')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'preextract':
        pre_extract_features(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        if not args.checkpoint:
            parser.error("--checkpoint is required for inference mode")
        if not args.image_path and not args.images_dir:
            parser.error("Either --image-path or --images-dir is required for inference")
        inference(args)


if __name__ == '__main__':
    main()
