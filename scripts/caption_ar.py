#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSG_C (Multi-Scale Semantic Guidance) Caption Generator
Tích hợp DeiT -> Encoder -> Decoder với gated fusion
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    import timm
except ImportError:
    timm = None
    warnings.warn("timm not installed. Install with: pip install timm")

try:
    import joblib
except ImportError:
    joblib = None
    warnings.warn("joblib not installed. Install with: pip install joblib")

try:
    from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
    from nltk.translate.meteor_score import meteor_score
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warnings.warn("NLTK not available. Metrics will be limited to BLEU only.")


class MSGC(nn.Module):
    """
    Multi-Scale Semantic Guidance Component
    Trích xuất đặc trưng đa tỷ lệ từ visual features
    """
    def __init__(self, in_dim: int, msgc_dim: int = 256, scales: List[int] = [1, 2, 4], 
                 num_heads: int = 4, use_keyword_head: bool = False):
        super().__init__()
        self.scales = scales
        self.msgc_dim = msgc_dim
        self.use_keyword_head = use_keyword_head
        
        # Multi-scale pooling projections
        self.scale_projections = nn.ModuleList([
            nn.Linear(in_dim, msgc_dim) for _ in scales
        ])
        
        # Lightweight self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=msgc_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Learned pooling to get msgc_vec
        self.pool_query = nn.Parameter(torch.randn(1, 1, msgc_dim))
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=msgc_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Optional keyword head
        if use_keyword_head:
            self.keyword_head = nn.Linear(msgc_dim, msgc_dim)
        
        self.layer_norm = nn.LayerNorm(msgc_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, in_dim) - visual features from DeiT
        Returns:
            msgc_tokens: (batch, num_scales*seq_len, msgc_dim)
            msgc_vec: (batch, msgc_dim)
        """
        batch_size = x.size(0)
        
        # Multi-scale pooling
        scale_features = []
        for i, scale in enumerate(self.scales):
            if scale == 1:
                pooled = x
            else:
                # Average pooling với stride = scale
                pooled = F.avg_pool1d(
                    x.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
            
            # Project to msgc_dim
            projected = self.scale_projections[i](pooled)
            scale_features.append(projected)
        
        # Concatenate all scales
        msgc_tokens = torch.cat(scale_features, dim=1)
        
        # Self-attention
        attn_out, _ = self.self_attention(msgc_tokens, msgc_tokens, msgc_tokens)
        msgc_tokens = self.layer_norm(msgc_tokens + attn_out)
        
        # Learned pooling to get msgc_vec
        pool_query = self.pool_query.expand(batch_size, -1, -1)
        msgc_vec, _ = self.pool_attention(pool_query, msgc_tokens, msgc_tokens)
        msgc_vec = msgc_vec.squeeze(1)  # (batch, msgc_dim)
        
        # Optional keyword extraction
        if self.use_keyword_head:
            msgc_vec = self.keyword_head(msgc_vec)
        
        return msgc_tokens, msgc_vec


class BiLSTMEncoder(nn.Module):
    """BiLSTM Encoder cho visual features"""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim // 2,  # Chia 2 vì bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            (batch, seq_len, hidden_dim)
        """
        output, _ = self.lstm(x)
        return output


class TransformerEncoder(nn.Module):
    """Lightweight Transformer Encoder"""
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


class AttentionDecoder(nn.Module):
    """
    Autoregressive LSTM Decoder với attention và gated fusion
    """
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, msgc_dim: int,
                 num_layers: int = 2, dropout: float = 0.1, 
                 fusion_type: str = 'gated', tie_embeddings: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.msgc_dim = msgc_dim
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        
        # Embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_dropout = nn.Dropout(dropout)
        
        # LSTM input: embedding + attended_context + msgc_vec (gated)
        lstm_input_dim = embed_dim
        if fusion_type == 'concat':
            lstm_input_dim += hidden_dim + msgc_dim
        elif fusion_type == 'gated':
            lstm_input_dim += hidden_dim + msgc_dim
        else:  # cross or default
            lstm_input_dim += hidden_dim
        
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_attention = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)
        
        # Gated fusion for MSG_C
        if fusion_type == 'gated':
            self.msgc_gate = nn.Sequential(
                nn.Linear(hidden_dim + msgc_dim, msgc_dim),
                nn.Sigmoid()
            )
        elif fusion_type == 'cross':
            # Cross attention to msgc_tokens
            self.msgc_cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=4, dropout=dropout, batch_first=True
            )
        
        # Output projection
        output_dim = hidden_dim
        if fusion_type == 'gated' or fusion_type == 'concat':
            output_dim += msgc_dim
            
        self.output_projection = nn.Linear(output_dim, vocab_size)
        
        # Weight tying
        if tie_embeddings:
            if embed_dim == vocab_size:
                self.output_projection.weight = self.embedding.weight
            else:
                warnings.warn("Cannot tie embeddings: embed_dim != vocab_size")
        
    def forward(self, encoder_outputs: torch.Tensor, msgc_vec: torch.Tensor,
                msgc_tokens: Optional[torch.Tensor], target_seq: torch.Tensor,
                hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, Tuple]:
        """
        Args:
            encoder_outputs: (batch, enc_seq_len, hidden_dim)
            msgc_vec: (batch, msgc_dim)
            msgc_tokens: (batch, msgc_seq_len, msgc_dim) - optional for cross attention
            target_seq: (batch, tgt_seq_len)
            hidden: Previous LSTM hidden state
        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
            hidden: Updated LSTM hidden state
        """
        batch_size, tgt_seq_len = target_seq.size()
        
        # Embedding
        embedded = self.embedding(target_seq)  # (batch, tgt_seq_len, embed_dim)
        embedded = self.embed_dropout(embedded)
        
        outputs = []
        
        for t in range(tgt_seq_len):
            # Current token embedding
            emb_t = embedded[:, t:t+1, :]  # (batch, 1, embed_dim)
            
            # Attention over encoder outputs
            if hidden is not None:
                query = hidden[0][-1].unsqueeze(1)  # (batch, 1, hidden_dim)
            else:
                query = torch.zeros(batch_size, 1, self.hidden_dim, device=emb_t.device)
            
            # Compute attention scores
            attn_scores = torch.tanh(
                self.attention(query) + self.context_attention(encoder_outputs)
            )
            attn_weights = F.softmax(self.v(attn_scores), dim=1)
            context = torch.sum(attn_weights * encoder_outputs, dim=1, keepdim=True)
            
            # Gated fusion with MSG_C
            if self.fusion_type == 'gated':
                # Gate to control MSG_C contribution
                msgc_vec_expanded = msgc_vec.unsqueeze(1)  # (batch, 1, msgc_dim)
                gate_input = torch.cat([query, msgc_vec_expanded], dim=-1)
                gate = self.msgc_gate(gate_input)
                gated_msgc = gate * msgc_vec_expanded
                
                # LSTM input
                lstm_input = torch.cat([emb_t, context, gated_msgc], dim=-1)
            elif self.fusion_type == 'concat':
                msgc_vec_expanded = msgc_vec.unsqueeze(1)
                lstm_input = torch.cat([emb_t, context, msgc_vec_expanded], dim=-1)
            elif self.fusion_type == 'cross' and msgc_tokens is not None:
                # Cross attention to msgc_tokens
                msgc_context, _ = self.msgc_cross_attn(query, msgc_tokens, msgc_tokens)
                lstm_input = torch.cat([emb_t, context], dim=-1)
            else:
                lstm_input = torch.cat([emb_t, context], dim=-1)
            
            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            
            # Output projection
            if self.fusion_type in ['gated', 'concat']:
                output_input = torch.cat([lstm_out, gated_msgc if self.fusion_type == 'gated' 
                                         else msgc_vec_expanded], dim=-1)
            else:
                output_input = lstm_out
                
            logits_t = self.output_projection(output_input)
            outputs.append(logits_t)
        
        logits = torch.cat(outputs, dim=1)  # (batch, tgt_seq_len, vocab_size)
        return logits, hidden


class CaptionARModel(nn.Module):
    """
    Full model: DeiT -> Encoder -> MSGC -> Decoder
    """
    def __init__(self, vocab_size: int, proj_dim: int = 512, msgc_dim: int = 256,
                 msgc_scales: List[int] = [1, 2, 4], enc_type: str = 'bilstm',
                 enc_layers: int = 2, dec_layers: int = 2, enc_dropout: float = 0.1,
                 dec_dropout: float = 0.1, emb_dim: int = 512, pretrained: bool = True,
                 freeze_deit: bool = False, use_msgc: bool = True,
                 msgc_fusion: str = 'gated', tie_embeddings: bool = False):
        super().__init__()
        
        self.use_msgc = use_msgc
        self.enc_type = enc_type
        
        # DeiT feature extractor
        if timm is not None:
            self.deit = timm.create_model('deit_base_patch16_224', pretrained=pretrained, num_classes=0)
            deit_dim = self.deit.embed_dim  # 768 for base
            
            if freeze_deit:
                for param in self.deit.parameters():
                    param.requires_grad = False
        else:
            # Fallback for when timm not available
            self.deit = None
            deit_dim = 768
            warnings.warn("timm not available, using dummy DeiT")
        
        # Projection from DeiT to encoder dim
        self.deit_projection = nn.Linear(deit_dim, proj_dim)
        
        # Encoder
        if enc_type == 'bilstm':
            self.encoder = BiLSTMEncoder(proj_dim, proj_dim, enc_layers, enc_dropout)
        elif enc_type == 'transformer':
            self.encoder = TransformerEncoder(proj_dim, num_layers=enc_layers, dropout=enc_dropout)
        elif enc_type == 'none':
            self.encoder = nn.Identity()
        else:
            raise ValueError(f"Unknown encoder type: {enc_type}")
        
        # MSG_C module
        if use_msgc:
            self.msgc = MSGC(proj_dim, msgc_dim, msgc_scales)
        else:
            self.msgc = None
            msgc_dim = 0
        
        # Decoder
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=emb_dim,
            hidden_dim=proj_dim,
            msgc_dim=msgc_dim if use_msgc else 0,
            num_layers=dec_layers,
            dropout=dec_dropout,
            fusion_type=msgc_fusion if use_msgc else 'none',
            tie_embeddings=tie_embeddings
        )
        
    def forward(self, images: torch.Tensor, target_seq: torch.Tensor,
                preextracted_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224) or None if using preextracted
            target_seq: (batch, tgt_seq_len)
            preextracted_features: (batch, num_patches, dim) optional
        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
        """
        # Extract features from DeiT
        if preextracted_features is not None:
            visual_features = preextracted_features
        else:
            if self.deit is not None:
                visual_features = self.deit.forward_features(images)  # (batch, num_patches+1, 768)
                # Remove CLS token
                visual_features = visual_features[:, 1:, :]
            else:
                # Dummy features for testing
                batch_size = images.size(0) if images is not None else target_seq.size(0)
                visual_features = torch.randn(batch_size, 196, 768, device=target_seq.device)
        
        print(f"  Visual features shape: {visual_features.shape}")
        
        # Project to encoder dim
        projected_features = self.deit_projection(visual_features)
        print(f"  Projected features shape: {projected_features.shape}")
        
        # Encoder
        encoder_outputs = self.encoder(projected_features)
        print(f"  Encoder outputs shape: {encoder_outputs.shape}")
        
        # MSG_C
        if self.use_msgc and self.msgc is not None:
            msgc_tokens, msgc_vec = self.msgc(encoder_outputs)
            print(f"  MSGC tokens shape: {msgc_tokens.shape}")
            print(f"  MSGC vec shape: {msgc_vec.shape}")
        else:
            msgc_tokens = None
            msgc_vec = torch.zeros(encoder_outputs.size(0), 0, device=encoder_outputs.device)
        
        # Decoder
        logits, _ = self.decoder(encoder_outputs, msgc_vec, msgc_tokens, target_seq)
        print(f"  Decoder logits shape: {logits.shape}")
        
        return logits
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters in each component"""
        counts = {}
        
        if self.deit is not None:
            counts['deit'] = sum(p.numel() for p in self.deit.parameters())
            counts['deit_trainable'] = sum(p.numel() for p in self.deit.parameters() if p.requires_grad)
        
        counts['deit_projection'] = sum(p.numel() for p in self.deit_projection.parameters())
        
        if self.encoder is not None and not isinstance(self.encoder, nn.Identity):
            counts['encoder'] = sum(p.numel() for p in self.encoder.parameters())
        
        if self.msgc is not None:
            counts['msgc'] = sum(p.numel() for p in self.msgc.parameters())
        
        counts['decoder'] = sum(p.numel() for p in self.decoder.parameters())
        counts['total'] = sum(p.numel() for p in self.parameters())
        counts['total_trainable'] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return counts


def load_vocab(vocab_path: str) -> Dict:
    """
    Robust vocab loader - hỗ trợ vocab.pkl và vocab.json
    """
    vocab_path = Path(vocab_path)
    
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    
    # Try loading vocab.json first
    if vocab_path.suffix == '.json':
        with open(vocab_path, 'r') as f:
            vocab_dict = json.load(f)
        return vocab_dict
    
    # Try joblib for .pkl
    if vocab_path.suffix == '.pkl':
        if joblib is not None:
            try:
                tokenizer = joblib.load(vocab_path)
                # Extract vocab from tokenizer
                if hasattr(tokenizer, 'vocab'):
                    vocab_obj = tokenizer.vocab
                    vocab_dict = {
                        'token2idx': vocab_obj.str_2_idx,
                        'idx2token': vocab_obj.idx_2_str,
                        'vocab_size': len(vocab_obj),
                        'pad_idx': vocab_obj.pad_idx,
                        'sos_idx': vocab_obj.sos_idx,
                        'eos_idx': vocab_obj.eos_idx,
                        'unk_idx': vocab_obj.unk_idx,
                    }
                    return vocab_dict
                else:
                    raise ValueError("Loaded object doesn't have 'vocab' attribute")
            except Exception as e:
                warnings.warn(f"Failed to load with joblib: {e}. Trying pickle...")
        
        # Fallback to pickle with safe unpickler
        try:
            with open(vocab_path, 'rb') as f:
                tokenizer = pickle.load(f)
            if hasattr(tokenizer, 'vocab'):
                vocab_obj = tokenizer.vocab
                vocab_dict = {
                    'token2idx': vocab_obj.str_2_idx,
                    'idx2token': vocab_obj.idx_2_str,
                    'vocab_size': len(vocab_obj),
                    'pad_idx': vocab_obj.pad_idx,
                    'sos_idx': vocab_obj.sos_idx,
                    'eos_idx': vocab_obj.eos_idx,
                    'unk_idx': vocab_obj.unk_idx,
                }
                return vocab_dict
        except Exception as e:
            raise RuntimeError(f"Failed to load vocab with pickle: {e}")
    
    raise ValueError(f"Unsupported vocab file format: {vocab_path.suffix}")


class DummyDataset(Dataset):
    """Dataset for smoke testing with random data"""
    def __init__(self, num_samples: int = 100, vocab_size: int = 5000, 
                 seq_len: int = 12, image_size: int = 224):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.image_size = image_size
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random image
        image = torch.randn(3, self.image_size, self.image_size)
        # Random caption (avoid 0 which is usually pad)
        caption = torch.randint(1, self.vocab_size, (self.seq_len,))
        return image, caption


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device, teacher_forcing_ratio: float = 1.0,
                pad_idx: int = 0) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, (images, captions) in enumerate(dataloader):
        images = images.to(device)
        captions = captions.to(device)
        
        # Teacher forcing: use ground truth as input
        # Input: <sos> w1 w2 ... wn-1
        # Target: w1 w2 ... wn <eos>
        input_seq = captions[:, :-1]
        target_seq = captions[:, 1:]
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(images, input_seq)
        
        # Compute loss
        loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    return total_loss / num_batches


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module,
             device: torch.device, compute_metrics: bool = False,
             idx2token: Optional[Dict] = None) -> Tuple[float, Dict]:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            
            input_seq = captions[:, :-1]
            target_seq = captions[:, 1:]
            
            logits = model(images, input_seq)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_seq.reshape(-1))
            
            total_loss += loss.item()
            num_batches += 1
            
            if compute_metrics:
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                all_predictions.extend(preds.cpu().numpy())
                all_references.extend(target_seq.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    
    metrics = {'loss': avg_loss}
    
    # Compute BLEU and METEOR if requested
    if compute_metrics and NLTK_AVAILABLE and idx2token is not None:
        # Convert to tokens
        pred_sentences = []
        ref_sentences = []
        
        for pred, ref in zip(all_predictions, all_references):
            pred_tokens = [idx2token.get(str(idx), '<unk>') for idx in pred if idx != 0]
            ref_tokens = [idx2token.get(str(idx), '<unk>') for idx in ref if idx != 0]
            
            pred_sentences.append(pred_tokens)
            ref_sentences.append([ref_tokens])  # BLEU expects list of references
        
        # BLEU score
        try:
            bleu_score = corpus_bleu(ref_sentences, pred_sentences)
            metrics['bleu'] = bleu_score
        except:
            metrics['bleu'] = 0.0
        
        # METEOR score (averaged over sentences)
        try:
            meteor_scores = []
            for pred, ref in zip(pred_sentences, ref_sentences):
                score = meteor_score([ref[0]], pred)
                meteor_scores.append(score)
            metrics['meteor'] = np.mean(meteor_scores)
        except:
            metrics['meteor'] = 0.0
    elif compute_metrics and not NLTK_AVAILABLE:
        warnings.warn("NLTK not available. Skipping BLEU/METEOR computation.")
    
    return avg_loss, metrics


def preextract_features(model: nn.Module, dataloader: DataLoader, 
                       device: torch.device, output_dir: str):
    """Pre-extract DeiT features to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            
            # Extract features
            if model.deit is not None:
                visual_features = model.deit.forward_features(images)
                visual_features = visual_features[:, 1:, :]  # Remove CLS token
            else:
                visual_features = torch.randn(images.size(0), 196, 768, device=device)
            
            # Save to disk
            for i, (feat, cap) in enumerate(zip(visual_features, captions)):
                sample_id = batch_idx * len(captions) + i
                torch.save({
                    'features': feat.cpu(),
                    'caption': cap.cpu()
                }, output_path / f'sample_{sample_id}.pt')
            
            if batch_idx % 10 == 0:
                print(f"  Extracted {batch_idx * len(captions)} samples")
    
    print(f"Feature extraction complete. Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='MSG_C Caption Training with Autoregressive Decoder')
    
    # Model architecture
    parser.add_argument('--proj-dim', type=int, default=512, help='Projection dimension')
    parser.add_argument('--msgc-dim', type=int, default=256, help='MSG_C dimension')
    parser.add_argument('--msgc-scales', type=int, nargs='+', default=[1, 2, 4], help='MSG_C scales')
    parser.add_argument('--enc-type', choices=['bilstm', 'transformer', 'none'], default='bilstm')
    parser.add_argument('--enc-layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dec-layers', type=int, default=2, help='Number of decoder layers')
    parser.add_argument('--enc-dropout', type=float, default=0.1, help='Encoder dropout')
    parser.add_argument('--dec-dropout', type=float, default=0.1, help='Decoder dropout')
    parser.add_argument('--emb-dim', type=int, default=512, help='Embedding dimension')
    
    # Training
    parser.add_argument('--vocab-path', type=str, default='data/vocab.pkl', help='Path to vocab file')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--tgt-seq-len', type=int, default=12, help='Target sequence length')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--teacher-forcing-ratio', type=float, default=1.0, help='Teacher forcing ratio')
    
    # DeiT options
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained DeiT')
    parser.add_argument('--freeze-deit', action='store_true', help='Freeze DeiT weights')
    
    # MSG_C options
    parser.add_argument('--use-msgc', action='store_true', default=True, help='Use MSG_C module')
    parser.add_argument('--msgc-fusion', choices=['gated', 'concat', 'cross'], default='gated',
                       help='MSG_C fusion type')
    
    # Other options
    parser.add_argument('--tie-embeddings', action='store_true', help='Tie input/output embeddings')
    parser.add_argument('--compute-metrics', action='store_true', help='Compute BLEU/METEOR')
    
    # Feature extraction
    parser.add_argument('--preextract-features', type=str, default=None,
                       help='Directory to save pre-extracted features')
    parser.add_argument('--use-preextracted', type=str, default=None,
                       help='Directory with pre-extracted features')
    
    # Mode
    parser.add_argument('--inference', action='store_true', help='Run inference mode')
    
    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocab_path}...")
    try:
        vocab_dict = load_vocab(args.vocab_path)
        vocab_size = vocab_dict['vocab_size']
        pad_idx = vocab_dict['pad_idx']
        idx2token = vocab_dict.get('idx2token', {})
        print(f"  Vocab size: {vocab_size}")
        print(f"  PAD idx: {pad_idx}")
    except Exception as e:
        print(f"  Warning: Could not load vocab ({e}). Using dummy vocab for smoke test.")
        vocab_size = 5000
        pad_idx = 0
        idx2token = {str(i): f'word_{i}' for i in range(vocab_size)}
    
    # Create model
    print("\nCreating model...")
    model = CaptionARModel(
        vocab_size=vocab_size,
        proj_dim=args.proj_dim,
        msgc_dim=args.msgc_dim,
        msgc_scales=args.msgc_scales,
        enc_type=args.enc_type,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        enc_dropout=args.enc_dropout,
        dec_dropout=args.dec_dropout,
        emb_dim=args.emb_dim,
        pretrained=args.pretrained,
        freeze_deit=args.freeze_deit,
        use_msgc=args.use_msgc,
        msgc_fusion=args.msgc_fusion,
        tie_embeddings=args.tie_embeddings
    )
    model = model.to(device)
    
    # Print parameter counts
    print("\nParameter counts:")
    param_counts = model.count_parameters()
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Create dummy dataset for smoke test
    print("\nCreating dataset...")
    train_dataset = DummyDataset(num_samples=100, vocab_size=vocab_size, seq_len=args.tgt_seq_len)
    val_dataset = DummyDataset(num_samples=20, vocab_size=vocab_size, seq_len=args.tgt_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Pre-extract features if requested
    if args.preextract_features:
        print(f"\nPre-extracting features to {args.preextract_features}...")
        preextract_features(model, train_loader, device, args.preextract_features)
        print("Feature extraction complete. Exiting.")
        return
    
    # Inference mode
    if args.inference:
        print("\nRunning inference demo...")
        model.eval()
        with torch.no_grad():
            # Get a sample batch
            images, captions = next(iter(val_loader))
            images = images.to(device)
            captions = captions.to(device)
            
            print("\nForward pass shapes:")
            input_seq = captions[:, :-1]
            logits = model(images, input_seq)
            print(f"  Input images: {images.shape}")
            print(f"  Input sequence: {input_seq.shape}")
            print(f"  Output logits: {logits.shape}")
            
            # Greedy decoding
            preds = torch.argmax(logits, dim=-1)
            print(f"  Predictions: {preds.shape}")
            print(f"\nSample prediction (first sequence):")
            print(f"  {preds[0].cpu().numpy()}")
        
        print("\nInference demo complete.")
        return
    
    # Training mode
    print("\nStarting training smoke test...")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_loss = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        print("\nTraining...")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device,
                                args.teacher_forcing_ratio, pad_idx)
        print(f"  Average train loss: {train_loss:.4f}")
        
        # Evaluate
        print("\nValidating...")
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device,
                                        args.compute_metrics, idx2token)
        print(f"  Validation loss: {val_loss:.4f}")
        if args.compute_metrics:
            for metric_name, metric_value in val_metrics.items():
                if metric_name != 'loss':
                    print(f"  {metric_name.upper()}: {metric_value:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, checkpoint_path)
            print(f"  Saved best model to {checkpoint_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
