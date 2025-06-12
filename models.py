import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize
import torch.utils.checkpoint as checkpoint


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float = 0.1,
                 maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class MemoryEfficientS6Compressor(nn.Module):
    """
    Memory-efficient S6-inspired History Compressor with optimized memory usage.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=1.2, dropout=0.1, chunk_size=32):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state  # Reduced from 64 to 16
        self.d_inner = int(expand * d_model)  # Reduced expansion
        self.d_conv = d_conv
        self.chunk_size = chunk_size  # Smaller chunks
        
        # Efficient projections
        self.input_proj = nn.Linear(d_model, self.d_inner)
        self.gate_proj = nn.Linear(d_model, self.d_inner)
        
        # Fix: Calculate valid groups that divide d_inner evenly
        def find_valid_groups(channels, max_groups=8):
            """Find the largest valid number of groups that divides channels evenly."""
            for groups in range(min(max_groups, channels), 0, -1):
                if channels % groups == 0:
                    return groups
            return 1  # Fallback to 1 if no valid groups found
        
        valid_groups = find_valid_groups(self.d_inner, max_groups=8)
        
        # Lightweight convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=valid_groups
        )
        
        # Efficient selective mechanism
        self.dt_rank = max(4, d_model // 64)  # Much smaller rank
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False)
        
        # Simple state space parameters
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)
        self.register_buffer('A_log', torch.log(A))
        
        # Skip connection
        self.D = nn.Parameter(torch.ones(self.d_inner) * 0.1)
        
        # Simple output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        self._init_parameters()
    
    def _init_parameters(self):
        # Lightweight initialization
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.gate_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.x_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.dt_proj.weight, gain=0.5)
    
    def _process_chunk_efficient(self, x_chunk):
        """Memory-efficient chunk processing without maintaining large states"""
        seq_len, batch_size, _ = x_chunk.shape
        
        if seq_len == 0:
            return torch.zeros(seq_len, batch_size, self.d_inner, device=x_chunk.device, dtype=x_chunk.dtype)
        
        # Process input
        x_inner = self.input_proj(x_chunk)
        z = self.gate_proj(x_chunk)
        
        # Apply convolution
        x_conv = x_inner.transpose(0, 1).transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :seq_len]
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)
        x_conv = self.activation(x_conv)
        
        # Simplified state processing - use local aggregation instead of full state maintenance
        outputs = []
        A = -torch.exp(self.A_log.float())  # (1, d_state)
        
        # Use sliding window approach for memory efficiency
        window_size = min(8, seq_len)  # Small sliding window
        
        for t in range(seq_len):
            x_t = x_conv[t]  # (batch_size, d_inner)
            
            # Get window context
            start_idx = max(0, t - window_size + 1)
            window_context = x_conv[start_idx:t+1]  # (window_len, batch_size, d_inner)
            
            # Simplified selective parameters
            x_proj_t = self.x_proj(x_t)  # (batch_size, dt_rank + d_state)
            dt_t, B_t = torch.split(x_proj_t, [self.dt_rank, self.d_state], dim=-1)
            
            dt_t = F.softplus(self.dt_proj(dt_t)) + 1e-4  # (batch_size, d_inner)
            
            # Efficient local state computation using window context
            # Instead of maintaining full state, use weighted sum of recent inputs
            weights = F.softmax(dt_t.unsqueeze(0) * torch.arange(window_context.size(0), device=x_t.device, dtype=x_t.dtype).unsqueeze(-1).unsqueeze(-1), dim=0)
            local_state = (weights * window_context).sum(dim=0)  # (batch_size, d_inner)
            
            # Output computation
            y_t = local_state + self.D * x_t
            y_t = y_t * torch.sigmoid(z[t])
            outputs.append(y_t)
            
            # Clear intermediate tensors
            del weights, local_state
        
        return torch.stack(outputs, dim=0)
    
    def forward(self, x, state=None):
        seq_len, batch_size, _ = x.shape
        
        if seq_len == 0:
            return torch.zeros(batch_size, self.d_model, device=x.device, dtype=x.dtype), None
        
        # Limit sequence length aggressively
        max_len = 128  # Much more aggressive limit
        if seq_len > max_len:
            # Use only the most recent part
            x = x[-max_len:]
            seq_len = max_len
        
        # Process in very small chunks
        all_outputs = []
        
        for i in range(0, seq_len, self.chunk_size):
            end_idx = min(i + self.chunk_size, seq_len)
            chunk = x[i:end_idx]
            
            # Use gradient checkpointing to save memory
            chunk_outputs = checkpoint.checkpoint(
                self._process_chunk_efficient,
                chunk,
                use_reentrant=False
            )
            all_outputs.append(chunk_outputs)
            
            # Explicit memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Combine outputs
        if all_outputs:
            output_seq = torch.cat(all_outputs, dim=0)
        else:
            output_seq = torch.zeros(seq_len, batch_size, self.d_inner, device=x.device, dtype=x.dtype)
        
        # Simple global compression - just use recent timesteps
        if seq_len > 8:
            compressed_output = output_seq[-8:].mean(dim=0)
        else:
            compressed_output = output_seq.mean(dim=0)
        
        # Final projection
        compressed_output = self.out_proj(compressed_output)
        compressed_output = self.norm(compressed_output)
        compressed_output = self.dropout(compressed_output)
        
        return compressed_output, None  # Don't maintain state


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        self.anchors = opt["anchors"]
        self.history_tokens = 12  # Reduced from 24 to 12
        dropout = 0.25  # Slightly increased for regularization
        
        # Helper function to find valid number of heads
        def find_valid_heads(embed_dim, preferred_heads=6):
            """Find the largest valid number of heads that divides embed_dim evenly."""
            for heads in range(min(preferred_heads, embed_dim), 0, -1):
                if embed_dim % heads == 0:
                    return heads
            return 1  # Fallback to 1 if no valid heads found
        
        valid_heads = find_valid_heads(n_embedding_dim, preferred_heads=6)
        
        # Memory-efficient S6 compressor
        self.s6_history_compressor = MemoryEfficientS6Compressor(
            d_model=n_embedding_dim,
            d_state=16,  # Small state
            d_conv=4,
            expand=1.2,  # Modest expansion
            dropout=dropout,
            chunk_size=24  # Small chunks
        )
        
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=200)   

        # Reduced transformer capacity but still improved from original
        self.history_encoder_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=valid_heads,  # Use valid number of heads
                dim_feedforward=n_embedding_dim * 2,  # Moderate increase
                dropout=dropout, 
                activation='gelu',
                batch_first=False
            ), 
            num_layers=2,  # Keep reasonable
            norm=nn.LayerNorm(n_embedding_dim)
        )  
        
        self.history_encoder_block2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim, 
                nhead=valid_heads,  # Use valid number of heads
                dim_feedforward=n_embedding_dim * 2,
                dropout=dropout, 
                activation='gelu',
                batch_first=False
            ), 
            num_layers=2, 
            norm=nn.LayerNorm(n_embedding_dim)
        )  

        # FIXED: Snippet classification with proper dimension handling
        snippet_feat_dim = n_embedding_dim // 3
        self.snip_head = nn.Sequential(
            nn.Linear(n_embedding_dim, snippet_feat_dim), 
            nn.GELU(),
            nn.Dropout(dropout)
        )     
        
        # FIXED: Use adaptive layer to handle variable input sizes dynamically
        self.snip_adaptive = nn.Linear(self.history_tokens * snippet_feat_dim, n_embedding_dim//2)
        
        # FIXED: Proper classifier that works with adaptive input
        self.snip_classifier = nn.Sequential(
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim//2, n_class)
        )                      

        self.history_token = nn.Parameter(torch.randn(self.history_tokens, 1, n_embedding_dim) * 0.02)
        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Simple adaptive mechanism
        self.adaptive_gate = nn.Linear(n_embedding_dim * 2, n_embedding_dim)

    def forward(self, long_x, encoded_x):
        batch_size = encoded_x.shape[1]
        device = encoded_x.device
        dtype = encoded_x.dtype
        
        # Aggressive sequence length limiting
        if long_x.size(0) > 0:
            max_seq_len = 100  # Very conservative limit
            if long_x.size(0) > max_seq_len:
                # Use only recent history
                long_x = long_x[-max_seq_len:]
            
            try:
                # S6 compression with error handling
                compressed_history, _ = self.s6_history_compressor(long_x, None)
                compressed_history = compressed_history.unsqueeze(0).expand(
                    self.history_tokens, -1, -1
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("Warning: OOM in S6 compressor, using zero history")
                    compressed_history = torch.zeros(
                        self.history_tokens, batch_size, long_x.shape[-1],
                        device=device, dtype=dtype
                    )
                else:
                    raise e
            
            # Traditional processing with conservative limits
            if long_x.size(0) <= 50:  # Very conservative
                try:
                    hist_pe_x = self.history_positional_encoding(long_x)
                    history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
                    
                    # Use gradient checkpointing
                    hist_encoded_x_1 = checkpoint.checkpoint(
                        self.history_encoder_block1,
                        history_token,
                        hist_pe_x,
                        use_reentrant=False
                    )
                    
                    # Simple adaptive fusion
                    gate_input = torch.cat([compressed_history, hist_encoded_x_1], dim=-1)
                    gate = torch.sigmoid(self.adaptive_gate(gate_input))
                    hist_encoded_x_1 = gate * compressed_history + (1 - gate) * hist_encoded_x_1
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("Warning: OOM in history encoder, using compressed only")
                        hist_encoded_x_1 = compressed_history
                    else:
                        raise e
            else:
                hist_encoded_x_1 = compressed_history
        else:
            hist_encoded_x_1 = torch.zeros(
                self.history_tokens, batch_size, encoded_x.shape[-1],
                device=device, dtype=dtype
            )
        
        # Second stage with memory management
        try:
            hist_encoded_x_2 = checkpoint.checkpoint(
                self.history_encoder_block2,
                hist_encoded_x_1,
                encoded_x,
                use_reentrant=False
            )
            hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
            hist_encoded_x = self.norm2(hist_encoded_x_2)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("Warning: OOM in second history encoder")
                hist_encoded_x = self.norm2(hist_encoded_x_1)
            else:
                raise e
   
        # FIXED: Snippet Classification with proper dimension handling
        snippet_feat = self.snip_head(hist_encoded_x_1)  # (history_tokens, batch_size, snippet_feat_dim)
        snippet_feat = snippet_feat.permute(1, 0, 2)  # (batch_size, history_tokens, snippet_feat_dim)
        
        # Flatten
        snippet_feat_flat = snippet_feat.contiguous().view(batch_size, -1)  # (batch_size, history_tokens * snippet_feat_dim)
        
        # FIXED: Apply adaptive layer that handles the exact flattened size
        snippet_adapted = self.snip_adaptive(snippet_feat_flat)  # (batch_size, n_embedding_dim//2)
        
        # FIXED: Apply classifier
        snip_cls = self.snip_classifier(snippet_adapted)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.n_feature = opt["feat_dim"]
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        self.anchors = opt["anchors"]
        self.history_tokens = 12
        self.short_window_size = 16  # Keep reasonable
        dropout = 0.25
        
        # Helper function to find valid number of heads
        def find_valid_heads(embed_dim, preferred_heads):
            """Find the largest valid number of heads that divides embed_dim evenly."""
            for heads in range(min(preferred_heads, embed_dim), 0, -1):
                if embed_dim % heads == 0:
                    return heads
            return 1  # Fallback to 1 if no valid heads found
        
        # Calculate valid heads for encoder and decoder
        valid_enc_heads = find_valid_heads(n_embedding_dim, n_enc_head)
        valid_dec_heads = find_valid_heads(n_embedding_dim, n_dec_head)
        
        # Enhanced but memory-efficient feature reduction
        self.feature_reduction_rgb = nn.Sequential(
            nn.Linear(self.n_feature // 2, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.feature_reduction_flow = nn.Sequential(
            nn.Linear(self.n_feature // 2, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Simple feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        # Moderately enhanced encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=n_embedding_dim,
                nhead=valid_enc_heads,  # Use valid number of heads
                dim_feedforward=n_embedding_dim * 3,  # Moderate increase
                dropout=dropout,
                activation='gelu',
                batch_first=False
            ), 
            num_layers=n_enc_layer,
            norm=nn.LayerNorm(n_embedding_dim)
        )
                                            
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim,
                nhead=valid_dec_heads,  # Use valid number of heads
                dim_feedforward=n_embedding_dim * 3,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            ), 
            num_layers=n_dec_layer,
            norm=nn.LayerNorm(n_embedding_dim)
        )  

        # Memory-efficient history unit
        self.history_unit = HistoryUnit(opt)

        # Enhanced but efficient anchor decoder
        valid_anchor_heads = find_valid_heads(n_embedding_dim, 6)
        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=n_embedding_dim,
                nhead=valid_anchor_heads,  # Use valid number of heads
                dim_feedforward=n_embedding_dim * 3,
                dropout=dropout,
                activation='gelu',
                batch_first=False
            ), 
            num_layers=3,  # Reasonable depth
            norm=nn.LayerNorm(n_embedding_dim)
        )  
            
        # Enhanced classifiers
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim // 2, n_class)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim // 2, 2)
        )    
                           
        self.decoder_token = nn.Parameter(torch.randn(len(self.anchors), 1, n_embedding_dim) * 0.02)

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.best_loss = 1000000
        self.best_map = 0

    def forward(self, inputs):
        # Feature processing with memory management
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        base_x = self.feature_fusion(base_x)
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

        # Conservative sequence splitting
        seq_len = base_x.shape[0]
        window_size = min(self.short_window_size, seq_len)
        
        short_x = base_x[-window_size:]
        long_x = base_x[:-window_size] if seq_len > window_size else torch.empty(0, base_x.shape[1], base_x.shape[2], device=base_x.device)
        
        # Anchor Feature Generator with gradient checkpointing
        pe_x = self.positional_encoding(short_x)
        encoded_x = checkpoint.checkpoint(self.encoder, pe_x, use_reentrant=False)
        
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = checkpoint.checkpoint(self.decoder, decoder_token, encoded_x, use_reentrant=False)

        # Memory-efficient history processing
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # History Driven Anchor Refinement
        decoded_anchor_feat = checkpoint.checkpoint(
            self.history_anchor_decoder_block1,
            decoded_x,
            hist_encoded_x,
            use_reentrant=False
        )
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Prediction Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        n_class = opt["num_of_class"]-1
        n_seglen = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen  # Moderate increase
        dropout = 0.25
        
        self.mlp1 = nn.Sequential(
            nn.Linear(n_seglen, n_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.mlp2 = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_embedding_dim // 2, 1)
        )
        
        self.norm = nn.LayerNorm(n_class)  # Better normalization
        self.sigmoid = nn.Sigmoid()
        
        self.best_loss = 1000000
        self.best_map = 0
        
    def forward(self, inputs):
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.mlp1(base_x)
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
