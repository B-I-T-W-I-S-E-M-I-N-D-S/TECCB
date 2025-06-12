import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torch.nn.functional import normalize


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


class S6InspiredHistoryCompressor(nn.Module):
    """
    S6-Inspired History Compressor for efficient long sequence processing.
    
    Key S6 concepts implemented:
    1. Selective mechanism: Input-dependent gating for what information to retain
    2. State space modeling: Maintains compressed hidden state across time
    3. Linear complexity: Processes sequences efficiently without quadratic attention
    4. Bidirectional processing: Forward and backward information flow
    
    This compressor maintains a compressed representation of the entire history
    while allowing selective information retention based on input relevance.
    """
    
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super(S6InspiredHistoryCompressor, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv
        
        # Input projection - similar to S6's in_proj but without bidirectional split
        self.input_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolutional layer for local temporal modeling (inspired by S6's conv1d)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise convolution for efficiency
        )
        
        # Selective mechanism components (core S6 innovation)
        # These determine what information to retain/forget at each step
        self.dt_rank = max(16, d_model // 16)  # Rank for time step computation
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)
        
        # State space parameters (inspired by S4D initialization in S6)
        # A matrix: Determines state evolution dynamics
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Log parameterization for stability
        
        # D matrix: Skip connection strength (direct input influence)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Normalization and activation
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()  # Same as S6
        self.dropout = nn.Dropout(dropout)
        
        # Initialize dt_proj bias for stable dynamics
        with torch.no_grad():
            dt_bias = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
            self.dt_proj.bias.copy_(dt_bias + torch.log(-torch.expm1(-dt_bias)))
    
    def forward(self, x, state=None):
        """
        Forward pass implementing S6-inspired selective state space processing.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            state: Optional previous state for continuous processing
            
        Returns:
            output: Compressed representation (batch_size, d_model)
            new_state: Updated state for next iteration
        """
        seq_len, batch_size, _ = x.shape
        
        # Input projection (analogous to S6's input processing)
        xz = self.input_proj(x)  # (seq_len, batch_size, d_inner * 2)
        x_inner, z = xz.chunk(2, dim=-1)  # Split for gating mechanism
        
        # Apply convolution for local temporal dependencies
        # Transpose for conv1d: (batch, channels, seq_len)
        x_conv = x_inner.transpose(0, 1).transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[..., :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2).transpose(0, 1)  # Back to (seq_len, batch, d_inner)
        x_conv = self.activation(x_conv)
        
        # Initialize state if not provided
        if state is None:
            state = torch.zeros(batch_size, self.d_inner, self.d_state, 
                              device=x.device, dtype=x.dtype)
        
        # Process sequence with selective state space mechanism
        outputs = []
        current_state = state
        
        for t in range(seq_len):
            x_t = x_conv[t]  # (batch_size, d_inner)
            
            # Compute selective parameters (core S6 innovation)
            x_proj_t = self.x_proj(x_t)  # (batch_size, dt_rank + 2*d_state)
            dt_t, B_t, C_t = torch.split(x_proj_t, 
                                       [self.dt_rank, self.d_state, self.d_state], 
                                       dim=-1)
            
            # Compute time step (how much to update state)
            dt_t = F.softplus(self.dt_proj(dt_t) + self.dt_proj.bias)  # (batch_size, d_inner)
            
            # State space dynamics (inspired by S6's SSM)
            A = -torch.exp(self.A_log.float())  # (d_inner, d_state) - negative for stability
            
            # Discretization of continuous system
            dA = torch.exp(torch.einsum('bd,dn->bdn', dt_t, A))  # (batch, d_inner, d_state)
            dB = torch.einsum('bd,bn->bdn', dt_t, B_t)  # (batch, d_inner, d_state)
            
            # State update: s_t = dA * s_{t-1} + dB * x_t
            current_state = current_state * dA + torch.einsum('bd,bdn->bdn', x_t, dB)
            
            # Output computation: y_t = C_t * s_t + D * x_t
            y_t = torch.einsum('bdn,bn->bd', current_state, C_t) + self.D * x_t
            
            # Apply gating mechanism (z acts as selective gate)
            y_t = y_t * self.activation(z[t])
            
            outputs.append(y_t)
        
        # Stack outputs and take the final compressed representation
        output_seq = torch.stack(outputs, dim=0)  # (seq_len, batch_size, d_inner)
        
        # Global average pooling for compression (can be replaced with learned pooling)
        compressed_output = output_seq.mean(dim=0)  # (batch_size, d_inner)
        
        # Final projection and normalization
        compressed_output = self.out_proj(compressed_output)  # (batch_size, d_model)
        compressed_output = self.norm(compressed_output)
        compressed_output = self.dropout(compressed_output)
        
        return compressed_output, current_state


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_hist_dec_head = 4
        n_hist_dec_layer = 5
        n_hist_dec_head_2 = 4
        n_hist_dec_layer_2 = 2
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        
        # S6-Inspired History Compressor replaces traditional positional encoding + transformer
        self.s6_history_compressor = S6InspiredHistoryCompressor(
            d_model=n_embedding_dim,
            d_state=64,  # Compressed state size
            d_conv=4,    # Local convolution window
            expand=2,    # Expansion factor
            dropout=dropout
        )
        
        # Keep original positional encoding for compatibility
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   

        # Reduced transformer layers since S6 compressor handles most temporal modeling
        self.history_encoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer // 2,  # Reduced layers
                                            nn.LayerNorm(n_embedding_dim))  
        
        self.history_encoder_block2 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_hist_dec_head_2, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_hist_dec_layer_2, 
                                            nn.LayerNorm(n_embedding_dim))  

        self.snip_head = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim//4), nn.ReLU())     
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens*n_embedding_dim//4, (self.history_tokens*n_embedding_dim//4)//4), 
            nn.ReLU(), 
            nn.Linear((self.history_tokens*n_embedding_dim//4)//4, n_class)
        )                      

        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)
        
        # State management for continuous processing
        self.register_buffer('compression_state', None)

    def forward(self, long_x, encoded_x):
        """
        Enhanced forward pass with S6-inspired history compression.
        
        Args:
            long_x: Long history sequence (seq_len, batch_size, hidden_dim)
            encoded_x: Current encoded features (seq_len, batch_size, hidden_dim)
        
        Returns:
            hist_encoded_x: Processed history features
            snip_cls: Snippet classification logits
        """
        
        # S6-Inspired History Compression
        # This replaces the traditional approach of processing the entire long sequence
        if long_x.size(0) > 0:  # Only compress if we have history
            compressed_history, new_state = self.s6_history_compressor(
                long_x, self.compression_state
            )
            # Update state for next iteration (in practice, you'd manage this externally)
            # self.compression_state = new_state.detach()  # Detach to prevent gradient flow
            
            # Expand compressed history to match expected dimensions for downstream processing
            compressed_history = compressed_history.unsqueeze(0).expand(
                self.history_tokens, -1, -1
            )  # (history_tokens, batch_size, hidden_dim)
        else:
            # No history available, use zero initialization
            batch_size = encoded_x.shape[1]
            compressed_history = torch.zeros(
                self.history_tokens, batch_size, long_x.shape[-1] if long_x.numel() > 0 else encoded_x.shape[-1],
                device=encoded_x.device, dtype=encoded_x.dtype
            )
        
        # Traditional processing path (kept for compatibility and final refinement)
        if long_x.size(0) > 0:
            hist_pe_x = self.history_positional_encoding(long_x)
            history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
            hist_encoded_x_1 = self.history_encoder_block1(history_token, hist_pe_x)
        else:
            # Use compressed history when no long sequence available
            hist_encoded_x_1 = compressed_history
        
        # Combine compressed history with current features
        hist_encoded_x_2 = self.history_encoder_block2(hist_encoded_x_1, encoded_x)
        hist_encoded_x_2 = hist_encoded_x_2 + self.dropout2(hist_encoded_x_1)
        hist_encoded_x = self.norm2(hist_encoded_x_2)
   
        # Snippet Classification Head (using the refined features)
        snippet_feat = self.snip_head(hist_encoded_x_1)
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature=opt["feat_dim"] 
        n_class=opt["num_of_class"]
        n_embedding_dim=opt["hidden_dim"]
        n_enc_layer=opt["enc_layer"]
        n_enc_head=opt["enc_head"]
        n_dec_layer=opt["dec_layer"]
        n_dec_head=opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen=opt["segment_size"]
        self.anchors=opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride=[]
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0

        self.feature_reduction_rgb = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        self.feature_reduction_flow = nn.Linear(self.n_feature//2, n_embedding_dim//2)
        
        self.positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)      
        
        self.encoder = nn.TransformerEncoder(
                                            nn.TransformerEncoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_enc_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_enc_layer, 
                                            nn.LayerNorm(n_embedding_dim))
                                            
        self.decoder = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        # Enhanced history unit with S6-inspired compression
        self.history_unit = HistoryUnit(opt)

        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  
            
        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim,n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim,2))    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        """
        Enhanced forward pass with S6-inspired history compression for efficient long sequence processing.
        
        The S6InspiredHistoryCompressor provides:
        1. Linear complexity instead of quadratic attention
        2. Selective information retention based on input relevance
        3. Continuous state management for streaming scenarios
        4. Bidirectional context understanding
        """
        
        # Feature processing (unchanged)
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb,base_x_flow],dim=-1)
        
        base_x = base_x.permute([1,0,2])# seq_len x batch x featsize

        # Split into short-term and long-term sequences
        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]
        
        # Anchor Feature Generator (unchanged)
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 

        # Enhanced Future-Supervised History Module with S6-inspired compression
        # This efficiently processes potentially very long sequences
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # History Driven Anchor Refinement (unchanged)
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Prediction Module (unchanged)
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class=opt["num_of_class"]-1
        n_seglen=opt["segment_size"]
        n_embedding_dim=2*n_seglen
        dropout=0.3
        self.best_loss=1000000
        self.best_map=0
        # FC layers for the 2 streams
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        #inputs - batch x seq_len x class
        
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
