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


class SelectiveStateSpace(nn.Module):
    """
    Selective State Space Model (S6) for bidirectional processing
    """
    def __init__(self, d_model, chunk_size=16):
        super(SelectiveStateSpace, self).__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Selection mechanism
        self.selection_linear = nn.Linear(d_model, d_model)
        self.gate = nn.Sigmoid()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        seq_len, batch, d_model = x.shape
        
        # Split into chunks for selective processing
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[i:i+self.chunk_size]
            chunks.append(chunk)
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Selection mechanism
            selection = self.gate(self.selection_linear(chunk))
            
            # Apply state space transformation
            chunk_t = chunk.permute(1, 0, 2)  # (batch, chunk_len, d_model)
            
            # State space computation (simplified)
            h = torch.zeros(batch, d_model, device=x.device)
            outputs = []
            
            for t in range(chunk_t.shape[1]):
                x_t = chunk_t[:, t, :]  # (batch, d_model)
                sel_t = selection.permute(1, 0, 2)[:, t, :]  # (batch, d_model)
                
                # State update with selection
                h = torch.tanh(x_t @ self.A.T + h @ self.B.T) * sel_t
                y = h @ self.C.T + x_t * self.D
                outputs.append(y)
            
            chunk_output = torch.stack(outputs, dim=1)  # (batch, chunk_len, d_model)
            chunk_output = chunk_output.permute(1, 0, 2)  # (chunk_len, batch, d_model)
            processed_chunks.append(chunk_output)
        
        # Concatenate processed chunks
        output = torch.cat(processed_chunks, dim=0)
        return self.norm(output)


class FABiS6Block(nn.Module):
    """
    Feature Aggregated Bi-S6 Block
    """
    def __init__(self, n_embedding_dim, history_tokens=16):
        super(FABiS6Block, self).__init__()
        self.n_embedding_dim = n_embedding_dim
        self.history_tokens = history_tokens
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        self.tfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding='same')
        self.tfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=3, padding='same')
        self.tfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=4, padding='same')
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        self.cfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding='same')
        self.cfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=4, padding='same')
        self.cfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=8, padding='same')
        
        # Layer normalization
        self.tfa_norm = nn.LayerNorm(n_embedding_dim)
        self.cfa_norm = nn.LayerNorm(n_embedding_dim)
        self.feature_norm = nn.LayerNorm(n_embedding_dim)
        
        # Selective State Space Models for bidirectional processing
        self.s6_forward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        self.s6_backward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        
        # Final layer norm for concatenated bidirectional features
        self.final_norm = nn.LayerNorm(2 * n_embedding_dim)
        
    def forward(self, x):
        """
        x: (seq_len, batch, n_embedding_dim)
        """
        seq_len, batch, n_embedding_dim = x.shape
        
        # Convert to (batch, n_embedding_dim, seq_len) for Conv1D
        x_conv = x.permute(1, 2, 0)
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        tfa1 = self.tfa_conv1(x_conv)
        tfa2 = self.tfa_conv2(x_conv)
        tfa3 = self.tfa_conv3(x_conv)
        
        # Concatenate and sum TFA features
        tfa_features = torch.cat([tfa1, tfa2, tfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        tfa_features = tfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        tfa_features = self.tfa_norm(tfa_features)
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        cfa1 = self.cfa_conv1(x_conv)
        cfa2 = self.cfa_conv2(x_conv)
        cfa3 = self.cfa_conv3(x_conv)
        
        # Concatenate and sum CFA features
        cfa_features = torch.cat([cfa1, cfa2, cfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        cfa_features = cfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        cfa_features = self.cfa_norm(cfa_features)
        
        # Feature Aggregation: Sum TFA and CFA outputs
        aggregated_features = tfa_features + cfa_features
        aggregated_features = self.feature_norm(aggregated_features)
        
        # Bidirectional S6 Processing
        # Forward pass
        forward_output = self.s6_forward(aggregated_features)
        
        # Backward pass (flip sequence)
        backward_input = torch.flip(aggregated_features, dims=[0])
        backward_output = self.s6_backward(backward_input)
        backward_output = torch.flip(backward_output, dims=[0])  # Flip back
        
        # Concatenate forward and backward outputs
        bidirectional_output = torch.cat([forward_output, backward_output], dim=-1)
        bidirectional_output = self.final_norm(bidirectional_output)
        
        return bidirectional_output


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Positional encoding
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   
        
        # Replace Transformer blocks with FA-Bi-S6 blocks
        self.fa_bi_s6_block1 = FABiS6Block(n_embedding_dim, self.history_tokens)
        self.fa_bi_s6_block2 = FABiS6Block(2 * n_embedding_dim, self.history_tokens)  # Input is 2*n_embedding_dim from block1
        
        # History token parameters
        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        
        # Projection layers to handle dimension changes - Fixed dimensions
        self.history_projection = nn.Linear(4 * n_embedding_dim, n_embedding_dim)  # 4*1024 -> 1024
        self.encoded_projection = nn.Linear(n_embedding_dim, 2 * n_embedding_dim)  # 1024 -> 2048
        self.hist_tokens_projection = nn.Linear(2 * n_embedding_dim, n_embedding_dim)  # 2048 -> 1024
        
        # Snippet classification head (adjusted for bidirectional features)
        self.snip_head = nn.Sequential(
            nn.Linear(2 * n_embedding_dim, n_embedding_dim // 2), 
            nn.ReLU()
        )     
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 2, (self.history_tokens * n_embedding_dim // 2) // 4), 
            nn.ReLU(), 
            nn.Linear((self.history_tokens * n_embedding_dim // 2) // 4, n_class)
        )                      
        
        # Layer normalization and dropout
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, encoded_x):
        # Apply positional encoding to history
        hist_pe_x = self.history_positional_encoding(long_x)
        
        # Expand history tokens
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        
        # First FA-Bi-S6 block with history tokens and long context
        # Concatenate history tokens with positional encoded history
        hist_input = torch.cat([history_token, hist_pe_x], dim=0)
        hist_encoded_x_1 = self.fa_bi_s6_block1(hist_input)
        
        # Extract only the history token part for snippet classification
        hist_tokens_encoded = hist_encoded_x_1[:self.history_tokens]  # (history_tokens, batch, 2*n_embedding_dim)
        
        # Project encoded_x to match the dimension for second block
        encoded_x_projected = self.encoded_projection(encoded_x)  # (seq_len, batch, 2*n_embedding_dim)
        
        # Second FA-Bi-S6 block with history tokens and current encoded features
        hist_encoded_x_2 = self.fa_bi_s6_block2(torch.cat([hist_tokens_encoded, encoded_x_projected], dim=0))
        
        # Extract the part corresponding to history tokens
        hist_final = hist_encoded_x_2[:self.history_tokens]  # (history_tokens, batch, 4*n_embedding_dim)
        
        # Project back to original dimension for compatibility - Fixed projection
        hist_encoded_x = self.history_projection(hist_final)  # (history_tokens, batch, n_embedding_dim)
        hist_encoded_x = hist_encoded_x + self.dropout2(self.hist_tokens_projection(hist_tokens_encoded))
        hist_encoded_x = self.norm2(hist_encoded_x)
   
        # Snippet Classification Head
        snippet_feat = self.snip_head(hist_tokens_encoded)  # Use bidirectional features
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0

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

        # Updated History Unit with FA-Bi-S6
        self.history_unit = HistoryUnit(opt)

        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 2))    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]
        
        # Anchor Feature Generator
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x

        # Future-Supervised History Module with FA-Bi-S6
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # History Driven Anchor Refinement
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Prediction Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class = opt["num_of_class"] - 1
        n_seglen = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        # inputs - batch x seq_len x class
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x
    





    #same code copy from colab 
    #same code copy from colab 
    #same code copy from colab 
    #same code copy from colab 
    #same code copy from colab 

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


class SelectiveStateSpace(nn.Module):
    """
    Selective State Space Model (S6) for bidirectional processing
    """
    def __init__(self, d_model, chunk_size=16):
        super(SelectiveStateSpace, self).__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Selection mechanism
        self.selection_linear = nn.Linear(d_model, d_model)
        self.gate = nn.Sigmoid()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        seq_len, batch, d_model = x.shape
        
        # Split into chunks for selective processing
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[i:i+self.chunk_size]
            chunks.append(chunk)
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Selection mechanism
            selection = self.gate(self.selection_linear(chunk))
            
            # Apply state space transformation
            chunk_t = chunk.permute(1, 0, 2)  # (batch, chunk_len, d_model)
            
            # State space computation (simplified)
            h = torch.zeros(batch, d_model, device=x.device)
            outputs = []
            
            for t in range(chunk_t.shape[1]):
                x_t = chunk_t[:, t, :]  # (batch, d_model)
                sel_t = selection.permute(1, 0, 2)[:, t, :]  # (batch, d_model)
                
                # State update with selection
                h = torch.tanh(x_t @ self.A.T + h @ self.B.T) * sel_t
                y = h @ self.C.T + x_t * self.D
                outputs.append(y)
            
            chunk_output = torch.stack(outputs, dim=1)  # (batch, chunk_len, d_model)
            chunk_output = chunk_output.permute(1, 0, 2)  # (chunk_len, batch, d_model)
            processed_chunks.append(chunk_output)
        
        # Concatenate processed chunks
        output = torch.cat(processed_chunks, dim=0)
        return self.norm(output)


class FABiS6Block(nn.Module):
    """
    Feature Aggregated Bi-S6 Block
    """
    def __init__(self, n_embedding_dim, history_tokens=16):
        super(FABiS6Block, self).__init__()
        self.n_embedding_dim = n_embedding_dim
        self.history_tokens = history_tokens
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        self.tfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding='same')
        self.tfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=3, padding='same')
        self.tfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=4, padding='same')
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        self.cfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding='same')
        self.cfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=4, padding='same')
        self.cfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=8, padding='same')
        
        # Layer normalization
        self.tfa_norm = nn.LayerNorm(n_embedding_dim)
        self.cfa_norm = nn.LayerNorm(n_embedding_dim)
        self.feature_norm = nn.LayerNorm(n_embedding_dim)
        
        # Selective State Space Models for bidirectional processing
        self.s6_forward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        self.s6_backward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        
        # Final layer norm for concatenated bidirectional features
        self.final_norm = nn.LayerNorm(2 * n_embedding_dim)
        
    def forward(self, x):
        """
        x: (seq_len, batch, n_embedding_dim)
        """
        seq_len, batch, n_embedding_dim = x.shape
        
        # Convert to (batch, n_embedding_dim, seq_len) for Conv1D
        x_conv = x.permute(1, 2, 0)
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        tfa1 = self.tfa_conv1(x_conv)
        tfa2 = self.tfa_conv2(x_conv)
        tfa3 = self.tfa_conv3(x_conv)
        
        # Concatenate and sum TFA features
        tfa_features = torch.cat([tfa1, tfa2, tfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        tfa_features = tfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        tfa_features = self.tfa_norm(tfa_features)
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        cfa1 = self.cfa_conv1(x_conv)
        cfa2 = self.cfa_conv2(x_conv)
        cfa3 = self.cfa_conv3(x_conv)
        
        # Concatenate and sum CFA features
        cfa_features = torch.cat([cfa1, cfa2, cfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        cfa_features = cfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        cfa_features = self.cfa_norm(cfa_features)
        
        # Feature Aggregation: Sum TFA and CFA outputs
        aggregated_features = tfa_features + cfa_features
        aggregated_features = self.feature_norm(aggregated_features)
        
        # Bidirectional S6 Processing
        # Forward pass
        forward_output = self.s6_forward(aggregated_features)
        
        # Backward pass (flip sequence)
        backward_input = torch.flip(aggregated_features, dims=[0])
        backward_output = self.s6_backward(backward_input)
        backward_output = torch.flip(backward_output, dims=[0])  # Flip back
        
        # Concatenate forward and backward outputs
        bidirectional_output = torch.cat([forward_output, backward_output], dim=-1)
        bidirectional_output = self.final_norm(bidirectional_output)
        
        return bidirectional_output


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Positional encoding
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   
        
        # Replace Transformer blocks with FA-Bi-S6 blocks
        self.fa_bi_s6_block1 = FABiS6Block(n_embedding_dim, self.history_tokens)
        self.fa_bi_s6_block2 = FABiS6Block(2 * n_embedding_dim, self.history_tokens)  # Input is 2*n_embedding_dim from block1
        
        # History token parameters
        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        
        # Projection layers to handle dimension changes - Fixed dimensions
        self.history_projection = nn.Linear(4 * n_embedding_dim, n_embedding_dim)  # 4*1024 -> 1024
        self.encoded_projection = nn.Linear(n_embedding_dim, 2 * n_embedding_dim)  # 1024 -> 2048
        self.hist_tokens_projection = nn.Linear(2 * n_embedding_dim, n_embedding_dim)  # 2048 -> 1024
        
        # Snippet classification head (adjusted for bidirectional features)
        self.snip_head = nn.Sequential(
            nn.Linear(2 * n_embedding_dim, n_embedding_dim // 2), 
            nn.ReLU()
        )     
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 2, (self.history_tokens * n_embedding_dim // 2) // 4), 
            nn.ReLU(), 
            nn.Linear((self.history_tokens * n_embedding_dim // 2) // 4, n_class)
        )                      
        
        # Layer normalization and dropout
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, encoded_x):
        # Apply positional encoding to history
        hist_pe_x = self.history_positional_encoding(long_x)
        
        # Expand history tokens
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        
        # First FA-Bi-S6 block with history tokens and long context
        # Concatenate history tokens with positional encoded history
        hist_input = torch.cat([history_token, hist_pe_x], dim=0)
        hist_encoded_x_1 = self.fa_bi_s6_block1(hist_input)
        
        # Extract only the history token part for snippet classification
        hist_tokens_encoded = hist_encoded_x_1[:self.history_tokens]  # (history_tokens, batch, 2*n_embedding_dim)
        
        # Project encoded_x to match the dimension for second block
        encoded_x_projected = self.encoded_projection(encoded_x)  # (seq_len, batch, 2*n_embedding_dim)
        
        # Second FA-Bi-S6 block with history tokens and current encoded features
        hist_encoded_x_2 = self.fa_bi_s6_block2(torch.cat([hist_tokens_encoded, encoded_x_projected], dim=0))
        
        # Extract the part corresponding to history tokens
        hist_final = hist_encoded_x_2[:self.history_tokens]  # (history_tokens, batch, 4*n_embedding_dim)
        
        # Project back to original dimension for compatibility - Fixed projection
        hist_encoded_x = self.history_projection(hist_final)  # (history_tokens, batch, n_embedding_dim)
        hist_encoded_x = hist_encoded_x + self.dropout2(self.hist_tokens_projection(hist_tokens_encoded))
        hist_encoded_x = self.norm2(hist_encoded_x)
   
        # Snippet Classification Head
        snippet_feat = self.snip_head(hist_tokens_encoded)  # Use bidirectional features
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0

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

        # Updated History Unit with FA-Bi-S6
        self.history_unit = HistoryUnit(opt)

        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 2))    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]
        
        # Anchor Feature Generator
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x

        # Future-Supervised History Module with FA-Bi-S6
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # History Driven Anchor Refinement
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Prediction Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class = opt["num_of_class"] - 1
        n_seglen = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        # inputs - batch x seq_len x class
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x












# Fixed Model Code Waring Same Karnel and Padding 

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


class SelectiveStateSpace(nn.Module):
    """
    Selective State Space Model (S6) for bidirectional processing
    """
    def __init__(self, d_model, chunk_size=16):
        super(SelectiveStateSpace, self).__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        
        # State space parameters
        self.A = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.B = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_model) * 0.1)
        self.D = nn.Parameter(torch.randn(d_model) * 0.1)
        
        # Selection mechanism
        self.selection_linear = nn.Linear(d_model, d_model)
        self.gate = nn.Sigmoid()
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        x: (seq_len, batch, d_model)
        """
        seq_len, batch, d_model = x.shape
        
        # Split into chunks for selective processing
        chunks = []
        for i in range(0, seq_len, self.chunk_size):
            chunk = x[i:i+self.chunk_size]
            chunks.append(chunk)
        
        # Process each chunk
        processed_chunks = []
        for chunk in chunks:
            # Selection mechanism
            selection = self.gate(self.selection_linear(chunk))
            
            # Apply state space transformation
            chunk_t = chunk.permute(1, 0, 2)  # (batch, chunk_len, d_model)
            
            # State space computation (simplified)
            h = torch.zeros(batch, d_model, device=x.device)
            outputs = []
            
            for t in range(chunk_t.shape[1]):
                x_t = chunk_t[:, t, :]  # (batch, d_model)
                sel_t = selection.permute(1, 0, 2)[:, t, :]  # (batch, d_model)
                
                # State update with selection
                h = torch.tanh(x_t @ self.A.T + h @ self.B.T) * sel_t
                y = h @ self.C.T + x_t * self.D
                outputs.append(y)
            
            chunk_output = torch.stack(outputs, dim=1)  # (batch, chunk_len, d_model)
            chunk_output = chunk_output.permute(1, 0, 2)  # (chunk_len, batch, d_model)
            processed_chunks.append(chunk_output)
        
        # Concatenate processed chunks
        output = torch.cat(processed_chunks, dim=0)
        return self.norm(output)


class FABiS6Block(nn.Module):
    """
    Feature Aggregated Bi-S6 Block
    """
    def __init__(self, n_embedding_dim, history_tokens=16):
        super(FABiS6Block, self).__init__()
        self.n_embedding_dim = n_embedding_dim
        self.history_tokens = history_tokens
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        # Changed to explicit padding values to avoid 'same' padding warning
        self.tfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding=1)  # padding=1 for kernel_size=2
        self.tfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=3, padding=1)  # padding=1 for kernel_size=3
        self.tfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=4, padding=2)  # padding=2 for kernel_size=4
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        # Changed to explicit padding values to avoid 'same' padding warning
        self.cfa_conv1 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=2, padding=1)  # padding=1 for kernel_size=2
        self.cfa_conv2 = nn.Conv1d(n_embedding_dim, n_embedding_dim // 3, kernel_size=4, padding=2)  # padding=2 for kernel_size=4
        self.cfa_conv3 = nn.Conv1d(n_embedding_dim, n_embedding_dim - 2 * (n_embedding_dim // 3), kernel_size=8, padding=4)  # padding=4 for kernel_size=8
        
        # Layer normalization
        self.tfa_norm = nn.LayerNorm(n_embedding_dim)
        self.cfa_norm = nn.LayerNorm(n_embedding_dim)
        self.feature_norm = nn.LayerNorm(n_embedding_dim)
        
        # Selective State Space Models for bidirectional processing
        self.s6_forward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        self.s6_backward = SelectiveStateSpace(n_embedding_dim, chunk_size=history_tokens)
        
        # Final layer norm for concatenated bidirectional features
        self.final_norm = nn.LayerNorm(2 * n_embedding_dim)
        
    def _trim_to_original_length(self, x, original_length):
        """Helper function to trim padded convolution output back to original length"""
        if x.size(-1) > original_length:
            # Calculate how much to trim from each side
            excess = x.size(-1) - original_length
            trim_left = excess // 2
            trim_right = excess - trim_left
            if trim_right == 0:
                return x[..., trim_left:]
            else:
                return x[..., trim_left:-trim_right]
        return x
        
    def forward(self, x):
        """
        x: (seq_len, batch, n_embedding_dim)
        """
        seq_len, batch, n_embedding_dim = x.shape
        
        # Convert to (batch, n_embedding_dim, seq_len) for Conv1D
        x_conv = x.permute(1, 2, 0)
        original_seq_len = x_conv.size(-1)
        
        # Temporal Feature Aggregation (TFA-Bi-S6)
        tfa1 = self.tfa_conv1(x_conv)
        tfa1 = self._trim_to_original_length(tfa1, original_seq_len)
        
        tfa2 = self.tfa_conv2(x_conv)
        tfa2 = self._trim_to_original_length(tfa2, original_seq_len)
        
        tfa3 = self.tfa_conv3(x_conv)
        tfa3 = self._trim_to_original_length(tfa3, original_seq_len)
        
        # Concatenate and sum TFA features
        tfa_features = torch.cat([tfa1, tfa2, tfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        tfa_features = tfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        tfa_features = self.tfa_norm(tfa_features)
        
        # Channel Feature Aggregation (CFA-Bi-S6)
        cfa1 = self.cfa_conv1(x_conv)
        cfa1 = self._trim_to_original_length(cfa1, original_seq_len)
        
        cfa2 = self.cfa_conv2(x_conv)
        cfa2 = self._trim_to_original_length(cfa2, original_seq_len)
        
        cfa3 = self.cfa_conv3(x_conv)
        cfa3 = self._trim_to_original_length(cfa3, original_seq_len)
        
        # Concatenate and sum CFA features
        cfa_features = torch.cat([cfa1, cfa2, cfa3], dim=1)  # (batch, n_embedding_dim, seq_len)
        cfa_features = cfa_features.permute(2, 0, 1)  # (seq_len, batch, n_embedding_dim)
        cfa_features = self.cfa_norm(cfa_features)
        
        # Feature Aggregation: Sum TFA and CFA outputs
        aggregated_features = tfa_features + cfa_features
        aggregated_features = self.feature_norm(aggregated_features)
        
        # Bidirectional S6 Processing
        # Forward pass
        forward_output = self.s6_forward(aggregated_features)
        
        # Backward pass (flip sequence)
        backward_input = torch.flip(aggregated_features, dims=[0])
        backward_output = self.s6_backward(backward_input)
        backward_output = torch.flip(backward_output, dims=[0])  # Flip back
        
        # Concatenate forward and backward outputs
        bidirectional_output = torch.cat([forward_output, backward_output], dim=-1)
        bidirectional_output = self.final_norm(bidirectional_output)
        
        return bidirectional_output


class HistoryUnit(torch.nn.Module):
    def __init__(self, opt):
        super(HistoryUnit, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        # Positional encoding
        self.history_positional_encoding = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)   
        
        # Replace Transformer blocks with FA-Bi-S6 blocks
        self.fa_bi_s6_block1 = FABiS6Block(n_embedding_dim, self.history_tokens)
        self.fa_bi_s6_block2 = FABiS6Block(2 * n_embedding_dim, self.history_tokens)  # Input is 2*n_embedding_dim from block1
        
        # History token parameters
        self.history_token = nn.Parameter(torch.zeros(self.history_tokens, 1, n_embedding_dim))
        
        # Projection layers to handle dimension changes - Fixed dimensions
        self.history_projection = nn.Linear(4 * n_embedding_dim, n_embedding_dim)  # 4*1024 -> 1024
        self.encoded_projection = nn.Linear(n_embedding_dim, 2 * n_embedding_dim)  # 1024 -> 2048
        self.hist_tokens_projection = nn.Linear(2 * n_embedding_dim, n_embedding_dim)  # 2048 -> 1024
        
        # Snippet classification head (adjusted for bidirectional features)
        self.snip_head = nn.Sequential(
            nn.Linear(2 * n_embedding_dim, n_embedding_dim // 2), 
            nn.ReLU()
        )     
        self.snip_classifier = nn.Sequential(
            nn.Linear(self.history_tokens * n_embedding_dim // 2, (self.history_tokens * n_embedding_dim // 2) // 4), 
            nn.ReLU(), 
            nn.Linear((self.history_tokens * n_embedding_dim // 2) // 4, n_class)
        )                      
        
        # Layer normalization and dropout
        self.norm2 = nn.LayerNorm(n_embedding_dim)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, long_x, encoded_x):
        # Apply positional encoding to history
        hist_pe_x = self.history_positional_encoding(long_x)
        
        # Expand history tokens
        history_token = self.history_token.expand(-1, hist_pe_x.shape[1], -1)  
        
        # First FA-Bi-S6 block with history tokens and long context
        # Concatenate history tokens with positional encoded history
        hist_input = torch.cat([history_token, hist_pe_x], dim=0)
        hist_encoded_x_1 = self.fa_bi_s6_block1(hist_input)
        
        # Extract only the history token part for snippet classification
        hist_tokens_encoded = hist_encoded_x_1[:self.history_tokens]  # (history_tokens, batch, 2*n_embedding_dim)
        
        # Project encoded_x to match the dimension for second block
        encoded_x_projected = self.encoded_projection(encoded_x)  # (seq_len, batch, 2*n_embedding_dim)
        
        # Second FA-Bi-S6 block with history tokens and current encoded features
        hist_encoded_x_2 = self.fa_bi_s6_block2(torch.cat([hist_tokens_encoded, encoded_x_projected], dim=0))
        
        # Extract the part corresponding to history tokens
        hist_final = hist_encoded_x_2[:self.history_tokens]  # (history_tokens, batch, 4*n_embedding_dim)
        
        # Project back to original dimension for compatibility - Fixed projection
        hist_encoded_x = self.history_projection(hist_final)  # (history_tokens, batch, n_embedding_dim)
        hist_encoded_x = hist_encoded_x + self.dropout2(self.hist_tokens_projection(hist_tokens_encoded))
        hist_encoded_x = self.norm2(hist_encoded_x)
   
        # Snippet Classification Head
        snippet_feat = self.snip_head(hist_tokens_encoded)  # Use bidirectional features
        snippet_feat = torch.flatten(snippet_feat.permute(1, 0, 2), start_dim=1)
        snip_cls = self.snip_classifier(snippet_feat)
        
        return hist_encoded_x, snip_cls


class MYNET(torch.nn.Module):
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature = opt["feat_dim"] 
        n_class = opt["num_of_class"]
        n_embedding_dim = opt["hidden_dim"]
        n_enc_layer = opt["enc_layer"]
        n_enc_head = opt["enc_head"]
        n_dec_layer = opt["dec_layer"]
        n_dec_head = opt["dec_head"]
        n_comb_dec_head = 4
        n_comb_dec_layer = 5
        n_seglen = opt["segment_size"]
        self.anchors = opt["anchors"]
        self.history_tokens = 16
        self.short_window_size = 16
        self.anchors_stride = []
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0

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

        # Updated History Unit with FA-Bi-S6
        self.history_unit = HistoryUnit(opt)

        self.history_anchor_decoder_block1 = nn.TransformerDecoder(
                                            nn.TransformerDecoderLayer(d_model=n_embedding_dim, 
                                                                        nhead=n_comb_dec_head, 
                                                                        dropout=dropout, 
                                                                        activation='gelu'), 
                                            n_comb_dec_layer, 
                                            nn.LayerNorm(n_embedding_dim))  

        self.classifier = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, n_class))
        self.regressor = nn.Sequential(nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(), nn.Linear(n_embedding_dim, 2))    
                           
        self.decoder_token = nn.Parameter(torch.zeros(len(self.anchors), 1, n_embedding_dim))

        self.norm1 = nn.LayerNorm(n_embedding_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.relu = nn.ReLU(True)
        self.softmaxd1 = nn.Softmax(dim=-1)

    def forward(self, inputs):
        base_x_rgb = self.feature_reduction_rgb(inputs[:,:,:self.n_feature//2].float())
        base_x_flow = self.feature_reduction_flow(inputs[:,:,self.n_feature//2:].float())
        base_x = torch.cat([base_x_rgb, base_x_flow], dim=-1)
        
        base_x = base_x.permute([1,0,2])  # seq_len x batch x featsize

        short_x = base_x[-self.short_window_size:]
        long_x = base_x[:-self.short_window_size]
        
        # Anchor Feature Generator
        pe_x = self.positional_encoding(short_x)
        encoded_x = self.encoder(pe_x)   
        decoder_token = self.decoder_token.expand(-1, encoded_x.shape[1], -1)  
        decoded_x = self.decoder(decoder_token, encoded_x) 
        decoded_x = decoded_x

        # Future-Supervised History Module with FA-Bi-S6
        hist_encoded_x, snip_cls = self.history_unit(long_x, encoded_x)

        # History Driven Anchor Refinement
        decoded_anchor_feat = self.history_anchor_decoder_block1(decoded_x, hist_encoded_x)
        decoded_anchor_feat = decoded_anchor_feat + self.dropout1(decoded_x)
        decoded_anchor_feat = self.norm1(decoded_anchor_feat)
        decoded_anchor_feat = decoded_anchor_feat.permute([1, 0, 2])
        
        # Prediction Module
        anc_cls = self.classifier(decoded_anchor_feat)
        anc_reg = self.regressor(decoded_anchor_feat)
        
        return anc_cls, anc_reg, snip_cls

 
class SuppressNet(torch.nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class = opt["num_of_class"] - 1
        n_seglen = opt["segment_size"]
        n_embedding_dim = 2 * n_seglen
        dropout = 0.3
        self.best_loss = 1000000
        self.best_map = 0
        
        self.mlp1 = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2 = nn.Linear(n_embedding_dim, 1)
        self.norm = nn.InstanceNorm1d(n_class)
        self.relu = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs):
        # inputs - batch x seq_len x class
        base_x = inputs.permute([0,2,1])
        base_x = self.norm(base_x)
        x = self.relu(self.mlp1(base_x))
        x = self.sigmoid(self.mlp2(x))
        x = x.squeeze(-1)
        
        return x