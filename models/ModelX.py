import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layers.RevIN import RevIN



# -----------------------------
# Haar Wavelet Decomposition
# -----------------------------
class HaarWaveletDecomp:
    def __init__(self):
       pass

    def decompose(self, x):
        """
        x: [B, C, L]
        Returns:
            approx: [B, C, L/2]
            detail: [B, C, L/2]
        """
        assert x.shape[-1] % 2 == 0, "Input length must be even for Haar decomposition."

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        # Haar filters
        approx = (x_even + x_odd) / math.sqrt(2)
        detail = (x_even - x_odd) / math.sqrt(2)

        return approx, detail

    def reconstruct(self, approx, detail):
        """
        approx, detail: [B, C, L/2]
        Returns:
            x_recon: [B, C, L]
        """
        assert approx.shape == detail.shape, "Approximation and detail must have the same shape for reconstruction."

        x_even = (approx + detail) / math.sqrt(2)
        x_odd  = (approx - detail) / math.sqrt(2)

        # Interleave
        B, C, L_half = approx.shape
        L = L_half * 2

        x = torch.zeros(B, C, L, device=approx.device, dtype=approx.dtype)
        x[..., 0::2] = x_even
        x[..., 1::2] = x_odd

        return x



# -----------------------------
# Harmonic Decomposition
# -----------------------------
class HarmonicDecomp(nn.Module):
    def __init__(self, modes):
        super(HarmonicDecomp, self).__init__()
        self.modes = modes

    def forward(self, x):
        x_fft = torch.fft.rfft(x, norm='ortho', dim=-1)
        x_fft[:,:, self.modes:] = 0
        x_mode = torch.fft.irfft(x_fft, norm='ortho', dim=-1, n=x.shape[-1])
        return x_mode, x - x_mode

# -----------------------------
# Patch Embedding
# -----------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, patch_dim):
        super(PatchEmbedding, self).__init__()

        self.patch_size = patch_size
        self.patch_dim = patch_dim

        self.embed = nn.Linear(patch_size, patch_dim)


    def forward(self, x):
        '''
        input x: [B, C, L]
        output: [B, C, N * d_P]
        '''
        # x: [B, C, L]
        batch, channels, length = x.shape
        x = x.contiguous().view(batch, channels, -1, self.patch_size)  # [B, C, N, P]
        out = self.embed(x)  # [B, C, N, d_P
        out = out.contiguous().view(batch, channels, -1)  # [B, C, N * d_P]

        return out  # [B, c, N * d_P]


# -----------------------------
# Convolution Attention
# -----------------------------
class ConvolutionalAttention(nn.Module):
    def __init__(self, channel_dim, kernel_size,
                dropout):
        super(ConvolutionalAttention, self).__init__()

        self.channel_dim = channel_dim

        # Depthwise Conv → Q,K,V (shared)
        self.DWConv_qk = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=2 * channel_dim,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            groups=channel_dim,
            bias=False
        )

        self.DWConv_v = nn.Conv1d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
            stride=1,
            padding="same",
            groups=channel_dim,
            bias=False
        )

        self.dropout = float(dropout)

    def forward(self, x):
        '''
        input x: [B, d_c, d_L]
        output: [B, d_c, d_L]
        '''
        # -------------------------
        # Stage 1: Q, K, V generation.
        # -------------------------

        x_conv = self.DWConv_qk(x) # [B, 2 * d_c, N * d_P]
        Q, K = x_conv[:, 0::2, :], x_conv[:, 1::2, :] # [B, d_c, N * d_P] each

        V = self.DWConv_v(x) # [B, d_c, N * d_P]

        # =========================
        # Stage 2: Channel-wise Attention
        # =========================
        out = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout
        ) # [B, d_c, N * d_P]

        return out # [B, d_c, N * d_P]

    

# -----------------------------
# MultiHead Convolution Attention Block
# -----------------------------  
class MHCABlock(nn.Module):
    def __init__(self, channel_dim,N,patch_dim, attn_kernel_size, dropout=0.1):
        super(MHCABlock, self).__init__()


        self.attention = ConvolutionalAttention(
                channel_dim=channel_dim,
                kernel_size=attn_kernel_size,
                dropout=dropout
            )
        
        self.pre_norm1 = nn.GroupNorm(
            num_groups=channel_dim, 
            num_channels=channel_dim
            )

        self.pre_norm2 = nn.GroupNorm(
            num_groups=channel_dim, 
            num_channels=channel_dim
            )
        
        self.mlp = PatchLinear(
            in_patch_size=patch_dim,
            out_patch_size=patch_dim,
            in_N=N,
            out_N=N)
        
        #self.alpha = nn.Parameter(torch.ones(1, channel_dim, 1))  # Learnable scaling factor for attention output
        #self.beta = nn.Parameter(torch.ones(1, channel_dim, 1))   # Learnable scaling factor for MLP output
    
    @torch.compile
    def forward(self, x):
        '''input x: [B, d_c, N * d_P]
           output: [B, d_c, N * d_P]
        '''
        # Pre-norm before attention.
        x = self.pre_norm1(x)

        # Convolutional Attention.
        out = self.attention(x)  # [B, d_c, N * d_P]

        # Residual connection.
        out = x + out

        # Pre-norm before MLP.
        out = self.pre_norm2(out)

        # MLP on the input sequence dimension.
        out1 = self.mlp(out)
        # out1 = F.gelu(out1)

        out = out + out1

        return out


# -----------------------------
# Main Model
# -----------------------------
class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.d_seq_len
        self.pred_len = configs.d_pred_len
        self.in_features = configs.d_in_features
        self.out_features = configs.d_out_features

        self.patch_size = configs.m_patch_size
        self.patch_dim = configs.m_patch_dim
        
       
        assert self.seq_len % self.patch_size == 0, "Sequence length should be divisible by patch size to create non-overlapping patches."
        assert self.pred_len % self.patch_size == 0, "Prediction length should be divisible by patch size for the patch-based prediction."
        self.N_L = self.seq_len // (2 * self.patch_size)
        self.N_H = self.pred_len // (2 * self.patch_size)

        # RevIN
        self.revin = RevIN(
            self.in_features,
            affine=configs.m_revin_affine,
            subtract_last=configs.m_revin_subtract_last
        )

        # Haar Wavelet Decomposition
        self.decomp = HaarWaveletDecomp()


        self.embed = PatchEmbedding(
            patch_size=self.patch_size,
            patch_dim=self.patch_dim
        )

        self.mhca_blocks = nn.Sequential(*[
            MHCABlock(
                channel_dim=2* self.in_features,
                N = self.N_L,
                patch_dim=self.patch_dim,
                attn_kernel_size=configs.m_attn_head_kernel_size,
                dropout=configs.m_attn_head_dropout,
                
            )
            for _ in range(configs.m_n_layers)
        ])

        # Base prediction from L to H.
        self.approx_pred = PatchLinear(
            in_patch_size=self.patch_dim,
            out_patch_size=self.patch_size,
            in_N=self.N_L,
            out_N=self.N_H
        )

        self.detail_pred = PatchLinear(
            in_patch_size=self.patch_dim,
            out_patch_size=self.patch_size,
            in_N=self.N_L,
            out_N=self.N_H
        )

    
    def forward(self, x):
        # x: [B, L, C]
        b, l, c = x.shape

        # RevIN normalization
        x = self.revin(x, mode='norm') # [B, L, C]
        x = x.permute(0, 2, 1) # [B, C, L]
        # Decomposition
        approx, detail = self.decomp.decompose(x)  # [B, C, L//2]

        x = torch.cat([approx, detail], dim=1)  # [B, 2C, L//2]

        # Embedding
        embed= self.embed(x)  # [B, d_c, N * d_P]
        

        # Multi-Head Convolutional Attention Blocks
        # for block in self.mhca_blocks:
        #     embed = block(embed)  # [B, d_c, N * d_P]

        embed = self.mhca_blocks(embed)  # [B, d_c, N * d_P]

        approx_out, detail_out = torch.split(embed, c, dim=1)
        approx_out = self.approx_pred(approx_out)  # [B, c, N_H * d_P]
        detail_out = self.detail_pred(detail_out)  # [B, c, N_H * d_P]

        out = self.decomp.reconstruct(approx_out, detail_out)  # [B, d_c, pred_len]

        # RevIN denormalization
        out = self.revin(out.permute(0, 2, 1), mode='denorm')

        return out
    
class PatchLinear(nn.Module):
    def __init__(self, in_patch_size, out_patch_size, in_N, out_N):
        super(PatchLinear, self).__init__()
        self.in_patch_size = in_patch_size
        self.out_patch_size = out_patch_size
        self.in_N = in_N
        self.out_N = out_N
        self.patch_proj = nn.Linear(in_patch_size, out_patch_size)
        self.len_proj = nn.Linear(in_N, out_N)

    def forward(self, x):
        '''
        input x: [B, C, N * d_P]
        output: [B, C, N * d_P]
        '''
        batch, channels, length = x.shape
        x = x.contiguous().view(batch, channels, self.in_N, self.in_patch_size)
        x = self.patch_proj(x)  # [B, C, N, out_P]
        x = F.gelu(x)
        x = x.permute(0, 1, 3, 2)
        x = self.len_proj(x)  # [B, C, out_P, out_N]
        out = x.contiguous().view(batch, channels, self.out_N * self.out_patch_size)  # [B, C, out_N * out_P]
        return out
