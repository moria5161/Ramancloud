import numpy as np
from scipy.signal import resample
import torch
import torch.nn as nn
from numpy import linalg as la

# =====================================================================================
# 1. 模型定义
# =====================================================================================
def sincos_pos_embed(embed_dim, num_patches):
    pos = np.arange(num_patches, dtype=np.float64).reshape(-1, 1)
    dim_t = np.arange(embed_dim // 2, dtype=np.float64)
    dim_t = 1. / (10000.**(dim_t / (embed_dim // 2)))
    pos_times_dim = pos * dim_t
    emb = np.concatenate((np.sin(pos_times_dim), np.cos(pos_times_dim)), axis=1)
    if embed_dim % 2 != 0:
        emb = np.concatenate((emb, np.zeros((num_patches, 1))), axis=1)
    return torch.from_numpy(emb).float()

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads)**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(x.shape)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SpecTransformer(nn.Module):
    def __init__(self, spectrum_length, patch_size, embed_dim, depth, heads):
        super().__init__()
        if spectrum_length % patch_size != 0:
            raise ValueError("光谱长度必须能被patch大小整除")
        self.num_patches = spectrum_length // patch_size
        self.patch_embed = PatchEmbedding(patch_size, embed_dim)
        self.pos_embed = nn.Parameter(sincos_pos_embed(embed_dim, self.num_patches), requires_grad=False)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(embed_dim, heads) for _ in range(depth)])
        self.decoder = nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 2),
                            nn.GELU(),
                            nn.Linear(embed_dim * 2, patch_size)
                                    )
        self.patch_size = patch_size

    def unpatchify(self, patches):
        B, n_patches, p = patches.shape
        C = 1
        L = n_patches * p
        return patches.reshape(B, n_patches, C, p).permute(0, 2, 1, 3).reshape(B, C, L)

    def forward(self, x):
        patches = self.patch_embed(x) + self.pos_embed
        for blk in self.transformer_blocks:
            patches = blk(patches)
        pred_patches = self.decoder(patches)
        return self.unpatchify(pred_patches)

# =====================================================================================
# 2. 辅助函数 (Utilities)
# =====================================================================================
def spectra_correction(raw_spec, denoised_spec):
    A = np.vstack([denoised_spec, np.ones(len(denoised_spec))]).T
    a, b = la.lstsq(A, raw_spec, rcond=None)[0]
    corrected_spec = a * denoised_spec + b
    mean_shift = np.mean(raw_spec) - np.mean(corrected_spec)
    return corrected_spec + mean_shift


# =====================================================================================
# 3. f2p单谱去噪处理函数
# =====================================================================================
def load_model(model_path='/media/ramancloud/api/Flask/methods/model_pt/f2p_ps16.pth'):
    spectrum_length = 1600
    patch_size = 16
    embed_dim = 256
    depth = 8
    heads = 8
    device = torch.device("cpu")

    model = SpecTransformer(spectrum_length, patch_size, embed_dim, depth, heads)
    state_dict = torch.load(model_path, map_location=device)
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, device


def f2p_process(spectra_noisy_orig, wavenumbers, model, device):
    spectrum_length = 1600

    original_length = len(spectra_noisy_orig)
    spectra_resampled = resample(spectra_noisy_orig, spectrum_length)
    min_val = np.min(spectra_resampled)
    max_val = np.max(spectra_resampled)
    spectra_norm = (spectra_resampled - min_val) / (max_val - min_val + 1e-8)

    with torch.no_grad():
        input_tensor = torch.from_numpy(spectra_norm).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
        denoised_norm = model(input_tensor).squeeze().cpu().numpy()

    denoised_unscaled = denoised_norm * (max_val - min_val + 1e-8) + min_val
    denoised_resampled = resample(denoised_unscaled, original_length)
    denoised_final = spectra_correction(spectra_noisy_orig, denoised_resampled)
    return denoised_final, wavenumbers
