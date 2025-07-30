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

    input_tensor = torch.from_numpy(spectra_noisy_orig).to(device, dtype=torch.float32)

    resampled_tensor = torch.nn.functional.interpolate(
        input_tensor.view(1, 1, -1),
        size=spectrum_length,
        mode='linear',
        align_corners=False
    )

    min_val = torch.min(resampled_tensor)
    max_val = torch.max(resampled_tensor)
    range_val = max_val - min_val
    normalized_tensor = (resampled_tensor - min_val) / (range_val + 1e-8)

    with torch.no_grad():
        denoised_norm = model(normalized_tensor)

    denoised_unscaled = denoised_norm * (range_val + 1e-8) + min_val

    denoised_resampled = torch.nn.functional.interpolate(
        denoised_unscaled,
        size=original_length,
        mode='linear',
        align_corners=False
    ).squeeze().cpu().numpy()

    denoised_final = spectra_correction(spectra_noisy_orig, denoised_resampled)
    
    return denoised_final, wavenumbers

def f2p_process_batch(spectra_batch, model, device):
    if not spectra_batch:
        return []
    
    spectrum_length = 1600 # 模型固定的输入长度
    
    # 记录每条光谱的原始长度，用于后续恢复
    original_lengths = [len(s) for s in spectra_batch]
    
    # --- 1. 批量预处理 ---
    
    # a. 将每条光谱插值到模型所需的固定长度(1600)
    # 因为原始长度可能不同，这里需要先单独处理再合并成批次
    resampled_list = [
        torch.nn.functional.interpolate(
            torch.from_numpy(spec).view(1, 1, -1),
            size=spectrum_length,
            mode='linear',
            align_corners=False
        ) for spec in spectra_batch
    ]
    
    # b. 将插值后的张量列表合并成一个大的批次张量
    # 形状变为: [批次大小, 1, 1600]
    batch_tensor = torch.cat(resampled_list, dim=0).to(device, dtype=torch.float32)

    # c. 向量化归一化：对批次中的每条光谱进行min-max归一化
    # keepdim=True 保持维度，便于后续广播计算
    min_vals, _ = torch.min(batch_tensor, dim=2, keepdim=True)
    max_vals, _ = torch.max(batch_tensor, dim=2, keepdim=True)
    ranges = max_vals - min_vals
    # 防止除以零
    ranges[ranges == 0] = 1e-8
    normalized_batch = (batch_tensor - min_vals) / ranges

    # --- 2. 批量模型推理 ---
    with torch.no_grad():
        denoised_norm_batch = model(normalized_batch)

    # --- 3. 批量后处理 ---

    # a. 向量化反归一化
    denoised_unscaled_batch = denoised_norm_batch * ranges + min_vals

    # b. 将结果插值回各自的原始长度
    # 由于原始长度不同，这一步必须逐一处理
    results = []
    for i in range(len(spectra_batch)):
        denoised_unscaled = denoised_unscaled_batch[i:i+1] # 切片以保持形状 [1, 1, 1600]
        original_len = original_lengths[i]
        
        # 插值回原始长度
        denoised_resampled = torch.nn.functional.interpolate(
            denoised_unscaled,
            size=original_len,
            mode='linear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # 应用最终校正
        final_spectrum = spectra_correction(spectra_batch[i], denoised_resampled)
        results.append(final_spectrum)
        
    return results