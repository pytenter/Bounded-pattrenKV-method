import math
import os
import warnings
warnings.filterwarnings("ignore")
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import (
    cuda_attn_v_fused_with_base,
    cuda_attn_v_fused_with_base_strided_v2,
    cuda_attn_v_mixed_fused_with_base,
    cuda_bmm_fA_qB_outer,
    cuda_bmm_fA_qB_outer_with_base,
    patternkv_cache_growth_backend,
    record_mixed_v_reference_call,
)
from quant.page_batch import patternkv_fused_page_batch_decode, record_patternkv_real_decode_counter
from quant.patternkv_profile import profile_range
from models.segmented_cache import (
    CHUNKED_CACHE_MODE,
    PatternQuantizedKVCache,
    ROLLING_CACHE_MODE,
    append_decode,
    append_decode_chunked_buffer_only,
    build_cache_from_prefill,
    cache_validate_enabled,
    deserialize_cache,
    flush_chunked_buffer,
    maybe_validate_cache,
    normalize_value_objective,
    normalize_value_precision_selector,
    pattern_gather_request_centroids,
    pattern_select_v_candidate,
    pattern_select_request_v_candidate,
    reconstruct_packed_v,
    serialize_cache,
    tensor_tokens,
    update_value_causal_importance,
    value_precision_is_mixed,
)
from insight.hook_metrics import (
    record_decode_k_window_metrics,
    record_decode_v_window_metrics,
    record_prefill_k_metrics,
    record_prefill_v_metrics,
)

from transformers.models.llama.configuration_llama import *
from transformers.models.llama.modeling_llama import *
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
try:
    from cuml.cluster import KMeans  # type: ignore
except ImportError:
    KMeans = None

try:
    import cupy  # type: ignore
except ImportError:
    cupy = None


_CONFIG_FOR_DOC = "LlamaConfig"


def patternkv_mixed_v_backend() -> str:
    backend = os.environ.get("PATTERNKV_MIXED_V_BACKEND", "fused").strip().lower()
    if backend not in {"fused", "reference", "fused_page"}:
        raise ValueError("PATTERNKV_MIXED_V_BACKEND must be 'fused', 'fused_page', or 'reference'")
    return backend


def cuda_attn_v_fused_with_base_strided_v2_compat(
    group_size: int,
    attn_q: torch.Tensor,
    vq: torch.Tensor,
    v_scale: torch.Tensor,
    v_zero: torch.Tensor,
    bits: int,
    v_centroids: torch.Tensor,
    v_mask_q: torch.Tensor,
    v_idx_q: torch.Tensor,
    nh: int,
    nh_kv: int,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(bits) != 2:
        raise RuntimeError("capacity strided compat reader currently supports non-mixed V2 only")
    return cuda_attn_v_fused_with_base_strided_v2(
        group_size,
        attn_q,
        vq,
        v_scale,
        v_zero,
        v_centroids,
        v_mask_q,
        v_idx_q,
        nh=nh,
        nh_kv=nh_kv,
        attn_f=attn_f,
        v_full=v_full,
    )


def patternkv_value_reader_fn(bits: int):
    if bits == 2 and patternkv_cache_growth_backend() in {"fixed_capacity", "chunked_capacity"}:
        return cuda_attn_v_fused_with_base_strided_v2_compat
    return cuda_attn_v_fused_with_base


def patternkv_mixed_value_attention(
    module: nn.Module,
    cache: PatternQuantizedKVCache,
    weights: torch.Tensor,
    v_mask: torch.Tensor | None,
    quant_tokens: int,
    *,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    backend = patternkv_mixed_v_backend()
    if backend == "reference":
        with profile_range("mixed_v_reference_attention", tokens=int(quant_tokens)):
            record_mixed_v_reference_call(quant_tokens)
            packed_v = reconstruct_packed_v(cache)
            if packed_v is None:
                if attn_f is None or v_full is None:
                    raise RuntimeError("reference mixed Value path has no quantized or full-precision Value")
                return torch.matmul(attn_f, repeat_kv(v_full, module.num_key_value_groups))
            packed_v = packed_v[:, :, :quant_tokens, :].contiguous()
            if cache.v_centroids is not None and cache.v_assignment_idx is not None and v_mask is not None:
                centroids = cache.v_centroids.unsqueeze(0).expand(packed_v.shape[0], -1, -1, -1)
                gathered = torch.gather(
                    centroids,
                    2,
                    cache.v_assignment_idx[:, :, :quant_tokens].long().unsqueeze(-1).expand(-1, -1, -1, packed_v.shape[-1]),
                ).to(packed_v.dtype)
                packed_v = packed_v + v_mask[:, :, :quant_tokens].unsqueeze(-1).to(packed_v.dtype) * gathered
            out = torch.matmul(weights, repeat_kv(packed_v, module.num_key_value_groups))
            if attn_f is not None and v_full is not None:
                out = out + torch.matmul(attn_f, repeat_kv(v_full, module.num_key_value_groups))
            return out

    if cache.v_centroids is None or cache.v_assignment_idx is None or v_mask is None:
        raise RuntimeError("fused mixed Value path requires Pattern centroid, assignment, and mask metadata")
    if cache.v_precision_mask is None:
        raise RuntimeError("fused mixed Value path requires v_precision_mask")
    if backend == "fused_page":
        pools = getattr(cache, "operator_ready_page_pools", None)
        if pools is None:
            raise RuntimeError("fused page Value path requires operator-ready page pools")
        record_patternkv_real_decode_counter("real_decode_steps", 1)
        out = patternkv_fused_page_batch_decode(weights, pools)
        if attn_f is not None and v_full is not None:
            with profile_range("value_fp16_tail", tokens=int(v_full.shape[2])):
                out = out + torch.matmul(attn_f, repeat_kv(v_full, module.num_key_value_groups))
        return out
    record_patternkv_real_decode_counter("legacy_mixed_v_operator_calls", 1)
    return cuda_attn_v_mixed_fused_with_base(
        module.group_size,
        weights,
        cache.packed_v,
        cache.packed_v_scale,
        cache.packed_v_zero,
        cache.packed_v4,
        cache.packed_v4_scale,
        cache.packed_v4_zero,
        cache.v_precision_mask[:, :quant_tokens],
        cache.v_centroids,
        v_mask[:, :, :quant_tokens],
        cache.v_assignment_idx[:, :, :quant_tokens],
        nh=module.num_heads,
        nh_kv=module.num_key_value_heads,
        attn_f=attn_f,
        v_full=v_full,
        v2_mask_q=getattr(cache, "v2_pattern_mask", None),
        v2_idx_q=getattr(cache, "v2_assignment_idx", None),
        v4_mask_q=getattr(cache, "v4_pattern_mask", None),
        v4_idx_q=getattr(cache, "v4_assignment_idx", None),
    )


class LlamaAttention_PatternKV(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length
        assert getattr(config, "use_flash", False), "currently PatternKV is only available for flash-attn. Please add ```config.use_flash = True```"

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.k_base = None
        self.num_k_bases = config.num_k_base
        self.kmeans_heads = [None] * self.num_key_value_heads
        self.layer_idx = None

        # self.lambda_proj = config.lambda_proj
        # self.n_subspaces = config.n_subspaces
        self.num_v_bases = config.num_v_base
        self.value_objective = normalize_value_objective(getattr(config, "patternkv_value_objective", "base"))
        self.v_precision_selector = normalize_value_precision_selector(getattr(config, "patternkv_v_precision_selector", "base_v2"))
        self.v4_budget_fraction = float(getattr(config, "patternkv_v4_budget_fraction", 0.0))
        self.random_selector_seed = int(getattr(config, "patternkv_random_selector_seed", 20260809))


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        with profile_range("qkv_projection"):
            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and past_key_value and past_key_value[0] == "patternkv_segmented_cache_v1":
                kv_seq_len += int(past_key_value[13])
            else:
                kv_seq_len += past_key_value[-1]
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(self.group_size, query_states, key_states_quant_trans, 
                                key_scale_trans, key_mn_trans, self.k_bits)
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, key_states_full.transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(key_states_full.transpose(2, 3).contiguous(), 
                                                                                                                            self.group_size, 
                                                                                                                            self.k_bits)
                key_states_full = None
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, value_states_full)
            else:
                attn_output = cuda_bmm_fA_qB_outer(self.group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant, 
                                                value_scale, value_mn, self.v_bits)
                attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], value_states_full)
            
            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(value_states_full[:, :, :1, :].contiguous(), 
                                                                                                self.group_size, 
                                                                                                self.v_bits)
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            attn_weights = torch.matmul(query_states, 
                                        key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            # quantize
            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-self.residual_length, :].contiguous()
                value_states_full = value_states[:, :, -self.residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(value_states_quant, 
                                                                                                self.group_size, 
                                                                                                self.v_bits)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)

            attn_output = torch.matmul(attn_weights, value_states) 
        past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len) if use_cache else None
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value



import torch
from contextlib import nullcontext

import torch

@torch.no_grad()
def batched_kmeans_fast(X, k, iters=0, tol=1e-4, seed=0):
    
    """
    X: [H, N, D]
    返回:
    assign:   [H, N]
    centroids:[H, K, D]
    """
    H, N, D = X.shape
    device = X.device
    dtype = X.dtype

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # ========== 初始化（无放回随机采样） ==========
    scores = torch.rand(H, N, generator=g, device=device)
    _, idx = scores.topk(k, dim=1)                          # [H, K]
    centroids = torch.gather(
        X, 1, idx.unsqueeze(-1).expand(-1, -1, D)
    ).contiguous()                                          # [H, K, D]

    # ========== 预计算 & 预分配 ==========
    # x2 与迭代次数无关，提前算好
    x2 = (X * X).sum(-1, keepdim=True)                      # [H, N, 1]

    # sums / counts 反复复用，减少内存分配
    sums = torch.empty(H, k, D, device=device, dtype=dtype)
    counts = torch.empty(H, k, device=device, dtype=dtype)

    # 统计用的 ones，避免每次 new tensor
    ones = torch.ones(H, N, device=device, dtype=dtype)

    last_shift = None

    for _ in range(iters):
        # ---------- 计算距离: d2 = ||x||^2 + ||c||^2 - 2 x·c ----------
        c2 = (centroids * centroids).sum(-1).unsqueeze(1)   # [H, 1, K]
        base = x2 + c2                                      # [H, N, K]

        # baddbmm: base - 2 * X @ C^T
        d2 = torch.baddbmm(base, X, centroids.transpose(1, 2),
                        beta=1.0, alpha=-2.0)            # [H, N, K]

        assign = d2.argmin(dim=-1)                          # [H, N]

        # ---------- 重新计算中心（分组均值） ----------
        sums.zero_()
        counts.zero_()

        # sums[h, k, :] 累加所有 assign==k 的点
        sums.scatter_add_(
            1,
            assign.unsqueeze(-1).expand(-1, -1, D),         # index: [H, N, D]
            X                                               # src:   [H, N, D]
        )

        # counts[h, k] 统计数量
        counts.scatter_add_(
            1,
            assign,                                         # [H, N]
            ones                                            # [H, N]
        )

        # 处理空簇：用随机样本填充
        empty = counts == 0                                 # [H, K]
        counts_safe = counts.clamp_min_(1.0).unsqueeze(-1)  # [H, K, 1]

        new_centroids = sums / counts_safe                  # [H, K, D]

        if empty.any():
            # 随机从各 head 的样本里抽 K 个作为候选
            rand_idx = torch.randint(
                0, N, (H, k), generator=g, device=device
            )                                               # [H, K]
            repl = torch.gather(
                X, 1, rand_idx.unsqueeze(-1).expand(-1, -1, D)
            )                                               # [H, K, D]
            new_centroids = torch.where(
                empty.unsqueeze(-1), repl, new_centroids
            )

        # 收敛判据：中心最大位移
        shift = (new_centroids - centroids).abs().amax()
        centroids = new_centroids

        if last_shift is not None and shift <= tol:
            break
        last_shift = shift

    # return assign, centroids
    return None, centroids


@torch.no_grad()
def batched_kmeans(X, k, iters=0, tol=1e-4, seed=0):
    """
    X: [H, N, D]  (H=n_kv, N=bz*seq_len, D=hd)
    返回:
      assign: [H, N]        每个 head、每个样本的簇编号
      centroids: [H, K, D]  每个 head 的簇中心
    """
    H, N, D = X.shape
    g = torch.Generator(device=X.device)
    g.manual_seed(seed)

    # --- 初始化（无放回随机采样，等价于各 head 自己 topk 随机分数）---
    scores = torch.rand(H, N, generator=g, device=X.device)
    _, idx = scores.topk(k, dim=1)                                      # [H, K]
    centroids = torch.gather(X, 1, idx.unsqueeze(-1).expand(-1, -1, D)) # [H, K, D]

    last_shift = None
    for _ in range(iters):
        # --- 计算距离并分配 ---
        # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
        x2 = (X * X).sum(-1, keepdim=True)                               # [H, N, 1]
        c2 = (centroids * centroids).sum(-1).unsqueeze(1)                # [H, 1, K]
        xc = torch.matmul(X, centroids.transpose(1, 2))                  # [H, N, K]
        d2 = x2 + c2 - 2.0 * xc                                          # [H, N, K]
        assign = d2.argmin(dim=-1)                                       # [H, N]

        # --- 重新计算中心（scatter_add 实现分组均值）---
        sums = torch.zeros(H, k, D, device=X.device, dtype=X.dtype)
        sums.scatter_add_(1, assign.unsqueeze(-1).expand(-1, -1, D), X)  # sum_x per (H, K, :)
        counts = torch.zeros(H, k, device=X.device, dtype=X.dtype)
        counts.scatter_add_(1, assign, torch.ones_like(assign, dtype=X.dtype))  # count per (H, K)

        # 处理空簇：用随机样本重置
        empty = counts == 0                                              # [H, K]
        if empty.any():
            # 从各自 head 的样本里重采 K 个替补
            repl_scores = torch.rand(H, N, generator=g, device=X.device)
            _, repl_idx = repl_scores.topk(k, dim=1)                     # [H, K]
            repl = torch.gather(X, 1, repl_idx.unsqueeze(-1).expand(-1, -1, D))  # [H, K, D]
        
        counts_safe = counts.clamp_min_(1.0).unsqueeze(-1)               # [H, K, 1]
        new_centroids = sums / counts_safe                                # [H, K, D]
        if empty.any():
            new_centroids = torch.where(empty.unsqueeze(-1), repl, new_centroids)

        # 收敛判据（最大中心位移）
        shift = (new_centroids - centroids).abs().amax()
        centroids = new_centroids
        if last_shift is not None and shift <= tol:
            break
        last_shift = shift

    return assign, centroids

# 可选：PyTorch 2.3+，让编译器生成 Triton 核心
# _compile = getattr(torch, "compile", None)
_compile = None
def _maybe_compile(fn):
    if _compile is None:
        return fn
    try:
        return _compile(
            fn,
            mode="max-autotune-no-cudagraphs",
            dynamic=True,  # 关键参数
        )
    except TypeError:
        # 兼容老版本 torch.compile 不支持 dynamic 的情况
        return _compile(fn, mode="max-autotune-no-cudagraphs")

batched_kmeans_fast_compiled = _maybe_compile(batched_kmeans_fast)

@torch.no_grad()
def batched_assign(X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    给定样本和已有的簇中心，做一次批量分配。

    参数：
    X          : [H, N, D]   （H = n_kv, N = bz*seq_len, D = hd）
    centroids  : [H, K, D]   （每个 head 自己有 K 个簇中心）

    返回：
    assign     : [H, N]      （每个 head、每个 token 的簇编号）
    """
    H, N, D = X.shape
    _, K, D2 = centroids.shape
    assert D == D2, "X 和 centroids 维度 D 不一致"

    # dist^2 = ||x||^2 + ||c||^2 - 2 x·c
    x2 = (X * X).sum(-1, keepdim=True)                    # [H, N, 1]
    c2 = (centroids * centroids).sum(-1).unsqueeze(1)     # [H, 1, K]
    xc = torch.matmul(X, centroids.transpose(1, 2))       # [H, N, K]
    d2 = x2 + c2 - 2.0 * xc                               # [H, N, K]

    assign = d2.argmin(dim=-1)                            # [H, N]
    return assign

batched_assign_compiled = _maybe_compile(batched_assign)


def patternkv_cache_path(config) -> str:
    path = str(os.environ.get("PATTERNKV_CACHE_PATH") or getattr(config, "patternkv_cache_path", "segmented")).strip().lower()
    if path not in ("legacy", "segmented"):
        raise ValueError(f"Unsupported PatternKV cache path: {path!r}")
    return path


def patternkv_cache_mode(config) -> str:
    explicit = os.environ.get("PATTERNKV_CACHE_MODE") or getattr(config, "patternkv_cache_mode", None)
    if explicit is not None:
        mode = str(explicit).strip().lower()
        aliases = {
            "legacy": "legacy_tuple_chunked",
            "legacy_tuple": "legacy_tuple_chunked",
            "segmented": ROLLING_CACHE_MODE,
            "rolling": ROLLING_CACHE_MODE,
            "chunked": CHUNKED_CACHE_MODE,
        }
        mode = aliases.get(mode, mode)
        if mode not in ("legacy_tuple_chunked", CHUNKED_CACHE_MODE, ROLLING_CACHE_MODE):
            raise ValueError(f"Unsupported PatternKV cache mode: {explicit!r}")
        return mode
    return "legacy_tuple_chunked" if patternkv_cache_path(config) == "legacy" else ROLLING_CACHE_MODE


def patternkv_equivalence_backend(config) -> str:
    explicit = os.environ.get("PATTERNKV_EQUIVALENCE_BACKEND") or getattr(config, "patternkv_equivalence_backend", None)
    if os.environ.get("PATTERNKV_FORCE_REFERENCE_ATTENTION") == "1":
        explicit = "reference"
    backend = str(explicit or "production").strip().lower()
    if backend not in ("production", "reference"):
        raise ValueError(f"Unsupported PatternKV equivalence backend: {explicit!r}")
    return backend


def reset_patternkv_runtime_state(model: nn.Module) -> None:
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            continue
        attn.k_base = None
        attn.v_centroids = None


def collect_patternkv_dynamic_stats(model: nn.Module, past_key_values=None) -> dict:
    layers = list(getattr(getattr(model, "model", None), "layers", []))
    stats = {
        "initial_k_centroids_per_layer": [],
        "final_k_centroids_per_layer": [],
        "initial_v_centroids_per_layer": [],
        "final_v_centroids_per_layer": [],
        "k_centroid_updates_per_layer": [],
        "v_centroid_updates_per_layer": [],
        "k_assignment_tokens_per_layer": [],
        "v_assignment_tokens_per_layer": [],
        "v_pattern_selected_tokens_per_layer": [],
        "v_pattern_rejected_tokens_per_layer": [],
        "v_precision_v4_tokens_per_layer": [],
        "v_precision_total_tokens_per_layer": [],
        "v_precision_v4_fraction_per_layer": [],
        "packed_k_tokens_per_layer": [],
        "packed_v_tokens_per_layer": [],
    }
    for idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        cache = None
        if past_key_values is not None and idx < len(past_key_values):
            layer_cache = past_key_values[idx]
            if isinstance(layer_cache, tuple) and layer_cache and layer_cache[0] == "patternkv_segmented_cache_v1":
                cache = deserialize_cache(layer_cache, pattern=True)
        k_centroids = getattr(cache, "k_centroids", None) if cache is not None else getattr(attn, "k_base", None)
        v_centroids = getattr(cache, "v_centroids", None) if cache is not None else getattr(attn, "v_centroids", None)
        k_count = int(k_centroids.shape[1]) if torch.is_tensor(k_centroids) else 0
        v_count = int(v_centroids.shape[1]) if torch.is_tensor(v_centroids) else 0
        initial_k = int(getattr(attn, "num_k_bases", 0) or 0)
        initial_v = int(getattr(attn, "num_v_bases", 0) or 0)
        k_assign = getattr(cache, "k_assignments", None) if cache is not None else None
        v_idx = getattr(cache, "v_assignment_idx", None) if cache is not None else None
        v_mask = getattr(cache, "v_pattern_mask", None) if cache is not None else None
        if v_mask is None and cache is not None:
            v_mask = getattr(cache, "v_assignments", None)
        v_precision_mask = getattr(cache, "v_precision_mask", None) if cache is not None else None
        packed_k_tokens = int(getattr(cache, "packed_k_tokens", 0) or 0) if cache is not None else 0
        packed_v_tokens = int(getattr(cache, "packed_v_tokens", 0) or 0) if cache is not None else 0
        v_selected = int(v_mask.sum().item()) if torch.is_tensor(v_mask) else 0
        v_precision_total = int(v_precision_mask.shape[1]) if torch.is_tensor(v_precision_mask) and v_precision_mask.dim() == 2 else 0
        v_precision_v4 = int(v_precision_mask.bool().sum().item()) if torch.is_tensor(v_precision_mask) else 0
        v_assignment_tokens = int(v_idx.shape[2]) if torch.is_tensor(v_idx) else 0
        stats["initial_k_centroids_per_layer"].append(initial_k if k_count else 0)
        stats["final_k_centroids_per_layer"].append(k_count)
        stats["initial_v_centroids_per_layer"].append(initial_v if v_count else 0)
        stats["final_v_centroids_per_layer"].append(v_count)
        stats["k_centroid_updates_per_layer"].append(int(getattr(cache, "centroid_updates_k", 0) or max(k_count - initial_k, 0)) if cache is not None else 0)
        stats["v_centroid_updates_per_layer"].append(int(getattr(cache, "centroid_updates_v", 0) or max(v_count - initial_v, 0)) if cache is not None else 0)
        stats["k_assignment_tokens_per_layer"].append(int(k_assign.shape[2]) if torch.is_tensor(k_assign) else 0)
        stats["v_assignment_tokens_per_layer"].append(v_assignment_tokens)
        stats["v_pattern_selected_tokens_per_layer"].append(v_selected)
        stats["v_pattern_rejected_tokens_per_layer"].append(int(v_mask.numel()) - v_selected if torch.is_tensor(v_mask) else 0)
        stats["v_precision_v4_tokens_per_layer"].append(v_precision_v4)
        stats["v_precision_total_tokens_per_layer"].append(v_precision_total)
        stats["v_precision_v4_fraction_per_layer"].append((v_precision_v4 / v_precision_total) if v_precision_total else None)
        stats["packed_k_tokens_per_layer"].append(packed_k_tokens)
        stats["packed_v_tokens_per_layer"].append(packed_v_tokens)
    return stats

class LlamaFlashAttention_PatternKV(LlamaAttention_PatternKV):


    def _assign_minmax_hnk(self, X: torch.Tensor, C: torch.Tensor, block_k: int = 32):
        """
        X: [H, N, D]    （一批样本：这里是窗口内 K 的样本）
        C: [H, K, D]    （当前所有质心）
        返回: assign [H, N]，按 minmax 距离选最近质心
        """
        H, N, D = X.shape
        K = C.size(1)
        best_dist = torch.full((H, N), float("inf"), device=X.device, dtype=X.dtype)
        best_idx  = torch.zeros((H, N), device=X.device, dtype=torch.long)
        for k0 in range(0, K, block_k):
            k1 = min(k0 + block_k, K)
            Cb = C[:, k0:k1, :]                      # [H, B, D]
            diff = X.unsqueeze(2) - Cb.unsqueeze(1)  # [H, N, B, D]
            r = diff.amax(-1) - diff.amin(-1)       # [H, N, B]  = minmax 距离
            cand, idx = r.min(-1)                   # [H, N]
            better = cand < best_dist
            best_dist[better] = cand[better]
            best_idx[better]  = (k0 + idx)[better]
        return best_idx

    def _chebyshev_center_per_head(self, X: torch.Tensor):
        """
        X: [H, N, D]  -> 返回 [H, 1, D]
        逐维 (max+min)/2 —— Chebyshev center（最小化 ℓ∞ 半径）
        """
        x_min = X.amin(dim=1, keepdim=True)   # [H, 1, D]
        x_max = X.amax(dim=1, keepdim=True)   # [H, 1, D]
        return (x_min + x_max) * 0.5

    def _append_v_centroid_from_window(self, Vw: torch.Tensor):
        """
        Vw: [bz, n_kv, Lr, hd]  —— 当前 decode 高精度窗口（长度==residual_length）
        动态生成一个 Chebyshev center 并追加到 self.v_centroids: [n_kv, m(+1), hd]
        """
        assert Vw.dim() == 4, f"Vw shape wrong: {Vw.shape}"
        bz, H, Lr, hd = Vw.shape
        device = Vw.device
        dtype  = Vw.dtype

        # [bz, H, Lr, hd] -> [H, bz*Lr, hd]
        Xw = Vw.permute(1, 0, 2, 3).reshape(H, bz * Lr, hd).contiguous()

        # Chebyshev center（逐维 (max+min)/2），与 K 侧保持一致
        cur = self._chebyshev_center_per_head(Xw).to(dtype).to(device)   # [H, 1, hd]

        if getattr(self, "v_centroids", None) is None:
            self.v_centroids = cur
        else:
            # 直接在“质心数”维度拼接
            self.v_centroids = torch.cat([self.v_centroids, cur.to(self.v_centroids.dtype)], dim=1)

    def _threshold_and_mask_given_base(self, x: torch.Tensor, base: torch.Tensor, *, use_approx: bool = False):
        """
        x:    [bz, n_kv, L, hd]
        base: [bz, n_kv, L, hd]
        return:
            rho : [bz, n_kv, L, 1]   （R(x-base)/R(x)）
            ms  : [bz, n_kv, L]      （是否残差化）
        说明：忽略 use_approx，统一按范围收缩检验。
        """
        device, dtype_vec = x.device, x.dtype
        bz, n_kv, L, hd = x.shape
        eps = 1e-12

        # R(x) 与 R(x-base)
        x_max = x.amax(dim=-1); x_min = x.amin(dim=-1)
        R_x   = (x_max - x_min).clamp_min(eps)                    # [bz, n_kv, L]
       
        diff  = x - base
        d_max = diff.amax(dim=-1); d_min = diff.amin(dim=-1)
        R_xy  = (d_max - d_min).clamp_min(eps)                    # [bz, n_kv, L]

        rho = (R_xy / R_x).clamp_min(0.0)                         # [bz, n_kv, L]
        rho4 = rho * rho; rho4 = rho4 * rho4                      # ρ^4

        # z_{0.95}
        f32 = torch.float32
        z = (torch.sqrt(torch.tensor(2.0, dtype=f32, device=device)) *
            torch.erfinv(torch.tensor(0.9, dtype=f32, device=device)))  # Φ^{-1}(0.95)
        z = z.to(dtype_vec)

        lhs = 1.0 - rho * rho
        rhs = (2.0 * z / torch.sqrt(torch.tensor(5.0 * float(hd), dtype=dtype_vec, device=device))) * torch.sqrt(1.0 + rho4)

        mask = (lhs >= rhs)                                       # True -> 进行残差化
        return rho.unsqueeze(-1), mask


    def _v_threshold_and_mask(self, v_states, *, use_approx=False, alpha=None, base_override: torch.Tensor | None = None):
        """
        v_states:     [bz, n_kv, seq, hd]
        base_override:[bz, n_kv, seq, hd] 或 None（None 时沿用 self.v_base）
        return:
            T:    [bz, n_kv, seq, 1]
            mask: [bz, n_kv, seq]
        """
        # assert hasattr(self, "v_base") and self.v_base is not None, "v_base must be set in prefill"
        if alpha is not None:
            self.alpha = alpha
        if base_override is None:
            raise NotImplementedError
        else:
            base = base_override
        return self._threshold_and_mask_given_base(v_states, base, use_approx=use_approx)

    # --- helpers (放在类内任意位置，比如阈值函数上方) ---
    def _gather_centroids(self, idx: torch.Tensor, centroids: torch.Tensor):
        """
        idx:        [bz, n_kv, L]
        centroids:  [n_kv, m, hd]
        return:     [bz, n_kv, L, hd]
        """
        bz, n_kv, L = idx.shape
        hd = centroids.size(-1)
        cent_exp = centroids.unsqueeze(0).expand(bz, -1, -1, -1)              # [bz, n_kv, m, hd]
        return torch.gather(cent_exp, 2, idx.unsqueeze(-1).expand(-1,-1,-1,hd))  # [bz, n_kv, L, hd]

    def _nearest_v_centroid(self, x: torch.Tensor, centroids: torch.Tensor):
        """
        x:         [bz, n_kv, L, hd]
        centroids: [n_kv, m,  hd]
        return:    [bz, n_kv, L]  以 minmax 距离选择最近质心
        """
        bz, n_kv, L, hd = x.shape
        # [bz, n_kv, 1, L, hd] vs [1, n_kv, m, 1, hd] -> [bz, n_kv, m, L]
        x_ = x.unsqueeze(2)                               # [bz, n_kv, 1, L, hd]
        c  = centroids.unsqueeze(0).unsqueeze(3)          # [1,  n_kv, m, 1, hd]
        diff = x_ - c                                     # [bz, n_kv, m, L, hd]
        r = diff.amax(-1) - diff.amin(-1)                 # [bz, n_kv, m, L]
        idx = r.argmin(dim=2)                             # [bz, n_kv, L]
        return idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]


        if past_key_value is not None:
            if isinstance(past_key_value, tuple) and past_key_value and past_key_value[0] == "patternkv_segmented_cache_v1":
                kv_seq_len += int(past_key_value[13])
            else:
                kv_seq_len += past_key_value[8]
        with profile_range("rope_position"):
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None and isinstance(past_key_value, tuple) and past_key_value and past_key_value[0] == "patternkv_segmented_cache_v1":
            cache = deserialize_cache(past_key_value, pattern=True)
            with profile_range("cache_append"):
                if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE:
                    append_decode_chunked_buffer_only(cache, key_states, value_states)
                else:
                    append_decode(cache, key_states, value_states)
                    maybe_validate_cache(cache)
            if patternkv_equivalence_backend(self.config) == "reference":
                from bench.patternkv_equivalence_reference import (
                    record_reference_metric_capture,
                    reference_metric_capture_enabled,
                    reference_patternkv_attention,
                    reference_view_from_segmented_cache,
                )

                view = reference_view_from_segmented_cache(cache, num_attention_heads=self.num_heads, num_key_value_heads=self.num_key_value_heads, head_dim=self.head_dim)
                capture_metrics = reference_metric_capture_enabled(int(self.layer_idx))
                if capture_metrics:
                    attn_output, _, _, _, metric_details = reference_patternkv_attention(query_states, view, attention_mask=attention_mask, return_details=True)
                else:
                    attn_output, _, _, _ = reference_patternkv_attention(query_states, view, attention_mask=attention_mask)
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)
                if capture_metrics:
                    record_reference_metric_capture(
                        decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                        layer_idx=int(self.layer_idx),
                        cache_mode="segmented_chunked",
                        details=metric_details,
                        post_o_proj=attn_output,
                    )
                if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE:
                    cache.trace_layer_idx = int(self.layer_idx)
                    flush_chunked_buffer(cache)
                    maybe_validate_cache(cache)
                return attn_output, None, serialize_cache(cache) if use_cache else None

            score_parts = []
            value_parts: list[tuple[str, int]] = []
            if cache.sink_k is not None:
                with profile_range("qk_fp16_regions", tokens=int(cache.sink_k.shape[2])):
                    sink_k = repeat_kv(cache.sink_k, self.num_key_value_groups)
                    score_parts.append(torch.matmul(query_states, sink_k.transpose(2, 3)))
                value_parts.append(("sink", cache.sink_k.shape[2]))
            if cache.packed_k is not None:
                if cache.k_centroids is not None and cache.k_assignments is not None:
                    packed_scores = cuda_bmm_fA_qB_outer_with_base(
                        self.group_size,
                        query_states,
                        cache.packed_k,
                        cache.packed_k_scale,
                        cache.packed_k_zero,
                        self.k_bits,
                        cache.k_centroids,
                        cache.k_assignments[:, :, : cache.packed_k_tokens],
                        self.num_heads,
                        self.num_key_value_heads,
                    )
                else:
                    packed_scores = cuda_bmm_fA_qB_outer(
                        self.group_size, query_states, cache.packed_k, cache.packed_k_scale, cache.packed_k_zero, self.k_bits
                    )
                score_parts.append(packed_scores[:, :, :, : cache.packed_k_tokens])
                value_parts.append(("packed", cache.packed_k_tokens))
            if cache.pending_k is not None:
                with profile_range("qk_fp16_regions", tokens=int(cache.pending_k.shape[2])):
                    pending_k = repeat_kv(cache.pending_k, self.num_key_value_groups)
                    score_parts.append(torch.matmul(query_states, pending_k.transpose(2, 3)))
                value_parts.append(("pending", cache.pending_k.shape[2]))
            if cache.recent_k is not None:
                with profile_range("qk_fp16_regions", tokens=int(cache.recent_k.shape[2])):
                    recent_k = repeat_kv(cache.recent_k, self.num_key_value_groups)
                    score_parts.append(torch.matmul(query_states, recent_k.transpose(2, 3)))
                value_parts.append(("recent", cache.recent_k.shape[2]))
            with profile_range("attention_score_concat", tokens=int(cache.total_tokens)):
                attn_weights = torch.cat(score_parts, dim=-1) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, cache.total_tokens):
                raise ValueError(f"segmented PatternKV attention weights shape mismatch: got {attn_weights.size()}, total={cache.total_tokens}")
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, cache.total_tokens):
                    raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, cache.total_tokens)}, but is {attention_mask.size()}")
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            with profile_range("attention_softmax", tokens=int(cache.total_tokens)):
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            update_value_causal_importance(cache, attn_weights)

            offset = 0
            attn_output = None
            if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE:
                full_tokens = tensor_tokens(cache.pending_v)
                quant_tokens = int(cache.packed_v_tokens)
                if quant_tokens and full_tokens:
                    v_mask = cache.v_pattern_mask if getattr(cache, "v_pattern_mask", None) is not None else cache.v_assignments
                    if value_precision_is_mixed(getattr(cache, "v_precision_selector", "base_v2")):
                        attn_output = patternkv_mixed_value_attention(
                            self,
                            cache,
                            attn_weights[:, :, :, :quant_tokens],
                            v_mask,
                            quant_tokens,
                            attn_f=attn_weights[:, :, :, quant_tokens:],
                            v_full=cache.pending_v,
                        )
                    else:
                        v_reader = patternkv_value_reader_fn(self.v_bits)
                        attn_output = v_reader(
                            self.group_size,
                            attn_weights[:, :, :, :quant_tokens],
                            cache.packed_v,
                            cache.packed_v_scale,
                            cache.packed_v_zero,
                            self.v_bits,
                            cache.v_centroids,
                            v_mask[:, :, :quant_tokens],
                            cache.v_assignment_idx[:, :, :quant_tokens],
                            nh=self.num_heads,
                            nh_kv=self.num_key_value_heads,
                            attn_f=attn_weights[:, :, :, quant_tokens:],
                            v_full=cache.pending_v,
                        )
                    offset = quant_tokens + full_tokens
                elif quant_tokens:
                    v_mask = cache.v_pattern_mask if getattr(cache, "v_pattern_mask", None) is not None else cache.v_assignments
                    if value_precision_is_mixed(getattr(cache, "v_precision_selector", "base_v2")):
                        attn_output = patternkv_mixed_value_attention(self, cache, attn_weights, v_mask, quant_tokens)
                    else:
                        v_reader = patternkv_value_reader_fn(self.v_bits)
                        attn_output = v_reader(
                            self.group_size,
                            attn_weights,
                            cache.packed_v,
                            cache.packed_v_scale,
                            cache.packed_v_zero,
                            self.v_bits,
                            cache.v_centroids,
                            v_mask[:, :, :quant_tokens],
                            cache.v_assignment_idx[:, :, :quant_tokens],
                            nh=self.num_heads,
                            nh_kv=self.num_key_value_heads,
                        )
                    offset = quant_tokens
                else:
                    with profile_range("value_fp16_tail", tokens=int(full_tokens)):
                        attn_output = torch.matmul(attn_weights, repeat_kv(cache.pending_v, self.num_key_value_groups))
                    offset = full_tokens
            else:
                for name, length in value_parts:
                    weights = attn_weights[:, :, :, offset : offset + length]
                    if name == "packed":
                        v_mask = cache.v_pattern_mask if getattr(cache, "v_pattern_mask", None) is not None else cache.v_assignments
                        if value_precision_is_mixed(getattr(cache, "v_precision_selector", "base_v2")):
                            part = patternkv_mixed_value_attention(self, cache, weights, v_mask, cache.packed_v_tokens)
                        elif cache.v_centroids is not None and cache.v_assignment_idx is not None and v_mask is not None:
                            v_reader = patternkv_value_reader_fn(self.v_bits)
                            part = v_reader(
                                self.group_size,
                                weights,
                                cache.packed_v,
                                cache.packed_v_scale,
                                cache.packed_v_zero,
                                self.v_bits,
                                cache.v_centroids,
                                v_mask[:, :, : cache.packed_v_tokens],
                                cache.v_assignment_idx[:, :, : cache.packed_v_tokens],
                                nh=self.num_heads,
                                nh_kv=self.num_key_value_heads,
                            )
                        else:
                            part = cuda_bmm_fA_qB_outer(self.group_size, weights, cache.packed_v, cache.packed_v_scale, cache.packed_v_zero, self.v_bits)
                    else:
                        source = {"sink": cache.sink_v, "pending": cache.pending_v, "recent": cache.recent_v}[name]
                        with profile_range("value_fp16_tail", tokens=int(length)):
                            part = torch.matmul(weights, repeat_kv(source, self.num_key_value_groups))
                    attn_output = part if attn_output is None else attn_output + part
                    offset += length
            if cache_validate_enabled() and offset != cache.total_tokens:
                raise ValueError(f"segmented PatternKV attention consumed {offset} tokens, expected {cache.total_tokens}")
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
            with profile_range("output_projection"):
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)
            if getattr(cache, "cache_mode", ROLLING_CACHE_MODE) == CHUNKED_CACHE_MODE:
                cache.trace_layer_idx = int(self.layer_idx)
                with profile_range("cache_flush"):
                    flush_chunked_buffer(cache)
                    maybe_validate_cache(cache)
            return attn_output, None, serialize_cache(cache) if use_cache else None
        # assert self.num_key_value_groups == 1
        # [bsz, nh, t, hd]
        if past_key_value is not None:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]
            assignments = past_key_value[9]  
            v_assignments = past_key_value[10]  # [bz, n_kv, L_quant] or None
            v_assignments_idx = past_key_value[11]


            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer_with_base(
                    self.group_size,
                    query_states,                      # [B, nh, 1, hd]
                    key_states_quant_trans,            # [B, nh_kv, hd, N//pack]
                    key_scale_trans, key_mn_trans,
                    self.k_bits,
                    self.k_base,                       # [nh_kv, M, hd]
                    assignments,                       # [B, nh_kv, N]
                    self.num_heads, self.num_key_value_heads
                )

            else:
                att_qkquant = None
            if key_states_full is not None:
                reference_key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                reference_key_states_full = key_states
            if value_states_full is not None:
                reference_value_states_full = torch.cat([value_states_full, value_states], dim=2)
            else:
                reference_value_states_full = value_states
            if patternkv_equivalence_backend(self.config) == "reference":
                from bench.patternkv_equivalence_reference import (
                    record_reference_metric_capture,
                    reference_metric_capture_enabled,
                    reference_patternkv_attention,
                    reference_view_from_legacy_tuple,
                )

                legacy_view = reference_view_from_legacy_tuple(
                    past_key_value,
                    chunk_k=reference_key_states_full,
                    chunk_v=reference_value_states_full,
                    k_centroids=self.k_base,
                    v_centroids=self.v_centroids,
                    total_tokens=kv_seq_len,
                    num_attention_heads=self.num_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                )
                capture_metrics = reference_metric_capture_enabled(int(self.layer_idx))
                if capture_metrics:
                    attn_output, _, _, _, metric_details = reference_patternkv_attention(query_states, legacy_view, attention_mask=attention_mask, return_details=True)
                else:
                    attn_output, _, _, _ = reference_patternkv_attention(query_states, legacy_view, attention_mask=attention_mask)
                if reference_key_states_full.shape[-2] == self.residual_length:
                    assert self.residual_length % self.group_size == 0
                    Lr = self.residual_length
                    H = self.num_key_value_heads
                    bz = reference_key_states_full.size(0)
                    hd = self.head_dim
                    Xw = reference_key_states_full.permute(1, 0, 2, 3).reshape(H, bz * Lr, hd).contiguous()
                    cur_centroid = self._chebyshev_center_per_head(Xw).to(self.k_base.dtype)
                    _insight_old_k_base = self.k_base
                    _insight_new_k_base = torch.cat([_insight_old_k_base, cur_centroid], dim=1)
                    record_decode_k_window_metrics(
                        window_raw=reference_key_states_full,
                        old_k_base=_insight_old_k_base,
                        new_k_base=_insight_new_k_base,
                        layer_idx=self.layer_idx,
                        window_idx=int(assignments.shape[-1] // self.residual_length) if assignments is not None else 0,
                        bits=self.k_bits,
                        group_size=self.group_size,
                    )
                    self.k_base = torch.cat([self.k_base, cur_centroid], dim=1)
                    assign_hn = self._assign_minmax_hnk(Xw, self.k_base, block_k=256)
                    cur_assignments = assign_hn.view(H, bz, Lr).permute(1, 0, 2).contiguous().to(torch.long)
                    k_base_per_pos = self.k_base.unsqueeze(0).expand(bz, -1, -1, -1)
                    k_base_per_pos = torch.gather(k_base_per_pos, 2, cur_assignments.unsqueeze(-1).expand(-1, -1, -1, hd))
                    if assignments is not None:
                        assignments = torch.cat([assignments, cur_assignments], dim=-1)
                    else:
                        assignments = cur_assignments
                    if os.environ.get("PATTERNKV_EQUIV_TRACE") == "1" and int(os.environ.get("PATTERNKV_EQUIV_TRACE_LAYER", "-1")) == int(self.layer_idx):
                        try:
                            from bench.patternkv_equivalence_reference import save_assignment_trace

                            save_assignment_trace(
                                mode="legacy_tuple_chunked",
                                layer_idx=int(self.layer_idx),
                                decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                                k_window=reference_key_states_full,
                                v_window=reference_value_states_full,
                                k_centroids=self.k_base,
                                k_assignments=cur_assignments,
                                v_centroids=self.v_centroids,
                                v_assignment_idx=None,
                                v_gate=None,
                            )
                        except Exception:
                            if os.environ.get("PATTERNKV_EQUIV_TRACE_STRICT") == "1":
                                raise
                    reference_key_states_full = reference_key_states_full - k_base_per_pos
                    key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(
                        reference_key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                    )
                    reference_key_states_full = None
                    if key_states_quant_trans is not None:
                        key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                        key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                        key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                    else:
                        key_states_quant_trans = key_states_quant_trans_new
                        key_scale_trans = key_scale_trans_new
                        key_mn_trans = key_mn_trans_new
                attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
                if reference_value_states_full.shape[-2] == self.residual_length:
                    _insight_old_v_centroids = self.v_centroids
                    self._append_v_centroid_from_window(reference_value_states_full)
                    idx_w = self._nearest_v_centroid(reference_value_states_full, self.v_centroids)
                    cent_w = self._gather_centroids(idx_w, self.v_centroids)
                    _T, v_mask_w = self._v_threshold_and_mask(reference_value_states_full, base_override=cent_w)
                    if _insight_old_v_centroids is not None:
                        old_idx_w = self._nearest_v_centroid(reference_value_states_full, _insight_old_v_centroids)
                        old_cent_w = self._gather_centroids(old_idx_w, _insight_old_v_centroids)
                        _old_T, old_v_mask_w = self._v_threshold_and_mask(reference_value_states_full, base_override=old_cent_w)
                        record_decode_v_window_metrics(
                            window_raw=reference_value_states_full,
                            old_v_centroids=_insight_old_v_centroids,
                            new_v_centroids=self.v_centroids,
                            old_idx=old_idx_w,
                            new_idx=idx_w,
                            old_mask=old_v_mask_w,
                            new_mask=v_mask_w,
                            layer_idx=self.layer_idx,
                            window_idx=int(v_assignments.shape[-1] // self.residual_length) if v_assignments is not None else 0,
                            bits=self.v_bits,
                            group_size=self.group_size,
                        )
                    value_states_full_adj = reference_value_states_full - v_mask_w.unsqueeze(-1).to(reference_value_states_full.dtype) * cent_w
                    value_states_quant_new, scale_new, mn_new = triton_quantize_and_pack_along_last_dim(
                        value_states_full_adj, self.group_size, self.v_bits
                    )
                    if value_states_quant is not None:
                        value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                        value_scale = torch.cat([value_scale, scale_new], dim=2)
                        value_mn = torch.cat([value_mn, mn_new], dim=2)
                        v_assignments = torch.cat([v_assignments, v_mask_w.to(torch.uint8)], dim=2)
                        v_assignments_idx = torch.cat([v_assignments_idx, idx_w], dim=2)
                    else:
                        value_states_quant = value_states_quant_new
                        value_scale = scale_new
                        value_mn = mn_new
                        v_assignments = v_mask_w.to(torch.uint8)
                        v_assignments_idx = idx_w
                    reference_value_states_full = None
                if self.config.pretraining_tp > 1:
                    attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                    o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                    attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
                else:
                    attn_output = self.o_proj(attn_output)
                if capture_metrics:
                    record_reference_metric_capture(
                        decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                        layer_idx=int(self.layer_idx),
                        cache_mode="legacy_tuple_chunked",
                        details=metric_details,
                        post_o_proj=attn_output,
                    )
                return attn_output, None, (key_states_quant_trans, reference_key_states_full, key_scale_trans, key_mn_trans, value_states_quant, reference_value_states_full, value_scale, value_mn, kv_seq_len, assignments, v_assignments, v_assignments_idx) if use_cache else None
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states
            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if key_states_full.shape[-2] == self.residual_length:
                assert self.residual_length % self.group_size == 0
                Lr = self.residual_length
                H  = self.num_key_value_heads
                bz = key_states_full.size(0)
                hd = self.head_dim

                # [bz, H, Lr, hd] -> [H, bz*Lr, hd]
                Xw = key_states_full.permute(1, 0, 2, 3).reshape(H, bz * Lr, hd).contiguous()

                # ---- (A) 用 minmax 的“Chebyshev center”当新质心（每个 head 1 个）----
                
                cur_centroid = self._chebyshev_center_per_head(Xw).to(self.k_base.dtype)   # [H, 1, D]
                _insight_old_k_base = self.k_base
                _insight_new_k_base = torch.cat([_insight_old_k_base, cur_centroid], dim=1)
                record_decode_k_window_metrics(
                    window_raw=key_states_full,
                    old_k_base=_insight_old_k_base,
                    new_k_base=_insight_new_k_base,
                    layer_idx=self.layer_idx,
                    window_idx=int(assignments.shape[-1] // self.residual_length) if assignments is not None else 0,
                    bits=self.k_bits,
                    group_size=self.group_size,
                )

                # ---- (B) 质心集合追加一列（K -> K+1）----
                self.k_base = torch.cat([self.k_base, cur_centroid], dim=1)                # [H, K+1, D]

                # ---- (C) 窗口样本按 minmax 距离重分配到最近质心（向量化 & 分块K）----
                assign_hn = self._assign_minmax_hnk(Xw, self.k_base, block_k=256)           # [H, bz*Lr]
                
                cur_assignments = assign_hn.view(H, bz, Lr).permute(1, 0, 2).contiguous().to(torch.long)  # [bz, H, Lr]

                # ---- (D) 逐位置 gather 基向量，做残差化并量化 ----
                k_base_per_pos = self.k_base.unsqueeze(0).expand(bz, -1, -1, -1)           # [bz, H, K+1, D]
                k_base_per_pos = torch.gather(
                    k_base_per_pos, 2, cur_assignments.unsqueeze(-1).expand(-1, -1, -1, hd)
                )  # [bz, H, Lr, D]

                # 维护历史 assignments
                if assignments is not None:
                    assignments = torch.cat([assignments, cur_assignments], dim=-1)        # [bz, H, ... + Lr]
                else:
                    assignments = cur_assignments
                if os.environ.get("PATTERNKV_EQUIV_TRACE") == "1" and int(os.environ.get("PATTERNKV_EQUIV_TRACE_LAYER", "-1")) == int(self.layer_idx):
                    try:
                        from bench.patternkv_equivalence_reference import save_assignment_trace

                        save_assignment_trace(
                            mode="legacy_tuple_chunked",
                            layer_idx=int(self.layer_idx),
                            decode_position=int(os.environ.get("PATTERNKV_EQUIV_TRACE_DECODE_POS", "-1")),
                            k_window=key_states_full,
                            v_window=value_states_full,
                            k_centroids=self.k_base,
                            k_assignments=cur_assignments,
                            v_centroids=self.v_centroids,
                            v_assignment_idx=None,
                            v_gate=None,
                        )
                    except Exception:
                        if os.environ.get("PATTERNKV_EQUIV_TRACE_STRICT") == "1":
                            raise

                # 残差化 + 量化 pack（沿最后一维）
                key_states_full = key_states_full - k_base_per_pos
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = \
                    triton_quantize_and_pack_along_last_dim(
                        key_states_full.transpose(2, 3).contiguous(), self.group_size, self.k_bits
                    )
                key_states_full = None  # 滚动窗口“吃掉”后清空 full，等待下一轮
                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(
                    attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            if value_states_full is None:
                value_states_full = value_states
            else:
                value_states_full = torch.cat([value_states_full, value_states], dim=2)

            value_full_length = value_states_full.shape[-2]
            # print(value_states_full.shape)
            total_kv = attn_weights.size(-1)
            value_full_length = value_states_full.shape[-2]
            L_quant = total_kv - value_full_length
            if L_quant < 0:
                raise RuntimeError(f"bad split: L_quant={L_quant}, total_kv={total_kv}, full={value_full_length}")

            if value_states_quant is None or L_quant == 0:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_weights_q = attn_weights[:, :, :, :L_quant]
                attn_weights_f = attn_weights[:, :, :, L_quant:]
                
                # v_mask_q  = v_assignments[:, :, :L_quant].to(attn_weights.dtype)
                # idx_q     = v_assignments_idx[:, :, :L_quant]
                # v_cent_q  = self._gather_centroids(idx_q, self.v_centroids)
                # v_mask_rep= v_mask_q.unsqueeze(2).repeat_interleave(self.num_key_value_groups, dim=1)
                # base_rep  = v_cent_q.repeat_interleave(self.num_key_value_groups, dim=1).to(out_old.dtype)
                # out_old  += torch.matmul(attn_weights_q * v_mask_rep, base_rep)
                
                # import copy
                # out_old_q = copy.deepcopy(out_old)

                # out_old += torch.matmul(attn_weights_f, repeat_kv(value_states_full, self.num_key_value_groups))

                # 新路径
                # zeros_cen = torch.zeros_like(self.v_centroids, dtype=self.v_centroids.dtype)
                attn_output = cuda_attn_v_fused_with_base(
                    self.group_size, attn_weights_q, value_states_quant, value_scale, value_mn, self.v_bits,
                    self.v_centroids, v_assignments[:, :, :L_quant], v_assignments_idx[:, :, :L_quant],
                    nh=self.num_heads, nh_kv=self.num_key_value_heads,
                    attn_f=attn_weights_f, v_full=value_states_full
                )

                # mae = (out_old - out_new).abs().max().item()
                # print("max|diff| =", mae)  # FP16 下一般 < 1e-3 ~ 1e-2

                # attn_output = out_new


            # print(value_full_length)
            attn_output = attn_output.transpose(1, 2).contiguous()
            if value_full_length == self.residual_length:
                # print("!")
                # 1) 先追加一个由当前满窗生成的 Chebyshev center（每个 head 1 个），与 K 侧语义一致
                #    注：_append_v_centroid_from_window 内部已处理 self.v_centroids 是否为 None 的情况
                
                _insight_old_v_centroids = self.v_centroids
                self._append_v_centroid_from_window(value_states_full)  # self.v_centroids: [n_kv, m(+1), hd]

                # 2) 用“最新质心集合”对整个窗口做最近质心分配（minmax 距离）
                
                idx_w, v_mask_w, _ = pattern_select_v_candidate(
                    value_states_full,
                    self.v_centroids,
                    value_objective=self.value_objective,
                    group_size=self.group_size,
                    bits=self.v_bits,
                )
                cent_w = self._gather_centroids(idx_w, self.v_centroids)                       # [bz, n_kv, Lr, hd]

                # 3) 阈值/掩码（范围收缩检验），并条件性做残差化
                _T, _ = self._v_threshold_and_mask(value_states_full, base_override=cent_w)  # v_mask_w: [bz, n_kv, Lr]
                if _insight_old_v_centroids is not None:
                    old_idx_w = self._nearest_v_centroid(value_states_full, _insight_old_v_centroids)
                    old_cent_w = self._gather_centroids(old_idx_w, _insight_old_v_centroids)
                    _old_T, old_v_mask_w = self._v_threshold_and_mask(value_states_full, base_override=old_cent_w)
                    record_decode_v_window_metrics(
                        window_raw=value_states_full,
                        old_v_centroids=_insight_old_v_centroids,
                        new_v_centroids=self.v_centroids,
                        old_idx=old_idx_w,
                        new_idx=idx_w,
                        old_mask=old_v_mask_w,
                        new_mask=v_mask_w,
                        layer_idx=self.layer_idx,
                        window_idx=int(v_assignments.shape[-1] // self.residual_length) if v_assignments is not None else 0,
                        bits=self.v_bits,
                        group_size=self.group_size,
                    )
                value_states_full_adj = value_states_full - v_mask_w.unsqueeze(-1).to(value_states_full.dtype) * cent_w

                # 4) 量化打包（沿最后一维），并把整窗拼接到量化段尾部
                value_states_quant_new, scale_new, mn_new = triton_quantize_and_pack_along_last_dim(
                    value_states_full_adj, self.group_size, self.v_bits
                )

                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale        = torch.cat([value_scale,        scale_new],            dim=2)
                    value_mn           = torch.cat([value_mn,           mn_new],               dim=2)
                    v_assignments      = torch.cat([v_assignments,      v_mask_w.to(torch.uint8)], dim=2)
                    v_assignments_idx  = torch.cat([v_assignments_idx,  idx_w],                    dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale        = scale_new
                    value_mn           = mn_new
                    v_assignments      = v_mask_w.to(torch.uint8)
                    v_assignments_idx  = idx_w

                value_states_full = None


        else:
            
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                logger.warning_once(
                    f"The input hidden states seems to be silently casted in float32, this might be related to"
                    f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                    f" {target_dtype}."
                )

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)
            if patternkv_equivalence_backend(self.config) == "reference":
                key_for_attention = repeat_kv(key_states, self.num_key_value_groups)
                value_for_attention = repeat_kv(value_states, self.num_key_value_groups)
                attn_weights = torch.matmul(query_states, key_for_attention.transpose(2, 3)) / math.sqrt(self.head_dim)
                if attention_mask is not None:
                    if attention_mask.size() != (bsz, 1, q_len, q_len):
                        raise ValueError(f"Attention mask should be of size {(bsz, 1, q_len, q_len)}, but is {attention_mask.size()}")
                    attn_weights = attn_weights + attention_mask
                    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_output = torch.matmul(attn_weights, value_for_attention)
            else:
                attn_output = self._flash_attention_forward(
                    query_states.transpose(1, 2), key_states.transpose(1, 2),
                    value_states.transpose(1, 2), None, q_len, dropout=0.0
                )
      

            bsz, n_kv, seq_len, hd = key_states.shape
            
            bz, n_kv, seq_len, hd = key_states.shape
   
            if bz == 1:
                key_states_means = key_states.mean(dim=0, keepdim=True)
                Xmk = key_states_means.permute(1, 0, 2, 3).reshape(n_kv, 1 * seq_len, hd).to(torch.float32)  # [H, N, D]
                Xk = key_states.permute(1, 0, 2, 3).reshape(n_kv, bz * seq_len, hd).to(torch.float32)
                assign_k, k_centroids = batched_kmeans_fast_compiled(Xmk, k=self.num_k_bases, iters=30, tol=1e-4, seed=0)
                assign_k = batched_assign_compiled(Xk, k_centroids)
                assignments = assign_k.view(n_kv, bz, seq_len).permute(1, 0, 2).contiguous().to(torch.long)  # [bz, n_kv, seq_len]
                self.k_base = k_centroids.to(key_states.dtype)
            else:
                k_bases = []
                assignment_rows = []
                for req_idx in range(bz):
                    Xmk = key_states[req_idx : req_idx + 1].permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
                    _assign_seed, k_centroids = batched_kmeans_fast_compiled(Xmk, k=self.num_k_bases, iters=30, tol=1e-4, seed=0)
                    assign_k = batched_assign_compiled(Xmk, k_centroids)
                    assignment_rows.append(assign_k.view(n_kv, 1, seq_len).permute(1, 0, 2).contiguous().to(torch.long))
                    k_bases.append(k_centroids.to(key_states.dtype))
                assignments = torch.cat(assignment_rows, dim=0)
                self.k_base = torch.stack(k_bases, dim=0).contiguous()

            
            bz, n_kv, seq_len, hd = value_states.shape
            
            if bz == 1:
                value_states_means = value_states.mean(dim=0, keepdim=True)
                Xm = value_states_means.permute(1, 0, 2, 3).reshape(n_kv, 1 * seq_len, hd).to(torch.float32)  # [H, N, D]
                X = value_states.permute(1, 0, 2, 3).reshape(n_kv, bz * seq_len, hd).to(torch.float32)
                assign, centroids = batched_kmeans_fast_compiled(Xm, k=self.num_v_bases, iters=30, tol=1e-4, seed=0)
                assign = batched_assign_compiled(X, centroids)
                v_assignments_idx_all = assign.view(n_kv, bz, seq_len).permute(1, 0, 2).contiguous().to(torch.long)  # [bz, n_kv, seq_len]
                self.v_centroids = centroids.to(value_states.dtype)                                                        # [n_kv, num_v_bases, hd]
            else:
                v_bases = []
                v_assignment_rows = []
                for req_idx in range(bz):
                    Xm = value_states[req_idx : req_idx + 1].permute(1, 0, 2, 3).reshape(n_kv, seq_len, hd).to(torch.float32)
                    _assign_seed, centroids = batched_kmeans_fast_compiled(Xm, k=self.num_v_bases, iters=30, tol=1e-4, seed=0)
                    assign = batched_assign_compiled(Xm, centroids)
                    v_assignment_rows.append(assign.view(n_kv, 1, seq_len).permute(1, 0, 2).contiguous().to(torch.long))
                    v_bases.append(centroids.to(value_states.dtype))
                v_assignments_idx_all = torch.cat(v_assignment_rows, dim=0)
                self.v_centroids = torch.stack(v_bases, dim=0).contiguous()

            if key_states.shape[-2] % self.residual_length != 0:
                if key_states.shape[-2] < self.residual_length:
                    key_states_quant = None
                    assignments = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % self.residual_length), :].contiguous()
                    assignments = assignments[:, :, :-(key_states.shape[-2] % self.residual_length)]
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % self.residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None
            if key_states_quant is not None:
                # build per-position base and subtract
                # assignments: [bsz, n_kv, seq_len]
                # centroids: [n_kv, num_k_bases, hd]
                # gather to [bsz, n_kv, seq_len, hd]
                if self.k_base.dim() == 4:
                    k_base_per_pos = pattern_gather_request_centroids(assignments, self.k_base)
                else:
                    k_base_per_pos = self.k_base.unsqueeze(0).expand(bsz, -1, -1, -1)
                    # index along num_k_bases dim via assignments
                    k_base_per_pos = torch.gather(
                        k_base_per_pos,
                        2,
                        assignments.unsqueeze(-1).expand(-1, -1, -1, hd)
                    )  # [bsz, n_kv, seq_len, hd]
                key_states_quant = key_states_quant - k_base_per_pos
                if self.k_base.dim() == 3:
                    record_prefill_k_metrics(
                        key_states=key_states,
                        key_states_quant=key_states_quant,
                        assignments=assignments,
                        k_base=self.k_base,
                        key_states_full=key_states_quant + k_base_per_pos,
                        layer_idx=self.layer_idx,
                        bits=self.k_bits,
                        group_size=self.group_size,
                    )
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(key_states_quant.transpose(2, 3).contiguous(), self.group_size, self.k_bits)
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None
            
            if value_states.shape[-2] <= self.residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
                v_assignments = None
                v_assignments_idx  = None         # 质心编号 (long)
            else:
                # print(value_states.shape)
                # qlen = value_states.shape[-2] - self.residual_length
                qlen = -(value_states.shape[-2] % self.residual_length)
                if qlen == 0:
                    value_states_quant = value_states.contiguous()
                    value_states_full  = None
                    idx_q = v_assignments_idx_all                    # [bz, n_kv, qlen]
                else:
                    value_states_quant = value_states[:, :, :qlen, :].contiguous()
                    value_states_full  = value_states[:, :, qlen:, :].contiguous()
                    idx_q = v_assignments_idx_all[:, :, :qlen]                     # [bz, n_kv, qlen]

                
                if self.v_centroids.dim() == 4:
                    idx_q, v_mask_q, _ = pattern_select_request_v_candidate(
                        value_states_quant,
                        self.v_centroids,
                        value_objective=self.value_objective,
                        group_size=self.group_size,
                        bits=self.v_bits,
                    )
                    v_cent_per_pos_q = pattern_gather_request_centroids(idx_q, self.v_centroids)
                else:
                    idx_q, v_mask_q, _ = pattern_select_v_candidate(
                        value_states_quant,
                        self.v_centroids,
                        value_objective=self.value_objective,
                        group_size=self.group_size,
                        bits=self.v_bits,
                    )
                    v_cent_per_pos_q = self._gather_centroids(idx_q, self.v_centroids)  # [bz, n_kv, qlen, hd]

                # 用已选 candidate 的生产阈值/掩码
                T1, _ = self._v_threshold_and_mask(value_states_quant, base_override=v_cent_per_pos_q)
                # print(v_mask_q.float().mean())
                # exit(0)

                # 条件性减质心（做残差化）
                _insight_value_states_quant_raw = value_states_quant
                value_states_quant = value_states_quant - v_mask_q.unsqueeze(-1).to(value_states.dtype) * v_cent_per_pos_q
                if self.v_centroids.dim() == 3:
                    record_prefill_v_metrics(
                        value_states_quant=value_states_quant,
                        idx_q=idx_q,
                        v_centroids=self.v_centroids,
                        v_mask_q=v_mask_q,
                        value_states_full=_insight_value_states_quant_raw,
                        layer_idx=self.layer_idx,
                        bits=self.v_bits,
                        group_size=self.group_size,
                        rho=T1,
                    )

                # 量化 pack（沿最后一维）
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_states_quant, self.group_size, self.v_bits
                )

                # 保存：mask 与 最近质心编号
                v_assignments     = v_mask_q.to(torch.uint8)   # [bz, n_kv, qlen]
                v_assignments_idx = idx_q                      # [bz, n_kv, qlen]


        cache_mode = patternkv_cache_mode(self.config)
        if use_cache and past_key_value is None and cache_mode in (CHUNKED_CACHE_MODE, ROLLING_CACHE_MODE):
            cache = build_cache_from_prefill(
                key_states,
                value_states,
                sink_length=0 if cache_mode == CHUNKED_CACHE_MODE else int(getattr(self.config, "sink_length", 0)),
                recent_length=0 if cache_mode == CHUNKED_CACHE_MODE else int(getattr(self.config, "recent_length", self.residual_length)),
                group_size=self.group_size,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                pattern=True,
                k_centroids=self.k_base,
                v_centroids=self.v_centroids,
                k_assignments=assignments,
                v_assignment_idx=v_assignments_idx,
                v_pattern_mask=v_assignments,
                cache_mode=cache_mode,
                chunk_length=self.residual_length,
                value_objective=self.value_objective,
                v_precision_selector=self.v_precision_selector,
                v4_budget_fraction=self.v4_budget_fraction,
                random_selector_seed=self.random_selector_seed,
                selector_task_key=getattr(self, "selector_task_key", getattr(self.config, "patternkv_selector_task_key", "task")),
                selector_layer_idx=self.layer_idx,
                v_causal_importance=getattr(self, "v_causal_importance", None),
                v_oracle_importance=getattr(self, "v_oracle_importance", None),
            )
            past_key_value = serialize_cache(cache)
        else:
            past_key_value = (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans, 
                              value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len, assignments, v_assignments, v_assignments_idx) if use_cache else None
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        try:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
        except ImportError:
            query_layer = query_states.transpose(1, 2)
            key_layer = key_states.transpose(1, 2)
            value_layer = value_states.transpose(1, 2)
            if key_layer.shape[1] != query_layer.shape[1]:
                key_layer = repeat_kv(key_layer, self.num_key_value_groups)
                value_layer = repeat_kv(value_layer, self.num_key_value_groups)
            q_len = query_layer.shape[-2]
            kv_len = key_layer.shape[-2]
            sdpa_mask = None
            if self.is_causal:
                query_positions = torch.arange(q_len, device=query_layer.device) + max(kv_len - q_len, 0)
                key_positions = torch.arange(kv_len, device=query_layer.device)
                sdpa_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
                sdpa_mask = sdpa_mask.view(1, 1, q_len, kv_len)
            if attention_mask is not None:
                key_padding_mask = attention_mask[:, None, None, :].bool()
                sdpa_mask = key_padding_mask if sdpa_mask is None else sdpa_mask & key_padding_mask
            kwargs = {"dropout_p": dropout if self.training else 0.0, "attn_mask": sdpa_mask}
            if softmax_scale is not None:
                kwargs["scale"] = softmax_scale
            attn_output = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, **kwargs)
            return attn_output.transpose(1, 2).contiguous()

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
            )

        return attn_output


    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    

class LlamaDecoderLayer_PatternKV(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = (
            LlamaAttention_PatternKV(config=config)
            if not getattr(config, "use_flash", False)
            else LlamaFlashAttention_PatternKV(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaModel_PatternKV(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer_PatternKV(config) for _ in range(config.num_hidden_layers)])
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.layer_idx = layer_idx
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            first_cache = past_key_values[0]
            if isinstance(first_cache, tuple) and first_cache and first_cache[0] == "patternkv_segmented_cache_v1":
                past_key_values_length = int(first_cache[13])
            else:
                past_key_values_length = past_key_values[0][8]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if getattr(self.config, "_flash_attn_2_enabled", False):
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_PatternKV(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel_PatternKV(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        phase = "decode" if (past_key_values is not None) else "prefill"
        
        
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with profile_range(f"{phase}_decoder_model_forward"):
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        hidden_states = outputs[0]
        with profile_range(f"{phase}_lm_head"):
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                logits = self.lm_head(hidden_states)
            logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if isinstance(past_key_values, DynamicCache):
            past_key_values = past_key_values.to_legacy_cache()
            if len(past_key_values) == 0:
                past_key_values = None
        if past_key_values is not None:
            first_cache = past_key_values[0]
            if isinstance(first_cache, tuple) and first_cache and first_cache[0] == "patternkv_segmented_cache_v1":
                past_length = int(first_cache[13])
            else:
                past_length = past_key_values[0][8]
            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
