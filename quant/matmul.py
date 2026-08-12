import torch
# import ipdb
import os
import random
import sys
from pathlib import Path
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import patternkv_gemv
from quant.patternkv_profile import profile_range, record_counter, record_temp_allocation

_MIXED_V_COUNTERS = {
	"mixed_v_fused_calls": 0,
	"mixed_v_reference_calls": 0,
	"v2_tokens_processed": 0,
	"v4_tokens_processed": 0,
	"fused_temp_bytes": 0,
	"gqa_v2_calls": 0,
	"gqa_v2_fallbacks": 0,
}

_PAGE_V_READER_COUNTERS = {
	"historical_materialize_calls": 0,
	"historical_torch_cat_calls": 0,
	"historical_materialized_bytes": 0,
	"page_native_kernel_calls": 0,
	"page_table_rebuilds": 0,
	"page_table_device_updates": 0,
	"page_table_bytes_uploaded": 0,
}


def reset_patternkv_mixed_v_counters() -> None:
	for key in _MIXED_V_COUNTERS:
		_MIXED_V_COUNTERS[key] = 0


def get_patternkv_mixed_v_counters() -> dict:
	return dict(_MIXED_V_COUNTERS)


def reset_patternkv_page_v_reader_counters() -> None:
	for key in _PAGE_V_READER_COUNTERS:
		_PAGE_V_READER_COUNTERS[key] = 0


def get_patternkv_page_v_reader_counters() -> dict:
	return dict(_PAGE_V_READER_COUNTERS)


def record_mixed_v_reference_call(tokens: int = 0) -> None:
	_MIXED_V_COUNTERS["mixed_v_reference_calls"] += 1
	_MIXED_V_COUNTERS["v2_tokens_processed"] += int(tokens)


def _tensor_bytes(value: torch.Tensor | None) -> int:
	return 0 if value is None else int(value.numel() * value.element_size())


def patternkv_gqa_v_backend() -> str:
	backend = os.environ.get("PATTERNKV_GQA_V_BACKEND", "baseline").strip().lower()
	if backend not in {"baseline", "gqa"}:
		raise ValueError("PATTERNKV_GQA_V_BACKEND must be 'baseline' or 'gqa'")
	return backend


def patternkv_page_v_reader_backend() -> str:
	backend = os.environ.get("PATTERNKV_PAGE_V_READER", "contiguous").strip().lower()
	if backend not in {"contiguous", "paged_v2"}:
		raise ValueError("PATTERNKV_PAGE_V_READER must be 'contiguous' or 'paged_v2'")
	return backend


class DevicePageTable:
	"""Device-resident pointer table for fixed pages.

	The table is refreshed only when page allocations change. Existing page
	contents may mutate without requiring a table update because the data_ptrs are
	stable for the lifetime of each page tensor.
	"""

	def __init__(self, stream: str = "page") -> None:
		self.stream = str(stream)
		self._signature: tuple[int, ...] = ()
		self.tensor: torch.Tensor | None = None

	def refresh(self, pages: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
		if not pages:
			raise ValueError(f"{self.stream} page table cannot be built from an empty page list")
		signature = tuple(int(page.data_ptr()) for page in pages)
		device = pages[0].device
		if self.tensor is not None and self._signature == signature and self.tensor.device == device:
			return self.tensor
		if not all(page.device == device for page in pages):
			raise ValueError(f"{self.stream} pages must live on the same device")
		host = torch.tensor(signature, dtype=torch.int64, device="cpu")
		table = torch.empty((len(signature),), dtype=torch.int64, device=device)
		table.copy_(host, non_blocking=False)
		self.tensor = table
		self._signature = signature
		_PAGE_V_READER_COUNTERS["page_table_rebuilds"] += 1
		_PAGE_V_READER_COUNTERS["page_table_device_updates"] += 1
		_PAGE_V_READER_COUNTERS["page_table_bytes_uploaded"] += int(table.numel() * table.element_size())
		return table


@triton.jit
def qbvm_kernel(
	bits,
	a_ptr, b_ptr, c_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_abatch, stride_am, stride_ak,
	stride_bbatch, stride_bk, stride_bn,
	stride_cbatch, stride_cm, stride_cn,
	stride_scales_b, stride_scales_k, stride_scales_g,
	stride_zeros_b, stride_zeros_k, stride_zeros_g,
	groupsize,
	BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
	"""
	Compute the batch matrix multiplication C = A x B.
	A is of shape (B, 1, K) float16
	B is of shape (B, K, N//feat_per_int) int32
	C is of shape (B, 1, N) float16
	scales is of shape (B, K, G) float16
	zeros is of shape (B, K, G) float16
	groupsize is an int specifying the size of groups for scales and zeros.
	G is N // groupsize.
	Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.

	WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
	WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
	WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
	"""
	pid_batch = tl.program_id(axis=0)
	pid = tl.program_id(axis=1)
	feat_per_int = 32 // bits
	num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
	pid_n = pid % num_pid_n
	offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	a_batch_offset = (pid_batch * stride_abatch)
	b_batch_offset = (pid_batch * stride_bbatch)
	c_batch_offset = (pid_batch * stride_cbatch)
	a_ptr = a_ptr + a_batch_offset 
	b_ptr = b_ptr + b_batch_offset 
	c_ptr = c_ptr + c_batch_offset
	a_ptrs = a_ptr + (offs_k[:, None] * stride_ak)   # (BLOCK_SIZE_K, 1)
	# a_mask = (offs_am[:, None] < M)
	# b_ptrs is set up such that it repeats elements along the N axis feat_per_int times
	b_ptrs = b_ptr  + (offs_k[:, None] * stride_bk + (offs_bn[None, :]//feat_per_int) * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
	# shifter is used to extract the # bits bits of each element in the 32-bit word from B
	shifter = (offs_bn % feat_per_int) * bits
	scales_ptr = scales_ptr + pid_batch*stride_scales_b + ((offs_bn[None, :] // groupsize)) * stride_scales_g   # (BLOCK_SIZE_N,)
	zeros_ptr = zeros_ptr + pid_batch*stride_zeros_b + ((offs_bn[None, :] // groupsize)) * stride_zeros_g   # (BLOCK_SIZE_N,)

	# Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
	# M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
	# So this loop is along the infeatures dimension (K)
	# It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel	
	# accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
	num = 0xFF >> (8-bits)
	for pid_k in range(0, num_pid_k):
		offs_bk = (offs_k[:, None] + pid_k * BLOCK_SIZE_K)
		# offs_k[None, :] < K - pid_k * BLOCK_SIZE_K
		a = tl.load(a_ptrs, mask=offs_bk < K, other=0.)   # (1, BLOCK_SIZE_K)
		b = tl.load(b_ptrs, mask=offs_bk < K, other=0.)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		ptr = scales_ptr + offs_bk * stride_scales_k 
		scales = tl.load(ptr, mask=offs_bk < K, other=0.)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		ptr = zeros_ptr + offs_bk * stride_zeros_k  
		zeros = tl.load(ptr, mask=offs_bk < K, other=0.)  # (BLOCK_SIZE_K, BLOCK_SIZE_N)
		# Now we need to unpack b into 32-bit values
		# tl.device_print("scale ",scales.dtype)
		# tl.device_print("zeros ",zeros.dtype)
		b = (b >> shifter[None, :]) & num  # For 4-bit values, bit_op_num is 0xF
		b = b * scales + zeros # Scale and shift
		accumulator += tl.sum(a * b, 0) # tl.dot(a, b)
		# if pid_m == 0 and pid_n == 0:
		# 	tl.device_print("hello ", tl.dot(a, b).shape)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += BLOCK_SIZE_K * stride_bk
	c = accumulator # .to(tl.float16)
	# c = accumulator
	# Store the result
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + stride_cn * offs_cn
	c_mask = (offs_cn < N)
	tl.store(c_ptrs, c, mask=c_mask)


def understand_code():
	M, N, K = 512, 256, 256
	BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M = 64, 64, 4
	total_program_id = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
	for pid in range(0, total_program_id):
		num_pid_m = triton.cdiv(M, BLOCK_SIZE_M)
		num_pid_n = triton.cdiv(N, BLOCK_SIZE_N)
		num_pid_in_group = GROUP_SIZE_M * num_pid_n
		group_id = pid // num_pid_in_group
		first_pid_m = group_id * GROUP_SIZE_M
		group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
		pid_m = first_pid_m + (pid % group_size_m)
		pid_n = (pid % num_pid_in_group) // group_size_m
		print(f"pid={pid}, pid_m={pid_m}, pid_n={pid_n}")
	

def triton_bmm_fA_qB_outer(group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh, M, K) float16
	qB is of shape (B, nh, K, N // feat_per_int) int32
	scales is of shape (B, nh, K, G) float16
	zeros is of shape (B, nh, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh, M, N) float16
	"""    
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape 
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K)
	N = qB.shape[-1] * feat_per_int
	qB = qB.reshape(-1, K, qB.shape[-1])
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
	flatten_B = B * nh
	c = torch.empty((flatten_B, M, N), device='cuda', dtype=torch.float16)
	# print(f'M {M} N {N} K {K}')
	grid = lambda META: (
		flatten_B, triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1])
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1])
	if N > K:
		BLOCK_SIZE_N = 128	
		BLOCK_SIZE_K = 32
		num_warps=4  #
	else:
		BLOCK_SIZE_N = 32
		BLOCK_SIZE_K = 128
		num_warps = 2
	num_stages= 7 if K > 64 else 3  #
	qbvm_kernel[grid](
		bits, 
		fA, qB, c,
		scales, zeros,
		M, N, K,
		fA.stride(0), fA.stride(1), fA.stride(2), 
		qB.stride(0), qB.stride(1), qB.stride(2),
		c.stride(0), c.stride(1), c.stride(2),
		scales.stride(0), scales.stride(1), scales.stride(2),
		zeros.stride(0), zeros.stride(1), scales.stride(2),
		group_size, BLOCK_SIZE_N, BLOCK_SIZE_K, 
		num_warps=num_warps, num_stages=num_stages
	)
	return c.view(B, nh, c.shape[-2], c.shape[-1])


def cuda_bmm_fA_qB_outer(group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh, M, K) float16
	qB is of shape (B, nh, K, N // feat_per_int) int32
	scales is of shape (B, nh, K, G) float16
	zeros is of shape (B, nh, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh, M, N) float16
	"""    
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh, M, K = fA.shape 
	nh_kv =  qB.shape[1]
	feat_per_int = 32 // bits
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	N = qB.shape[-1] * feat_per_int
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	# This is based on the possible BLOCK_SIZE_Ks
	# assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	# assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	# assert group_size % 64 == 0, "groupsize must be a multiple of 64, and 128"
	flatten_B = B * nh_kv
	scales = scales.view(flatten_B, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(flatten_B, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	assert bits in [2, 4]
	assert nh % nh_kv == 0
	c = patternkv_gemv.gemv_forward_cuda_outer_dim(fA, qB, scales, zeros, bits, group_size, nh, nh_kv)
	c = c.view(B, nh, c.shape[-2], c.shape[-1])
	return c



# def cuda_bmm_fA_qB_outer_with_base(
#     group_size: int,
#     fA: torch.FloatTensor,            # [B, nh, 1, K]  (q_len=1)
#     qB: torch.IntTensor,              # [B, nh_kv, K, N // feat_per_int]
#     scales: torch.FloatTensor,        # [B, nh_kv, K, G]
#     zeros: torch.FloatTensor,         # [B, nh_kv, K, G]
#     bits: int,                        # 2 or 4
#     centroids: torch.FloatTensor,     # [nh_kv, M, K]
#     assignments: torch.Tensor,        # [B, nh_kv, N] (uint8/uint16/int32均可)
#     nh: int,
#     nh_kv: int,
# ) -> torch.FloatTensor:
#     """
#     计算 logits = Q @ K_residual^T + Q @ C[assign]^T
#     返回 [B, nh, 1, N]
#     """
#     assert len(fA.shape) == 4 and fA.size(2) == 1, "decode下q_len必须为1"
#     B, nh_in, M, K = fA.shape
#     assert nh_in == nh and M == 1
#     assert qB.dim() == 4 and qB.size(0) == B and qB.size(1) == nh_kv
#     assert centroids.shape == (nh_kv, centroids.shape[1], K), "centroids应为[nh_kv, Mc, K]"
#     assert assignments.shape == (B, nh_kv, (qB.shape[-1] * (32 // bits))), "assignments尺寸与N匹配"
#     assert nh % nh_kv == 0

#     feat_per_int = 32 // bits
#     N = qB.shape[-1] * feat_per_int

#     # 视图/转置与现有C++代码的假设保持一致
#     fA_ = fA.view(-1, 1, K).contiguous()  # [B*nh, 1, K]
#     qB_ = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()  # [B*nh_kv, N/pack, K]

#     flatten_B_kv = B * nh_kv
#     scales_ = scales.view(flatten_B_kv, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
#     zeros_  = zeros.view(flatten_B_kv,  zeros.shape[-2],  zeros.shape[-1]).transpose(1, 2).contiguous()

#     # assignments转成最紧凑整型（uint8/uint16），减少带宽
#     if assignments.dtype not in (torch.uint8, torch.int16, torch.int32):
#         assignments_ = assignments.to(torch.int16).contiguous()
#     else:
#         assignments_ = assignments.contiguous()

#     out = patternkv_gemv.gemv_forward_cuda_outer_dim_with_base(
#         fA_, qB_, scales_, zeros_, bits, group_size, nh, nh_kv, centroids.contiguous(), assignments_
#     )
#     # 还原为 [B, nh, 1, N]
#     return out.view(B, nh, 1, N)

def cuda_bmm_fA_qB_outer_with_base(
    group_size: int,
    fA: torch.FloatTensor,            # [B, nh, 1, K] (decode: q_len=1)
    qB: torch.IntTensor,              # [B, nh_kv, K, N // feat_per_int]
    scales: torch.FloatTensor,        # [B, nh_kv, K, G]
    zeros: torch.FloatTensor,         # [B, nh_kv, K, G]
    bits: int,                        # 2 or 4
    centroids: torch.FloatTensor,     # [nh_kv, M, K]
    assignments: torch.Tensor,        # [B, nh_kv, N]  (uint8/uint16/int32)
    nh: int,
    nh_kv: int,
) -> torch.FloatTensor:
    """
    计算 logits = Q @ K_residual^T + Q @ C[assign]^T
    返回 [B, nh, 1, N]
    """
    with profile_range("qk_quantized_history"):
        assert fA.dim() == 4 and fA.size(2) == 1, "decode 路径 q_len 必须为 1"
        B, nh_in, _, K = fA.shape
        assert nh_in == nh

        feat_per_int = 32 // bits
        N = qB.shape[-1] * feat_per_int

        # 展平到 C++ 期望的视图
        fA_ = fA.view(-1, 1, K).contiguous()  # [B*nh, 1, K]
        qB_ = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()  # [B*nh_kv, N/pack, K]

        flatten_B_kv = B * nh_kv
        scales_ = scales.view(flatten_B_kv, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
        zeros_  = zeros.view(flatten_B_kv,  zeros.shape[-2],  zeros.shape[-1]).transpose(1, 2).contiguous()

        # assignments 尽量用紧凑整数类型
        if assignments.dtype not in (torch.uint8, torch.int16, torch.int32):
            assignments_ = assignments.to(torch.int16).contiguous()
        else:
            assignments_ = assignments.contiguous()

        out = patternkv_gemv.gemv_forward_cuda_outer_dim_with_base(
            fA_, qB_, scales_, zeros_, bits, group_size, nh, nh_kv,
            centroids.contiguous(), assignments_
        )  # [B*nh, 1, N]

        return out.view(B, nh, 1, N)

def cuda_attn_v_fused_with_base(
    group_size: int,
    attn_q: torch.Tensor,          # [B, nh, 1, K]   (float16/bfloat16/float32 -> 将强制转 float16)
    vq: torch.Tensor,              # [B, nh_kv, K, OC/pack]  (int32)
    v_scale: torch.Tensor,         # [B, nh_kv, K, OC/group] (float* -> 将强制转 float16)
    v_zero: torch.Tensor,          # [B, nh_kv, K, OC/group] (float* -> 将强制转 float16)
    bits: int,                     # 2 or 4
    v_centroids: torch.Tensor,     # [nh_kv, Mcent, OC]      (float* -> 将强制转 float16)
    v_mask_q: torch.Tensor,        # [B, nh_kv, K]           (uint8/其它 -> 将转 uint8)
    v_idx_q: torch.Tensor,         # [B, nh_kv, K]           (uint8/uint16/int32)
    nh: int,
    nh_kv: int,
    attn_f: torch.Tensor | None = None,   # [B, nh, 1, Lf] (float* -> 将强制转 float16)
    v_full: torch.Tensor | None = None    # [B, nh_kv, Lf, OC] (float* -> 将强制转 float16)
) -> torch.Tensor:
    """
    返回: [B, nh, 1, OC]，dtype 为 float16（调用处可再 cast 回原 dtype）
    """
    # ---------- 形状快速检查 ----------
    assert attn_q.dim() == 4 and attn_q.size(2) == 1, f"attn_q must be [B,nh,1,K], got {attn_q.shape}"
    B, nh_in, _, K = attn_q.shape
    assert nh_in == nh, f"nh mismatch: attn_q has {nh_in}, arg nh={nh}"
    assert v_centroids.dim() == 3, f"v_centroids shape wrong: {v_centroids.shape}"
    OC = v_centroids.size(-1)

    pack = 32 // bits
    assert bits in (2, 4), f"bits must be 2/4, got {bits}"

    # # vq 兼容两种顺序：[B,nh_kv,K,OC/pack] 或 [B,nh_kv,OC/pack,K]
    # if vq.dim() != 4:
    #     raise RuntimeError(f"vq must be 4D, got {vq.shape}")
    # if vq.size(-1) == K and vq.size(-2) == OC // pack:
    #     # vq is [B, nh_kv, OC/pack, K] -> 转成 [B, nh_kv, K, OC/pack]
    #     vq = vq.transpose(-1, -2)
    assert vq.shape == (B, nh_kv, K, OC // pack), f"vq expected [B,{nh_kv},{K},{OC//pack}], got {vq.shape}"
    assert vq.dtype in (torch.int32, torch.int), "vq must be int32"

    # v_scale / v_zero 也需 [B,nh_kv,K,OC/group]
    group = group_size
    assert (OC % group) == 0, f"OC({OC}) not divisible by group_size({group})"
    assert v_scale.shape == (B, nh_kv, K, OC // group), f"v_scale shape mismatch: {v_scale.shape}"
    assert v_zero .shape == (B, nh_kv, K, OC // group), f"v_zero  shape mismatch: {v_zero.shape}"

    # ---------- 强制 dtype: 所有浮点 -> float16 ----------
    attn_q     = attn_q.to(torch.float16).contiguous()
    v_centroids= v_centroids.to(torch.float16).contiguous()
    v_scale    = v_scale.to(torch.float16).contiguous()
    v_zero     = v_zero .to(torch.float16).contiguous()
    if attn_f is not None:
        attn_f = attn_f.to(torch.float16).contiguous()
    if v_full is not None:
        v_full = v_full.to(torch.float16).contiguous()

    # 其它张量转成需要的 dtype/布局
    vq      = vq.contiguous()
    v_mask_q= v_mask_q.to(torch.uint8).contiguous()
    if v_idx_q.dtype not in (torch.uint8, torch.int16, torch.int32):
        v_idx_q = v_idx_q.to(torch.uint8 if v_centroids.size(1) <= 256 else torch.int16)
    v_idx_q = v_idx_q.contiguous()

    # ---------- 展平成 C++ 接口期望的视图 ----------
    # alpha_q: [B*nh, 1, K]
    alpha_q = attn_q.view(-1, 1, K).contiguous()
    # vq_: [B*nh_kv, OC/pack, K]
    vq_     = vq.reshape(-1, K, vq.shape[-1]).transpose(1, 2).contiguous()
    # v_scale_/v_zero_: [B*nh_kv, OC/group, K]
    flat_kv = B * nh_kv
    v_scale_= v_scale.view(flat_kv, v_scale.shape[-2], v_scale.shape[-1]).transpose(1, 2).contiguous()
    v_zero_ = v_zero .view(flat_kv, v_zero .shape[-2], v_zero .shape[-1]).transpose(1, 2).contiguous()

    # alpha_f / v_full：若无则传空
    if (attn_f is None) or (v_full is None):
        alpha_f = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
        v_full_ = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
    else:
        # alpha_f: [B*nh, Lf]
        Lf = attn_f.shape[-1] if attn_f.size(-2) == 1 else attn_f.size(-2)
        assert v_full.size(2) == Lf and v_full.size(-1) == OC, f"v_full shape mismatch: {v_full.shape}, Lf={Lf}, OC={OC}"
        alpha_f = attn_f.view(-1, Lf).contiguous()
        v_full_ = v_full.contiguous()

    # ---------- 调 C++ 扩展 ----------
    out16 = patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base(
        alpha_q, vq_, v_scale_, v_zero_,
        int(bits), int(group_size), int(nh), int(nh_kv),
        v_centroids.contiguous(), v_mask_q, v_idx_q,
        alpha_f, v_full_
    )  # [B*nh, 1, OC]
    # c = patternkv_gemv.gemv_forward_cuda_outer_dim(alpha_q, vq_, v_scale_, v_zero_, int(bits), int(group_size), int(nh), int(nh_kv))

    return out16.view(B, nh, 1, OC) 
    # return c


def _page_token_length(buffer) -> int:
	if hasattr(buffer, "logical_length"):
		return int(buffer.logical_length())
	return int(getattr(buffer, "num_tokens"))


def _require_page_buffer(name: str, buffer, *, tokens: int) -> list[torch.Tensor]:
	if buffer is None or not hasattr(buffer, "pages"):
		raise TypeError(f"{name} must be a FixedPageBuffer-like object with .pages")
	pages = list(buffer.pages)
	if not pages:
		raise RuntimeError(f"{name} has no pages")
	if _page_token_length(buffer) != tokens:
		raise RuntimeError(f"{name} token mismatch: expected {tokens}, got {_page_token_length(buffer)}")
	return pages


def _idx_bytes_for_page_reader(idx_dtype: torch.dtype) -> int:
	if idx_dtype == torch.uint8:
		return 1
	if idx_dtype == torch.int16:
		return 2
	if idx_dtype == torch.int32:
		return 4
	raise RuntimeError("paged V reader requires v_idx_q pages to be uint8, int16, or int32")


def cuda_attn_v_fused_with_base_paged_v2(
	group_size: int,
	attn_q: torch.Tensor,
	vq_pages,
	v_scale_pages,
	v_zero_pages,
	v_centroids: torch.Tensor,
	v_mask_pages,
	v_idx_pages,
	nh: int,
	nh_kv: int,
	attn_f: torch.Tensor | None = None,
	v_full: torch.Tensor | None = None,
	page_tables: dict[str, DevicePageTable] | None = None,
) -> torch.Tensor:
	"""Experimental V2 page-native Value attention reader.

	Page tensors must use the extension-native layout:
	- vq: [B*nh_kv, OC/16, page_size]
	- scale/zero: [B*nh_kv, OC/group_size, page_size]
	- mask/idx: [B, nh_kv, page_size]
	"""
	assert attn_q.dim() == 4 and attn_q.size(2) == 1, f"attn_q must be [B,nh,1,K], got {attn_q.shape}"
	B, nh_in, _, K = attn_q.shape
	assert nh_in == nh, f"nh mismatch: attn_q has {nh_in}, arg nh={nh}"
	assert v_centroids.dim() == 3, f"v_centroids shape wrong: {v_centroids.shape}"
	OC = v_centroids.size(-1)
	if OC % 16 != 0:
		raise RuntimeError("V2 page reader requires OC divisible by 16")
	if OC % group_size != 0:
		raise RuntimeError("OC must be divisible by group_size")

	vq_page_list = _require_page_buffer("vq_pages", vq_pages, tokens=K)
	scale_page_list = _require_page_buffer("v_scale_pages", v_scale_pages, tokens=K)
	zero_page_list = _require_page_buffer("v_zero_pages", v_zero_pages, tokens=K)
	mask_page_list = _require_page_buffer("v_mask_pages", v_mask_pages, tokens=K)
	idx_page_list = _require_page_buffer("v_idx_pages", v_idx_pages, tokens=K)
	page_size = int(getattr(vq_pages, "page_size"))
	for name, buffer in (
		("v_scale_pages", v_scale_pages),
		("v_zero_pages", v_zero_pages),
		("v_mask_pages", v_mask_pages),
		("v_idx_pages", v_idx_pages),
	):
		if int(getattr(buffer, "page_size")) != page_size:
			raise RuntimeError(f"{name} page_size mismatch")

	flat_kv = B * nh_kv
	expected = {
		"vq": (flat_kv, OC // 16, page_size),
		"scale": (flat_kv, OC // group_size, page_size),
		"zero": (flat_kv, OC // group_size, page_size),
		"mask": (B, nh_kv, page_size),
		"idx": (B, nh_kv, page_size),
	}
	first_pages = {
		"vq": vq_page_list[0],
		"scale": scale_page_list[0],
		"zero": zero_page_list[0],
		"mask": mask_page_list[0],
		"idx": idx_page_list[0],
	}
	for name, page in first_pages.items():
		if tuple(page.shape) != expected[name]:
			raise RuntimeError(f"{name} page layout mismatch: expected {expected[name]}, got {tuple(page.shape)}")
		if not page.is_cuda:
			raise RuntimeError(f"{name} pages must be CUDA tensors")
		if not page.is_contiguous():
			raise RuntimeError(f"{name} pages must be contiguous")

	idx_bytes = _idx_bytes_for_page_reader(idx_page_list[0].dtype)
	attn_q = attn_q.to(torch.float16).contiguous()
	v_centroids = v_centroids.to(torch.float16).contiguous()
	if attn_f is not None:
		attn_f = attn_f.to(torch.float16).contiguous()
	if v_full is not None:
		v_full = v_full.to(torch.float16).contiguous()

	if (attn_f is None) or (v_full is None):
		alpha_f = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
		v_full_ = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
	else:
		Lf = attn_f.shape[-1] if attn_f.size(-2) == 1 else attn_f.size(-2)
		assert v_full.size(2) == Lf and v_full.size(-1) == OC, f"v_full shape mismatch: {v_full.shape}, Lf={Lf}, OC={OC}"
		alpha_f = attn_f.view(-1, Lf).contiguous()
		v_full_ = v_full.contiguous()

	table_cache = {} if page_tables is None else page_tables
	vq_table = table_cache.setdefault("vq", DevicePageTable("vq")).refresh(vq_page_list)
	scale_table = table_cache.setdefault("scale", DevicePageTable("scale")).refresh(scale_page_list)
	zero_table = table_cache.setdefault("zero", DevicePageTable("zero")).refresh(zero_page_list)
	mask_table = table_cache.setdefault("mask", DevicePageTable("mask")).refresh(mask_page_list)
	idx_table = table_cache.setdefault("idx", DevicePageTable("idx")).refresh(idx_page_list)

	alpha_q = attn_q.view(-1, 1, K).contiguous()
	_PAGE_V_READER_COUNTERS["page_native_kernel_calls"] += 1
	out16 = patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base_paged_v2(
		alpha_q,
		vq_table,
		scale_table,
		zero_table,
		int(group_size),
		int(nh),
		int(nh_kv),
		v_centroids.contiguous(),
		mask_table,
		idx_table,
		alpha_f,
		v_full_,
		int(K),
		int(page_size),
		int(idx_bytes),
	)
	return out16.view(B, nh, 1, OC)


def _supports_gqa_v2_kernel(
    *,
    group_size: int,
    bits: int,
    attn_q: torch.Tensor,
    v_centroids: torch.Tensor,
    nh: int,
    nh_kv: int,
) -> bool:
    if bits != 2 or nh_kv <= 0 or nh % nh_kv != 0 or (nh // nh_kv) != 4:
        return False
    if attn_q.dim() != 4 or attn_q.size(0) < 1 or attn_q.size(1) != nh:
        return False
    if group_size != 128:
        return False
    return bool(v_centroids.dim() == 3 and v_centroids.size(-1) == 128)


def cuda_attn_v_fused_with_base_gqa_v2(
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
    """Experimental S2B-3 V2 GQA backend with safe production fallback."""
    if not _supports_gqa_v2_kernel(
        group_size=group_size,
        bits=bits,
        attn_q=attn_q,
        v_centroids=v_centroids,
        nh=nh,
        nh_kv=nh_kv,
    ):
        _MIXED_V_COUNTERS["gqa_v2_fallbacks"] += 1
        return cuda_attn_v_fused_with_base(
            group_size,
            attn_q,
            vq,
            v_scale,
            v_zero,
            bits,
            v_centroids,
            v_mask_q,
            v_idx_q,
            nh,
            nh_kv,
            attn_f=attn_f,
            v_full=v_full,
        )

    assert attn_q.dim() == 4 and attn_q.size(2) == 1, f"attn_q must be [B,nh,1,K], got {attn_q.shape}"
    B, nh_in, _, K = attn_q.shape
    assert nh_in == nh, f"nh mismatch: attn_q has {nh_in}, arg nh={nh}"
    OC = v_centroids.size(-1)
    pack = 16
    assert vq.shape == (B, nh_kv, K, OC // pack), f"vq expected [B,{nh_kv},{K},{OC//pack}], got {vq.shape}"
    assert vq.dtype in (torch.int32, torch.int), "vq must be int32"
    assert v_scale.shape == (B, nh_kv, K, OC // group_size), f"v_scale shape mismatch: {v_scale.shape}"
    assert v_zero.shape == (B, nh_kv, K, OC // group_size), f"v_zero shape mismatch: {v_zero.shape}"

    attn_q = attn_q.to(torch.float16).contiguous()
    v_centroids = v_centroids.to(torch.float16).contiguous()
    v_scale = v_scale.to(torch.float16).contiguous()
    v_zero = v_zero.to(torch.float16).contiguous()
    if attn_f is not None:
        attn_f = attn_f.to(torch.float16).contiguous()
    if v_full is not None:
        v_full = v_full.to(torch.float16).contiguous()
    vq = vq.contiguous()
    v_mask_q = v_mask_q.to(torch.uint8).contiguous()
    if v_idx_q.dtype not in (torch.uint8, torch.int16, torch.int32):
        v_idx_q = v_idx_q.to(torch.uint8 if v_centroids.size(1) <= 256 else torch.int16)
    v_idx_q = v_idx_q.contiguous()

    alpha_q = attn_q.view(-1, 1, K).contiguous()
    vq_ = vq.reshape(-1, K, vq.shape[-1]).transpose(1, 2).contiguous()
    flat_kv = B * nh_kv
    v_scale_ = v_scale.view(flat_kv, v_scale.shape[-2], v_scale.shape[-1]).transpose(1, 2).contiguous()
    v_zero_ = v_zero.view(flat_kv, v_zero.shape[-2], v_zero.shape[-1]).transpose(1, 2).contiguous()

    if (attn_f is None) or (v_full is None):
        alpha_f = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
        v_full_ = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
    else:
        Lf = attn_f.shape[-1] if attn_f.size(-2) == 1 else attn_f.size(-2)
        assert v_full.size(2) == Lf and v_full.size(-1) == OC, f"v_full shape mismatch: {v_full.shape}, Lf={Lf}, OC={OC}"
        alpha_f = attn_f.view(-1, Lf).contiguous()
        v_full_ = v_full.contiguous()

    _MIXED_V_COUNTERS["gqa_v2_calls"] += 1
    out16 = patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base_gqa_v2(
        alpha_q,
        vq_,
        v_scale_,
        v_zero_,
        int(group_size),
        int(nh),
        int(nh_kv),
        v_centroids.contiguous(),
        v_mask_q,
        v_idx_q,
        alpha_f,
        v_full_,
    )
    return out16.view(B, nh, 1, OC)


_ATTN_V_DEBUG_MODES = {
    "FULL": 0,
    "RESIDUAL_ONLY": 1,
    "NO_CENTROID_HISTOGRAM": 2,
    "CENTROID_ONLY": 3,
    "WARP_AGG_FULL": 4,
    "PER_WARP_HIST_FULL": 5,
    "NO_TABLE_CONTRIBUTION": 6,
    "LANE0_TABLE_FULL": 7,
}


def cuda_attn_v_fused_with_base_debug(
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
    *,
    debug_mode: str,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Benchmark-only Value attention ablation path.

    Production callers should use cuda_attn_v_fused_with_base. This wrapper is
    intentionally explicit so unset environment variables cannot affect the
    frozen production path.
    """
    if debug_mode not in _ATTN_V_DEBUG_MODES:
        valid = ", ".join(sorted(_ATTN_V_DEBUG_MODES))
        raise ValueError(f"invalid debug_mode={debug_mode!r}; expected one of: {valid}")
    assert attn_q.dim() == 4 and attn_q.size(2) == 1, f"attn_q must be [B,nh,1,K], got {attn_q.shape}"
    B, nh_in, _, K = attn_q.shape
    assert nh_in == nh, f"nh mismatch: attn_q has {nh_in}, arg nh={nh}"
    assert v_centroids.dim() == 3, f"v_centroids shape wrong: {v_centroids.shape}"
    OC = v_centroids.size(-1)

    pack = 32 // bits
    assert bits in (2, 4), f"bits must be 2/4, got {bits}"
    assert vq.shape == (B, nh_kv, K, OC // pack), f"vq expected [B,{nh_kv},{K},{OC//pack}], got {vq.shape}"
    assert vq.dtype in (torch.int32, torch.int), "vq must be int32"

    group = group_size
    assert (OC % group) == 0, f"OC({OC}) not divisible by group_size({group})"
    assert v_scale.shape == (B, nh_kv, K, OC // group), f"v_scale shape mismatch: {v_scale.shape}"
    assert v_zero.shape == (B, nh_kv, K, OC // group), f"v_zero shape mismatch: {v_zero.shape}"

    attn_q = attn_q.to(torch.float16).contiguous()
    v_centroids = v_centroids.to(torch.float16).contiguous()
    v_scale = v_scale.to(torch.float16).contiguous()
    v_zero = v_zero.to(torch.float16).contiguous()
    if attn_f is not None:
        attn_f = attn_f.to(torch.float16).contiguous()
    if v_full is not None:
        v_full = v_full.to(torch.float16).contiguous()

    vq = vq.contiguous()
    v_mask_q = v_mask_q.to(torch.uint8).contiguous()
    if v_idx_q.dtype not in (torch.uint8, torch.int16, torch.int32):
        v_idx_q = v_idx_q.to(torch.uint8 if v_centroids.size(1) <= 256 else torch.int16)
    v_idx_q = v_idx_q.contiguous()

    alpha_q = attn_q.view(-1, 1, K).contiguous()
    vq_ = vq.reshape(-1, K, vq.shape[-1]).transpose(1, 2).contiguous()
    flat_kv = B * nh_kv
    v_scale_ = v_scale.view(flat_kv, v_scale.shape[-2], v_scale.shape[-1]).transpose(1, 2).contiguous()
    v_zero_ = v_zero.view(flat_kv, v_zero.shape[-2], v_zero.shape[-1]).transpose(1, 2).contiguous()

    if (attn_f is None) or (v_full is None):
        alpha_f = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
        v_full_ = torch.empty(0, device=attn_q.device, dtype=attn_q.dtype)
    else:
        Lf = attn_f.shape[-1] if attn_f.size(-2) == 1 else attn_f.size(-2)
        assert v_full.size(2) == Lf and v_full.size(-1) == OC, f"v_full shape mismatch: {v_full.shape}, Lf={Lf}, OC={OC}"
        alpha_f = attn_f.view(-1, Lf).contiguous()
        v_full_ = v_full.contiguous()

    out16 = patternkv_gemv.attn_v_forward_cuda_outer_dim_with_base_debug(
        alpha_q,
        vq_,
        v_scale_,
        v_zero_,
        int(bits),
        int(group_size),
        int(nh),
        int(nh_kv),
        v_centroids.contiguous(),
        v_mask_q,
        v_idx_q,
        alpha_f,
        v_full_,
        int(_ATTN_V_DEBUG_MODES[debug_mode]),
    )
    return out16.view(B, nh, 1, OC)


def cuda_attn_v_mixed_fused_with_base(
    group_size: int,
    attn_q: torch.Tensor,
    vq2: torch.Tensor | None,
    v2_scale: torch.Tensor | None,
    v2_zero: torch.Tensor | None,
    vq4: torch.Tensor | None,
    v4_scale: torch.Tensor | None,
    v4_zero: torch.Tensor | None,
    precision_mask: torch.Tensor,
    v_centroids: torch.Tensor,
    v_mask_q: torch.Tensor,
    v_idx_q: torch.Tensor,
    nh: int,
    nh_kv: int,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compressed-domain mixed V2/V4 Value attention.

    This Phase S1 path keeps the frozen split payload representation. It gathers
    logical-order attention/Pattern metadata into V2 and V4 compact order, then
    runs the existing CUDA fused Value-attention kernel separately for each
    precision and sums the outputs. It never materializes the full historical
    FP16 Value tensor.
    """
    with profile_range("mixed_v_fused_attention"):
        return _cuda_attn_v_mixed_fused_with_base_impl(
            group_size,
            attn_q,
            vq2,
            v2_scale,
            v2_zero,
            vq4,
            v4_scale,
            v4_zero,
            precision_mask,
            v_centroids,
            v_mask_q,
            v_idx_q,
            nh,
            nh_kv,
            attn_f=attn_f,
            v_full=v_full,
        )


def _cuda_attn_v_mixed_fused_with_base_impl(
    group_size: int,
    attn_q: torch.Tensor,
    vq2: torch.Tensor | None,
    v2_scale: torch.Tensor | None,
    v2_zero: torch.Tensor | None,
    vq4: torch.Tensor | None,
    v4_scale: torch.Tensor | None,
    v4_zero: torch.Tensor | None,
    precision_mask: torch.Tensor,
    v_centroids: torch.Tensor,
    v_mask_q: torch.Tensor,
    v_idx_q: torch.Tensor,
    nh: int,
    nh_kv: int,
    attn_f: torch.Tensor | None = None,
    v_full: torch.Tensor | None = None,
) -> torch.Tensor:
    assert attn_q.dim() == 4 and attn_q.size(2) == 1, f"attn_q must be [B,nh,1,T], got {attn_q.shape}"
    B, nh_in, _, total_tokens = attn_q.shape
    assert nh_in == nh, f"nh mismatch: attn_q has {nh_in}, arg nh={nh}"
    if B != 1:
        raise RuntimeError("Phase S1 mixed fused Value attention currently supports B=1, matching frozen mixed cache packing")
    if precision_mask.dim() != 2 or precision_mask.shape != (B, total_tokens):
        raise RuntimeError(f"precision_mask must be [B,T]={B,total_tokens}, got {tuple(precision_mask.shape)}")
    if v_mask_q.shape != (B, nh_kv, total_tokens):
        raise RuntimeError(f"v_mask_q must be [B,nh_kv,T]={B,nh_kv,total_tokens}, got {tuple(v_mask_q.shape)}")
    if v_idx_q.shape != (B, nh_kv, total_tokens):
        raise RuntimeError(f"v_idx_q must be [B,nh_kv,T]={B,nh_kv,total_tokens}, got {tuple(v_idx_q.shape)}")
    if v_centroids is None or v_centroids.dim() != 3:
        raise RuntimeError("fused mixed Value attention requires Pattern centroids")

    with profile_range("mixed_v_mapping_prepare", tokens=int(total_tokens)):
        mask = precision_mask[0].bool()
        low_mask = ~mask
        high_mask = mask
        v2_tokens = int(low_mask.sum().item())
        v4_tokens = int(high_mask.sum().item())
        gqa_backend = patternkv_gqa_v_backend()
    if v2_tokens + v4_tokens != total_tokens:
        raise RuntimeError("precision mask token count mismatch")

    _MIXED_V_COUNTERS["mixed_v_fused_calls"] += 1
    _MIXED_V_COUNTERS["v2_tokens_processed"] += v2_tokens
    _MIXED_V_COUNTERS["v4_tokens_processed"] += v4_tokens
    record_counter("mixed_v_kernel_launches", calls=int(bool(v2_tokens)) + int(bool(v4_tokens)))

    out = None
    full_attached = False
    if v2_tokens:
        if vq2 is None or v2_scale is None or v2_zero is None:
            raise RuntimeError("V2 payload/scale/zero are required for V2 tokens")
        if vq2.shape[2] != v2_tokens:
            raise RuntimeError(f"V2 payload token mismatch: payload={vq2.shape[2]} mask={v2_tokens}")
        with profile_range("mixed_v_layout_prepare_v2", tokens=v2_tokens):
            attn2 = attn_q[..., low_mask].contiguous()
            mask2 = v_mask_q[:, :, low_mask].contiguous()
            idx2 = v_idx_q[:, :, low_mask].contiguous()
        record_temp_allocation("mixed_v_attn2_compact", attn2)
        record_temp_allocation("mixed_v_mask2_compact", mask2)
        record_temp_allocation("mixed_v_idx2_compact", idx2)
        _MIXED_V_COUNTERS["fused_temp_bytes"] += _tensor_bytes(attn2) + _tensor_bytes(mask2) + _tensor_bytes(idx2)
        with profile_range("mixed_v_v2_compute", tokens=v2_tokens):
            v2_fn = cuda_attn_v_fused_with_base_gqa_v2 if gqa_backend == "gqa" else cuda_attn_v_fused_with_base
            out = v2_fn(
                group_size,
                attn2,
                vq2,
                v2_scale,
                v2_zero,
                2,
                v_centroids,
                mask2,
                idx2,
                nh=nh,
                nh_kv=nh_kv,
                attn_f=attn_f,
                v_full=v_full,
            )
        full_attached = attn_f is not None and v_full is not None

    if v4_tokens:
        if vq4 is None or v4_scale is None or v4_zero is None:
            raise RuntimeError("V4 payload/scale/zero are required for V4 tokens")
        if vq4.shape[2] != v4_tokens:
            raise RuntimeError(f"V4 payload token mismatch: payload={vq4.shape[2]} mask={v4_tokens}")
        with profile_range("mixed_v_layout_prepare_v4", tokens=v4_tokens):
            attn4 = attn_q[..., high_mask].contiguous()
            mask4 = v_mask_q[:, :, high_mask].contiguous()
            idx4 = v_idx_q[:, :, high_mask].contiguous()
        record_temp_allocation("mixed_v_attn4_compact", attn4)
        record_temp_allocation("mixed_v_mask4_compact", mask4)
        record_temp_allocation("mixed_v_idx4_compact", idx4)
        _MIXED_V_COUNTERS["fused_temp_bytes"] += _tensor_bytes(attn4) + _tensor_bytes(mask4) + _tensor_bytes(idx4)
        with profile_range("mixed_v_v4_compute", tokens=v4_tokens):
            part4 = cuda_attn_v_fused_with_base(
                group_size,
                attn4,
                vq4,
                v4_scale,
                v4_zero,
                4,
                v_centroids,
                mask4,
                idx4,
                nh=nh,
                nh_kv=nh_kv,
                attn_f=None if full_attached else attn_f,
                v_full=None if full_attached else v_full,
            )
        with profile_range("mixed_v_output_reduce"):
            out = part4 if out is None else out + part4
        full_attached = full_attached or (attn_f is not None and v_full is not None)

    if out is None:
        if attn_f is None or v_full is None:
            raise RuntimeError("mixed fused Value attention received no quantized or full-precision Value tokens")
        out = torch.matmul(attn_f, torch.repeat_interleave(v_full, nh // nh_kv, dim=1))
    return out
