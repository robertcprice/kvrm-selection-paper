#!/usr/bin/env python3
"""
KVRM-GPU Neural Kernel Selector

LLM-native kernel selection for GPU operations:
- Learns optimal kernel variants for given operations
- Considers input shapes, memory patterns, hardware constraints
- Emits keys from verified registry (zero hallucination)
- Adapts to workload patterns over time

This is the core control plane component:
- Takes operation context as input
- Outputs kernel selection as keys (not code)
- Keys map to verified primitives in data plane

Research reference:
"KVRM-GPU: Model-Native Control Plane for GPU Execution"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# KERNEL REGISTRY (VERIFIED PRIMITIVES)
# =============================================================================

class GPUKernelType(Enum):
    """Types of GPU kernels."""
    # GEMM variants
    GEMM_NAIVE = "GEMM_NAIVE"
    GEMM_TILED_16 = "GEMM_TILED_16"
    GEMM_TILED_32 = "GEMM_TILED_32"
    GEMM_TILED_64 = "GEMM_TILED_64"
    GEMM_TILED_128 = "GEMM_TILED_128"
    GEMM_TENSOR_CORE = "GEMM_TENSOR_CORE"
    GEMM_STRIDED = "GEMM_STRIDED"
    GEMM_BATCHED = "GEMM_BATCHED"

    # Attention variants
    ATTENTION_NAIVE = "ATTENTION_NAIVE"
    ATTENTION_FLASH = "ATTENTION_FLASH"
    ATTENTION_MEMORY_EFFICIENT = "ATTENTION_MEMORY_EFFICIENT"
    ATTENTION_BLOCK_SPARSE = "ATTENTION_BLOCK_SPARSE"

    # Softmax variants
    SOFTMAX_STABLE = "SOFTMAX_STABLE"
    SOFTMAX_ONLINE = "SOFTMAX_ONLINE"
    SOFTMAX_BLOCKWISE = "SOFTMAX_BLOCKWISE"

    # Layer norm variants
    LAYERNORM_FUSED = "LAYERNORM_FUSED"
    LAYERNORM_ELEMENTWISE = "LAYERNORM_ELEMENTWISE"

    # Activation variants
    GELU_EXACT = "GELU_EXACT"
    GELU_APPROX = "GELU_APPROX"
    SILU = "SILU"
    RELU = "RELU"


# Valid kernel keys per operation (zero hallucination enforcement)
VALID_KERNELS = {
    'gemm': [
        GPUKernelType.GEMM_NAIVE,
        GPUKernelType.GEMM_TILED_16,
        GPUKernelType.GEMM_TILED_32,
        GPUKernelType.GEMM_TILED_64,
        GPUKernelType.GEMM_TILED_128,
        GPUKernelType.GEMM_TENSOR_CORE,
        GPUKernelType.GEMM_STRIDED,
        GPUKernelType.GEMM_BATCHED,
    ],
    'attention': [
        GPUKernelType.ATTENTION_NAIVE,
        GPUKernelType.ATTENTION_FLASH,
        GPUKernelType.ATTENTION_MEMORY_EFFICIENT,
        GPUKernelType.ATTENTION_BLOCK_SPARSE,
    ],
    'softmax': [
        GPUKernelType.SOFTMAX_STABLE,
        GPUKernelType.SOFTMAX_ONLINE,
        GPUKernelType.SOFTMAX_BLOCKWISE,
    ],
    'layernorm': [
        GPUKernelType.LAYERNORM_FUSED,
        GPUKernelType.LAYERNORM_ELEMENTWISE,
    ],
    'activation': [
        GPUKernelType.GELU_EXACT,
        GPUKernelType.GELU_APPROX,
        GPUKernelType.SILU,
        GPUKernelType.RELU,
    ],
}


# =============================================================================
# CONTEXT ENCODING
# =============================================================================

@dataclass
class KernelSelectionContext:
    """Context for kernel selection."""
    operation: str  # 'gemm', 'attention', etc.
    input_shapes: List[Tuple[int, ...]]
    output_shape: Tuple[int, ...]
    memory_state: Optional[torch.Tensor] = None
    hardware_constraints: Optional[Dict[str, Any]] = None
    recent_kernels: Optional[List[str]] = None


# =============================================================================
# NEURAL KERNEL SELECTOR
# =============================================================================

class NeuralKernelSelector(nn.Module):
    """
    Neural kernel selector that learns to choose optimal kernel variants.

    This is a micro-LLM that:
    1. Encodes operation context (shapes, memory, history)
    2. Predicts best kernel variant
    3. Emits KEY from verified set (not code!)
    4. Learns from execution feedback

    Zero hallucination: Can only output valid kernel keys.
    """

    def __init__(
        self,
        operation: str,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        context_features: int = 32,
    ):
        super().__init__()

        self.operation = operation
        self.valid_kernels = VALID_KERNELS.get(operation, [])
        self.num_kernels = len(self.valid_kernels)

        if self.num_kernels == 0:
            raise ValueError(f"Unknown operation: {operation}")

        # Shape encoder (encodes input/output tensor shapes)
        self.shape_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),  # [M, N, K, aspect_ratios]
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Memory context encoder
        self.memory_encoder = nn.Sequential(
            nn.Linear(context_features, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # History encoder (recent kernel selections)
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Kernel embedding (learned representation of each kernel)
        self.kernel_embeddings = nn.Parameter(
            torch.randn(self.num_kernels, hidden_dim)
        )

        # Cross-attention between context and kernels
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Selection head (outputs logits over valid kernels only)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_kernels),
        )

        # Learnable temperature for Gumbel-Softmax
        self.register_buffer('temperature', torch.tensor(5.0))

    def _encode_shapes(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        """Encode tensor shapes into feature vector."""
        device = self.shape_encoder[0].weight.device

        # For GEMM: encode M, N, K and ratios
        if self.operation == 'gemm' and len(input_shapes) == 2:
            M, K = input_shapes[0][:2]
            K2, N = input_shapes[1][:2]
            features = torch.tensor([
                [M, N, K, M / (N + 1e-8)]
            ], dtype=torch.float32, device=device)
        else:
            # Generic encoding: use max dimensions
            max_dims = [max(s) if s else 0 for s in input_shapes + [output_shape]]
            features = torch.tensor([max_dims[:4]], dtype=torch.float32, device=device)

        # Normalize
        features = features / (features.max() + 1e-8)
        features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        return self.shape_encoder(features)  # [1, hidden_dim]

    def _encode_memory(
        self,
        memory_context: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode memory state."""
        device = self.memory_encoder[0].weight.device

        if memory_context is None:
            memory_context = torch.zeros(
                1, self.memory_encoder[0].in_features,
                device=device
            )

        return self.memory_encoder(memory_context)  # [1, hidden_dim]

    def _encode_history(
        self,
        recent_kernels: Optional[List[str]],
    ) -> torch.Tensor:
        """Encode recent kernel selections."""
        device = self.kernel_embeddings.device
        history_len = 4

        if recent_kernels is None or len(recent_kernels) == 0:
            return torch.zeros(1, history_len, self.kernel_embeddings.shape[1], device=device)

        # Map kernel names to embeddings
        embeddings = []
        for kernel_name in recent_kernels[-history_len:]:
            try:
                idx = self.valid_kernels.index(GPUKernelType(kernel_name))
                emb = self.kernel_embeddings[idx]
            except (ValueError, IndexError):
                emb = torch.zeros(self.kernel_embeddings.shape[1], device=device)
            embeddings.append(emb)

        # Pad to history_len
        while len(embeddings) < history_len:
            embeddings.append(torch.zeros(self.kernel_embeddings.shape[1], device=device))

        history = torch.stack(embeddings).unsqueeze(0)  # [1, history_len, hidden_dim]
        return self.history_encoder(history)  # [1, history_len, hidden_dim]

    def forward(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        memory_context: Optional[torch.Tensor] = None,
        recent_kernels: Optional[List[str]] = None,
        hard: bool = False,
        return_scores: bool = False,
    ) -> Tuple[GPUKernelType, torch.Tensor]:
        """
        Select optimal kernel for given context.

        Args:
            input_shapes: Shapes of input tensors
            output_shape: Shape of output tensor
            memory_context: Optional memory state features
            recent_kernels: Recently selected kernels (for temporal context)
            hard: If True, use hard selection (argmax)
            return_scores: If True, return selection scores

        Returns:
            (selected_kernel, selection_logits)
        """
        # Encode context
        shape_encoded = self._encode_shapes(input_shapes, output_shape)  # [1, hidden_dim]
        memory_encoded = self._encode_memory(memory_context)  # [1, hidden_dim]
        history_encoded = self._encode_history(recent_kernels)  # [1, history_len, hidden_dim]

        # Combine shape and memory context
        context = (shape_encoded + memory_encoded) / 2  # [1, hidden_dim]

        # Cross-attention with kernel embeddings
        kernel_embs = self.kernel_embeddings.unsqueeze(0)  # [1, num_kernels, hidden_dim]
        context_seq = context.unsqueeze(1)  # [1, 1, hidden_dim]

        attended, attn_weights = self.cross_attn(
            query=kernel_embs,
            key=context_seq,
            value=context_seq,
        )  # [1, num_kernels, hidden_dim]

        # Pool and add history context
        pooled = attended.mean(dim=1)  # [1, hidden_dim]
        history_pooled = history_encoded.mean(dim=1)  # [1, hidden_dim]
        combined = (pooled + history_pooled) / 2  # [1, hidden_dim]

        # Select kernel
        logits = self.selector(combined)  # [1, num_kernels]

        if hard or not self.training:
            # Hard selection
            kernel_idx = logits.argmax(dim=-1)
            selected_kernel = self.valid_kernels[kernel_idx.item()]
        else:
            # Soft selection with Gumbel-Softmax
            soft_selection = F.gumbel_softmax(
                logits,
                tau=self.temperature.item(),
                hard=False,
                dim=-1,
            )
            kernel_idx = soft_selection.argmax(dim=-1)
            selected_kernel = self.valid_kernels[kernel_idx.item()]

        if return_scores:
            return selected_kernel, logits

        return selected_kernel, logits

    def get_kernel_rankings(
        self,
        input_shapes: List[Tuple[int, ...]],
        output_shape: Tuple[int, ...],
        memory_context: Optional[torch.Tensor] = None,
        recent_kernels: Optional[List[str]] = None,
    ) -> List[Tuple[GPUKernelType, float]]:
        """
        Get ranked list of kernel options with confidence scores.

        Returns:
            List of (kernel, score) tuples, sorted by score
        """
        _, logits = self.forward(
            input_shapes,
            output_shape,
            memory_context,
            recent_kernels,
            hard=False,
            return_scores=True,
        )

        probs = F.softmax(logits, dim=-1)[0]  # [num_kernels]

        rankings = [
            (kernel, probs[i].item())
            for i, kernel in enumerate(self.valid_kernels)
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)

        return rankings


# =============================================================================
# MULTI-OPERATION SELECTOR
# =============================================================================

class MultiOperationKernelSelector(nn.Module):
    """
    Manages kernel selection for multiple operation types.

    This is the top-level selector that routes to appropriate
    operation-specific selectors.
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()

        self.selectors = nn.ModuleDict({
            op: NeuralKernelSelector(op, hidden_dim=hidden_dim)
            for op in VALID_KERNELS.keys()
        })

    def forward(
        self,
        context: KernelSelectionContext,
        hard: bool = False,
    ) -> GPUKernelType:
        """Select kernel for given context."""
        op = context.operation.lower()

        if op not in self.selectors:
            raise ValueError(f"Unsupported operation: {op}")

        selector = self.selectors[op]
        kernel, _ = selector(
            context.input_shapes,
            context.output_shape,
            context.memory_state,
            context.recent_kernels,
            hard=hard,
        )

        return kernel

    def get_rankings(
        self,
        context: KernelSelectionContext,
    ) -> List[Tuple[GPUKernelType, float]]:
        """Get ranked kernel options for given context."""
        op = context.operation.lower()

        if op not in self.selectors:
            raise ValueError(f"Unsupported operation: {op}")

        selector = self.selectors[op]
        return selector.get_kernel_rankings(
            context.input_shapes,
            context.output_shape,
            context.memory_state,
            context.recent_kernels,
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_kernel_selector(
    operation: str = 'gemm',
    device: str = 'cpu',
) -> NeuralKernelSelector:
    """Create a kernel selector for specific operation."""
    selector = NeuralKernelSelector(operation)
    selector.to(device)
    selector.eval()
    return selector


if __name__ == "__main__":
    print("=" * 60)
    print("KVRM-GPU NEURAL KERNEL SELECTOR")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test GEMM selector
    print("\n" + "-" * 60)
    print("GEMM KERNEL SELECTOR")
    print("-" * 60)

    gemm_selector = NeuralKernelSelector('gemm', hidden_dim=128)
    gemm_selector.to(device)
    print(f"  GEMM kernels available: {len(VALID_KERNELS['gemm'])}")

    # Test selection
    input_shapes = [(128, 128), (128, 128)]
    output_shape = (128, 128)

    selected, logits = gemm_selector(
        input_shapes,
        output_shape,
        hard=True,
        return_scores=True,
    )

    print(f"  Input shapes: {input_shapes}")
    print(f"  Selected kernel: {selected.value}")

    rankings = gemm_selector.get_kernel_rankings(input_shapes, output_shape)
    print(f"\n  Top 3 kernel choices:")
    for kernel, score in rankings[:3]:
        print(f"    {kernel.value:30s}: {score:.3f}")

    print("\n" + "=" * 60)
    print("KERNEL SELECTOR READY")
    print("=" * 60)
