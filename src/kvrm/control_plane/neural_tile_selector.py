#!/usr/bin/env python3
"""
NEURAL TILE SIZE SELECTOR - KVRM Research

Learns optimal tile size per attention head for FlashAttention.

This is a Key-Value Response Mapping (KVRM) module:
- Input: Head context (index, sequence length, head dim, memory)
- Output: TILE_SIZE key from verified registry
- Zero hallucination: Can only output valid tile sizes

Research: "Neural Tile Selection for Memory-Efficient Attention"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


# =============================================================================
# TILE SIZE REGISTRY (VERIFIED PRIMITIVES)
# =============================================================================

class TileSize(Enum):
    """Valid tile sizes for FlashAttention."""
    TILE_16 = 16
    TILE_32 = 32
    TILE_64 = 64
    TILE_128 = 128
    TILE_256 = 256
    TILE_512 = 512


# Valid tile size keys (zero hallucination enforcement)
VALID_TILES = [
    TileSize.TILE_16,
    TileSize.TILE_32,
    TileSize.TILE_64,
    TileSize.TILE_128,
    TileSize.TILE_256,
    TileSize.TILE_512,
]


# =============================================================================
# CONTEXT ENCODING
# =============================================================================

@dataclass
class TileSelectionContext:
    """Context for tile size selection."""
    head_idx: int
    num_heads: int
    seq_len: int
    head_dim: int
    memory_available_mb: float
    batch_size: int
    device_memory_bandwidth: Optional[float] = None  # GB/s
    recent_performance: Optional[Dict[int, float]] = None  # head_idx -> speed


# =============================================================================
# NEURAL TILE SIZE SELECTOR
# =============================================================================

class NeuralTileSelector(nn.Module):
    """
    Neural tile size selector that learns optimal tile per head.

    KVRM Pattern:
    1. Encode head context (index, seq_len, head_dim, memory)
    2. Predict best tile size
    3. Emit TILE_SIZE key from verified set
    4. Learn from execution feedback

    Zero hallucination: Can only output valid tile size keys.
    """

    def __init__(
        self,
        num_heads: int = 32,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_tiles = len(VALID_TILES)

        # Head identifier encoding (learnable embedding per head)
        self.head_embedding = nn.Embedding(num_heads, hidden_dim)

        # Context encoder (seq_len, head_dim, memory, batch_size)
        self.context_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Performance history encoder (recent performance per head)
        # First project 1D performance to hidden_dim
        self.history_projector = nn.Linear(1, hidden_dim)
        self.history_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=2,
                dim_feedforward=hidden_dim * 2,
                batch_first=True,
            ),
            num_layers=2,
        )

        # Tile size embedding (learnable representation of each tile)
        self.tile_embeddings = nn.Parameter(
            torch.randn(self.num_tiles, hidden_dim)
        )

        # Cross-attention between head context and tiles
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )

        # Selection head (outputs logits over valid tiles only)
        self.selector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_tiles),
        )

        # Learnable temperature for Gumbel-Softmax
        self.register_buffer('temperature', torch.tensor(5.0))

    def _encode_context(
        self,
        context: TileSelectionContext,
    ) -> torch.Tensor:
        """Encode selection context into feature vector."""
        device = self.context_encoder[0].weight.device

        # Normalize features
        seq_len_norm = context.seq_len / 4096.0  # Normalize to [0, 1]
        head_dim_norm = context.head_dim / 256.0
        memory_norm = context.memory_available_mb / 10000.0  # Assume 10GB max
        batch_norm = context.batch_size / 32.0

        features = torch.tensor([
            [seq_len_norm, head_dim_norm, memory_norm, batch_norm]
        ], dtype=torch.float32, device=device)

        return self.context_encoder(features)  # [1, hidden_dim]

    def _encode_history(
        self,
        recent_performance: Optional[Dict[int, float]],
    ) -> torch.Tensor:
        """Encode performance history."""
        device = self.history_projector.weight.device

        if recent_performance is None or len(recent_performance) == 0:
            # Return zero history with correct dimensions
            return torch.zeros(1, 1, 128, device=device)  # hidden_dim

        # Create history sequence: [perf_head_0, perf_head_1, ..., perf_head_N]
        history = []
        for head_idx in range(self.num_heads):
            perf = recent_performance.get(head_idx, 0.0)
            history.append([perf])

        history_tensor = torch.tensor(
            [history],  # [1, num_heads, 1]
            dtype=torch.float32,
            device=device,
        )

        # Project to hidden_dim
        history_projected = self.history_projector(history_tensor)  # [1, num_heads, hidden_dim]

        # Encode with transformer
        encoded = self.history_encoder(history_projected)  # [1, num_heads, hidden_dim]

        # Aggregate heads
        return encoded.mean(dim=1, keepdim=True)  # [1, 1, hidden_dim]

    def forward(
        self,
        context: TileSelectionContext,
        training: bool = False,
    ) -> TileSize:
        """
        Select tile size for given context.

        Args:
            context: Tile selection context
            training: If True, use Gumbel-Softmax for differentiable sampling

        Returns:
            Selected tile size (enum)
        """
        # Encode head identifier
        device = self.head_embedding.weight.device
        head_idx_tensor = torch.tensor(
            [context.head_idx],
            dtype=torch.long,
            device=device,
        )
        head_encoded = self.head_embedding(head_idx_tensor)  # [1, hidden_dim]

        # Encode context
        context_encoded = self._encode_context(context)  # [1, hidden_dim]

        # Encode history
        history_encoded = self._encode_history(context.recent_performance)  # [1, 1, hidden_dim]
        history_encoded = history_encoded.squeeze(1)  # [1, hidden_dim]

        # Combine encodings
        combined = head_encoded + context_encoded + history_encoded  # [1, hidden_dim]

        # Attend to tile embeddings
        combined_expanded = combined.unsqueeze(1)  # [1, 1, hidden_dim]
        tile_embeddings_expanded = self.tile_embeddings.unsqueeze(0)  # [1, num_tiles, hidden_dim]

        attended, _ = self.cross_attn(
            combined_expanded,
            tile_embeddings_expanded,
            tile_embeddings_expanded,
        )  # [1, 1, hidden_dim]

        attended = attended.squeeze(1)  # [1, hidden_dim]

        # Select tile
        logits = self.selector(attended)  # [1, num_tiles]

        if training:
            # Gumble-Softmax for differentiable sampling
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True)
            tile_idx = probs.argmax(dim=-1)
        else:
            # Greedy selection
            tile_idx = logits.argmax(dim=-1)

        # Return tile size enum
        return VALID_TILES[tile_idx.item()]

    def get_tile_size(self, context: TileSelectionContext) -> int:
        """Get tile size as integer (convenience method)."""
        tile_enum = self.forward(context, training=False)
        return tile_enum.value


# =============================================================================
# TILE SELECTION PROFILER
# =============================================================================

class TileSelectionProfiler:
    """
    Profiles tile size performance to train the neural selector.

    Measures execution time for different tile sizes and contexts.
    """

    def __init__(self):
        self.performance_history: Dict[Tuple[int, int, int], Dict[int, float]] = {}

    def profile_tile_size(
        self,
        head_idx: int,
        seq_len: int,
        head_dim: int,
        tile_size: int,
        attention_fn,
        num_iterations: int = 10,
    ) -> float:
        """
        Profile a specific tile size configuration.

        Returns:
            Throughput (tokens/second)
        """
        import time

        # Warmup
        for _ in range(3):
            _ = attention_fn(tile_size=tile_size)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            _ = attention_fn(tile_size=tile_size)
        elapsed = time.time() - start

        # Calculate throughput
        throughput = seq_len / elapsed

        # Store in history
        key = (head_idx, seq_len, head_dim)
        if key not in self.performance_history:
            self.performance_history[key] = {}

        self.performance_history[key][tile_size] = throughput

        return throughput

    def get_best_tile_size(
        self,
        head_idx: int,
        seq_len: int,
        head_dim: int,
    ) -> int:
        """Get best tile size for given context from history."""
        key = (head_idx, seq_len, head_dim)
        if key not in self.performance_history:
            return 64  # Default

        history = self.performance_history[key]
        return max(history.items(), key=lambda x: x[1])[0]

    def get_training_data(
        self,
        head_idx: int,
        seq_len: int,
        head_dim: int,
    ) -> Dict[int, float]:
        """Get performance data for training neural selector."""
        key = (head_idx, seq_len, head_dim)
        if key not in self.performance_history:
            return {}

        return self.performance_history[key]


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_tile_selector(
    model: NeuralTileSelector,
    profiler: TileSelectionProfiler,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
):
    """
    Train neural tile selector using profiling data.

    Loss: Negative throughput (we want to maximize throughput)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_samples = 0

        for (head_idx, seq_len, head_dim), perf_data in profiler.performance_history.items():
            # Create context
            context = TileSelectionContext(
                head_idx=head_idx,
                num_heads=model.num_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                memory_available_mb=8000,  # Assume 8GB
                batch_size=1,
                recent_performance=None,  # Could add history
            )

            # Get best tile size and throughput
            best_tile = max(perf_data.items(), key=lambda x: x[1])[0]
            best_throughput = perf_data[best_tile]

            # Forward pass
            model.train()
            selected_tile = model(context, training=True)

            # Calculate loss (negative throughput if wrong tile selected)
            if selected_tile.value != best_tile:
                loss = -best_throughput  # Penalize wrong selection
            else:
                loss = 0.0  # Correct selection

            # Backward pass
            optimizer.zero_grad()
            if loss != 0.0:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            num_samples += 1

        if epoch % 10 == 0 and num_samples > 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss / num_samples:.4f}")


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Create neural tile selector
    selector = NeuralTileSelector(num_heads=32)

    # Create context for head 5 with 2048 sequence length
    context = TileSelectionContext(
        head_idx=5,
        num_heads=32,
        seq_len=2048,
        head_dim=128,
        memory_available_mb=8000,
        batch_size=2,
    )

    # Select tile size
    selected_tile = selector(context)
    print(f"Selected tile size: {selected_tile.value}")

    # Get tile size as integer
    tile_size = selector.get_tile_size(context)
    print(f"Tile size for FlashAttention: {tile_size}")
