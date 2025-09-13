"""Training script for ORCHID-D4 distributed diffusion system."""

import os
import time
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DeviceProfile:
    """Device characteristics for distributed inference."""
    device_id: str
    device_type: str  # 'pixel8', 'quest3', 'rtx4090', 'a10g', 'a100'
    compute_power: float  # TFLOPS
    memory_gb: float
    bandwidth_gbps: float
    thermal_limit: float  # degrees C
    energy_per_step: float  # Joules


@dataclass
class NetworkProfile:
    """Network characteristics between devices."""
    source: str
    target: str
    rtt_ms: float
    bandwidth_mbps: float
    jitter_ms: float
    packet_loss: float


class PiecewiseKoopmanAdapter(nn.Module):
    """Piece-wise Koopman Adapter for cross-backbone warm-start."""

    def __init__(self, num_pieces: int = 16, feature_dim: int = 1280):
        super().__init__()
        self.num_pieces = num_pieces
        self.feature_dim = feature_dim

        # Dictionary of piece-wise linear maps
        self.koopman_maps = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim, bias=False)
            for _ in range(num_pieces)
        ])

        # Decision tree implemented as learned routing
        self.router = nn.Sequential(
            nn.Linear(feature_dim + 3, 64),  # +3 for depth, channel_group, condition_type
            nn.ReLU(),
            nn.Linear(64, num_pieces),
            nn.Softmax(dim=-1)
        )

        # RLS adaptation parameters
        self.register_buffer('P', torch.eye(feature_dim) * 100)  # Covariance matrix
        self.register_buffer('forgetting_factor', torch.tensor(0.99))

    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """Apply piece-wise Koopman transformation."""
        batch_size = features.shape[0]

        # Route to appropriate Koopman map
        routing_input = torch.cat([features.mean(dim=(2, 3)), metadata], dim=1)
        routing_weights = self.router(routing_input)

        # Apply weighted combination of Koopman maps
        output = torch.zeros_like(features)
        for i in range(self.num_pieces):
            weight = routing_weights[:, i:i + 1].unsqueeze(-1).unsqueeze(-1)
            transformed = self.koopman_maps[i](features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            output += weight * transformed

        return output

    def rls_update(self, features: torch.Tensor, target: torch.Tensor, map_idx: int):
        """Recursive Least Squares update for selected Koopman map."""
        with torch.no_grad():
            x = features.flatten()
            y = target.flatten()

            # RLS update equations
            k = self.P @ x / (self.forgetting_factor + x @ self.P @ x)
            self.P = (self.P - torch.outer(k, x) @ self.P) / self.forgetting_factor

            # Update weights
            error = y - self.koopman_maps[map_idx].weight @ x
            self.koopman_maps[map_idx].weight += torch.outer(error, k)


class HierarchicalRenyiCodec:
    """Hierarchical Rényi-Coded State for efficient transmission."""

    def __init__(self, num_tiers: int = 3):
        self.num_tiers = num_tiers
        self.gmm_prior = self._init_gmm_prior()

    def _init_gmm_prior(self):
        """Initialize Gaussian Mixture Model prior for entropy coding."""
        return {
            'means': torch.randn(8, 4),
            'covs': torch.eye(4).unsqueeze(0).repeat(8, 1, 1),
            'weights': torch.ones(8) / 8
        }

    def encode(self, state: torch.Tensor, tier_level: int) -> bytes:
        """Encode state into hierarchical representation."""
        if tier_level == 1:
            # Tier 1: 96-byte per-tile mean
            tile_means = F.adaptive_avg_pool2d(state, (8, 8))
            return tile_means.cpu().numpy().astype(np.float16).tobytes()
        elif tier_level == 2:
            # Tier 2: 4-bit low-rank basis
            u, s, v = torch.svd(state.view(state.shape[0], -1))
            basis = u[:, :16] @ torch.diag(s[:16])
            quantized = (basis * 15).round().clamp(0, 15).byte()
            return quantized.cpu().numpy().tobytes()
        else:
            # Tier 3: eigen-noise seeds
            seeds = torch.randint(0, 2 ** 32, (4,), dtype=torch.int32)
            return seeds.cpu().numpy().tobytes()

    def decode(self, encoded: bytes, tier_level: int, shape: Tuple) -> torch.Tensor:
        """Decode hierarchical representation back to state."""
        if tier_level == 1:
            tile_means = torch.from_numpy(
                np.frombuffer(encoded, dtype=np.float16).reshape(8, 8)
            ).float()
            return F.interpolate(
                tile_means.unsqueeze(0).unsqueeze(0),
                size=shape[-2:],
                mode='bilinear'
            ).squeeze()
        elif tier_level == 2:
            basis = torch.from_numpy(np.frombuffer(encoded, dtype=np.uint8)).float() / 15
            # Reconstruct approximation (simplified)
            return basis.view(1, 16, 1, 1).repeat(1, shape[1] // 16, shape[2], shape[3])
        else:
            # Generate noise from seeds
            seeds = np.frombuffer(encoded, dtype=np.int32)
            torch.manual_seed(int(seeds[0]))
            return torch.randn(shape) * 0.01


class TileAwareQuantSkip:
    """Tile-Aware Quantization and Skipping scheduler."""

    def __init__(self, tile_size: int = 8):
        self.tile_size = tile_size
        self.skip_probs = {}
        self.bit_widths = {}

    def partition_attention(self, attention_map: torch.Tensor) -> List[Dict]:
        """Partition attention maps into tiles."""
        h, w = attention_map.shape[-2:]
        tiles = []

        for i in range(0, h, self.tile_size):
            for j in range(0, w, self.tile_size):
                tile = attention_map[..., i:i + self.tile_size, j:j + self.tile_size]
                tile_id = f"{i}_{j}"

                # Compute importance score
                importance = tile.abs().mean().item()

                tiles.append({
                    'id': tile_id,
                    'data': tile,
                    'importance': importance,
                    'skip_prob': min(0.8, 1.0 - importance),
                    'bit_width': 16 if importance > 0.7 else (8 if importance > 0.3 else 4)
                })

        return tiles

    def solve_knapsack(self, tiles: List[Dict], memory_budget: int) -> List[Dict]:
        """Solve bi-criteria knapsack for tile selection."""
        # Sort by importance/cost ratio
        for tile in tiles:
            tile['cost'] = tile['bit_width'] * np.prod(tile['data'].shape) / 8
            tile['value'] = tile['importance'] * (1 - tile['skip_prob'])

        tiles.sort(key=lambda x: x['value'] / x['cost'], reverse=True)

        # Greedy packing
        selected = []
        current_memory = 0

        for tile in tiles:
            if current_memory + tile['cost'] <= memory_budget:
                selected.append(tile)
                current_memory += tile['cost']

        return selected


class ChannelSelectiveFairnessNoise:
    """Channel-selective fairness noise injection."""

    def __init__(self, top_p: float = 0.1, noise_sigma: float = 0.05):
        self.top_p = top_p
        self.noise_sigma = noise_sigma
        self.fairness_monitor = self._init_monitor()

    def _init_monitor(self):
        """Initialize chi-square fairness monitor."""
        return {
            'protected_attributes': ['gender', 'skin_tone'],
            'chi2_threshold': 5.99,  # 95% confidence
            'window_size': 100,
            'history': defaultdict(list)
        }

    def compute_renyi_contribution(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Rényi entropy contribution per channel."""
        # Simplified Rényi entropy calculation
        probs = F.softmax(features.mean(dim=(2, 3)), dim=1)
        alpha = 2.0
        renyi = (1 / (1 - alpha)) * torch.log((probs ** alpha).sum(dim=1))
        return renyi

    def detect_bias(self, features: torch.Tensor, attributes: Dict) -> bool:
        """Detect bias using chi-square test."""
        # Simplified bias detection
        for attr in self.fairness_monitor['protected_attributes']:
            if attr in attributes:
                self.fairness_monitor['history'][attr].append(attributes[attr])

                if len(self.fairness_monitor['history'][attr]) >= self.fairness_monitor['window_size']:
                    # Compute chi-square statistic
                    observed = torch.tensor(self.fairness_monitor['history'][attr][-100:])
                    expected = observed.mean()
                    chi2 = ((observed - expected) ** 2 / expected).sum()

                    if chi2 > self.fairness_monitor['chi2_threshold']:
                        return True
        return False

    def inject_noise(self, features: torch.Tensor, attributes: Dict) -> torch.Tensor:
        """Inject noise into top-p offending channels."""
        if self.detect_bias(features, attributes):
            renyi_scores = self.compute_renyi_contribution(features)

            # Select top-p channels
            num_channels = int(features.shape[1] * self.top_p)
            top_channels = torch.topk(renyi_scores, num_channels).indices

            # Inject noise only in selected channels
            noise = torch.zeros_like(features)
            for c in top_channels:
                noise[:, c] = torch.randn_like(features[:, c]) * self.noise_sigma

            features = features + noise
            logger.info(f"Injected fairness noise into {num_channels} channels")

        return features


class StepLevelDynamicOffloader:
    """Step-Level Dynamic Off-Loading scheduler."""

    def __init__(self, devices: List[DeviceProfile], networks: List[NetworkProfile]):
        self.devices = {d.device_id: d for d in devices}
        self.networks = self._build_network_graph(networks)
        self.current_assignments = {}
        self.profiling_data = defaultdict(list)

    def _build_network_graph(self, networks: List[NetworkProfile]) -> Dict:
        """Build network connectivity graph."""
        graph = defaultdict(dict)
        for net in networks:
            graph[net.source][net.target] = net
            graph[net.target][net.source] = net  # Assume symmetric
        return graph

    def solve_mip(self, num_steps: int, num_tiles: int,
                  current_conditions: Dict) -> Dict[Tuple[int, int], str]:
        """Solve Mixed-Integer Program for step/tile assignment."""
        # Simplified greedy assignment based on current conditions
        assignments = {}

        for step in range(num_steps):
            for tile in range(num_tiles):
                # Compute cost for each device
                costs = {}
                for device_id, device in self.devices.items():
                    # Latency cost
                    compute_cost = 1.0 / device.compute_power

                    # Network cost (if not local)
                    network_cost = 0
                    if device_id != 'pixel8':
                        if 'pixel8' in self.networks and device_id in self.networks['pixel8']:
                            network_cost = self.networks['pixel8'][device_id].rtt_ms / 1000

                    # Thermal cost
                    thermal_cost = 0
                    if device.device_type == 'pixel8':
                        current_temp = current_conditions.get('pixel8_temp', 35)
                        if current_temp > 36:
                            thermal_cost = (current_temp - 36) * 0.1

                    # Energy cost
                    energy_cost = device.energy_per_step * 0.01

                    # Privacy regularizer
                    privacy_cost = 0.05 if device_id in ['a10g', 'rtx4090'] else 0

                    costs[device_id] = compute_cost + network_cost + thermal_cost + energy_cost + privacy_cost

                # Assign to minimum cost device
                best_device = min(costs, key=costs.get)
                assignments[(step, tile)] = best_device

        return assignments

    def profile_execution(self, step: int, tile: int, device: str, latency: float):
        """Record profiling data for online adaptation."""
        self.profiling_data[device].append({
            'step': step,
            'tile': tile,
            'latency': latency,
            'timestamp': time.time()
        })


class ORCHIDModel(nn.Module):
    """Main ORCHID-D4 model combining all components."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

        # Initialize components
        self.pka = PiecewiseKoopmanAdapter(
            num_pieces=config['pka']['num_pieces'],
            feature_dim=config['pka']['feature_dim']
        )

        self.hrcs = HierarchicalRenyiCodec(
            num_tiers=config['hrcs']['num_tiers']
        )

        self.taqs = TileAwareQuantSkip(
            tile_size=config['taqs']['tile_size']
        )

        self.cf_noise = ChannelSelectiveFairnessNoise(
            top_p=config['cf_noise']['top_p'],
            noise_sigma=config['cf_noise']['noise_sigma']
        )

        # Device and network profiles
        self.devices = self._init_devices()
        self.networks = self._init_networks()

        self.sldo = StepLevelDynamicOffloader(
            devices=self.devices,
            networks=self.networks
        )

        # Metrics tracking
        self.metrics = defaultdict(list)

    def _init_devices(self) -> List[DeviceProfile]:
        """Initialize device profiles."""
        return [
            DeviceProfile('pixel8', 'pixel8', 2.0, 12, 100, 38, 0.5),
            DeviceProfile('quest3', 'quest3', 1.5, 8, 50, 45, 0.3),
            DeviceProfile('rtx4090', 'rtx4090', 82.6, 24, 1000, 85, 2.0),
            DeviceProfile('a10g', 'a10g', 31.2, 24, 600, 80, 1.5),
            DeviceProfile('a100', 'a100', 19.5, 80, 2000, 85, 3.0),
        ]

    def _init_networks(self) -> List[NetworkProfile]:
        """Initialize network profiles."""
        return [
            NetworkProfile('pixel8', 'quest3', 40, 100, 5, 0.01),
            NetworkProfile('pixel8', 'rtx4090', 8, 1000, 2, 0.001),
            NetworkProfile('pixel8', 'a10g', 120, 500, 10, 0.02),
            NetworkProfile('quest3', 'rtx4090', 15, 800, 3, 0.005),
            NetworkProfile('quest3', 'a10g', 130, 400, 12, 0.025),
            NetworkProfile('rtx4090', 'a10g', 5, 10000, 1, 0.0001),
        ]

    def forward(self, latents: torch.Tensor, prompt_embeds: torch.Tensor,
                timesteps: torch.Tensor, attributes: Optional[Dict] = None) -> torch.Tensor:
        """Forward pass with distributed orchestration."""
        batch_size = latents.shape[0]
        num_steps = len(timesteps)

        # Current system conditions
        current_conditions = {
            'pixel8_temp': 35.0 + torch.randn(1).item() * 3,  # Simulated temperature
            'network_rtt': {net.target: net.rtt_ms for net in self.networks if net.source == 'pixel8'}
        }

        # Solve MIP for step/tile assignment
        h, w = latents.shape[-2] // 32, latents.shape[-1] // 32  # 32x32 tiles
        num_tiles = h * w
        assignments = self.sldo.solve_mip(num_steps, num_tiles, current_conditions)

        # Track metrics
        device_usage = defaultdict(int)
        transmitted_bytes = 0

        # Process each step
        for step_idx, t in enumerate(timesteps):
            step_start = time.time()

            # Tile-wise processing
            tile_outputs = []
            for tile_idx in range(num_tiles):
                tile_i = tile_idx // w
                tile_j = tile_idx % w

                # Extract tile
                tile_latent = latents[:, :, tile_i * 32:(tile_i + 1) * 32, tile_j * 32:(tile_j + 1) * 32]

                # Get assigned device
                device = assignments.get((step_idx, tile_idx), 'pixel8')
                device_usage[device] += 1

                # Encode for transmission if remote
                if device != 'pixel8':
                    tier_level = 1 if device == 'quest3' else 2  # Lower tier for slower links
                    encoded = self.hrcs.encode(tile_latent, tier_level)
                    transmitted_bytes += len(encoded)
                    tile_latent = self.hrcs.decode(encoded, tier_level, tile_latent.shape)

                # Apply TAQS
                if hasattr(self, 'attention_maps') and step_idx < len(self.attention_maps):
                    tiles_data = self.taqs.partition_attention(self.attention_maps[step_idx])
                    selected_tiles = self.taqs.solve_knapsack(tiles_data, memory_budget=1024 * 1024)  # 1MB

                    # Skip or quantize based on selection
                    if tile_idx not in [int(t['id'].split('_')[0]) * w + int(t['id'].split('_')[1])
                                        for t in selected_tiles]:
                        tile_latent = tile_latent * 0.1  # Heavily reduce contribution

                # Process tile (simplified - would be actual diffusion step on device)
                tile_output = tile_latent * 0.99  # Placeholder for actual processing

                # Apply CF-Noise if needed
                if attributes is not None:
                    tile_output = self.cf_noise.inject_noise(tile_output, attributes)

                tile_outputs.append(tile_output)

                # Profile execution
                self.sldo.profile_execution(step_idx, tile_idx, device, time.time() - step_start)

            # Reconstruct full latent
            reconstructed = torch.zeros_like(latents)
            for tile_idx, tile_output in enumerate(tile_outputs):
                tile_i = tile_idx // w
                tile_j = tile_idx % w
                reconstructed[:, :, tile_i * 32:(tile_i + 1) * 32, tile_j * 32:(tile_j + 1) * 32] = tile_output

            latents = reconstructed

            # Apply PKA for warm-start (between steps)
            metadata = torch.tensor([step_idx / num_steps, 0.5, 0.5]).unsqueeze(0).repeat(batch_size, 1)
            if hasattr(latents, 'device'):
                metadata = metadata.to(latents.device)
            latents = self.pka(latents, metadata)

            # Track step metrics
            step_latency = time.time() - step_start
            self.metrics['step_latency'].append(step_latency)

        # Record session metrics
        self.metrics['device_usage'] = dict(device_usage)
        self.metrics['transmitted_kb'] = transmitted_bytes / 1024
        self.metrics['total_steps'] = num_steps

        return latents


def train_orchid(config: Dict, output_dir: Path):
    """Main training function for ORCHID-D4."""
    logger.info("Initializing ORCHID-D4 training...")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Initialize model
    model = ORCHIDModel(config).to(device)

    # ------------------------------------------------------------------
    #  FIX: Cast learning_rate to float to avoid string-related TypeError
    # ------------------------------------------------------------------
    try:
        learning_rate = float(config['training']['learning_rate'])
    except (TypeError, ValueError):
        raise ValueError("training.learning_rate must be numeric (int/float or numeric string)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Simulated training loop (actual training would use real diffusion models)
    num_epochs = int(config['training']['num_epochs'])
    batch_size = int(config['training']['batch_size'])

    training_metrics = []

    for epoch in range(num_epochs):
        epoch_metrics = {
            'epoch': epoch,
            'latencies': [],
            'transmitted_kb': [],
            'device_usage': defaultdict(int)
        }

        # Simulate batches
        num_batches = int(config['training'].get('num_batches', 10))
        for batch_idx in range(num_batches):
            # Generate dummy data (in real implementation, would load actual images)
            latents = torch.randn(batch_size, 4, 128, 128).to(device)
            prompt_embeds = torch.randn(batch_size, 77, 1024).to(device)
            timesteps = torch.linspace(1000, 0, config['inference']['num_steps']).to(device)

            # Simulate attributes for fairness testing
            attributes = {
                'gender': torch.randint(0, 2, (batch_size,)).tolist(),
                'skin_tone': torch.randint(0, 5, (batch_size,)).tolist()
            }

            # Forward pass
            optimizer.zero_grad()
            output = model(latents, prompt_embeds, timesteps, attributes)

            # Compute loss (simplified)
            loss = F.mse_loss(output, torch.randn_like(output))

            # Backward pass
            loss.backward()
            optimizer.step()

            # Collect metrics
            if hasattr(model, 'metrics'):
                epoch_metrics['latencies'].extend(model.metrics['step_latency'])
                epoch_metrics['transmitted_kb'].append(model.metrics.get('transmitted_kb', 0))
                for device_name, count in model.metrics.get('device_usage', {}).items():
                    epoch_metrics['device_usage'][device_name] += count

            if batch_idx % 5 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Aggregate epoch metrics
        epoch_metrics['avg_latency'] = np.mean(epoch_metrics['latencies']) if epoch_metrics['latencies'] else 0
        epoch_metrics['total_transmitted_kb'] = sum(epoch_metrics['transmitted_kb'])
        training_metrics.append(epoch_metrics)

        # Save checkpoint
        if (epoch + 1) % config['training'].get('save_every', 5) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch + 1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': training_metrics
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = output_dir / 'orchid_final.pt'
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Save training metrics
    metrics_path = output_dir / 'training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2, default=str)
    logger.info(f"Saved training metrics to {metrics_path}")

    return model, training_metrics


def main(config_path: str):
    """Main entry point for training."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run training
    model, metrics = train_orchid(config, output_dir)

    # Print summary
    print("\n" + "=" * 50)
    print("ORCHID-D4 Training Complete")
    print("=" * 50)
    print(f"Final model saved to: {output_dir / 'orchid_final.pt'}")
    print(f"Training metrics saved to: {output_dir / 'training_metrics.json'}")

    if metrics:
        last_epoch = metrics[-1]
        print(f"\nFinal Epoch Metrics:")
        print(f"  Average Latency: {last_epoch.get('avg_latency', 0):.3f}s")
        print(f"  Total Transmitted: {last_epoch.get('total_transmitted_kb', 0):.1f} KB")
        print(f"  Device Usage: {dict(last_epoch.get('device_usage', {}))}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main("config/full_experiment.yaml")