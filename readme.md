# RLHF - Reinforcement Learning from Human Feedback

A flexible and efficient implementation of Reinforcement Learning from Human Feedback (RLHF) supporting multiple policy gradient algorithms for training language models.

## Overview

This project implements various state-of-the-art policy gradient optimization algorithms including REINFORCE, PPO, GRPO, GSPO, CISPO, SAPO, DAPO, and MaxRL. It's designed for fine-tuning language models using reward signals and is based on the RLHF Book by Nathan Lambert.

**Credits:**
- Original implementation by Zafir Stojanovski (@zafstojano)
- Source: https://github.com/zafstojano/policy-gradients
- License: Apache 2.0
- Adapted for RLHF Book (https://rlhfbook.com)

## Code Structure

```
├── train.py              # Main training script - entry point for running experiments
├── config.py             # Configuration classes and settings management
├── grpo.yaml            # Example configuration file (GRPO algorithm)
├── rollout.py           # RolloutEngine for generating experiences during training
├── Buffer.py            # Experience buffer and replay buffer management
├── loss.py              # Policy gradient loss function implementations
├── utils.py             # Utility functions (model loading, dataset creation, etc.)
└── __init__.py          # Package initialization
```

### File Descriptions

- **train.py**: Main training loop that orchestrates data loading, model initialization, rollouts, and optimization steps
- **config.py**: Pydantic-based configuration system supporting both YAML files and programmatic configuration
- **rollout.py**: Handles generation of training experiences, reward computation, and advantage estimation
- **Buffer.py**: Data structures for storing experiences and managing replay buffers
- **loss.py**: Implements multiple policy gradient loss functions (REINFORCE, PPO, GRPO, GSPO, CISPO, SAPO, DAPO, MaxRL)
- **utils.py**: Helper functions for model loading, dataset management, logging, and evaluation

## Requirements

### System Requirements
- Python 3.8+
- CUDA-capable GPU (for efficient training)
- 16+ GB VRAM (depending on model size and batch configuration)

### Python Dependencies

```
torch>=2.0.0
transformers>=4.35.0
pydantic>=2.0.0
pyyaml>=6.0
wandb>=0.15.0
reasoning_gym>=0.1.0
tensordict>=0.1.0
rich>=13.0.0
```

## Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd RLHF
```

2. **Install dependencies:**
```bash
pip install torch transformers pydantic pyyaml wandb reasoning_gym tensordict rich
```

3. **Verify installation:**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Configuration

### Configuration Format

Configuration uses YAML files with the following structure:

```yaml
data:
  size: 3000                          # Total dataset size
  specs:
    - name: spell_backward            # Dataset name
      weight: 1                       # Weight in data mixture
      config:                         # Dataset-specific config
        min_word_len: 3
        max_word_len: 10

loss: grpo                            # Loss function: reinforce, rloo, ppo, grpo, drgrpo, gspo, cispo, sapo, dapo, maxrl
model_name: Qwen/Qwen3-1.7B          # HuggingFace model identifier

# Clipping parameters (GRPO, DrGRPO, GSPO, CISPO, PPO, DAPO)
clip_eps_lo: 0.2
clip_eps_hi: 0.2

# KL penalty configuration
beta: 0.0                             # KL coefficient (0 = disabled, >0 to enable)
kl_estimator: kl3                     # KL estimator: kl1, kl2, or kl3

# Generation parameters
temperature: 0.6                      # Sampling temperature
top_p: 0.95                           # Top-p (nucleus) sampling
top_k: 20                             # Top-k sampling
min_p: 0.0                            # Minimum probability
max_new_tokens: 512                   # Max generation length

# Training parameters
lr: 5e-6                              # Learning rate
prompts_per_step: 4                   # Prompts per training step
num_rollouts: 8                       # Rollouts per prompt
train_batch_size: 2                   # Training batch size
batch_acc: 4                          # Gradient accumulation steps
max_norm: 1.0                         # Gradient clipping norm
seed: 42                              # Random seed

# Device configuration
model_device_id: 0                    # GPU ID for policy model
ref_model_device_id: 0                # GPU ID for reference model
val_model_device_id: 0                # GPU ID for value model

# Logging
wandb_project: rlhf-book              # Weights & Biases project
wandb_run_name: grpo_spell_backwards   # Weights & Biases run name
```

## Running the Training

### Basic Usage

```bash
# Run with default GRPO configuration
python train.py

# Run with custom config file
python train.py --config custom_config.yaml

# Run with specific arguments
python train.py --config grpo.yaml --seed 123 --lr 1e-5
```

### Command-Line Arguments

```bash
python train.py [OPTIONS]

Options:
  --config FILE             Path to YAML configuration file (default: grpo.yaml)
  --seed INT               Random seed (default: from config)
  --lr FLOAT               Learning rate (default: from config)
  --loss {reinforce,rloo,ppo,grpo,drgrpo,gspo,cispo,sapo,dapo,maxrl}
                           Loss function (default: from config)
  --model_name STRING      HuggingFace model ID (default: from config)
  --num_rollouts INT       Rollouts per prompt (default: from config)
  --train_batch_size INT   Training batch size (default: from config)
  --max_new_tokens INT     Max generation length (default: from config)
```

## Parameter Guide

### Data Parameters

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `data.size` | Total training dataset size | int | 3000 |
| `data.specs[].name` | Dataset identifier | str | "spell_backward" |
| `data.specs[].weight` | Dataset weight in mixture | int | 1 |

### Model Parameters

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `model_name` | HuggingFace model identifier | str | "Qwen/Qwen3-1.7B" |
| `max_new_tokens` | Maximum tokens to generate | int | 512 |
| `temperature` | Sampling temperature (0=deterministic, >1=more random) | float | 0.6 |
| `top_p` | Nucleus sampling probability | float | 0.95 |
| `top_k` | Top-k sampling | int | 20 |
| `min_p` | Minimum probability threshold | float | 0.0 |

### Training Parameters

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `loss` | Policy gradient algorithm | str | "grpo" |
| `lr` | Learning rate | float | 5e-6 |
| `prompts_per_step` | Prompts sampled per training step | int | 4 |
| `num_rollouts` | Rollouts generated per prompt | int | 8 |
| `train_batch_size` | Batch size during training | int | 2 |
| `batch_acc` | Gradient accumulation steps | int | 4 |
| `max_norm` | Maximum gradient norm for clipping | float | 1.0 |
| `seed` | Random seed for reproducibility | int | 42 |

### Algorithm-Specific Parameters

**PPO-specific:**
- `clip_eps_val`: Value function clipping bound (float)
- `gamma`: Discount factor for GAE (float, typical: 0.99)
- `lam`: Lambda for GAE (float, typical: 0.95)
- `vf_coef`: Value function loss coefficient (float)
- `val_model_device_id`: GPU for value model (int)

**Clipping-based algorithms (GRPO, DrGRPO, GSPO, CISPO, PPO, DAPO):**
- `clip_eps_lo`: Lower clipping bound (float, e.g., 0.2)
- `clip_eps_hi`: Upper clipping bound (float, e.g., 0.2)

**SAPO-specific:**
- `sapo_temp_pos`: Sigmoid temperature for positive advantages (float)
- `sapo_temp_neg`: Sigmoid temperature for negative advantages (float)

**DAPO-specific:**
- `l_cache`: Cache length for overlong penalty (int)
- `l_max`: Maximum length for penalty (int)
- `accuracy_min_reward`: Minimum correctness reward (float)
- `accuracy_max_reward`: Maximum correctness reward (float)

**KL penalty (REINFORCE, RLOO, GRPO, etc.):**
- `beta`: KL penalty coefficient (float, 0=disabled)
- `kl_estimator`: KL estimator variant ('kl1', 'kl2', or 'kl3')
- `ref_model_device_id`: GPU for reference model (int)

### Device Configuration

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `model_device_id` | GPU ID for policy model | int | 0 |
| `ref_model_device_id` | GPU ID for reference model | int | 0 |
| `val_model_device_id` | GPU ID for value model | int | 0 |

### Logging Parameters

| Parameter | Description | Type | Example |
|-----------|-------------|------|---------|
| `wandb_project` | Weights & Biases project name | str | "rlhf-book" |
| `wandb_run_name` | Weights & Biases run name | str | "grpo_spell_backwards" |

## Usage Examples

### Example 1: Basic GRPO Training

```bash
python train.py --config grpo.yaml
```

This runs training with the default GRPO configuration targeting the spell-backward task.

### Example 2: REINFORCE with KL Penalty

Create `reinforce_kl.yaml`:
```yaml
data:
  size: 2000
  specs:
    - name: spell_backward
      weight: 1

loss: reinforce
model_name: Qwen/Qwen3-1.7B
beta: 0.05
kl_estimator: kl3
lr: 1e-5
prompts_per_step: 8
num_rollouts: 1
train_batch_size: 4
seed: 42
wandb_run_name: reinforce_with_kl
```

Then run:
```bash
python train.py --config reinforce_kl.yaml
```

### Example 3: PPO Training

Create `ppo_config.yaml`:
```yaml
data:
  size: 3000
  specs:
    - name: spell_backward
      weight: 1

loss: ppo
model_name: Qwen/Qwen3-1.7B
clip_eps_val: 0.2
gamma: 0.99
lam: 0.95
vf_coef: 0.1
lr: 5e-6
prompts_per_step: 4
num_rollouts: 4
train_batch_size: 2
batch_acc: 2
seed: 42
```

Then run:
```bash
python train.py --config ppo_config.yaml
```

### Example 4: Multi-GPU Setup

For distributed training across multiple GPUs:
```yaml
model_device_id: 0
ref_model_device_id: 1
val_model_device_id: 2
```

Then run:
```bash
python train.py --config multi_gpu_config.yaml
```

## Training Loop Overview

1. **Initialization**: Load model, tokenizer, and reference model
2. **Rollout**: Generate completions using the policy model
3. **Reward Computation**: Compute rewards from the environment
4. **Advantage Estimation**: Calculate advantages and GAE
5. **Loss Computation**: Calculate policy gradient loss
6. **Optimization**: Update model with gradient descent
7. **Logging**: Log metrics to Weights & Biases

## Supported Loss Functions

| Algorithm | Type | Use Case |
|-----------|------|----------|
| **REINFORCE** | Policy gradient | Simple baseline, high variance |
| **RLOO** | Policy gradient | Reduced variance version of REINFORCE |
| **PPO** | On-policy | Stable, clip-based training |
| **GRPO** | On-policy | Group-based policy optimization |
| **DrGRPO** | On-policy | Distributed GRPO variant |
| **GSPO** | On-policy | General stochastic policy optimization |
| **CISPO** | On-policy | Clipped importance sampling |
| **SAPO** | On-policy | Sigmoid-based advantage |
| **DAPO** | On-policy | Dynamic advantage policy optimization |
| **MaxRL** | On-policy | Maximum reinforcement learning |

## Performance Tips

1. **Batch Accumulation**: Use `batch_acc > 1` to increase effective batch size without GPU memory constraints
2. **Gradient Checkpointing**: Automatically enabled to reduce memory usage
3. **KL Penalty**: Set `beta > 0` to prevent model drift from reference
4. **Num Rollouts**: Increase `num_rollouts` for better advantage estimation (only for GRPO/RLOO)
5. **Learning Rate**: Start with lower learning rates (1e-6 to 1e-5) for fine-tuning

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `train_batch_size`, increase `batch_acc`, or reduce `max_new_tokens` |
| Poor training convergence | Reduce `lr`, increase `num_rollouts`, set `beta > 0` for KL penalty |
| Model diverges | Lower `lr`, enable KL penalty (`beta > 0`), reduce `max_norm` |
| Slow training | Use GPU with higher VRAM, increase `prompts_per_step` for better throughput |

## License

Apache 2.0 - See original source for details

## References

- [RLHF Book](https://rlhfbook.com)
- [Original Repository](https://github.com/zafstojano/policy-gradients)
- [Reasoning Gym](https://github.com/open-thoughts/reasoning_gym)

## Contributing

Contributions are welcome! Please ensure code follows the existing style and includes appropriate tests.
