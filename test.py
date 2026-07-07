from utils import create_dataset
import argparse
from config import Config, load_config
from reasoning_gym.dataset import ProceduralDataset
# import reasoning_gym
# from reasoning_gym.factory import DATASETS

# # Sort and print all operational dataset string tags
# valid_tasks = sorted(list(DATASETS.keys()))
# print(f"Total available tasks: {len(valid_tasks)}\n")

# for task in valid_tasks:
#     print(f" - {task}")


parser = argparse.ArgumentParser(description="Train policy gradient models for RLHF")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()
cfg = load_config(args.config)
data = create_dataset(cfg)
print(f"[bold cyan]Dataset created with {len(data)} samples.[/bold cyan]")
print(f"[bold cyan]First sample: {data[0]}[/bold cyan]")