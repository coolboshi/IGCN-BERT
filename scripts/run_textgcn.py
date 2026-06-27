import argparse
import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.dataloader import TextGCNDataLoader
from models.textgcn.textgcn import TextGCN
from train.trainer import Trainer
from utils.utils import setup_seed


def train_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "textgcn", "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", "textgcn")
    config["model_type"] = "textgcn"
    
    if args.dataset:
        config["dataset"] = args.dataset
    if args.log_interval:
        config["log_interval"] = args.log_interval
    
    os.makedirs(config["output_path"], exist_ok=True)
    
    setup_seed(config["seed"])
    
    print(f"Starting TextGCN training for dataset: {config['dataset']}")
    print(f"Hidden dim: {config['nhid']}")
    print(f"Epochs: {config['max_epoch']}")
    print(f"Learning rate: {config['lr']}")
    print(f"Log interval: {config['log_interval']}")
    
    data_loader = TextGCNDataLoader(config)
    data = data_loader.load_data()
    
    model = TextGCN(config)
    
    trainer = Trainer(model, config, data)
    trainer.fit()


def test_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "textgcn", "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", "textgcn")
    config["model_type"] = "textgcn"
    
    if args.dataset:
        config["dataset"] = args.dataset
    
    setup_seed(config["seed"])
    
    data_loader = TextGCNDataLoader(config)
    data = data_loader.load_data()
    
    model = TextGCN(config)
    model_path = args.model_path if args.model_path else os.path.join(config["output_path"], "best_model", "model_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path))
    
    trainer = Trainer(model, config, data)
    test_metrics = trainer._eval_graph(trainer.test_lst, prefix="test")
    print(f"\nTest results:")
    print(f"  >>> Test | loss: {test_metrics['test_loss']:.4f} | acc: {test_metrics['acc']:.4f} | f1: {test_metrics['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="TextGCN Training and Testing")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test (default: train)")
    parser.add_argument("--dataset", type=str, default="mr", help="Dataset name (mr, R52, R8, 20ng, ohsumed)")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training results every N steps (default: 100)")
    parser.add_argument("--model_path", type=str, help="Model checkpoint path (for test mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)


if __name__ == "__main__":
    main()