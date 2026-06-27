import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dataloader.dataloader import TextGCNDataLoader, preprocess_adj
from models.bert_gcn.bert_gcn import BERT_GCN
from train.trainer import Trainer
from utils.utils import setup_seed


def load_text_features(config):
    text_dataset_path = os.path.join(config["data_dir"], "text_dataset")
    dataset = config["dataset"]
    target_fn = os.path.join(text_dataset_path, f"{dataset}.txt")
    
    import pandas as pd
    df = pd.read_csv(target_fn, sep="\t", header=None)
    texts = df[1].tolist()
    labels = df[2].tolist()
    
    target2id = {label: idx for idx, label in enumerate(set(labels))}
    labels = [target2id[label] for label in labels]
    
    return texts, labels, len(target2id)


def train_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", args.model, "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", args.model)
    config["log_interval"] = args.log_interval
    
    if args.dataset:
        config["dataset"] = args.dataset
    
    model_type = config.get("model_type", "")
    if model_type.startswith(".") or model_type.startswith(".."):
        config["model_type"] = os.path.abspath(os.path.join(project_root, model_type))
    
    os.makedirs(config["output_path"], exist_ok=True)
    
    setup_seed(config["seed"])
    
    print(f"Starting training process for BERT-GCN...")
    print(f"Dataset: {config['dataset']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"BERT freeze: {config['bert_freeze']}")
    print(f"Log interval: {config['log_interval']}")
    
    data_loader = TextGCNDataLoader(config)
    graph_data = data_loader.load_data()
    
    texts, labels, nclass = load_text_features(config)
    config["nclass"] = nclass
    config["nfeat"] = config.get("nfeat", 768)
    
    data = {
        "adj": graph_data["adj"],
        "features": graph_data["features"],
        "target": labels,
        "nclass": nclass,
        "train_lst": graph_data["train_lst"],
        "val_lst": graph_data["val_lst"],
        "test_lst": graph_data["test_lst"],
        "texts": texts
    }
    
    model = BERT_GCN(config)
    
    trainer = Trainer(model, config, data)
    trainer.fit()


def test_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", args.model, "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", args.model)
    
    if args.dataset:
        config["dataset"] = args.dataset
    
    setup_seed(config["seed"])
    
    data_loader = TextGCNDataLoader(config)
    graph_data = data_loader.load_data()
    
    texts, labels, nclass = load_text_features(config)
    config["nclass"] = nclass
    
    data = {
        "adj": graph_data["adj"],
        "features": graph_data["features"],
        "target": labels,
        "nclass": nclass,
        "train_lst": graph_data["train_lst"],
        "val_lst": graph_data["val_lst"],
        "test_lst": graph_data["test_lst"],
        "texts": texts
    }
    
    model = BERT_GCN(config)
    model_path = args.model_path if args.model_path else os.path.join(config["output_path"], "best_model", "model_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path))
    
    trainer = Trainer(model, config, data)
    test_metrics = trainer._eval_graph(data["test_lst"], prefix="test")
    print(f"\nTest results:")
    print(f"  >>> Test | loss: {test_metrics['test_loss']:.4f} | acc: {test_metrics['acc']:.4f} | f1: {test_metrics['macro_f1']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="BERT_GCN Training and Testing")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test (default: train)")
    parser.add_argument("--model", type=str, default="bert_gcn", help="Model name")
    parser.add_argument("--dataset", type=str, default="mr", help="Dataset (mr, R8, R52, 20NG, Ohsumed)")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training results every N steps (default: 100)")
    parser.add_argument("--model_path", type=str, help="Model checkpoint path (for test mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)


if __name__ == "__main__":
    main()