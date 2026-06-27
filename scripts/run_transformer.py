import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dataloader.dataloader import DataLoader
from models.transformer.transformer import Transformer
from train.trainer import Trainer
from utils.utils import setup_seed


def build_vocab(texts, max_vocab_size=10000):
    from collections import Counter
    
    word_counts = Counter()
    for text in texts:
        tokens = text.split()
        word_counts.update(tokens)
    
    vocab = ['[UNK]', '[PAD]']
    for word, _ in word_counts.most_common(max_vocab_size - 2):
        vocab.append(word)
    
    stoi = {word: idx for idx, word in enumerate(vocab)}
    return vocab, stoi


def tokenize(text, stoi, max_seq_length):
    tokens = text.split()[:max_seq_length]
    ids = [stoi.get(token, 0) for token in tokens]
    
    padding = [1] * (max_seq_length - len(ids))
    ids += padding
    
    return ids


def train_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", args.model, "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", "transformer")
    config["max_seq_length"] = args.max_seq_length
    config["log_interval"] = args.log_interval
    
    if args.data_source:
        config["data_source"] = args.data_source
    
    os.makedirs(config["output_path"], exist_ok=True)
    
    setup_seed(config["seed"])
    
    print(f"Starting training process for Transformer...")
    print(f"Data source: {config['data_source']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Log interval: {config['log_interval']}")
    print(f"Max sequence length: {config['max_seq_length']}")
    print(f"Number of encoder layers: {config.get('num_encoder', 3)}")
    print(f"Model dimension: {config.get('dim_model', 256)}")
    
    data_loader = DataLoader(config)
    train_contents, train_labels, _ = data_loader.load_data("train")
    val_contents, val_labels, _ = data_loader.load_data("val")
    test_contents, test_labels, _ = data_loader.load_data("test")
    
    import pandas as pd
    train_df = pd.DataFrame({"text": train_contents, "label": train_labels})
    val_df = pd.DataFrame({"text": val_contents, "label": val_labels})
    test_df = pd.DataFrame({"text": test_contents, "label": test_labels})
    
    all_texts = train_contents + val_contents + test_contents
    vocab, stoi = build_vocab(all_texts, max_vocab_size=config.get('vocab_size', 10000))
    config["vocab_size"] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    
    from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
    import torch
    
    def create_dataset(df):
        input_ids = []
        labels = []
        for _, row in df.iterrows():
            ids = tokenize(row['text'], stoi, config['max_seq_length'])
            input_ids.append(ids)
            labels.append(row['label'])
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return TensorDataset(input_ids, labels)
    
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    test_dataset = create_dataset(test_df)
    
    train_loader = TorchDataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    model = Transformer(config)
    
    trainer = Trainer(model, config)
    trainer.fit(train_loader, val_loader, test_loader)


def test_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", args.model, "config.json")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", "transformer")
    
    if args.data_source:
        config["data_source"] = args.data_source
    if args.max_seq_length:
        config["max_seq_length"] = args.max_seq_length
    
    setup_seed(config["seed"])
    
    data_loader = DataLoader(config)
    _, _, test_loader = data_loader.get_dataloaders()
    
    model = Transformer(config)
    model_path = args.model_path if args.model_path else os.path.join(config["output_path"], "best_model", "model_best.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    model.load_state_dict(torch.load(model_path))
    
    trainer = Trainer(model, config)
    test_metrics = trainer._eval_epoch_bert(test_loader)
    print(f"\nTest results:")
    print(f"  >>> Test | loss: {test_metrics['loss']:.4f} | acc: {test_metrics['acc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Transformer Training and Testing")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test (default: train)")
    parser.add_argument("--model", type=str, default="transformer", help="Model name")
    parser.add_argument("--data_source", type=str, help="Data source (AAPR or PeerRead)")
    parser.add_argument("--log_interval", type=int, default=100, help="Log training results every N steps (default: 100)")
    parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    parser.add_argument("--model_path", type=str, help="Model checkpoint path (for test mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)


if __name__ == "__main__":
    main()