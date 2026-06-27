import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dataloader.dataloader import DataLoader
from models.bert_cls.bert_cls import BERT_CLS
from train.trainer import Trainer
from utils.utils import setup_seed, setup_logger


def train_model(args):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "configs", "bert_cls", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(project_root, "data")
    config["output_path"] = os.path.join(project_root, "checkpoints", "bert_cls")
    
    if args.data_source:
        config["data_source"] = args.data_source
    
    if args.max_seq_length:
        config["max_seq_length"] = args.max_seq_length
    
    if args.log_interval:
        config["log_interval"] = args.log_interval
    
    model_type = config.get("model_type", "")
    if model_type.startswith(".") or model_type.startswith(".."):
        config["model_type"] = os.path.abspath(os.path.join(project_root, model_type))
    
    os.makedirs(config["output_path"], exist_ok=True)
    
    logger = setup_logger(config["output_path"])
    setup_seed(config["seed"])
    
    logger.info(f"Starting training process for {config['model_name']}...")
    logger.info(f"Data source: {config['data_source']}")
    logger.info(f"Model type: {config['model_type']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    
    data_loader = DataLoader(config)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    
    model = BERT_CLS(config)
    
    trainer = Trainer(model, config)
    trainer.fit(train_loader, val_loader, test_loader)
    
    logger.info("Training completed!")


def test_model(args):
    config_path = os.path.join("configs", "bert_cls", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config["output_path"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", "bert_cls")
    
    if args.data_source:
        config["data_source"] = args.data_source
    
    if args.max_seq_length:
        config["max_seq_length"] = args.max_seq_length
    
    logger = setup_logger(config["output_path"])
    setup_seed(config["seed"])
    
    data_loader = DataLoader(config)
    _, _, test_loader = data_loader.get_dataloaders()
    
    model = BERT_CLS(config)
    model_path = args.model_path if args.model_path else os.path.join(config["output_path"], "best_model", "model_best.pth")
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return
    
    model.load_model(model_path)
    model.eval()
    
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_correct = 0
    total_samples = 0
    
    logger.info(f"Starting testing process...")
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    logger.info(f"Test accuracy: {accuracy:.4f}")


def main():
    import sys
    if len(sys.argv) == 1:
        sys.argv.append("train")
    
    parser = argparse.ArgumentParser(description="BERT_CLS Training and Testing")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    train_parser = subparsers.add_parser("train", help="Train BERT_CLS model")
    train_parser.add_argument("--data_source", type=str, help="Data source (AAPR or PeerRead)")
    train_parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    train_parser.add_argument("--log_interval", type=int, default=100, help="Log training info every N steps (default: 100)")
    
    test_parser = subparsers.add_parser("test", help="Test BERT_CLS model")
    test_parser.add_argument("--data_source", type=str, help="Data source (AAPR or PeerRead)")
    test_parser.add_argument("--model_path", type=str, help="Model checkpoint path")
    test_parser.add_argument("--max_seq_length", type=int, default=256, help="Maximum sequence length (default: 256)")
    
    args = parser.parse_args()
    
    if args.command == "train" or args.command is None:
        train_model(args)
    elif args.command == "test":
        test_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
