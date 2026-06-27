import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from dataloader.dataloader import DataLoader
from models.scibert_concat.scibert_concat import SciBERT_Concat
from train.trainer import Trainer
from utils.utils import setup_seed, setup_logger


def train_model(args):
    config_path = os.path.join("configs", args.model, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config["output_path"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", args.model)
    
    if args.data_source:
        config["data_source"] = args.data_source
    
    os.makedirs(config["output_path"], exist_ok=True)
    
    logger = setup_logger(config["output_path"])
    setup_seed(config["seed"])
    
    logger.info(f"Starting training process for {config['model_name']}...")
    logger.info(f"Data source: {config['data_source']}")
    logger.info(f"Model type: {config['model_type']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Batch size: {config['batch_size']}")
    logger.info(f"Learning rate: {config['learning_rate']}")
    logger.info(f"Number of layers: {config['num_layers']}")
    
    data_loader = DataLoader(config)
    train_loader, val_loader, test_loader = data_loader.get_dataloaders()
    
    model = SciBERT_Concat(config)
    
    trainer = Trainer(model, config)
    trainer.fit(train_loader, val_loader)
    
    logger.info("Training completed!")


def test_model(args):
    config_path = os.path.join("configs", args.model, "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    config["data_dir"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config["output_path"] = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints", args.model)
    
    if args.data_source:
        config["data_source"] = args.data_source
    
    logger = setup_logger(config["output_path"])
    setup_seed(config["seed"])
    
    data_loader = DataLoader(config)
    _, _, test_loader = data_loader.get_dataloaders()
    
    model = SciBERT_Concat(config)
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
    parser = argparse.ArgumentParser(description="SciBERT_Concat Training and Testing")
    
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="Mode: train or test (default: train)")
    parser.add_argument("--model", type=str, default="scibert_concat", help="Model name")
    parser.add_argument("--data_source", type=str, help="Data source (AAPR or PeerRead)")
    parser.add_argument("--model_path", type=str, help="Model checkpoint path (for test mode)")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)


if __name__ == "__main__":
    main()