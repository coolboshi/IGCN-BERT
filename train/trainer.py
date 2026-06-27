import os
import csv
import math
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def log(msg):
    print(msg, flush=True)

class Trainer:
    def __init__(self, model, config, data=None):
        self.model = model
        self.config = config
        self.data = data
        self.device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.max_epochs = config.get("epochs", config.get("max_epoch", 20))
        self.learning_rate = config.get("learning_rate", config.get("lr", 1e-3))
        self.weight_decay = config.get("weight_decay", 1e-4)
        self.warmup_ratio = config.get("warmup_ratio", 0.1)
        self.patience = config.get("patience", config.get("early_stopping", 10))
        self.output_path = config.get("output_path", "./checkpoints")
        self.log_interval = config.get("log_interval", 100)
        
        self.best_val_acc = float('-inf')
        self.global_step = 0
        self.early_stop_counter = 0
        self.optimizer = None
        self.scheduler = None
        
        model_name = config.get("model_name", "")
        self.is_graph_model = config.get("model_type") == "textgcn" or model_name == "BERT_GCN"
        
        if self.is_graph_model and data is not None:
            self.adj = data["adj"].to(self.device)
            self.features = data["features"].to(self.device)
            self.target = torch.tensor(data["target"]).long().to(self.device)
            self.train_lst = torch.tensor(data["train_lst"]).long().to(self.device)
            self.val_lst = torch.tensor(data["val_lst"]).long().to(self.device)
            self.test_lst = torch.tensor(data["test_lst"]).long().to(self.device)
            self.texts = data.get("texts", None)
            
            if model_name == "BERT_GCN" and self.texts is not None:
                from transformers import BertTokenizer
                model_type = config.get("model_type", "bert-base-uncased")
                try:
                    tokenizer = BertTokenizer.from_pretrained(model_type)
                except:
                    tokenizer = BertTokenizer.from_pretrained(model_type, local_files_only=True)
                
                max_seq_length = config.get("max_seq_length", 256)
                batch_size = config.get("batch_size", 32)
                
                encoded_inputs = tokenizer(
                    self.texts, 
                    padding='max_length', 
                    truncation=True, 
                    max_length=max_seq_length,
                    return_tensors='pt'
                )
                input_ids = encoded_inputs["input_ids"]
                
                num_docs = input_ids.shape[0]
                bert_features = []
                
                self.model.eval()
                with torch.no_grad():
                    for i in range(0, num_docs, batch_size):
                        batch_input_ids = input_ids[i:i+batch_size].to(self.device)
                        outputs = self.model.bert(batch_input_ids, return_dict=True)
                        batch_features = torch.mean(outputs.last_hidden_state, dim=1)
                        bert_features.append(batch_features.cpu())
                
                self.features = torch.cat(bert_features, dim=0).to(self.device)
                self.model.train()
                
                adj_dense = self.adj.to_dense()
                self.adj = adj_dense[:num_docs, :num_docs].to_sparse()
                self.train_lst = self.train_lst[self.train_lst < num_docs]
                self.val_lst = self.val_lst[self.val_lst < num_docs]
                self.test_lst = self.test_lst[self.test_lst < num_docs]
        
        os.makedirs(self.output_path, exist_ok=True)
        self.log_path = os.path.join(self.output_path, "training_log.csv")
        self._init_csv_log()
    
    def _init_csv_log(self):
        with open(self.log_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "test_loss", "test_acc", "test_f1", "lr"])
    
    def _log_csv(self, row):
        with open(self.log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([row.get(k, "") for k in ["epoch", "step", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "test_loss", "test_acc", "test_f1", "lr"]])
    
    def _build_optimizer(self, total_steps):
        params = list(self.model.named_parameters())
        
        if params and any('bert' in n.lower() for n, _ in params):
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in params if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
            
            warmup_steps = int(total_steps * self.warmup_ratio)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
    
    def _compute_accuracy(self, logits, labels):
        preds = torch.argmax(logits, dim=1)
        return (preds == labels).float().mean()
    
    def fit(self, train_loader=None, val_loader=None, test_loader=None):
        if self.is_graph_model:
            return self._fit_graph()
        else:
            return self._fit_bert(train_loader, val_loader, test_loader)
    
    def _fit_graph(self):
        log(f"训练设备: {self.device}")
        log(f"总步数: {self.max_epochs}, Epochs: {self.max_epochs}")
        
        criterion = nn.CrossEntropyLoss()
        self._build_optimizer(self.max_epochs)
        
        for epoch in range(1, self.max_epochs + 1):
            log(f"\n{'='*50}")
            log(f"Epoch {epoch}/{self.max_epochs}")
            log(f"{'='*50}")
            
            self.model.train()
            self.optimizer.zero_grad()
            
            logits = self.model(self.features, self.adj)
            loss = criterion(logits[self.train_lst], self.target[self.train_lst])
            
            loss.backward()
            self.optimizer.step()
            
            train_acc = self._compute_accuracy(logits[self.train_lst], self.target[self.train_lst]).item()
            log(f"  >>> Train | loss: {loss.item():.4f} | acc: {train_acc:.4f}")
            
            val_metrics = self._eval_graph(self.val_lst, prefix="val")
            log(f"  >>> Val   | loss: {val_metrics['val_loss']:.4f} | acc: {val_metrics['acc']:.4f} | f1: {val_metrics['macro_f1']:.4f}")
            
            test_metrics = self._eval_graph(self.test_lst, prefix="test")
            log(f"  >>> Test  | loss: {test_metrics['test_loss']:.4f} | acc: {test_metrics['acc']:.4f} | f1: {test_metrics['macro_f1']:.4f}")
            
            self._log_csv({
                "epoch": epoch, "step": epoch,
                "train_loss": f"{loss.item():.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_metrics['val_loss']:.4f}",
                "val_acc": f"{val_metrics['acc']:.4f}",
                "val_f1": f"{val_metrics['macro_f1']:.4f}",
                "test_loss": f"{test_metrics['test_loss']:.4f}",
                "test_acc": f"{test_metrics['acc']:.4f}",
                "test_f1": f"{test_metrics['macro_f1']:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            if val_metrics['acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['acc']
                self._save_best_model(val_metrics['acc'])
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                log(f"  早停计数: {self.early_stop_counter}/{self.patience}")
            
            if self.early_stop_counter >= self.patience:
                log(f"  早停触发！{self.patience} 个 epoch 无提升")
                break
        
        log(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.4f}")
        return self.model
    
    def _fit_bert(self, train_loader, val_loader=None, test_loader=None):
        total_steps = len(train_loader) * self.max_epochs
        self._build_optimizer(total_steps)
        log(f"训练设备: {self.device}")
        log(f"总步数: {total_steps}, Epochs: {self.max_epochs}, Batch: {train_loader.batch_size}")
        
        for epoch in range(1, self.max_epochs + 1):
            log(f"\n{'='*50}")
            log(f"Epoch {epoch}/{self.max_epochs}")
            log(f"{'='*50}")
            
            train_metrics = self._train_epoch_bert(train_loader)
            log(f"  >>> Train | loss: {train_metrics['loss']:.4f} | acc: {train_metrics['acc']:.4f}")
            
            if val_loader:
                val_metrics = self._eval_epoch_bert(val_loader)
                log(f"  >>> Val   | loss: {val_metrics['loss']:.4f} | acc: {val_metrics['acc']:.4f}")
                
                if test_loader:
                    test_metrics = self._eval_epoch_bert(test_loader)
                    log(f"  >>> Test  | loss: {test_metrics['loss']:.4f} | acc: {test_metrics['acc']:.4f}")
                else:
                    test_metrics = None
                
                self._log_csv({
                    "epoch": epoch, "step": self.global_step,
                    "train_loss": f"{train_metrics['loss']:.4f}",
                    "train_acc": f"{train_metrics['acc']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.4f}",
                    "val_acc": f"{val_metrics['acc']:.4f}",
                    "val_f1": "",
                    "test_loss": f"{test_metrics['loss']:.4f}" if test_metrics else "",
                    "test_acc": f"{test_metrics['acc']:.4f}" if test_metrics else "",
                    "test_f1": "",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                
                if val_metrics['acc'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['acc']
                    self._save_best_model(val_metrics['acc'])
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    log(f"  早停计数: {self.early_stop_counter}/{self.patience}")
                
                if self.early_stop_counter >= self.patience:
                    log(f"  早停触发！{self.patience} 个 epoch 无提升")
                    break
        
        log(f"\n训练完成！最佳验证准确率: {self.best_val_acc:.4f}")
        return self.model
    
    def _train_epoch_bert(self, train_loader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"  Train", leave=False)
        for batch in pbar:
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device)
            else:
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                attention_mask = None
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            batch_acc = (preds == labels).float().mean().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{batch_acc:.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            if self.global_step % self.log_interval == 0:
                log(f"  Step {self.global_step:5d} | loss: {loss.item():.4f} | "
                    f"acc: {batch_acc:.4f} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e}")
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / max(total_samples, 1)
        return {"loss": avg_loss, "acc": avg_acc}
    
    @torch.no_grad()
    def _eval_epoch_bert(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        for batch in tqdm(val_loader, desc=f"  Val", leave=False):
            if isinstance(batch, dict):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device) if "attention_mask" in batch else None
                labels = batch["labels"].to(self.device)
            else:
                input_ids = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                attention_mask = None
            
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / max(total_samples, 1)
        return {"loss": avg_loss, "acc": accuracy}
    
    @torch.no_grad()
    def _eval_graph(self, indices, prefix="val"):
        self.model.eval()
        logits = self.model(self.features, self.adj)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits[indices], self.target[indices])
        
        preds = torch.argmax(logits[indices], dim=1)
        acc = (preds == self.target[indices]).float().mean().item()
        
        f1 = f1_score(self.target[indices].cpu().numpy(), preds.cpu().numpy(), average='macro')
        
        return {
            f"{prefix}_loss": loss.item(),
            "acc": acc,
            "macro_f1": f1,
        }
    
    def _save_best_model(self, val_acc):
        save_dir = os.path.join(self.output_path, "best_model")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "model_best.pth")
        torch.save(self.model.state_dict(), save_path)
        log(f"  >>> 最佳模型已保存 (val_acc={val_acc:.4f}) -> {save_path}")