import os
import json
import torch
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .dataset import DGCBERTDataset


def preprocess_adj(adj, is_sparse=False):
    adj = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    if is_sparse:
        adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    else:
        adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    
    return torch.FloatTensor(adj)

class SimpleVocab:
    def __init__(self, specials=['[UNK]', '[PAD]']):
        self.stoi = {}
        self.itos = []
        for special in specials:
            if special not in self.stoi:
                self.stoi[special] = len(self.itos)
                self.itos.append(special)
        self.default_index = self.stoi['[UNK]']
    
    def __len__(self):
        return len(self.itos)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)
    
    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    
    def set_default_index(self, index):
        self.default_index = index


class TextGCNDataLoader:
    def __init__(self, config):
        self.config = config
        self.graph_path = config.get("graph_path", os.path.join(config.get("data_dir", "data"), "graph"))
        self.text_dataset_path = config.get("text_dataset_path", os.path.join(config.get("data_dir", "data"), "text_dataset"))
        self.dataset = config.get("dataset", "mr")
        self.val_ratio = config.get("val_ratio", 0.1)
        self.seed = config.get("seed", 42)
        
        self.adj = None
        self.features = None
        self.target = None
        self.nclass = None
        self.nfeat_dim = None
        self.train_lst = None
        self.val_lst = None
        self.test_lst = None
    
    def get_train_test(self, target_fn):
        train_lst = []
        test_lst = []
        with open(target_fn, 'r', encoding='utf-8') as fin:
            for idx, item in enumerate(fin):
                parts = item.split("\t")
                if len(parts) > 1 and parts[1] in {"train", "training", "20news-bydate-train"}:
                    train_lst.append(idx)
                else:
                    test_lst.append(idx)
        return train_lst, test_lst
    
    def load_data(self):
        cache_dir = os.path.join(self.text_dataset_path, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{self.dataset}_data.pkl")
        
        if os.path.exists(cache_path):
            print(f"[TextGCNDataLoader] Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print("[TextGCNDataLoader] Cached data loaded successfully!")
            return data
        
        print("[TextGCNDataLoader] Processing graph data...")
        with tqdm(total=5, desc="Loading data", unit="step") as pbar:
            graph = nx.read_weighted_edgelist(
                os.path.join(self.graph_path, f"{self.dataset}.txt"),
                nodetype=int
            )
            pbar.update(1)
            
            self.nfeat_dim = graph.number_of_nodes()
            adj = nx.adjacency_matrix(
                graph,
                nodelist=list(range(self.nfeat_dim)),
                weight='weight'
            )
            pbar.update(1)
            
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            self.adj = preprocess_adj(adj, is_sparse=True)
            pbar.update(1)
            
            row = list(range(self.nfeat_dim))
            col = list(range(self.nfeat_dim))
            value = [1.] * self.nfeat_dim
            shape = (self.nfeat_dim, self.nfeat_dim)
            indices = torch.from_numpy(np.vstack((row, col)).astype(np.int64))
            values = torch.FloatTensor(value)
            self.features = torch.sparse.FloatTensor(indices, values, torch.Size(shape))
            pbar.update(1)
            
            target_fn = os.path.join(self.text_dataset_path, f"{self.dataset}.txt")
            target = np.array(pd.read_csv(target_fn, sep="\t", header=None)[2])
            target2id = {label: idx for idx, label in enumerate(set(target))}
            self.target = [target2id[label] for label in target]
            self.nclass = len(target2id)
            
            self.train_lst, self.test_lst = self.get_train_test(target_fn)
            self.train_lst, self.val_lst = train_test_split(
                self.train_lst,
                test_size=self.val_ratio,
                shuffle=True,
                random_state=self.seed
            )
            pbar.update(1)
        
        self.config["nfeat"] = self.nfeat_dim
        self.config["nclass"] = self.nclass
        
        data = {
            "adj": self.adj,
            "features": self.features,
            "target": self.target,
            "nclass": self.nclass,
            "nfeat_dim": self.nfeat_dim,
            "train_lst": self.train_lst,
            "val_lst": self.val_lst,
            "test_lst": self.test_lst
        }
        
        print(f"[TextGCNDataLoader] Saving data to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print("[TextGCNDataLoader] Data cached successfully!")
        
        return data


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.data_dir = os.path.join(config["data_dir"], config["data_source"])
        self.max_seq_length = config["max_seq_length"]
        self.batch_size = config["batch_size"]
        self.vocab = None
        self.tokenizer = None
        self.num_class = config["num_class"]
        
    def load_data(self, phase):
        content_path = os.path.join(self.data_dir, f"{phase}_contents.list")
        label_path = os.path.join(self.data_dir, f"{phase}_labels.list")
        index_path = os.path.join(self.data_dir, f"{phase}_indexes.list")
        
        with open(content_path, 'r', encoding='utf-8') as f:
            contents = [line.strip() for line in f.readlines()]
        
        with open(label_path, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f.readlines()]
        
        with open(index_path, 'r', encoding='utf-8') as f:
            indexes = [int(line.strip()) for line in f.readlines()]
        
        return contents, labels, indexes
    
    def build_vocab(self, train_contents):
        vocab_path = os.path.join(self.data_dir, "vocab.pth")
        
        if os.path.exists(vocab_path):
            print(f"[DataLoader] Loading vocabulary from {vocab_path}")
            self.vocab = torch.load(vocab_path, weights_only=False)
        else:
            print(f"[DataLoader] Building vocabulary from {len(train_contents)} documents...")
            self.vocab = SimpleVocab(specials=['[UNK]', '[PAD]'])
            for content in tqdm(train_contents, desc="Building vocab", unit="doc"):
                tokens = self.tokenizer.tokenize(content)
                self.vocab.add_tokens(tokens)
            self.vocab.set_default_index(self.vocab['[UNK]'])
            torch.save(self.vocab, vocab_path)
            print(f"[DataLoader] Vocabulary saved to {vocab_path} (size: {len(self.vocab)})")
        
        return self.vocab
    
    def get_dataloaders(self):
        label_map_path = os.path.join(self.data_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r', encoding='utf-8') as f:
                label_map = json.load(f)
            self.num_class = len(label_map)
        
        cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"_{self.max_seq_length}"
        
        train_cache_path = os.path.join(cache_dir, f"train{cache_suffix}.pth")
        val_cache_path = os.path.join(cache_dir, f"val{cache_suffix}.pth")
        test_cache_path = os.path.join(cache_dir, f"test{cache_suffix}.pth")
        
        train_contents, train_labels, train_indexes = self.load_data("train")
        val_contents, val_labels, val_indexes = self.load_data("val")
        test_contents, test_labels, test_indexes = self.load_data("test")
        
        model_type = self.config["model_type"]
        if model_type.startswith(".") or model_type.startswith(".."):
            model_type = os.path.abspath(os.path.join(self.data_dir, "..", model_type))
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_type)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(model_type, local_files_only=True)
        self.build_vocab(train_contents)
        
        print(f"[DataLoader] Processing training data...")
        if os.path.exists(train_cache_path):
            print(f"[DataLoader] Loading training cache from {train_cache_path}")
            train_dataset = torch.load(train_cache_path, weights_only=False)
        else:
            train_dataset = DGCBERTDataset(
                train_contents, train_labels, train_indexes,
                self.vocab, self.tokenizer, self.max_seq_length
            )
            torch.save(train_dataset, train_cache_path)
            print(f"[DataLoader] Training cache saved to {train_cache_path}")
        
        print(f"[DataLoader] Processing validation data...")
        if os.path.exists(val_cache_path):
            print(f"[DataLoader] Loading validation cache from {val_cache_path}")
            val_dataset = torch.load(val_cache_path, weights_only=False)
        else:
            val_dataset = DGCBERTDataset(
                val_contents, val_labels, val_indexes,
                self.vocab, self.tokenizer, self.max_seq_length
            )
            torch.save(val_dataset, val_cache_path)
            print(f"[DataLoader] Validation cache saved to {val_cache_path}")
        
        print(f"[DataLoader] Processing test data...")
        if os.path.exists(test_cache_path):
            print(f"[DataLoader] Loading test cache from {test_cache_path}")
            test_dataset = torch.load(test_cache_path, weights_only=False)
        else:
            test_dataset = DGCBERTDataset(
                test_contents, test_labels, test_indexes,
                self.vocab, self.tokenizer, self.max_seq_length
            )
            torch.save(test_dataset, test_cache_path)
            print(f"[DataLoader] Test cache saved to {test_cache_path}")
        
        print(f"[DataLoader] Creating DataLoaders...")
        train_loader = TorchDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"[DataLoader] Done! Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader