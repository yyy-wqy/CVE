# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 20:39:11 2025

@author: hanly2
"""

import pickle

with open('dataset.pickle', 'rb') as file:
# 从文件中加载数据
    dataset = pickle.load(file)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import random
import numpy as np



import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
import random
from collections import defaultdict

# 示例数据结构（需替换为真实数据）
train_data = dataset[:15000]

test_data = dataset[15000:]

# 配置参数
MODEL_NAME = 'microsoft/codebert-base'
BATCH_SIZE = 16
MAX_LENGTH = 512
EPOCHS = 1
LEARNING_RATE = 2e-5

class CVEDataset(Dataset):
    def __init__(self, data):
        self.cve_groups = self._group_by_cve(data)
        self.all_cves = list(self.cve_groups.keys())
        self.positive_pairs = self._generate_positive_pairs()
        self.negative_pairs = self._generate_negative_pairs()
        self.pairs = self.positive_pairs + self.negative_pairs
        self.labels = [1] * len(self.positive_pairs) + [0] * len(self.negative_pairs)

    def _group_by_cve(self, data):
        groups = defaultdict(list)
        for cve_id, code in data:
            groups[cve_id].append(code)
        return groups

    def _generate_positive_pairs(self):
        pairs = []
        for codes in self.cve_groups.values():
            if len(codes) >= 2:
                # 生成所有可能的正样本对组合
                for i in range(len(codes)):
                    for j in range(i+1, len(codes)):
                        pairs.append((codes[i], codes[j]))
        return pairs

    def _generate_negative_pairs(self):
        pairs = []
        num_neg = len(self.positive_pairs)  # 保持正负样本平衡
        for _ in range(num_neg):
            # 随机选择两个不同的CVE
            cve1, cve2 = random.sample(self.all_cves, 2)
            code1 = random.choice(self.cve_groups[cve1])
            code2 = random.choice(self.cve_groups[cve2])
            pairs.append((code1, code2))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1], self.labels[idx]

# 初始化tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def collate_fn(batch):
    code1, code2, labels = zip(*batch)
    inputs1 = tokenizer(
        code1, 
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    inputs2 = tokenizer(
        code2,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='pt'
    )
    return inputs1, inputs2, torch.FloatTensor(labels)

class DualEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModel.from_pretrained(MODEL_NAME)
        
    def forward(self, inputs1, inputs2):
        emb1 = self.model(**inputs1).last_hidden_state[:, 0, :]
        emb2 = self.model(**inputs2).last_hidden_state[:, 0, :]
        return torch.cosine_similarity(emb1, emb2)

# 准备数据
train_dataset = CVEDataset(train_data)
test_dataset = CVEDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = DualEncoder().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCEWithLogitsLoss()

# 在文件开头添加tqdm导入
from tqdm.auto import tqdm

# 修改后的训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    # 创建进度条
    progress_bar = tqdm(
        train_loader,
        desc=f"Training Epoch {epoch+1}",
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    for batch_idx, (inputs1, inputs2, labels) in enumerate(progress_bar):
        inputs1 = {k: v.to(device) for k, v in inputs1.items()}
        inputs2 = {k: v.to(device) for k, v in inputs2.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs1, inputs2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 更新进度条信息
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    print(f'\nEpoch {epoch+1} | Average Loss: {total_loss/len(train_loader):.4f}')

# 评估函数（保持不变）
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs1, inputs2, labels in dataloader:
            inputs1 = {k: v.to(device) for k, v in inputs1.items()}
            inputs2 = {k: v.to(device) for k, v in inputs2.items()}
            labels = labels.to(device)
            
            outputs = torch.sigmoid(model(inputs1, inputs2))
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# 执行评估
test_acc = evaluate(model, test_loader)
print(f'Test Accuracy: {test_acc:.4f}')