import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml

class MLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),              # input BN

            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)          # logits
        )

    def forward(self, x):
        return self.net(x)

class BLS(nn.Module):
    def __init__(self, in_dim, num_classes, feature_nodes=20, enhancement_nodes=40):
        super().__init__()
        # 特征节点
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, feature_nodes),
            nn.Tanh()
        )
        
        # 增强节点
        self.enhancement_layer = nn.Sequential(
            nn.Linear(feature_nodes, enhancement_nodes),
            nn.ReLU()
        )
        
        # 输出层
        self.output_layer = nn.Linear(feature_nodes + enhancement_nodes, num_classes)
        
    def forward(self, x):
        features = self.feature_layer(x)
        enhancements = self.enhancement_layer(features)
        
        # 连接特征和增强节点
        combined = torch.cat((features, enhancements), dim=1)
        return self.output_layer(combined)

class AttentionMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4, 
            dropout=0.2,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 添加虚拟序列维度 (batch_size, seq_len=1, features)
        x = x.unsqueeze(1)  
        
        # 特征嵌入
        embedded = self.embedding(x)
        
        # 自注意力
        attn_output, _ = self.attention(embedded, embedded, embedded)
        attn_output = attn_output.squeeze(1)  # 移除序列维度
        
        return self.classifier(attn_output)
    
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-3, save_path="best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best = np.inf
        self.best_path = save_path
        self.stopped = False

    def step(self, val_loss, model):
        improved = (self.best - val_loss) > self.min_delta
        if improved:
            self.best = val_loss
            self.wait = 0
            torch.save(model.state_dict(), self.best_path)  # checkpoint
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True
        return improved

# 训练函数 (包含Early Stopping和学习率调度)
def train_model(model, train_loader, valid_loader, criterion, optimizer, 
               scheduler, early_stopping, epochs=200, verbose=1):
    """
    训练模型并返回历史记录
    :param model: 要训练的模型
    :param train_loader: 训练数据加载器
    :param valid_loader: 验证数据加载器
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param scheduler: 学习率调度器
    :param early_stopping: 早停回调
    :param epochs: 最大训练轮数
    :param verbose: 日志详细程度 (0: 无, 1: 每epoch, 2: 每batch)
    :return: 包含训练历史的字典
    """
    history = {
        'epoch': [],
        'lr': [],
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    print(f"开始训练，使用设备: {device}")
    print(f"模型结构:\n{model}")
    print(f"训练样本: {len(train_loader.dataset)} | 验证样本: {len(valid_loader.dataset)}")
    
    for epoch in range(epochs):
        # ===== 训练阶段 =====
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 计算指标
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # 详细日志
            if verbose > 1 and (batch_idx % 50 == 0 or batch_idx == len(train_loader)-1):
                batch_acc = 100. * correct / total
                print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {batch_acc:.2f}%')
        
        # 计算epoch训练指标
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # ===== 验证阶段 =====
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        history['epoch'].append(epoch+1)
        
        # 打印epoch摘要
        if verbose > 0:
            print(f'\nEpoch {epoch+1}/{epochs} | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f}')
        
        # Early Stopping检查
        improved = early_stopping.step(val_loss, model)
        if improved:
            print(f'🌟 Validation loss improved to {val_loss:.4f}, saving model...')
        
        # 检查是否早停
        if early_stopping.stopped:
            print(f'\n🚨 Early stopping triggered at epoch {epoch+1}')
            print(f'Best validation loss: {early_stopping.best:.4f}')
            break
    
    # 训练结束总结
    print(f"\n训练完成! 最佳验证损失: {early_stopping.best:.4f}")
    if early_stopping.stopped:
        print(f"在 {len(history['epoch'])} 个epoch后早停")
    else:
        print(f"完成所有 {epochs} 个epoch")
    
    # 加载最佳模型权重
    model.load_state_dict(torch.load(early_stopping.best_path))
    print(f"已加载最佳模型权重: {early_stopping.best_path}")
    
    return history



# 获取设备数量
device_count = torch_directml.device_count()
print(f"可用的 DirectML 设备数量: {device_count}")

# 列出所有设备信息
print("设备详细信息:")
amd_index = None
for i in range(device_count):
    # 注意：device_name() 需要整数索引作为参数
    name = torch_directml.device_name(i)
    print(f"设备 {i}: {name}")
    
    # 检查是否为 AMD 设备
    if "Radeon" in name or "AMD" in name or "MI60" in name:
        amd_index = i
        print("   --> 检测到 AMD 设备!")

# 选择 AMD 设备
if amd_index is not None:
    # 使用索引创建设备
    device = torch_directml.device(amd_index)
    print(f"\n✅ 已选择 AMD 设备 [索引 {amd_index}]: {torch_directml.device_name(amd_index)}")
else:
    device = torch_directml.device(0)
    print(f"\n⚠️ 未找到 AMD 设备，使用默认设备 [索引 0]: {torch_directml.device_name(0)}")

# 执行计算测试
print("\n运行计算测试...")
try:
    size = 10000
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.mm(a, b)
    c_cpu = c.cpu()  # 将结果复制回 CPU
    
    print(f"测试成功！结果形状: {c.shape}")
    print(f"使用的设备: {a.device}")
    
    # 获取设备索引
    device_index = device.index if device.index is not None else 0
    print(f"设备名称: {torch_directml.device_name(device_index)}")
    
except Exception as e:
    print(f"❌ 计算测试失败: {str(e)}")


# filename="StressLevelDataset.csv"
# 特征选择数据
filename="dataset_top10_features.csv"

num_classes = 3  # 假设有3个类别
StressLevelDataset = pd.read_csv(filename)
X=StressLevelDataset.copy()
y=X.pop('stress_level')

torch.manual_seed(42); np.random.seed(42)

# 确保 y 为 0..num_classes-1 的 long tensor
if not np.issubdtype(y.dtype, np.integer):
    y = y.astype('category').cat.codes
else:
    # 万一不是从 0 开始，做一个映射
    classes = np.sort(y.unique())
    mapping = {c:i for i,c in enumerate(classes)}
    y = y.map(mapping).astype(int)

# 1. 首先分出测试集（15%）
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# 2. 然后分出训练集（70%）和验证集（15%）
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.15/(1-0.15)≈0.176
)

# 转换为Tensor
X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long)

X_val_tensor = torch.tensor(X_valid.to_numpy(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_valid.to_numpy(), dtype=torch.long)

X_test_tensor = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)

#创建Dataset和DataLoader
batch_size = 128

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

input_shape = X_train.shape[1]
print(f"输入形状: {input_shape}")

# 3. 创建类似Keras的Sequential模型
# model = MLP(in_dim=input_shape, num_classes=num_classes)
# model = BLS(in_dim=input_shape, num_classes=num_classes)
model = AttentionMLP(in_dim=input_shape, num_classes=num_classes)

# 4. 设备设置
model = model.to(device)
# 多分类使用CrossEntropyLoss (包含softmax)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3, verbose=True
)
early_stopping = EarlyStopping(patience=10, min_delta=0.001, save_path="best_model.pt")

epochs=200 
patience=10
min_delta=0.001



# 6. 训练模型 (对应Keras的model.fit)
history = train_model(
    model, train_loader, val_loader, 
    criterion, optimizer,scheduler, early_stopping,
    epochs=200, 
    verbose=1   
)

# 7. 创建历史数据框 (对应Keras的history_df)
history_df = pd.DataFrame(history)
history_df.to_csv('training_history.csv', index=False)
print("训练历史已保存到 training_history.csv")


# ========= 8. 用最佳模型评估测试集，并导出结果 =========
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import json

def evaluate_on_loader(model, loader, criterion, device):
    model.eval()
    test_loss = 0.0
    total = 0
    correct = 0

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)                    # (B, C) logits
            loss = criterion(outputs, labels)

            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            probs = torch.softmax(outputs, dim=1)

            all_logits.append(outputs.detach().cpu())
            all_probs.append(probs.detach().cpu())
            all_preds.append(predicted.detach().cpu())
            all_labels.append(labels.detach().cpu())

    test_loss /= total
    test_acc = 100.0 * correct / total

    all_logits = torch.cat(all_logits).numpy()
    all_probs  = torch.cat(all_probs).numpy()
    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    return {
        "loss": test_loss,
        "acc": test_acc,
        "logits": all_logits,
        "probs": all_probs,
        "preds": all_preds,
        "labels": all_labels,
    }

# 如果你在训练末尾已经 load_state_dict(best) 了，当前 model 就是最佳模型；
# 若想“保险”可取消注释下一行从磁盘再加载一次
model.load_state_dict(torch.load("best_model.pt", map_location=device))

test_out = evaluate_on_loader(model, test_loader, criterion, device)

# —— 计算更细指标 —— 
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    test_out["labels"], test_out["preds"], average="macro", zero_division=0
)
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
    test_out["labels"], test_out["preds"], average="weighted", zero_division=0
)

# 多分类 AUC（若 num_classes>2 用 OVR/宏平均；概率必须有）
auc_macro_ovr = None
try:
    if test_out["probs"].shape[1] >= 2:
        auc_macro_ovr = roc_auc_score(
            test_out["labels"],
            test_out["probs"],
            multi_class="ovr",
            average="macro"
        )
except Exception:
    pass  # 某些数据/类别不平衡会报错，跳过即可

# —— 导出 1) 指标汇总（test_summary.csv）——
summary_row = {
    "test_loss": round(test_out["loss"], 6),
    "test_acc(%)": round(test_out["acc"], 4),
    "precision_macro": round(precision_macro, 6),
    "recall_macro": round(recall_macro, 6),
    "f1_macro": round(f1_macro, 6),
    "precision_weighted": round(precision_weighted, 6),
    "recall_weighted": round(recall_weighted, 6),
    "f1_weighted": round(f1_weighted, 6),
    "auc_macro_ovr": (None if auc_macro_ovr is None else round(auc_macro_ovr, 6)),
    "num_classes": int(num_classes),
    "num_test_examples": int(len(test_loader.dataset)),
}
pd.DataFrame([summary_row]).to_csv("test_summary.csv", index=False)
print("已保存测试集指标 -> test_summary.csv")

# —— 导出 2) 预测明细（test_results.csv）——
# 将概率列命名为 prob_class_i
prob_cols = [f"prob_class_{i}" for i in range(test_out["probs"].shape[1])]
df_probs = pd.DataFrame(test_out["probs"], columns=prob_cols)
df_preds = pd.DataFrame({
    "y_true": test_out["labels"],
    "y_pred": test_out["preds"],
    "correct": (test_out["labels"] == test_out["preds"]).astype(int),
})

# 若想把原始特征一起导出，拼上 X_test
X_test_reset = X_test.reset_index(drop=True)
test_results_df = pd.concat([X_test_reset, df_preds, df_probs], axis=1)
test_results_df.to_csv("test_results.csv", index=False)
print("已保存测试集预测明细 -> test_results.csv")

# —— 导出 3) 混淆矩阵（confusion_matrix.csv）——
cm = confusion_matrix(test_out["labels"], test_out["preds"])
cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in range(num_classes)],
                        columns=[f"pred_{i}" for i in range(num_classes)])
cm_df.to_csv("confusion_matrix.csv")
print("已保存混淆矩阵 -> confusion_matrix.csv")

# —— 额外：分类报告（文本），便于快速查看 —— 
report = classification_report(test_out["labels"], test_out["preds"], digits=4, zero_division=0)
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)
print("已保存分类报告 -> classification_report.txt")

print("\n【测试集结果摘要】")
print(pd.DataFrame([summary_row]))

# 收集模型预测结果
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 创建错误分析DataFrame
error_df = pd.DataFrame({
    'true_label': y_test,
    'predicted_label': all_preds,
    'is_correct': [1 if t == p else 0 for t, p in zip(y_test, all_preds)]
})

# 添加原始特征
error_df = pd.concat([error_df, X_test.reset_index(drop=True)], axis=1)

# 分析错误样本
incorrect_samples = error_df[error_df['is_correct'] == 0]
print(f"错误样本比例: {len(incorrect_samples)/len(error_df):.2%}")

# 检查错误样本的特征分布
plt.figure(figsize=(12, 6))
sns.boxplot(data=incorrect_samples.drop(columns=['true_label', 'predicted_label', 'is_correct']))
plt.title('wrong predictions feature distribution')
plt.xticks(rotation=45)
plt.savefig('error_boxplog.png')
plt.show()

# 检查特定特征与错误的关系
# for feature in ['most_important_feature1', 'most_important_feature2']:
#     plt.figure(figsize=(10, 4))
#     sns.kdeplot(data=error_df, x=feature, hue='is_correct', common_norm=False)
#     plt.title(f'feature"{feature}"distribution by correctness')
#     plt.show()





