import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 读取数据
train_df = pd.read_csv('ppp/train.csv')
test_df = pd.read_csv('ppp/test.csv')

# 保存测试集的客户ID
test_customer_ids = test_df['id']

# 移除无关的特征
train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

# 将特征和标签分离
X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

# 合并训练集和测试集以便进行独热编码
combined_df = pd.concat([X, test_df], keys=['train', 'test'])

# 独热编码
encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(combined_df[['Geography', 'Gender']]).toarray()
encoded_feature_names = encoder.get_feature_names_out(['Geography', 'Gender'])

# 将独热编码特征转换为DataFrame并与其他特征合并
encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=combined_df.index)
combined_df = combined_df.drop(['Geography', 'Gender'], axis=1)
combined_df = pd.concat([combined_df, encoded_df], axis=1)

# 分离处理后的训练集和测试集
X = combined_df.loc['train'].values
X_test = combined_df.loc['test'].values

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量并移动到GPU
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float32).reshape(-1, 1).to(device)
X_test = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = []
        layer_sizes = [input_size] + hidden_layers
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(layer_sizes[-1], output_size))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# 模型参数
input_size = X_train.shape[1]
hidden_layers = [100, 50]
output_size = 1
learning_rate = 0.001

# 初始化模型、损失函数和优化器
model = MLP(input_size, hidden_layers, output_size).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
epochs = 100000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()
            val_auc = roc_auc_score(y_val.cpu().numpy(), val_outputs.cpu().numpy())
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
# 训练模型
# epochs = 100000
# for epoch in range(epochs):
#     model.train()
#     running_loss = 0.0
#     for i, (inputs, labels) in enumerate(train_loader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     if epoch % 100 == 0:
#         model.eval()
#         with torch.no_grad():
#             val_outputs = torch.sigmoid(model(X_val))
#             val_loss = criterion(val_outputs, y_val).item()
#             val_auc = roc_auc_score(y_val.cpu().numpy(), val_outputs.cpu().numpy())
#             print(f'Epoch {epoch}, Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')


# 在测试集上进行预测
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).cpu().numpy()

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions.ravel()})
submission_df.to_csv('submission.csv', index=False)
