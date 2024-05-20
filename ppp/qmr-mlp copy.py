import cupy as cp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score
# from joblib import dump
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

class MLP:
    
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(cp.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(cp.zeros((1, layer_sizes[i + 1])))
        
    def relu(self, x):
        return cp.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(int)
    
    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def score(self, X, y):
        y_pred = self.predict_proba(X)
        return roc_auc_score(y.get(), y_pred.get())

    def forward(self, X):
        activations = [X]
        inputs = []
    
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = cp.dot(activations[-1], W) + b
            inputs.append(z)
            if i == len(self.weights) - 1:  # 如果是最后一层，使用sigmoid函数
                a = self.sigmoid(z)
            else:  # 否则，使用relu函数
                a = self.relu(z)
            activations.append(a)
        
        return activations, inputs
    def backward(self, activations, inputs, y):
        m = y.shape[0]
        deltas = [activations[-1] - y]
        
    # Calculate deltas for all layers
        for i in range(len(self.weights) - 1, 0, -1):
            if i == len(self.weights) - 1:  # 如果是最后一层，使用sigmoid函数的导数
                delta = cp.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(activations[i])
            else:  # 否则，使用relu函数的导数
                delta = cp.dot(deltas[0], self.weights[i].T) * self.relu_derivative(activations[i])
            deltas.insert(0, delta)
        
        dW = []
        db = []
    
    # Calculate gradients for weights and biases
        for i in range(len(self.weights)):
            dW.append(np.dot(activations[i].T, deltas[i]) / m)
            db.append(np.sum(deltas[i], axis=0, keepdims=True) / m)
        
        return dW, db
    def update_params(self, dW, db, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * dW[i]
            self.biases[i] -= learning_rate * db[i]
    
    def fit(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            activations, inputs = self.forward(X)
            dW, db = self.backward(activations, inputs, y)
            self.update_params(dW, db, learning_rate)
            
            if epoch % 100 == 0:
                predictions = self.predict_proba(X)
                loss = -cp.mean(y * np.log(predictions) + (1 - y) * cp.log(1 - predictions))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict_proba(self, X):
        activations, _ = self.forward(X)
        return activations[-1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
    
# 准备标签
y_train = cp.asarray(y_train.values.reshape(-1, 1))  # 使用cp.asarray将pandas数据转换为cupy数组
y_val = cp.asarray(y_val.values.reshape(-1, 1))

# 准备特征
X_train_cp = cp.asarray(X_train)
X_val_cp = cp.asarray(X_val)

# 定义MLP模型
input_size = X_train_cp.shape[1]
hidden_layers = [500, 200]
output_size = 1
mlp = MLP(input_size, hidden_layers, output_size)

# 训练模型
mlp.fit(X_train_cp, y_train, epochs=2000, learning_rate=0.01)

# 在验证集上进行预测
y_val_pred = mlp.predict_proba(X_val_cp)
val_auc = roc_auc_score(y_val.get(), y_val_pred.get())
print("Validation AUC:", val_auc)

# # 使用交叉验证评估模型
# X_scaled_cp = cp.asarray(X_scaled)
# y_cp = cp.asarray(y.values.reshape(-1, 1))
# scores = cross_val_score(mlp, X_scaled_cp, y_cp, cv=5, scoring=make_scorer(roc_auc_score))
# print("Cross-validated AUC:", np.mean(scores))
# 在测试集上进行预测
test_predictions = mlp.predict_proba(X_test_scaled)

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions.ravel()})
submission_df.to_csv('submission.csv', index=False)
