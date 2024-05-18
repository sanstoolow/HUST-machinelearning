import numpy as np
import cupy as cp
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

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

# 定义MLP模型
class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        self.weights_gpu = cp.zeros((sum(hidden_layers), input_size + sum(hidden_layers)), dtype=cp.float32)
        self.biases_gpu = cp.zeros((1, output_size), dtype=cp.float32)
        
    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.weights_gpu = cp.array(self.weights)
        self.biases_gpu = cp.array(self.biases)
        activations = [X]
        inputs = []
        
        for i, (W, b) in enumerate(zip(self.weights_gpu, self.biases_gpu)):
            z = cp.dot(activations[-1], W) + b
            inputs.append(z)
            a = self.sigmoid(z)
            activations.append(a)
        
        return activations, inputs
    
    def backward(self, activations, inputs, y):
        m = y.shape[0]
        # 注意：这里使用cupy计算梯度，但你需要确保cupy版本支持广播操作
        # 捕获可能的cupy运行时错误
        try:
            dW = cp.zeros_like(self.weights_gpu)
            db = cp.zeros_like(self.biases_gpu)
            
            # Calculate gradients for weights and biases
            for i in range(len(self.weights_gpu) - 1, 0, -1):
                delta = (activations[i] - y) * self.sigmoid_derivative(activations[i])
                dW[i] = cp.dot(delta, activations[i - 1].T)
                db[i] = cp.sum(delta, axis=0, keepdims=True)
            
            return dW, db
        except cp.cuda.runtime.CUDARuntimeError:
            print("GPU error occurred, falling back to CPU processing")
            dW, db = self._calculate_gradients_cpu(activations, inputs, y)
            return dW, db
    
    def _calculate_gradients_cpu(self, activations, inputs, y):
        dW = []
        db = []
        
        for i in range(len(self.weights)):
            dW.append((activations[i] - y) * self.sigmoid_derivative(activations[i]) @ inputs[i].T)
            db.append((activations[i] - y) * self.sigmoid_derivative(activations[i]).sum(axis=0, keepdims=True))
        
        return dW, db
    
    def update_params(self, dW, db, learning_rate):
        self.weights_gpu += -learning_rate * dW
        self.biases_gpu += -learning_rate * db
    
    def fit(self, X, y, epochs, learning_rate):
        self.weights = self.weights_gpu.get()
        self.biases = self.biases_gpu.get()
        
        for epoch in range(epochs):
            self.forward(X)
            dW, db = self.backward(activations, inputs, y)
            self.update_params(dW, db, learning_rate)
            
            if epoch % 100 == 0:
                predictions = self.predict_proba(X)
                loss = -cp.mean(y * cp.log(predictions) + (1 - y) * cp.log(1 - predictions))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
    
    def predict_proba(self, X):
        self.forward(X)
        return self.activations[-1]
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)
# 准备标签
y_train = y_train.values.reshape(-1, 1)
y_val = y_val.values.reshape(-1, 1)

# 将类别标签转换为0/1
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

# 定义MLP模型
input_size = X_train.shape[1]
hidden_layers = [100, 50]
output_size = 1

mlp = MLP(input_size, hidden_layers, output_size)

# 训练模型
mlp.fit(X_train, y_train, epochs=1000, learning_rate=0.01)

# 在验证集上进行预测
y_val_pred = mlp.predict_proba(X_val)
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation AUC:", val_auc)

# 在测试集上进行预测
test_predictions = mlp.predict(X_test_scaled)

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions.ravel()})
submission_df.to_csv('submission_cupy.csv', index=False)

