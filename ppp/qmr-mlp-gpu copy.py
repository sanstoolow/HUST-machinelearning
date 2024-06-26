import numpy as np
import cupy as cp
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 读取数据
train_df = pd.read_csv('ppp/train.csv')
test_df = pd.read_csv('ppp/test.csv')

test_customer_ids = test_df['id']

train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
#
X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

combined_df = pd.concat([X, test_df], keys=['train', 'test'])

encoder = OneHotEncoder()
encoded_features = encoder.fit_transform(combined_df[['Geography', 'Gender']]).toarray()
encoded_feature_names = encoder.get_feature_names_out(['Geography', 'Gender'])

encoded_df = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=combined_df.index)
combined_df = combined_df.drop(['Geography', 'Gender'], axis=1)
combined_df = pd.concat([combined_df, encoded_df], axis=1)

X = combined_df.loc['train'].values
X_test = combined_df.loc['test'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)



# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#CuPy数组
X_train = cp.asarray(X_train, dtype=cp.float32)
y_train = cp.asarray(y_train.values, dtype=cp.float32).reshape(-1, 1)
X_val = cp.asarray(X_val, dtype=cp.float32)
y_val = cp.asarray(y_val.values, dtype=cp.float32).reshape(-1, 1)
X_test = cp.asarray(X_test_scaled, dtype=cp.float32)

def sigmoid(x):
        return 1 / (1 + cp.exp(-x))

class MLP:
    def __init__(self, input_size, hidden_layers, output_size):
        self.weights = []
        self.biases = []
        layer_sizes = [input_size] + hidden_layers + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.weights.append(cp.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(cp.random.randn(layer_sizes[i + 1]))
    
    

    def forward(self, x):
        for weight, bias in zip(self.weights, self.biases):
            x = cp.dot(x, weight) + bias
            x = cp.maximum(0, x) 
        return sigmoid(x)

    def backward(self, x, y, learning_rate):
        grads = []
        for weight, bias in zip(self.weights[::-1], self.biases[::-1]):
            grad = cp.dot(x.T, (y - x) * x * (1 - x))
            grads.append(grad)
            x = cp.dot((y - x) * x * (1 - x), weight.T)
        for weight, grad in zip(self.weights, grads[::-1]):
            weight += learning_rate * grad.T

input_size = X_train.shape[1]
hidden_layers = [100, 50]
output_size = 1
learning_rate = 0.001

model = MLP(input_size, hidden_layers, output_size)

epochs = 100000
for epoch in range(epochs):
    outputs = model.forward(X_train)
    model.backward(outputs, y_train, learning_rate)

    if epoch % 100 == 0:
        val_outputs = model.forward(X_val)
        val_auc = roc_auc_score(cp.asnumpy(y_val), cp.asnumpy(val_outputs))
        print(f'Epoch {epoch}, Val AUC: {val_auc:.4f}')

test_predictions = model.forward(X_test)

submission_df = pd.DataFrame({'id': cp.asnumpy(test_customer_ids), 'Exited': cp.asnumpy(test_predictions).ravel()})
submission_df.to_csv('submission.csv', index=False)