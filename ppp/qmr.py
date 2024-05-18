import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
# 读取数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 数据预处理
# 将分类变量转换为数值型
test_customer_ids = test_df['id']
train_df['Gender'] = train_df['Gender'].map({'Male': 1, 'Female': 0})
test_df['Gender'] = test_df['Gender'].map({'Male': 1, 'Female': 0})

# 将Geography转换为数值型
geo_dict = {'France': 0, 'Germany': 1, 'Spain': 2}
train_df['Geography'] = train_df['Geography'].map(geo_dict)
test_df['Geography'] = test_df['Geography'].map(geo_dict)

# 移除无关的特征
train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

# 将特征和标签分离
X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 逻辑回归模型实现，包括L2正则化和Nesterov Momentum
class LogisticRegressionWithSGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization_strength=0.1, momentum_factor=0.9):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization_strength = regularization_strength
        self.momentum_factor = momentum_factor
        self.weights = None
        self.bias = None
        self.weight_momentums = None
        self.bias_momentum = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.weight_momentums = np.zeros(n_features)
        self.bias_momentum = 0

        for _ in range(self.n_iterations):
            # Nesterov Momentum
            weights_next = self.weights - self.learning_rate * (self.weight_momentums * self.momentum_factor
                                                                  + self.regularization_strength * self.weights
                                                                  + np.dot(X.T, (self._sigmoid(np.dot(X, self.weights + self.weight_momentums * self.momentum_factor) + self.bias) - y)) / n_samples)
            bias_next = self.bias - self.learning_rate * (self.bias_momentum * self.momentum_factor
                                                           + self.regularization_strength * self.bias
                                                           + np.sum(self._sigmoid(np.dot(X, self.weights + self.weight_momentums * self.momentum_factor) + self.bias) - y) / n_samples)

            self.weight_momentums = self.weights - weights_next
            self.bias_momentum = self.bias - bias_next

            self.weights = weights_next
            self.bias = bias_next

    def predict(self, X):
        model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

def random_search(X, y, num_iterations=100):
    best_auc = -np.inf
    best_params = {}
    
    for i in range(num_iterations):
        learning_rate = 10 ** np.random.uniform(-3, 0)
        n_iterations = np.random.randint(500, 2000)
        regularization_strength = 10 ** np.random.uniform(-3, 1)
        momentum_factor = np.random.uniform(0.5, 0.99)
        
        model = LogisticRegressionWithSGD(learning_rate=learning_rate, 
                                           n_iterations=n_iterations, 
                                           regularization_strength=regularization_strength, 
                                           momentum_factor=momentum_factor)
        model.fit(X, y)
        
        y_val_pred = model.predict(X_val_scaled)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = {'learning_rate': learning_rate,
                           'n_iterations': n_iterations,
                           'regularization_strength': regularization_strength,
                           'momentum_factor': momentum_factor}
            print(f'Iteration {i + 1}: Best AUC so far: {best_auc:.4f}, Best params: {best_params}')    
    return best_params

# 划分训练集和验证集
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用随机搜索来找到最佳超参数组合
best_params = random_search(X_train_scaled, y_train)

# 使用最佳超参数组合来训练模型
best_model = LogisticRegressionWithSGD(**best_params)
best_model.fit(X_train_scaled, y_train)

# 在验证集上评估模型
y_val_pred = best_model.predict(X_val_scaled)
val_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC with best hyperparameters: {val_auc}')

# 在测试集上进行预测
test_df_scaled = scaler.transform(test_df)
test_predictions = best_model.predict(test_df_scaled)

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)
