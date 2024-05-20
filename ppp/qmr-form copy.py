import cupy as cp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize

# 读取数据
train_df = pd.read_csv('ppp/train.csv')
test_df = pd.read_csv('ppp/test.csv')

# 数据预处理
# 将分类变量转换为独热编码
train_df = pd.get_dummies(train_df, columns=['Gender', 'Geography'])
test_df = pd.get_dummies(test_df, columns=['Gender', 'Geography'])

test_customer_ids = test_df['id']

# 移除无关的特征
train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

# 将特征和标签分离
X = train_df.drop('Exited', axis=1)
y = train_df['Exited']

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_val_scaled, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

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
        self.weights = cp.zeros(n_features)
        self.bias = 0
        self.weight_momentums = cp.zeros(n_features)
        self.bias_momentum = 0

        for _ in range(self.n_iterations):
            # Nesterov Momentum
            weights_next = self.weights - self.learning_rate * (self.weight_momentums * self.momentum_factor
                                                                  + self.regularization_strength * self.weights
                                                                  + cp.dot(X.T, (self._sigmoid(cp.dot(X, self.weights + self.weight_momentums * self.momentum_factor) + self.bias) - y)) / n_samples)
            bias_next = self.bias - self.learning_rate * (self.bias_momentum * self.momentum_factor
                                                           + self.regularization_strength * self.bias
                                                           + cp.sum(self._sigmoid(cp.dot(X, self.weights + self.weight_momentums * self.momentum_factor) + self.bias) - y) / n_samples)

            self.weight_momentums = self.weights - weights_next
            self.bias_momentum = self.bias - bias_next

            self.weights = weights_next
            self.bias = bias_next

    def predict(self, X):
        model = cp.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + cp.exp(-z))

# 定义一个函数，该函数接受超参数作为输入，并返回模型的负AUC分数
def objective(params):
    learning_rate, n_iterations, regularization_strength, momentum_factor = params

    model = LogisticRegressionWithSGD(learning_rate=learning_rate, n_iterations=n_iterations, regularization_strength=regularization_strength, momentum_factor=momentum_factor)

    model.fit(cp.asarray(X_train_scaled), cp.asarray(y_train))

    y_val_pred = model.predict(cp.asarray(X_val_scaled))
    val_auc = roc_auc_score(y_val, y_val_pred)

    # 我们返回负AUC分数，因为我们希望最小化这个函数
    return -val_auc

# 定义超参数的范围
space = [(0.0001, 0.5, 'log-uniform'),  # learning_rate
         (100, 3000),  # n_iterations
         (1e-6, 5e-1, 'log-uniform'),  # regularization_strength
         (0.1, 0.9)]  # momentum_factor

# 使用贝叶斯优化来找到最佳的超参数
res = gp_minimize(objective, space, n_calls=50, random_state=0)

# 输出最佳的超参数
print("Best parameters: {}".format(res.x))

# 使用最佳的超参数来训练模型
model = LogisticRegressionWithSGD(learning_rate=res.x[0], n_iterations=res.x[1], regularization_strength=res.x[2], momentum_factor=res.x[3])
model.fit(cp.asarray(X_train_scaled), cp.asarray(y_train))

# 在验证集上评估模型
y_val_pred = model.predict(cp.asarray(X_val_scaled))
val_auc = roc_auc_score(y_val, y_val_pred)
print(f'Validation AUC: {val_auc}')

# 在测试集上进行预测
test_df_scaled = scaler.transform(test_df)
test_predictions = model.predict(cp.asarray(test_df_scaled))

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)