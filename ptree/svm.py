import numpy as np
import pandas as pd

# 读取数据
train_df = pd.read_csv('ppp/train.csv')
test_df = pd.read_csv('ppp/test.csv')

# 数据预处理
# 将分类变量转换为数值型
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
X_train = train_df.drop('Exited', axis=1).values
y_train = train_df['Exited'].values
X_test = test_df.values

# 特征缩放
def feature_scaling(X):
    X_scaled = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_scaled

X_train_scaled = feature_scaling(X_train)
X_test_scaled = feature_scaling(X_test)

# 实现支持向量机（SVM）模型
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        y_pred = np.sign(np.dot(X, self.weights) - self.bias)
        return np.where(y_pred == -1, 0, 1)

# 训练支持向量机（SVM）模型
svm = SVM(learning_rate=0.01, lambda_param=0.01, num_epochs=1000)
svm.fit(X_train_scaled, y_train)

# 在测试集上进行预测
y_test_pred = svm.predict(X_test_scaled)

# 保存结果到文件
submission_df = pd.DataFrame({'id': test_df.index, 'Exited': y_test_pred})
submission_df.to_csv('submission.csv', index=False)
