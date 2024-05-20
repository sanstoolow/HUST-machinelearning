import cupy as cp
import pandas as pd
from multiprocessing import Pool

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
X_train = cp.asarray(train_df.drop('Exited', axis=1).values)
y_train = cp.asarray(train_df['Exited'].values)
X_test = cp.asarray(test_df.values)

# 特征缩放
def feature_scaling(X):
    X_scaled = (X - cp.mean(X, axis=0)) / cp.std(X, axis=0)
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
        self.weights = cp.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (cp.dot(x_i, self.weights) - self.bias) >= 1
                if condition:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - cp.dot(x_i, y[idx]))
                    self.bias -= self.learning_rate * y[idx]

    def predict(self, X):
        y_pred = cp.sign(cp.dot(X, self.weights) - self.bias)
        return cp.where(y_pred == -1, 0, 1)

# 训练支持向量机（SVM）模型
def train_model(learning_rate, lambda_param, num_epochs):
    svm = SVM(learning_rate=learning_rate, lambda_param=lambda_param, num_epochs=num_epochs)
    svm.fit(X_train_scaled, y_train)
    return svm

# 使用多进程并行训练多个 SVM 模型
def parallel_training(params_list):
    with Pool() as pool:
        models = pool.starmap(train_model, params_list)
    return models

# 定义参数列表
learning_rates = [0.001, 0.01, 0.1]
lambda_params = [0.01, 0.1, 1]
num_epochs_list = [1000, 2000, 3000]

params_list = [(lr, lp, ne) for lr in learning_rates for lp in lambda_params for ne in num_epochs_list]

if __name__ == '__main__':
    # 并行训练多个 SVM 模型
    models = parallel_training(params_list)

    # 在测试集上进行预测，并对结果进行投票
    def predict_and_vote(models, X_test):
        y_preds = [model.predict(X_test) for model in models]
        y_pred_combined = cp.sum(y_preds, axis=0)
        y_test_pred = cp.where(y_pred_combined >= 0, 1, 0)
        return y_test_pred

    y_test_pred = predict_and_vote(models, X_test_scaled)

    # 保存结果到文件
    submission_df = pd.DataFrame({'id': test_df.index, 'Exited': cp.asnumpy(y_test_pred)})
    submission_df.to_csv('submission.csv', index=False)