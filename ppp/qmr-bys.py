import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from bayes_opt import BayesianOptimization

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
X = train_df.drop('Exited', axis=1).values
y = train_df['Exited'].values

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 定义逻辑回归模型
class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 更新权重和偏差
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义逻辑回归模型评估函数
def evaluate_model(learning_rate, n_iterations):
    model = LogisticRegressionCustom(learning_rate=learning_rate, n_iterations=int(n_iterations))
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    val_auc = roc_auc_score(y_val, y_val_pred)
    print(f'Learning Rate = {learning_rate:.4f}, Iterations = {int(n_iterations)}, Validation AUC = {val_auc:.4f}')
    return val_auc

# 使用贝叶斯优化来搜索最佳超参数
bayes_optimizer = BayesianOptimization(evaluate_model, {'learning_rate': (0.0001, 0.2), 'n_iterations': (500, 2000)})
bayes_optimizer.maximize(init_points=5, n_iter=10)

best_learning_rate = bayes_optimizer.max['params']['learning_rate']
best_n_iterations = bayes_optimizer.max['params']['n_iterations']

# 使用最佳超参数来训练模型
best_model = LogisticRegressionCustom(learning_rate=best_learning_rate, n_iterations=int(best_n_iterations))
best_model.fit(X_scaled, y)

# 在测试集上进行预测
test_df_scaled = scaler.transform(test_df.values)
test_predictions = best_model.predict(test_df_scaled)

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)
