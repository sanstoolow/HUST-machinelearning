import cupy as cp
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


train_df = pd.read_csv('ppp/train.csv')
test_df = pd.read_csv('ppp/test.csv')

# 数据处理

test_customer_ids = test_df['id']
train_df = pd.get_dummies(train_df, columns=['Gender', 'Geography'])
test_df = pd.get_dummies(test_df, columns=['Gender', 'Geography'])

train_df = train_df.drop(['id', 'CustomerId', 'Surname'], axis=1)
test_df = test_df.drop(['id', 'CustomerId', 'Surname'], axis=1)

X_train = cp.asarray(train_df.drop('Exited', axis=1).values.astype(np.float32))
y_train = cp.asarray(train_df['Exited'].values.astype(np.float32))
X_test = cp.asarray(test_df.values.astype(np.float32))

def feature_scaling(X):
    X_scaled = (X - cp.mean(X, axis=0)) / cp.std(X, axis=0)
    return X_scaled

X_train_scaled = feature_scaling(X_train)
X_test_scaled = feature_scaling(X_test)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, num_epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.num_epochs = num_epochs
        self.weights = None
        self.bias = None
        self.platt_coef = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = cp.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            idx = cp.random.randint(0, n_samples) #随机选择一个样本，如果递归每一个就会很慢
            x_i = X[idx]
            condition = y[idx] * (cp.dot(x_i, self.weights) - self.bias) >= 1
            if condition:
                self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
            else:
                self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - cp.dot(x_i, y[idx]))
                self.bias -= self.learning_rate * y[idx]

        #Platt缩放
        decision_values = cp.dot(X, self.weights) - self.bias
        self.platt_coef = self._sigmoid_fit(decision_values.get(), y.get())

    def _sigmoid_fit(self, decision_values, y):
        lr = LogisticRegression()
        lr.fit(decision_values.reshape(-1, 1), y)
        return lr.coef_, lr.intercept_

    def predict_proba(self, X):
        decision_values = cp.dot(X, self.weights) - self.bias
        return expit(self.platt_coef[0] * decision_values.get() + self.platt_coef[1])

    def predict(self, X):
        proba = self.predict_proba(X)
        proba = cp.asarray(proba)  
        return cp.where(proba > 0.5, 1, 0)

svm = SVM(learning_rate=0.01, lambda_param=0.01, num_epochs=1000)
svm.fit(X_train_scaled, y_train)

y_test_pred = svm.predict(X_test_scaled)


submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': cp.asnumpy(y_test_pred).ravel()})
submission_df.to_csv('submission.csv', index=False)