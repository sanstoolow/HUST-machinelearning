import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# 从train.csv读取数据
train_data = pd.read_csv("ppp/train.csv")

# 数据预处理
train_data['Gender'] = train_data['Gender'].map({'Male': 0, 'Female': 1})
train_data['Geography'] = train_data['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
X = train_data.drop(columns=['Exited', 'id', 'CustomerId', 'Surname'])
y = train_data['Exited']

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 从test.csv读取数据
test_data = pd.read_csv("ppp/test.csv")
test_data['Gender'] = test_data['Gender'].map({'Male': 0, 'Female': 1})
test_data['Geography'] = test_data['Geography'].map({'France': 0, 'Spain': 1, 'Germany': 2})
test_data = test_data.drop(columns=['id', 'CustomerId', 'Surname'])

# 使用随机森林进行训练
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 预测验证集
y_pred_proba = clf.predict_proba(X_val)[:,1]

# 评估ROC AUC
roc_auc = roc_auc_score(y_val, y_pred_proba)
print("Validation ROC AUC:", roc_auc)

# 预测测试集
test_pred_proba = clf.predict_proba(test_data)[:,1]

# 保存结果
test_results = pd.DataFrame({'id': test_data.index, 'Exited': test_pred_proba})
test_results.to_csv("submission.csv", index=False)
