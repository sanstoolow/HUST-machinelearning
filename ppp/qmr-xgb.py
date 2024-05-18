import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb

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
X = combined_df.loc['train']
X_test = combined_df.loc['test']

# 特征缩放
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 使用XGBoost进行训练
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# 交叉验证
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-Validation AUC scores:", cv_scores)
print("Mean Cross-Validation AUC score:", np.mean(cv_scores))

# 训练模型
xgb_model.fit(X_train, y_train)

# 在验证集上进行预测
y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred)
print("Validation AUC:", val_auc)

# 在测试集上进行预测
test_predictions = xgb_model.predict_proba(X_test_scaled)[:, 1]

# 准备提交
submission_df = pd.DataFrame({'id': test_customer_ids, 'Exited': test_predictions})
submission_df.to_csv('submission.csv', index=False)
