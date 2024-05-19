import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('ppp/train.csv')

# 定义需要绘图的特征列表
features = ['Age', 'Balance', 'CreditScore', 'Tenure', 'EstimatedSalary', 'NumOfProducts', 'Gender']

for feature in features:
    # 计算特征的总频数
    total_counts = data[feature].value_counts().sort_index()

    # 计算Exited为1的特征的频数
    exited_counts = data[data['Exited'] == 1][feature].value_counts().sort_index()

    # 使用matplotlib进行绘图
    plt.figure(figsize=(10, 6))
    plt.plot(total_counts.index, total_counts.values, label='Total')
    plt.plot(exited_counts.index, exited_counts.values, label='Exited=1')

    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(f'{feature} Distribution')
    plt.legend()

    plt.show()