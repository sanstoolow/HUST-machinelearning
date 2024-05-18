import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 读取数据集
data = pd.read_csv('ppp/train.csv')

# 展示数据集的前几行
print(data.head())

# 检查数据集的基本信息
print(data.info())

# 可视化年龄分布
sns.histplot(data['Age'], kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 可视化账户余额分布
sns.histplot(data['Balance'], kde=True)
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()

# 可视化性别分布
sns.countplot(data['Gender'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# 可视化是否流失分布
sns.countplot(data['Exited'])
plt.title('Churn Distribution')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()

# 可视化地理位置分布
sns.countplot(data['Geography'])
plt.title('Geography Distribution')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()

# 可视化信用评分分布
sns.histplot(data['CreditScore'], kde=True)
plt.title('Credit Score Distribution')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.show()

# 可视化服务年限分布
sns.histplot(data['Tenure'], kde=True)
plt.title('Tenure Distribution')
plt.xlabel('Tenure (years)')
plt.ylabel('Frequency')
plt.show()

# 可视化产品数量分布
sns.countplot(data['NumOfProducts'])
plt.title('Number of Products Distribution')
plt.xlabel('Number of Products')
plt.ylabel('Count')
plt.show()

# 可视化是否持有信用卡分布
sns.countplot(data['HasCrCard'])
plt.title('Has Credit Card Distribution')
plt.xlabel('Has Credit Card')
plt.ylabel('Count')
plt.show()

# 可视化是否为活跃会员分布
sns.countplot(data['IsActiveMember'])
plt.title('Is Active Member Distribution')
plt.xlabel('Is Active Member')
plt.ylabel('Count')
plt.show()

# 可视化预估薪资分布
sns.histplot(data['EstimatedSalary'], kde=True)
plt.title('Estimated Salary Distribution')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()
