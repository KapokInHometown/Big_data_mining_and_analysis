import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
data = pd.read_csv('D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv')

# 2. 处理日期时间字段
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')  # 转换为日期时间格式
    data['Timestamp'] = data['Timestamp'].map(lambda x: x.timestamp() if pd.notnull(x) else np.nan)  # 转换为时间戳

# 3. 检查并处理 NaN 值
if data.isnull().values.any():
    print("Dataset contains NaN values. Filling or dropping them...")
    # 填充 NaN 值（数值列用均值填充，非数值列用众数填充）
    for col in data.columns:
        if data[col].dtype == 'object':
            # 对字符串列填充众数
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            # 对数值列填充均值
            data[col] = data[col].fillna(data[col].mean())

# 4. 转换所有非数值列为数值
for col in data.select_dtypes(include=['object']).columns:
    if col != 'Beach_Name':  # 不转换目标列
        data[col] = data[col].astype('category').cat.codes

# 确保所有列类型一致
print("Data types after conversion:")
print(data.dtypes)

# 5. 提取特征和目标
features = data.drop(columns=['Beach_Name'])  # 删除目标列
target = data['Beach_Name']  # 目标列

# 数据标准化
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 6. K-Means 聚类
kmeans = KMeans(n_clusters=len(target.unique()), random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)

# 7. DBSCAN 聚类
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(scaled_features)

# 8. 计算 Silhouette Score
kmeans_silhouette = silhouette_score(scaled_features, kmeans_labels)
print(f"Silhouette Score for K-Means: {kmeans_silhouette}")

dbscan_silhouette = silhouette_score(scaled_features, dbscan_labels)
print(f"Silhouette Score for DBSCAN: {dbscan_silhouette}")

# 9. 每个聚类的海滩信息和样本数量
def print_cluster_info(labels, clustering_algorithm):
    print(f"\nCluster info for {clustering_algorithm}:")
    clusters = pd.DataFrame({'Beach_Name': data['Beach_Name'], 'Cluster_Label': labels})
    cluster_counts = clusters.groupby('Cluster_Label')['Beach_Name'].value_counts()
    for cluster, count in cluster_counts.groupby(level=0).count().items():
        print(f"Cluster {cluster} contains {count} samples.")
        print(cluster_counts[cluster])

# 输出每个聚类的样本数量和主要海滩
print_cluster_info(kmeans_labels, 'K-Means')
print_cluster_info(dbscan_labels, 'DBSCAN')

# 10. 可视化
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(scaled_features)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=kmeans_labels, palette='Set2')
plt.title("K-Means Clustering")

plt.subplot(1, 2, 2)
sns.scatterplot(x=reduced_features[:, 0], y=reduced_features[:, 1], hue=dbscan_labels, palette='Set2')
plt.title("DBSCAN Clustering")
plt.show()