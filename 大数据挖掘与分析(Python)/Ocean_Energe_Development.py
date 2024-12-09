import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 加载数据集
file_path = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv'  # 请替换为实际文件路径
data = pd.read_csv(file_path)

# 显示数据列名
print("Columns in dataset:", data.columns)

# 检查数据结构
print(data.head())

# 定义不同海洋能类型及相关特征
energy_features = {
    'Tidal Energy': ['Transducer_Depth', 'Wave_Period'],
    'Wave Energy': ['Wave_Height', 'Wave_Period'],
    'Thermal Energy': ['Water_Temperature']
}

# 标准化特征
scaler = StandardScaler()

# 存储最佳开发位置
best_locations = {}

# 分析每种海洋能
for energy_type, features in energy_features.items():
    # 检查是否有缺失的特征列
    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        print(f"Missing features for {energy_type}: {missing_features}")
        continue

    # 提取相关特征
    selected_features = data[features].dropna()
    scaled_features = scaler.fit_transform(selected_features)

    # 简单回归模型（假设潜力与特征正相关，目标是找到潜力最大的海滩）
    # 创建虚拟潜力列 (加权平均方法)
    weights = [1.0] * len(features)  # 可以根据实际需求调整权重
    data.loc[selected_features.index, f'{energy_type}_potential'] = (
        selected_features.values @ weights
    )

    # 找到潜力最大的海滩
    max_potential_index = data[f'{energy_type}_potential'].idxmax()
    best_beach = data.loc[max_potential_index, 'Beach_Name']
    best_locations[energy_type] = best_beach

# 输出结果
print("\nBest locations for different types of ocean energy development:")
for energy_type, beach_name in best_locations.items():
    print(f"{energy_type}: {beach_name}")

# 保存结果到文件
output_file = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\ best_ocean_energy_locations.csv'
data.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")