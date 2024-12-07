# 导入必要的库
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取数据集
file_path = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\data.csv'
data = pd.read_csv(file_path)

# 1. 处理日期字段，转换为datetime格式，并提取季节、月份和星期等信息
data['Measurement_Date_And_Time'] = pd.to_datetime(data['Measurement_Date_And_Time'], errors='coerce')

# 提取时间特征：季节、月份、星期等
data['Month'] = data['Measurement_Date_And_Time'].dt.month
data['Weekday'] = data['Measurement_Date_And_Time'].dt.weekday
data['Hour'] = data['Measurement_Date_And_Time'].dt.hour

# 为季节创建一个简单的规则：春季：3-5月；夏季：6-8月；秋季：9-11月；冬季：12-2月
def get_season(month):
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'

data['Season'] = data['Month'].apply(get_season)

# 2. 按海滩名称进行分组，针对每个海滩进行独立的数据预处理
def preprocess_beach_group(beach_data):
    # 2.1 处理缺失值：使用均值填充数值型数据
    imputer = SimpleImputer(strategy='mean')
    numeric_columns = ['Water_Temperature', 'Turbidity', 'Transducer_Depth', 'Wave_Height', 'Wave_Period', 'Battery_Life']
    beach_data[numeric_columns] = imputer.fit_transform(beach_data[numeric_columns])

    # 2.2 异常值处理：检查波高（Wave_Height）和水温（Water_Temperature）的异常值
    # 设置一个合理的范围，去除明显的异常值
    beach_data = beach_data[(beach_data['Wave_Height'] > 0) & (beach_data['Wave_Height'] < 15)]  # 假设波高合理范围是 0-15 米
    beach_data = beach_data[(beach_data['Water_Temperature'] > 0) & (beach_data['Water_Temperature'] < 40)]  # 假设水温合理范围是 0-40°C

    # 2.3 数据标准化：对数值特征进行标准化处理
    scaler = StandardScaler()
    beach_data[numeric_columns] = scaler.fit_transform(beach_data[numeric_columns])

    return beach_data

# 3. 按海滩名称分组并进行独立处理
beach_groups = data.groupby('Beach_Name')
processed_beach_data = []

# 对每个海滩的数据进行预处理
for beach_name, beach_data in beach_groups:
    processed_data = preprocess_beach_group(beach_data.copy())  # 复制数据避免修改原数据
    processed_beach_data.append(processed_data)

# 4. 合并所有处理后的海滩数据
final_data = pd.concat(processed_beach_data)

# 5. 输出处理后的数据集
print("处理后的数据集：")
print(final_data.head())

# 将处理后的数据集保存到文件，供后续分析使用
processed_file_path = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv'
final_data.to_csv(processed_file_path, index=False)