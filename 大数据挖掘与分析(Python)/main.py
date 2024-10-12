import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘/data.csv')

# 数据清洗：填补缺失值
df['Transducer_Depth'].fillna(df['Transducer_Depth'].mean(), inplace=True)
df['Turbidity'].fillna(df['Turbidity'].mean(), inplace=True)

# 数据规约：选择所需列并删除重复行
df = df[['Beach_Name', 'Measurement_Date_And_Time', 'Water_Temperature', 'Turbidity', 'Transducer_Depth', 'Wave_Height', 'Wave_Period', 'Battery_Life']]
df.drop_duplicates(inplace=True)

# 数据转换：将日期列转换为datetime类型
df['Measurement_Date_And_Time'] = pd.to_datetime(df['Measurement_Date_And_Time'])

# 数据可视化
plt.figure(figsize=(12, 6))
for beach in df['Beach_Name'].unique():
    subset = df[df['Beach_Name'] == beach]
    plt.plot(subset['Measurement_Date_And_Time'], subset['Water_Temperature'], label=beach)

plt.xlabel('Date and Time')
plt.ylabel('Water Temperature (°C)')
plt.title('Water Temperature over Time by Beach')
plt.legend()
#plt.show()
plt.savefig('D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\data_svg.svg', dpi=600)

# 保存为新CSV文件
df.to_csv('D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘/data_done.csv', index=False)