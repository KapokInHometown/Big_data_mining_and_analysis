# 导入必要库
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 读取数据
data = pd.read_csv('D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv')

# 确保时间序列有序
data = data.sort_values(by=['Month', 'Weekday'])

# 选择用于预测的特征和目标
features = ['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life']
target = 'Water_Temperature'

# 填补缺失值（如果有）
data[features] = data[features].fillna(method='ffill')

# 标准化特征值
scaler = StandardScaler()
data[features] = scaler.fit_transform(data[features])

# 创建时间序列数据：生成时间窗口
def create_sequences(data, feature_columns, target_column, window_size=5):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[feature_columns].iloc[i:i + window_size].values)
        y.append(data[target_column].iloc[i + window_size])
    return np.array(X), np.array(y)

# 生成序列
window_size = 5
X, y = create_sequences(data, features, target, window_size)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# 方法一：线性回归
print("\n--- Linear Regression ---")
# 展平时间窗口数据
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# 训练线性回归模型
lr_model = LinearRegression()
lr_model.fit(X_train_flat, y_train)

# 预测
y_pred_lr = lr_model.predict(X_test_flat)

# 评估
print(f"Mean Squared Error (Linear Regression): {mean_squared_error(y_test, y_pred_lr):.4f}")

# 方法二：支持向量回归（SVR）
print("\n--- Support Vector Regression ---")
# 使用支持向量回归模型
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_flat, y_train)

# 预测
y_pred_svr = svr_model.predict(X_test_flat)

# 评估
print(f"Mean Squared Error (Support Vector Regression): {mean_squared_error(y_test, y_pred_svr):.4f}")

# 方法三：LSTM
print("\n--- LSTM ---")
# 构建 LSTM 模型
lstm_model = Sequential([
    LSTM(64, activation='relu', input_shape=(window_size, len(features))),
    Dense(1)
])

# 编译模型
lstm_model.compile(optimizer='adam', loss='mse')

# 训练模型
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# 预测
y_pred_lstm = lstm_model.predict(X_test).flatten()

# 评估
print(f"Mean Squared Error (LSTM): {mean_squared_error(y_test, y_pred_lstm):.4f}")