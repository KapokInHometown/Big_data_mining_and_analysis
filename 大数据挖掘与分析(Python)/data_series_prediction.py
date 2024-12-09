import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 加载数据集
file_path = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)

# 选择目标沙滩和特征
beach_name = "63rd Street Beach"  # 替换为实际沙滩名称
feature = "Water_Temperature"
time_col = "Measurement_Date_And_Time"

# 筛选数据
beach_data = data[data['Beach_Name'] == beach_name][[time_col, feature]].dropna()

# 转换时间列为时间戳，并设置为索引
beach_data[time_col] = pd.to_datetime(beach_data[time_col])
beach_data.set_index(time_col, inplace=True)

# 确保索引为时间序列并按时间排序
beach_data.sort_index(inplace=True)

# 填充缺失值并检查时间频率
beach_data = beach_data.asfreq('H')  # 假设数据是小时级别采样
beach_data[feature].fillna(method='ffill', inplace=True)

# 划分训练集和测试集
train_size = int(len(beach_data) * 0.8)
train_data = beach_data.iloc[:train_size]
test_data = beach_data.iloc[train_size:]

# ---------------- ARIMA 模型 ----------------

def train_arima(train, test):
    """
    使用 ARIMA 模型对时间序列进行训练和预测。
    :param train: 训练数据（时间序列）
    :param test: 测试数据（时间序列）
    :return: 预测结果、MSE、MAE
    """
    order = (1, 1, 1)
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    return forecast, mse, mae

# 调用 ARIMA
arima_forecast, arima_mse, arima_mae = train_arima(train_data[feature], test_data[feature])
print(f"ARIMA MSE: {arima_mse}, MAE: {arima_mae}")

# ---------------- LSTM 模型 ----------------

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(beach_data[feature].values.reshape(-1, 1))

# 创建序列数据
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 24
X, y = create_sequences(scaled_data, sequence_length)

# 划分训练集和测试集
X_train, y_train = X[:train_size - sequence_length], y[:train_size - sequence_length]
X_test, y_test = X[train_size - sequence_length:], y[train_size - sequence_length:]

# 构建 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))

# 训练 LSTM 模型
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 预测测试集
lstm_forecast = lstm_model.predict(X_test)
lstm_forecast = scaler.inverse_transform(lstm_forecast)
y_test = scaler.inverse_transform(y_test)

# 计算误差
lstm_mse = mean_squared_error(y_test, lstm_forecast)
lstm_mae = mean_absolute_error(y_test, lstm_forecast)
print(f"LSTM MSE: {lstm_mse}, MAE: {lstm_mae}")

# ---------------- 模型对比 ----------------

if arima_mse < lstm_mse:
    best_model = "ARIMA"
    future_model = ARIMA(beach_data[feature], order=(1, 1, 1)).fit()
    future_forecast = future_model.forecast(steps=24)
else:
    best_model = "LSTM"
    last_sequence = scaled_data[-sequence_length:]
    future_X = last_sequence.reshape(1, sequence_length, 1)
    future_forecast = []
    for _ in range(24):
        future_pred = lstm_model.predict(future_X)
        future_forecast.append(future_pred[0, 0])
        future_X = np.roll(future_X, -1, axis=1)
        future_X[0, -1, 0] = future_pred
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

print(f"Best Model: {best_model}")

# ---------------- 结果可视化 ----------------

# 绘制测试集和预测结果
plt.figure(figsize=(14, 7))
plt.plot(test_data.index[:len(arima_forecast)], test_data[feature][:len(arima_forecast)], label='Actual Data', color='blue')
plt.plot(test_data.index[:len(arima_forecast)], arima_forecast, label='ARIMA Forecast', color='orange')
plt.plot(test_data.index[:len(lstm_forecast)], lstm_forecast, label='LSTM Forecast', color='green')
plt.title(f"{feature} Prediction for {beach_name}")
plt.xlabel("Time")
plt.ylabel(feature)
plt.legend()
plt.grid()
plt.show()

# 绘制未来预测结果
future_index = pd.date_range(beach_data.index[-1], periods=24 + 1, freq='H')[1:]

plt.figure(figsize=(14, 7))
plt.plot(beach_data.index[-100:], beach_data[feature][-100:], label='Historical Data', color='blue')
plt.plot(future_index, future_forecast, label=f"{best_model} Future Forecast", color='red')
plt.title(f"Future {24} Hours Forecast for {beach_name}")
plt.xlabel("Time")
plt.ylabel(feature)
plt.legend()
plt.grid()
plt.show()