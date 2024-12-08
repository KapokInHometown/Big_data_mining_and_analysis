# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# 读取处理后的数据集
processed_file_path = 'D:\电子科技大学\大三上\大数据挖掘与分析——鲁才\Big_data_mining_and_analysis\天气自动传感器大数据（Beach Water Quality）挖掘\processed_data.csv'
data = pd.read_csv(processed_file_path)

# 1. 处理缺失值和无穷大值
# 使用 SimpleImputer 填充数值型数据中的缺失值
imputer = SimpleImputer(strategy='mean')  # 使用均值填充
numeric_columns = ['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life', 'Month', 'Weekday']
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])

# 检查并处理无穷大值（infinity 或 -infinity）
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # 将无穷大替换为NaN
data[numeric_columns] = imputer.fit_transform(data[numeric_columns])  # 重新填充无穷大值为均值

# 2. 特征选择与标签分配
features = ['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life', 'Month', 'Weekday', 'Season']
X = data[features]

# 目标变量：海滩名称（即分类标签）
y = data['Beach_Name']

# 使用LabelEncoder将海滩名称转为数字标签
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 4. 数据预处理：对类别特征进行One-Hot编码，数值特征进行标准化
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Water_Temperature', 'Turbidity', 'Wave_Height', 'Wave_Period', 'Battery_Life', 'Month', 'Weekday']),
        ('cat', OneHotEncoder(), ['Season'])  # 对Season进行One-Hot编码
    ])

# 5. 创建模型训练管道
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# 调试信息：确保数据分割正常
print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Testing data shape: {X_test.shape}, {y_test.shape}")

# 对每个模型进行训练并评估
for model_name, model in models.items():
    print(f"Training model: {model_name}")  # Debug print
    # 创建管道：包含预处理和模型
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)
    print(f"Model {model_name} trained successfully.")  # Debug print

    # 预测并评估模型
    y_pred = pipeline.predict(X_test)
    print(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")

    # 输出分类报告，处理零除错误
    print(f"Classification Report ({model_name}):\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

# 提示：如果遇到FutureWarning（例如KNN的mode方法），
# 可通过升级相关库版本或显式设置keepdims参数来解决。