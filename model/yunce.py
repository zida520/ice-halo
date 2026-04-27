import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# -------------------- 定义相同模型结构 --------------------
class Fansen(nn.Module):
    def __init__(self):
        super().__init__()
        self.mo = nn.Sequential(
            nn.Linear(8, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.mo(x)

# -------------------- 1. 加载模型和预处理器 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Fansen()
state_dict = torch.load("food_time_model_best.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

encoders = joblib.load("encoders.joblib")
scaler_X = joblib.load("scaler_X.joblib")
scaler_y = joblib.load("scaler_y.joblib")

# 加载训练时使用的特征顺序
with open("feature_cols.txt", "r") as f:
    feature_cols = f.read().strip().split(",")

# -------------------- 2. 读取新数据 --------------------
df = pd.read_csv("delivery_dataset.csv")   # 可以是训练/验证/新数据

# 确保数据中包含所有特征列（不包含目标也可以）
for col in feature_cols:
    if col not in df.columns:
        raise ValueError(f"数据缺少必需的特征列: {col}")

# -------------------- 3. 对分类列进行编码（使用训练时的编码器） --------------------
categorical_cols = ["Traffic_Level", "weather_description", "Type_of_order", "Type_of_vehicle"]
for col in categorical_cols:
    if col in df.columns:
        # 将未知类别映射为 -1 或训练时见过的最小类别（这里简单处理，实际可用 le.transform 若遇到未知会报错）
        le = encoders[col]
        # 为保证健壮，将未出现过的类别设为训练集中第一个类别（或报错）
        def encode_with_fallback(x):
            try:
                return le.transform([str(x)])[0]
            except ValueError:
                # 如果遇到未知类别，可以用训练集中最频繁的类别（或用 le.classes_[0]）
                return le.transform([le.classes_[0]])[0]   # 用第一个类别代替
        df[col] = df[col].astype(str).apply(encode_with_fallback)
    else:
        raise ValueError(f"数据缺少分类列: {col}")

# -------------------- 4. 提取特征（按训练时的顺序） --------------------
X = df[feature_cols].values.astype(np.float32)

# -------------------- 5. 标准化特征（使用训练时的 scaler_X） --------------------
X_scaled = scaler_X.transform(X)   # 注意：只 transform，不 fit

# -------------------- 6. 模型预测 --------------------
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
with torch.no_grad():
    pred_scaled = model(X_tensor).cpu().squeeze().numpy()

# 反标准化得到原始分钟数
pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

# -------------------- 7. 如果数据中包含真实标签，则计算误差并输出对比表 --------------------
if "TARGET" in df.columns:
    y_true = df["TARGET"].values.astype(np.float32)
    mae = mean_absolute_error(y_true, pred_original)
    rmse = np.sqrt(mean_squared_error(y_true, pred_original))
    print(f"平均绝对误差 (MAE) : {mae:.2f} 分钟")
    print(f"均方根误差 (RMSE) : {rmse:.2f} 分钟\n")

    # 输出对比表（前20行示例）
    results = pd.DataFrame({
        "真实时间": y_true,
        "预测时间": np.round(pred_original, 2),
        "绝对误差": np.round(np.abs(y_true - pred_original), 2)
    })
    pd.set_option('display.max_rows', 20)
    print("预测对比表（前20行）：")
    print(results.head(20))
else:
    # 没有真实标签，只输出预测值
    print("预测结果（单位：分钟）：")
    for i, val in enumerate(pred_original[:10]):
        print(f"样本{i+1}: {val:.2f} 分钟")

# 可选：保存预测结果到 CSV
df_output = df.copy()
df_output["Predicted_TARGET"] = np.round(pred_original, 2)
df_output.to_csv("prediction_results.csv", index=False)
print("\n预测结果已保存到 prediction_results.csv")