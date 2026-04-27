import os
import sys
import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib

# ---------- 解决打包后路径问题的辅助函数 ----------
def resource_path(relative_path):
    """获取资源的绝对路径，兼容开发环境和 PyInstaller 打包后的环境"""
    try:
        # PyInstaller 会创建临时文件夹，将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        # 开发环境下，使用当前文件的绝对路径
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------- 初始化 Flask 应用 ----------
# 注意：static 文件夹的路径也需要用 resource_path 获取
app = Flask(__name__,
            static_folder=resource_path('static'),
            template_folder=resource_path('templates'))  # 如果没有 templates 可以忽略
CORS(app)

# ---------- 首页路由 ----------
@app.route('/')
def index():
    # 使用 send_file 直接返回 HTML 文件，避免 send_from_directory 路径问题
    html_path = resource_path(os.path.join('static', 'qian.html'))
    return send_file(html_path)

# ---------- 定义模型结构（与训练时完全一致） ----------
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

# ---------- 加载模型和预处理器 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Fansen()

# 模型文件路径（使用 resource_path）
model_path = resource_path(os.path.join("model", "food_time_model.pth"))
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("模型加载成功")

# 加载预处理器
encoders_path = resource_path(os.path.join("model", "encoders.joblib"))
scaler_X_path = resource_path(os.path.join("model", "scaler_X.joblib"))
scaler_y_path = resource_path(os.path.join("model", "scaler_y.joblib"))
feature_cols_path = resource_path(os.path.join("model", "feature_cols.txt"))

encoders = joblib.load(encoders_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

with open(feature_cols_path, "r") as f:
    feature_cols = f.read().strip().split(",")

print("特征顺序:", feature_cols)

categorical_cols = ["Traffic_Level", "weather_description", "Type_of_order", "Type_of_vehicle"]

# ---------- 预处理函数 ----------
def preprocess_input(data_dict):
    # 字段映射（前端 -> 后端）
    field_mapping = {
        "Delivery_distance": "Distance (km)",
        "Preparation_time": None,
        "Order_hour": None,
        "Previous_orders": None,
        "Traffic_Level": "Traffic_Level",
        "weather_description": "weather_description",
        "Type_of_order": "Type_of_order",
        "Type_of_vehicle": "Type_of_vehicle",
    }
    
    default_values = {
        "Delivery_person_Age": 30.0,
        "Delivery_person_Ratings": 4.5,
        "temperature": 22.0,
        "Distance (km)": 5.0,
        "Traffic_Level": "Medium",
        "weather_description": "Clear",
        "Type_of_order": "Lunch",
        "Type_of_vehicle": "Car",
    }
    
    input_row = {feat: None for feat in feature_cols}
    
    # 映射前端数据
    for front_field, value in data_dict.items():
        if front_field in field_mapping:
            back_feat = field_mapping[front_field]
            if back_feat is not None:
                input_row[back_feat] = value
        else:
            if front_field in feature_cols:
                input_row[front_field] = value
    
    # 填充缺失值
    for feat in feature_cols:
        if input_row[feat] is None:
            if feat in default_values:
                input_row[feat] = default_values[feat]
                print(f"信息: 特征 '{feat}' 未提供，使用默认值 {default_values[feat]}")
            else:
                raise ValueError(f"缺少特征: {feat}，且未设置默认值")
    
    # 编码分类特征
    for col in categorical_cols:
        le = encoders[col]
        raw_val = str(input_row[col])
        try:
            encoded = le.transform([raw_val])[0]
        except ValueError:
            fallback = le.classes_[0]
            encoded = le.transform([fallback])[0]
            print(f"警告: 类别 '{raw_val}' 未在训练集中，已替换为 '{fallback}'")
        input_row[col] = encoded
    
    # 构建特征向量（顺序与 feature_cols 一致）
    feature_vector = []
    for col in feature_cols:
        val = input_row[col]
        # 确保数值类型
        if isinstance(val, str):
            # 简单尝试转换为 float
            try:
                val = float(val)
            except:
                pass
        feature_vector.append(val)
    
    X_raw = np.array(feature_vector).reshape(1, -1).astype(np.float32)
    X_scaled = scaler_X.transform(X_raw)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    return X_tensor

# ---------- 预测路由 ----------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "未收到数据"}), 400
        
        print("收到的数据:", data)
        input_tensor = preprocess_input(data)
        print("预处理后的张量形状:", input_tensor.shape)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).cpu().squeeze().numpy()
        
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
        
        return jsonify({
            "status": "success",
            "predicted_minutes": round(float(pred_original), 2)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# ---------- 启动 ----------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # 打包后建议 debug=False