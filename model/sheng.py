# generate_preprocessors.py
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Food_Time new.csv")
df = pd.read_csv(csv_path)

# 1. 数据清洗（与训练时完全一致）
df = df.replace("#VALUE!", np.nan)
def clean_two_point(val):
    if pd.isna(val):
        return np.nan
    try:
        return float(val)
    except:
        return np.nan

cols_to_clean = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude",
                 "Delivery_location_longitude", "TARGET"]
for i in cols_to_clean:
    if i in df.columns:
        df[i] = df[i].apply(clean_two_point)

categorical_cols = ["Traffic_Level", "weather_description", "Type_of_order", "Type_of_vehicle"]
encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

drop_cols = ["ID", "Delivery_person_ID", "Restaurant_latitude", "Restaurant_longitude",
             "Delivery_location_latitude", "Delivery_location_longitude", "humidity", "precipitation"]
existing_drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(existing_drop_cols, axis=1).dropna()

feature_cols = [c for c in df.columns if c != "TARGET"]
data_train = df[feature_cols].values.astype(np.float32)
label_train = df["TARGET"].values.astype(np.float32)

# 3. 拟合标准化器（使用全部数据，因为训练时也是从整个数据集划分后拟合的，这里近似）
sc_x = StandardScaler()
sc_x.fit(data_train)

sc_y = StandardScaler()
sc_y.fit(label_train.reshape(-1, 1))

# 4. 保存
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, "model")  # 可选子目录
os.makedirs("model", exist_ok=True)
joblib.dump(encoders, "encoders.joblib")
joblib.dump(sc_x, "scaler_X.joblib")
joblib.dump(sc_y, "scaler_Y.joblib")  # 注意大小写，之前代码是 scaler_y.joblib，保持一致
with open("feature_cols.txt", "w") as f:
    f.write(",".join(feature_cols))

print("预处理器生成完成！")