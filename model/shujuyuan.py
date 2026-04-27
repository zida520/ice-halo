import os
import pandas as pd

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "Food_Time new.csv")

# 读取 CSV 文件
df = pd.read_csv(csv_path)

# 查看分类列的唯一值
categorical_cols = ["Traffic_Level", "weather_description", "Type_of_order", "Type_of_vehicle"]
for col in categorical_cols:
    print(col, ":", df[col].unique())