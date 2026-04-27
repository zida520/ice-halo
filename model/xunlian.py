import os
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Mydata(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        return x, y


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


# --- 3. 数据预处理 ---
df = pd.read_csv("Food_Time new.csv")
if "TARGET" in df.columns:
    #把错误值改成空
    df = df.replace("#VALUE!", np.nan)
    #数据集中有有两个小数点的错误值，处理他们
    def clean_two_point(val):
        if pd.isna(val):
            return np.nan
        try:
            return float(val)
        except:
            return np.nan
    # 清洗数据
    cols_to_clean = ["Restaurant_latitude", "Restaurant_longitude", "Delivery_location_latitude",
                     "Delivery_location_longitude", "TARGET"]
    for i in cols_to_clean:
        if i in df.columns:
            df[i] = df[i].apply(clean_two_point)

    text = ["Traffic_Level", "weather_description", "Type_of_order","Type_of_vehicle"]
    #把上述文本类型转换为数字编码
    for i in text:
        if i in df.columns:
            df[i] = df[i].astype("category").cat.codes
    #删除对模型无意义的数据
    drop_cols = ["ID", "Delivery_person_ID", "Restaurant_latitude", "Restaurant_longitude","Delivery_location_latitude","Delivery_location_longitude","humidity","precipitation"]
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(existing_drop_cols, axis=1).dropna()

    #数据提取与类型转换
    data_train = df.iloc[:, :-1].values
    label_train = df.iloc[:, -1].values

    data_train = data_train.astype("float32")
    label_train = label_train.astype("float32")
    #划分数据集和训练集
    x_train, x_val, y_train, y_val = train_test_split(data_train, label_train, test_size=0.2, random_state=42)
    #标准化
    sc_x = StandardScaler()
    sc_y = StandardScaler()

    x_train = sc_x.fit_transform(x_train)
    x_val = sc_x.transform(x_val)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_train = sc_y.fit_transform(y_train).ravel()
    y_val = sc_y.transform(y_val).ravel()
    #pytorch数据集与dataloader
    dataset_train = Mydata(x_train, y_train)
    dataset_val = Mydata(x_val, y_val)

    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False)
    # 模型、损失函数、优化器、学习率调度器
    fansen = Fansen().to(device) #指定设备
    criterion = nn.MSELoss()   #损失函数
    optim = torch.optim.Adam(fansen.parameters(), lr=0.001, weight_decay=1e-5)
    #损失不下降的时候自动降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=20)

    best = float('inf')
    #循环训练
    for epoch in range(300):
        fansen.train()
        train_loss = 0.0
        for datas, targets in dataloader_train:
            datas = datas.to(device)
            targets = targets.to(device)
            targets = targets.unsqueeze(1).float()

            outs = fansen(datas)
            batch_loss = criterion(outs, targets)

            optim.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(fansen.parameters(),max_norm=1.0)
            optim.step()
            train_loss += batch_loss.item()

        train_loss /= len(dataloader_train)
        #评估模式，把Dropout、batchnorm给关闭
        fansen.eval()
        val_loss = 0.0
        with torch.no_grad():
            for datas, targets in dataloader_val:
                datas = datas.to(device)
                targets = targets.to(device)
                targets = targets.unsqueeze(1).float()
                outs = fansen(datas)
                val_loss += criterion(outs, targets).item()

        val_loss /= len(dataloader_val)

        scheduler.step(val_loss)
        #保留损失最低的模型
        if val_loss < best:
            best = val_loss
            torch.save(fansen.state_dict(), "food_time_model.pth")
        #50轮打印一次训练信息
        if epoch % 50 == 0:
            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optim.param_groups[0]['lr']:.6f}")
    print(f"训练完成，最佳验证 Loss: {best:.4f}")

    