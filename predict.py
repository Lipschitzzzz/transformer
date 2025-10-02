import torch
import numpy as np
import matplotlib.pyplot as plt
import vit

# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = vit.ViTAutoencoder().to(device)
model.load_state_dict(torch.load("vit_autoencoder_final.pth", map_location='cpu'))
model.eval()

# 3. 加载数据
data = np.load("Training300.npz")['data'][0,:300,:300]  # (31, 640, 720)
x = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # (1, 31, 640, 720)
print(x.shape)

# 模型移到设备
# 4. 预测
with torch.no_grad():
    pred = model(x)

# 5. 保存结果
pred_np = pred.squeeze(0).cpu().numpy()  # 去掉 batch 维度
np.savez_compressed("output300.npz", data = pred_np)
print("Prediction saved to output300.npz")