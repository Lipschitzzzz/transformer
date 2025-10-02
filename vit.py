import torch
from tqdm import tqdm
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block  # 可以复用你的代码中的类
from torch.utils.data import Dataset, DataLoader
import glob
import os

class MultiNPZDataset(Dataset):
    def __init__(self, npz_dir, key='data', transform=None):
        self.key = key
        self.transform = transform
        with np.load(npz_dir) as data:
            self.data = data['data']
            self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-8)

        # npz_files = os.listdir(npz_dir)
        # if len(npz_files) == 0:
        #     raise ValueError(f"No .npz files found in {npz_dir}")

        # print(f"Loading {len(npz_files)} .npz files...")

        # data_list = []
        # for npz_file in npz_files:
        #     with np.load(npz_dir + '/' + npz_file) as data:
        #         arr = data[key]
        #         data_list.append(arr)

        # 合并所有数组
        # self.data = np.concatenate(data_list, axis=0)  # 假设 shape: (N, C, H, W)
        # self.data = np.expand_dims(self.data, axis=0)
        # self.data = np.expand_dims(data_list, axis=0)
        print(f"Total dataset size: {self.data.shape}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # n_start = 2400
        # n_end = 2700
        # e_start = 6000
        # e_end = 6300
        # print(self.data.shape)
        img = self.data[idx]
         # 检查 NaN 和 inf
        if np.isnan(img).any():
            print(f"Found NaN in sample {idx}")
        if np.isinf(img).any():
            print(f"Found inf in sample {idx}")
        if self.transform:
            img = self.transform(img)
        # print(img)
        # print(img.shape)
        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)



class ViTAutoencoder(nn.Module):
    def __init__(self, img_size=300, patch_size=15, in_chans=1, embed_dim=768, depth=12,
                 num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        # self.num_patches_h = 224 // 20  # 32
        # self.num_patches_w = 224 // 20  # 360

        # H = W = img_size
        # assert H % patch_size == 0 and W % patch_size == 0, f"Image size {img_size} not divisible by patch_size {patch_size}"

        self.num_patches_h = img_size // patch_size
        self.num_patches_w = img_size // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # --- Encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        # Encoder Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

         # --- 新增：预测头（Head for prediction）---
        # 目标：从 [B, N, D] → 每个 patch 预测出 patch_size**2 个像素值
        self.pred_head = nn.Linear(embed_dim, patch_size ** 2 * 1)  # 输出单通道（未来 SST）

        '''

        # --- Decoder ---
        # 将编码后的特征映射回 patch 空间
        self.decoder_embed = nn.Linear(embed_dim, embed_dim, bias=False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 可选：用于掩码重建

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.decoder_blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(8)  # 解码器可以更浅
        ])
        self.decoder_norm = norm_layer(embed_dim)
        '''

        # 输出头：将每个 patch 映射回像素空间
        # self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)  # 每个 patch 重构

        # 权重初始化
        self.initialize_weights()

    def initialize_weights(self):
        # nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        # nn.init.normal_(self.decoder_pos_embed, std=.02)
        # nn.init.normal_(self.mask_token, std=.02)

        # 其他线性层和 LayerNorm 初始化
        w_pos_embed = self.pos_embed.data.reshape(-1, self.embed_dim).T  # 去掉 cls token
        # w_decoder_pos_embed = self.decoder_pos_embed.data[:, 1:, :].reshape(-1, self.embed_dim).T
        torch.nn.init.kaiming_uniform_(w_pos_embed, mode='fan_in', nonlinearity='linear')
        # torch.nn.init.kaiming_uniform_(w_decoder_pos_embed, mode='fan_in', nonlinearity='linear')

        # Output projection
        # nn.init.normal_(self.decoder_pred.weight, std=0.02)
        # nn.init.constant_(self.decoder_pred.bias, 0)

    def forward_encoder(self, x):
        B, T, H, W = x.shape
        x = x.reshape(B, T, H, W)
        x = self.patch_embed(x)  # (B, N, D), N = num_patches

        # cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        return x  # 包含 [CLS] 的编码特征

    '''
    def forward_decoder(self, x):
        # x: (B, N+1, D)
        x = self.decoder_embed(x)  # 投影到 decoder 维度（可省略或改变维度）

        x = x + self.decoder_pos_embed  # 加位置编码

        x = self.decoder_blocks(x)
        x = self.decoder_norm(x)

        # 预测每个 patch 的像素值（去掉 [CLS] token）
        pred = self.decoder_pred(x[:, 1:])  # (B, N, patch_size**2 * C)

        # 重塑为图像
        pred = pred.reshape(shape=(x.shape[0],
                                   self.img_size // self.patch_size,
                                   self.img_size // self.patch_size,
                                   self.patch_size, self.patch_size, self.in_chans))
        pred = torch.einsum('nhwpqc->nchpwq', pred)  # 重排
        pred = pred.reshape(x.shape[0], self.in_chans, self.img_size, self.img_size)
        return pred
    '''

    def forward(self, x):
        # 编码
        latent = self.forward_encoder(x)
        # 解码
        # pred = self.forward_decoder(latent)
        B, T, H, W = x.shape


        # 每个 patch 预测其对应区域的像素值
        pred = self.pred_head(latent)  # (B, N, patch_size**2 * 1)

        # 重塑为完整图像
        pred = pred.reshape(B,
                            self.num_patches_h, self.num_patches_w,
                            self.patch_size, self.patch_size, 1)
        pred = pred.permute(0, 5, 1, 3, 2, 4).reshape(B, 1, H, W)  # 使用 permute 替代 einsum 更清晰
        # 或者用你原来的 einsum:
        # pred = torch.einsum('nhwpqc->nchpwq', pred).reshape(B, 1, H, W)

        return pred  # 返回重建图像
    
class SSTDataset(Dataset):
    def __init__(self, data, seq_len=5, pred_len=1):
        """
        data: numpy array (T, H, W)
        seq_len: 输入时间步数
        pred_len: 预测时间步数
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]  # (seq_len, H, W)
        target_seq = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]  # (pred_len, H, W)
        return torch.tensor(input_seq, dtype=torch.float32), torch.tensor(target_seq, dtype=torch.float32)


import numpy as np
def show_tensor_image(tensor, title="Image"):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.max() <= 1.0:
        img = (img * 255).clip(0, 255).astype(np.uint8)
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 示例：显示第一张
# ============================
# 使用示例
# ============================
if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    # from PIL import Image
    # from torchvision import transforms

    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 将PIL图像转换为tensor，并将值缩放到[0, 1]
    ])

    # 应用变换
    # input_tensor = transform(img)  # 形状: (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = MultiNPZDataset("./Training300.npz")
    # mean = dataset.mean()
    # std = dataset.std()
    # dataset = (dataset - mean) / std

    split_ratio = 0.9
    split_idx = int(len(dataset) * split_ratio)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = ViTAutoencoder().to(device)

    # 构建 Dataset
    seq_len = 5
    pred_len = 1
    train_dataset = SSTDataset(train_data, seq_len=seq_len, pred_len=pred_len)
    val_dataset = SSTDataset(val_data, seq_len=seq_len, pred_len=pred_len)

    # DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


    criterion = torch.nn.MSELoss()  # 使用均方误差作为重建损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    num_epochs = 300
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0.0
        # progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # (B, 5, H, W), (B, 1, H, W)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

    # -------------------------------
    # 保存模型
    # -------------------------------
    torch.save(model.state_dict(), "vit_autoencoder_final.pth")
    print("Training finished. Model saved.")
    
    model.eval()

    # 显示原始和重建后的图像
    import matplotlib.pyplot as plt


    '''
    img = transform(img)
    print(img.shape)

    # 输入一张图
    x = torch.unsqueeze(img, 0).to(device)  # batch=4
    # x = torch.randn(4, 3, 224, 224).to(device)  # batch=4

    # 前向传播
    recon = model(x)  # (4, 3, 224, 224)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    show_tensor_image(img, "Reconstructed Image 1")

    # 计算重建损失（例如 MSE）
    loss = nn.MSELoss()(recon, x)
    # 假设 recon 是模型输出，形状为 (1, 3, 224, 224)，值在 [0, 1] 之间
    recon = torch.sigmoid(torch.randn(1, 3, 224, 224))  # 示例数据

    # 提取第一张图像 (3, 224, 224)
    img_tensor = recon[0]  # 或 recon.squeeze(0)

    # 转为 numpy array，范围 [0, 1] → [0, 255]
    img_np = img_tensor.detach().cpu().numpy()  # (3, 224, 224)

    # 转置维度：C,H,W → H,W,C
    img_np = np.transpose(img_np, (1, 2, 0))

    # 缩放到 0-255 并转为整数类型
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)

    # 使用 PIL 保存为 JPG
    image_pil = Image.fromarray(img_np)
    image_pil.save("output_image.jpg", "JPEG")
    print("output_image.jpg")

    print(f"Reconstruction loss: {loss.item():.4f}")
    '''