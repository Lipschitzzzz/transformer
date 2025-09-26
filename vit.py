import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, Block  # 可以复用你的代码中的类
from torch.utils.data import Dataset, DataLoader

class SingleImageDataset(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        return self.tensor, self.tensor  # 输入与标签相同

class ViTAutoencoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=256, depth=12,
                 num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # --- Encoder ---
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)

        # Encoder Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

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

        # 输出头：将每个 patch 映射回像素空间
        self.decoder_pred = nn.Linear(embed_dim, patch_size ** 2 * in_chans, bias=True)  # 每个 patch 重构

        # 权重初始化
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.decoder_pos_embed, std=.02)
        nn.init.normal_(self.mask_token, std=.02)

        # 其他线性层和 LayerNorm 初始化
        w_pos_embed = self.pos_embed.data[:, 1:, :].reshape(-1, self.embed_dim).T  # 去掉 cls token
        w_decoder_pos_embed = self.decoder_pos_embed.data[:, 1:, :].reshape(-1, self.embed_dim).T
        torch.nn.init.kaiming_uniform_(w_pos_embed, mode='fan_in', nonlinearity='linear')
        torch.nn.init.kaiming_uniform_(w_decoder_pos_embed, mode='fan_in', nonlinearity='linear')

        # Output projection
        nn.init.normal_(self.decoder_pred.weight, std=0.02)
        nn.init.constant_(self.decoder_pred.bias, 0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)  # (B, N, D), N = num_patches

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        return x  # 包含 [CLS] 的编码特征

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

    def forward(self, x):
        # 编码
        latent = self.forward_encoder(x)
        # 解码
        pred = self.forward_decoder(latent)
        return pred  # 返回重建图像
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
    # img = Image.open("Korean sea.jpg")
    # transform = transforms.Compose([transforms.RandomResizedCrop(224),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    from PIL import Image
    import torch
    import torchvision.transforms as transforms

    # 加载图片
    image_path = "1.png"  # 替换为你的图片路径
    img = Image.open(image_path).convert('RGB')
    # print(img.shape)

    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # 将PIL图像转换为tensor，并将值缩放到[0, 1]
    ])

    # 应用变换
    input_tensor = transform(img)  # 形状: (3, 224, 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = SingleImageDataset(input_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = ViTAutoencoder().to(device)

    criterion = torch.nn.MSELoss()  # 使用均方误差作为重建损失
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.to(device)

    num_epochs = 100  # 训练轮数
    for epoch in range(num_epochs):
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    model.eval()
    with torch.no_grad():
        reconstructed = model(input_tensor.unsqueeze(0).to(device))

    # 显示原始和重建后的图像
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(transforms.ToPILImage()(input_tensor))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(transforms.ToPILImage()(reconstructed.squeeze().cpu()))
    plt.axis('off')

    plt.show()





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