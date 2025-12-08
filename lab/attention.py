import torch.nn as nn
import torch
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Setup dữ liệu
# 300 words
vocab = [
    "I",
    "love",
    "machine",
    "learning",
    "because",
    "it",
    "is",
    "fun",
    "so",
    "excited",
]
# Batch size = 1 để dễ hình dung, Seq = 4, Dim = 32
# (Mình đổi Batch=1 để khi vẽ bạn thấy rõ từng Head của 1 câu)
embedding = torch.randn(1, len(vocab), 32)

embed_size = 32
heads = 4


class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttentionWrapper, self).__init__()
        self.embed_size = embed_size
        self.heads = heads

        # SỬA 1: Bỏ các lớp Linear thừa (self.q, k, v).
        # Chỉ dùng nn.MultiheadAttention là đủ.
        self.mutihead = nn.MultiheadAttention(embed_size, heads, batch_first=True)

    def forward(self, values, keys, query, mask=None):
        # SỬA 2: Thêm average_attn_weights=False để lấy trọng số riêng của từng Head
        # Output weight shape: (Batch, Heads, Target_Seq, Source_Seq)
        atten_out, atten_weights = self.mutihead(
            query, keys, values, key_padding_mask=mask, average_attn_weights=False
        )
        return atten_out, atten_weights


# --- Training Data ---
epochs = 500  # Giảm epoch chút cho nhanh
lr = 0.001

# Input: "I", "love", "machine"
src = embedding[:, :-1]
# Target: "love", "machine", "learning"
tgt = embedding[:, 1:]

loss_fn = nn.MSELoss()
model = MultiheadAttentionWrapper(embed_size, heads)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
for epoch in range(epochs):
    optimizer.zero_grad()

    atten_out, atten_weights = model(src, src, src, None)

    loss_value = loss_fn(atten_out, tgt)
    loss_value.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value.item():.6f}")

# --- Visualization ---
print("\nShape của Attention Weights:", atten_weights.shape)
# Kỳ vọng: (Batch=1, Heads=4, Seq=3, Seq=3)

# SỬA 3: Vẽ 4 Heads trên cùng một hàng để so sánh
fig, axes = plt.subplots(1, heads, figsize=(16, 4))
labels = vocab[:-1]  # Nhãn trục: I, love, machine

# Lấy dữ liệu của mẫu đầu tiên trong batch (batch index = 0)
batch_idx = 0

for i in range(heads):
    # atten_weights[batch, head, row, col]
    heatmap_data = atten_weights[batch_idx, i].detach().numpy()

    sns.heatmap(
        heatmap_data,
        ax=axes[i],
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        annot=True,  # Hiện số để dễ nhìn
        fmt=".2f",
        cbar=False,
    )
    axes[i].set_title(f"Head {i + 1}")
    axes[i].set_xlabel("Keys (Input)")
    if i == 0:
        axes[i].set_ylabel("Queries (Output)")

plt.tight_layout()
plt.show()
