import torch
import torch.nn as nn

class Embedder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.operator_embedder = nn.Embedding(26, embed_dim)
        self.device_embedder = nn.Embedding(22, embed_dim)
        self.provider_embedder = nn.Embedding(4, embed_dim)
        self.sta_id_embedder = nn.Embedding(3, embed_dim)
        # outputs of above: (B, embed_dim, L)

        self.ff = nn.Sequential(
            nn.Conv1d(embed_dim * 4, embed_dim * 2, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(embed_dim * 2, embed_dim, 3, 1, 1)
        )

    def forward(self, meta):
        operate_id, device_id, provider_id, sta_id = meta.split(1, dim=1)   # (B, 1, L) for each
        operate_embed = self.operator_embedder(operate_id)
        device_embed = self.device_embedder(device_id)
        provider_embed = self.provider_embedder(provider_id)
        sta_embed = self.sta_id_embedder(sta_id)

        # After embedding, (B, 1, L, embed_dim) for each
        mix_embed = torch.cat([operate_embed, device_embed, provider_embed, sta_embed], dim=-1)

        return self.ff(mix_embed.transpose(1, 3).squeeze(3))
