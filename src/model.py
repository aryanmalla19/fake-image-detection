from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        embedding_dim: int = 128,
        transformer_layers: int = 2,
        attention_heads: int = 4,
        mlp_ratio: int = 2,
        dropout: float = 0.2,
        image_size: int = 32,
        use_transformer: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % attention_heads:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        channels = [3, *cnn_channels]
        blocks = []
        for idx in range(len(channels) - 1):
            blocks.append(ConvBlock(channels[idx], channels[idx + 1], pool=idx < len(channels) - 2))
        self.cnn = nn.Sequential(*blocks)
        feature_channels = channels[-1]

        self.patch_projection = nn.Linear(feature_channels, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        pooled_blocks = max(0, len(cnn_channels) - 1)
        feature_size = image_size // (2**pooled_blocks)
        if feature_size < 1 or image_size % (2**pooled_blocks):
            raise ValueError("image_size is incompatible with the CNN pooling depth")
        self.position_embedding = nn.Parameter(torch.zeros(1, 1 + feature_size**2, embedding_dim))
        self.use_transformer = use_transformer

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=attention_heads,
            dim_feedforward=embedding_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(feature_channels + embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.cnn(x)
        cnn_vector = self.cnn_pool(feature_map).flatten(1)

        patches = feature_map.flatten(2).transpose(1, 2)
        patch_embeddings = self.patch_projection(patches)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls_tokens, patch_embeddings], dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.size(1)]
        transformer_vector = self.transformer(tokens)[:, 0] if self.use_transformer else tokens[:, 1:].mean(1)

        fused = torch.cat([cnn_vector, transformer_vector], dim=1)
        return self.classifier(fused)


class CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        cnn_channels: list[int] | tuple[int, ...] = (32, 64, 128),
        dropout: float = 0.2,
        **_: object,
    ) -> None:
        super().__init__()
        channels = [3, *cnn_channels]
        self.cnn = nn.Sequential(
            *(ConvBlock(channels[i], channels[i + 1], pool=i < len(channels) - 2) for i in range(len(channels) - 1))
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.cnn(x))


def build_model(config: dict) -> nn.Module:
    model_config = dict(config["model"])
    model_type = model_config.pop("type", "hybrid")
    model_config.setdefault("image_size", int(config["data"]["image_size"]))
    if model_type == "cnn":
        return CNNClassifier(**model_config)
    if model_type == "hybrid":
        return HybridCNNTransformer(**model_config)
    raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
