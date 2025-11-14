# =====================================================
# src/models.py
# =====================================================

import timm
import torch
import torch.nn as nn
from math import sqrt

# =====================================================
# КАСТОМНЫЕ МОДУЛИ ДЛЯ ViT
# =====================================================

class MLP(nn.Module):
    """MLP блок с GELU активацией"""
    
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention механизм"""
    
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5  # 1/sqrt(d_k)
        
        # Линейные проекции для Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v shape: (B, heads, N, head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer блок с residual connections"""
    
    def __init__(self, dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout)
    
    def forward(self, x):
        # Multi-head attention с residual
        x = x + self.attn(self.norm1(x))
        # MLP с residual
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """Кастомная реализация Vision Transformer для NAS"""

    @staticmethod
    def find_compatible_dim(target_dim, heads, config=None):
        """
        Находит ближайший `dim`, который делится на `heads` без остатка.
        Это необходимо для корректной работы Multi-Head Attention.
        Использует `dim_min` и `dim_max` из конфига, если они предоставлены.
        """
        if target_dim % heads == 0:
            # Если dim уже совместим, проверяем, находится ли он в диапазоне (если он есть)
            if config:
                try:
                    dim_min, dim_max = config.nas.architecture_search.model_params.dim.range
                    if dim_min <= target_dim <= dim_max:
                        return target_dim
                except Exception:
                    return target_dim # Если диапазона нет, возвращаем как есть
            else:
                return target_dim

        # Ищем ближайшие совместимые значения
        lower_dim = target_dim - (target_dim % heads)
        upper_dim = lower_dim + heads

        # Если есть конфиг, выбираем лучший вариант в пределах диапазона
        if config:
            try:
                dim_min, dim_max = config.nas.architecture_search.model_params.dim.range
                
                is_lower_valid = lower_dim >= dim_min
                is_upper_valid = upper_dim <= dim_max

                if is_lower_valid and is_upper_valid:
                    return lower_dim if (target_dim - lower_dim) < (upper_dim - target_dim) else upper_dim
                elif is_lower_valid:
                    return lower_dim
                elif is_upper_valid:
                    return upper_dim
                else:
                    # Если оба варианта вне диапазона, используем fallback
                    return int(config.nas.architecture_search.fallback_dim)

            except Exception:
                # Если конфиг есть, но в нем нет диапазона, выбираем просто ближайший
                return lower_dim if (target_dim - lower_dim) < (upper_dim - target_dim) else upper_dim
        
        # Если конфига нет, просто выбираем ближайший
        return lower_dim if (target_dim - lower_dim) < (upper_dim - target_dim) else upper_dim

    def __init__(self, image_size=224, patch_size=16, num_classes=101, 
                 dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        
        # Сохраняем параметры для NAS
        self.config = {
            'image_size': image_size,
            'patch_size': patch_size,
            'num_classes': num_classes,
            'dim': dim,
            'depth': depth,
            'heads': heads,
            'mlp_dim': mlp_dim,
            'dropout': dropout
        }
        
        # Расчет количества патчей
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding (обучаемый)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: (B, 3, 224, 224) -> (B, dim, H', W') -> (B, num_patches, dim)
        x = self.patch_embed(x)  # (B, dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, dim, H', W') -> (B, dim, num_patches) -> (B, num_patches, dim)
        
        # Добавляем class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, dim)
        
        # Добавляем positional embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Классификация: берем class token и пропускаем через head
        x = self.norm(x[:, 0])  # (B, dim)
        x = self.dropout(x)
        x = self.head(x)  # (B, num_classes)
        return x

    def get_num_params(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# =====================================================
# МОДУЛИ ДЛЯ CNN
# =====================================================

class ConvBlock(nn.Module):
    """
    Базовый сверточный блок: Conv2d -> BatchNorm2d -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        return x

class ResBlock(nn.Module):
    """
    Residual Block в стиле ResNet.
    Содержит несколько сверточных слоев и опциональный skip-connection.
    """
    def __init__(self, in_channels, out_channels, num_conv_layers=2, stride=1, use_skip=True, dropout_rate_for_convs: float = 0.0):
        super().__init__()
        self.use_skip = use_skip

        layers = []
        # Первая свертка в блоке меняет пространственный размер (downsampling)
        layers.append(ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, dropout_rate=dropout_rate_for_convs))
        
        # Последующие свертки в блоке сохраняют пространственный размер
        for j in range(1, num_conv_layers):
            layers.append(ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=dropout_rate_for_convs))
        
        self.conv_layers = nn.Sequential(*layers)

        self.shortcut = nn.Identity()
        if use_skip:
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv_layers(x)
        
        if self.use_skip:
            out += self.shortcut(identity)
        
        out = self.final_relu(out)
        return out


class CustomCNN(nn.Module):
    """
    Кастомная реализация сверточной нейронной сети в стиле ResNet с ResBlock.
    Параметры:
        in_channels (int): Количество входных каналов (обычно 3 для RGB изображений).
        num_classes (int): Количество классов для классификации.
        stem_channels (int): Количество каналов в первом сверточном слое (stem).
        stem_conv_kernel_size (int): Размер ядра для первой свертки в stem.
        stem_conv_stride (int): Шаг для первой свертки в stem.
        stem_conv_padding (int): Паддинг для первой свертки в stem.
        stem_pool_kernel_size (int): Размер ядра для MaxPool2d в stem.
        stem_pool_stride (int): Шаг для MaxPool2d в stem.
        stem_pool_padding (int): Паддинг для MaxPool2d в stem.
        block_channels (list): Список количества выходных каналов для каждого ResBlock.
        block_strides (list): Список значений stride для каждого ResBlock (для downsampling).
        convs_per_block (list): Список количества сверточных слоев внутри каждого ResBlock.
        use_skip_per_block (list): Список булевых значений для использования skip-connection в каждом ResBlock.
        hidden_dims (list): Список размеров скрытых слоев для полносвязных слоев.
        dropout (float): Процент dropout для полносвязных слоев.
        dropout_rates_per_block (list): Список процентов Dropout2d для каждого ResBlock.
    """
    def __init__(self, in_channels=3, num_classes=101, stem_channels=64,
                 stem_conv_kernel_size=7, stem_conv_stride=2, stem_conv_padding=3,
                 stem_pool_kernel_size=3, stem_pool_stride=2, stem_pool_padding=1,
                 block_channels=[64, 128, 256, 512],
                 block_strides=[1, 2, 2, 2], convs_per_block=[2, 2, 2, 2], use_skip_per_block=[True, True, True, True],
                 hidden_dims=[512], dropout=0.3, dropout_rates_per_block = None):
        super().__init__()
        
        if not (len(block_channels) == len(block_strides) == len(convs_per_block) == len(use_skip_per_block)):
            raise ValueError("All block configuration lists must have the same length.")

        if dropout_rates_per_block is None:
            dropout_rates_per_block = [0.0] * len(block_channels)
        if len(dropout_rates_per_block) != len(block_channels):
            raise ValueError(f"Length of dropout_rates_per_block ({len(dropout_rates_per_block)}) must match number of blocks ({len(block_channels)}).")

        # Stem: Начальная сверточная часть
        self.stem = nn.Sequential(
            ConvBlock(in_channels, stem_channels, 
                      kernel_size=stem_conv_kernel_size, stride=stem_conv_stride, padding=stem_conv_padding, dropout_rate=0.0), # Stem usually doesn't have dropout
            nn.MaxPool2d(kernel_size=stem_pool_kernel_size, stride=stem_pool_stride, padding=stem_pool_padding)
        )

        # Feature extraction stages using ResBlocks
        self.feature_stages = nn.ModuleList()
        current_in_channels = stem_channels
        
        for i in range(len(block_channels)):
            self.feature_stages.append(ResBlock(
                current_in_channels,
                block_channels[i],
                num_conv_layers=convs_per_block[i],
                stride=block_strides[i],
                use_skip=use_skip_per_block[i],
                dropout_rate_for_convs=dropout_rates_per_block[i]
            ))
            current_in_channels = block_channels[i]
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier (MLP)
        classifier_layers = []
        flattened_size = current_in_channels

        for hidden_dim in hidden_dims:
            classifier_layers.append(nn.Linear(flattened_size, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(dropout))
            flattened_size = hidden_dim
        
        classifier_layers.append(nn.Linear(flattened_size, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.feature_stages:
            x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_num_params(self):
        """Возвращает количество параметров модели"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

def get_model(cfg):
    """Универсальная функция для создания моделей"""
    model_type = cfg.model.model_type

    if model_type == 'custom':
        if cfg.model.custom_model_type == 'vit':
            mlp_dim = cfg.model.mlp_dim if hasattr(cfg.model, 'mlp_dim') else int(cfg.model.dim * cfg.model.mlp_ratio)
            model = ViT(
                image_size=cfg.model.image_size,
                patch_size=cfg.model.patch_size,
                num_classes=cfg.model.num_classes,
                dim=cfg.model.dim,
                depth=cfg.model.depth,
                heads=cfg.model.heads,
                mlp_dim=mlp_dim,
                dropout=cfg.model.dropout
            )
        elif cfg.model.custom_model_type == 'cnn':
            model = CustomCNN(
                in_channels=cfg.model.in_channels,
                num_classes=cfg.model.num_classes,
                stem_channels=cfg.model.stem_channels,
                stem_conv_kernel_size=getattr(cfg.model, 'stem_conv_kernel_size', 7),
                stem_conv_stride=getattr(cfg.model, 'stem_conv_stride', 2),
                stem_conv_padding=getattr(cfg.model, 'stem_conv_padding', 3),
                stem_pool_kernel_size=getattr(cfg.model, 'stem_pool_kernel_size', 3),
                stem_pool_stride=getattr(cfg.model, 'stem_pool_stride', 2),
                stem_pool_padding=getattr(cfg.model, 'stem_pool_padding', 1),
                block_channels=cfg.model.block_channels,
                block_strides=cfg.model.block_strides,
                convs_per_block=cfg.model.convs_per_block,
                use_skip_per_block=cfg.model.use_skip_per_block,
                hidden_dims=cfg.model.hidden_dims,
                dropout=cfg.model.dropout,
                dropout_rates_per_block=getattr(cfg.model, 'dropout_rates_per_block', None) # New parameter
            )
        else:
            raise ValueError(f"Unknown custom model type: {cfg.model.custom_model_type}")
    elif model_type == 'timm':
        model = timm.create_model(
            cfg.model.timm_model_name,
            pretrained=cfg.model.pretrained,
            num_classes=cfg.model.num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model
