"""
Vision Transformer (ViT) Implementation
"""
import torch
import torch.nn as nn


class PatchExtractor(nn.Module):
    """Extract patches from images."""
    
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
    
    def forward(self, images):
        """
        Args:
            images: (batch_size, channels, height, width)
        Returns:
            patches: (batch_size, num_patches, patch_dim)
        """
        batch_size, channels, height, width = images.shape
        
        # Extract patches using unfold
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)
        patches = patches.contiguous().view(batch_size, -1, channels * self.patch_size * self.patch_size)
        
        return patches


class PatchEncoder(nn.Module):
    """Encode patches with linear projection and positional embeddings."""
    
    def __init__(self, num_patches, patch_dim, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = nn.Linear(patch_dim, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
    
    def forward(self, patches):
        """
        Args:
            patches: (batch_size, num_patches, patch_dim)
        Returns:
            encoded: (batch_size, num_patches, projection_dim)
        """
        batch_size = patches.shape[0]
        positions = torch.arange(0, self.num_patches, device=patches.device)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention and MLP."""
    
    def __init__(self, projection_dim, num_heads, mlp_units, dropout_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(projection_dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            projection_dim, 
            num_heads, 
            dropout=dropout_rate, 
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(projection_dim, eps=1e-6)
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, mlp_units[0]),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_units[0], mlp_units[1]),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_patches, projection_dim)
        Returns:
            x: (batch_size, num_patches, projection_dim)
        """
        # Multi-head attention with residual connection
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer for image classification."""
    
    def __init__(self, config):
        super().__init__()
        
        # Extract config parameters
        image_size = config['data']['image_size']
        patch_size = config['model']['patch_size']
        num_classes = config['data']['num_classes']
        projection_dim = config['model']['projection_dim']
        num_heads = config['model']['num_heads']
        transformer_layers = config['model']['transformer_layers']
        transformer_units = config['model']['transformer_units']
        mlp_head_units = config['model']['mlp_head_units']
        dropout_rate = config['model']['dropout_rate']
        mlp_dropout_rate = config['model']['mlp_dropout_rate']
        
        # Calculate dimensions
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        
        # Model components
        self.patch_extractor = PatchExtractor(patch_size)
        self.patch_encoder = PatchEncoder(self.num_patches, self.patch_dim, projection_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(projection_dim, num_heads, transformer_units, dropout_rate)
            for _ in range(transformer_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(projection_dim, eps=1e-6)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(mlp_dropout_rate)
        
        # MLP head
        self.mlp_head = nn.Sequential(
            nn.Linear(self.num_patches * projection_dim, mlp_head_units[0]),
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.GELU(),
            nn.Dropout(mlp_dropout_rate)
        )
        
        self.classifier = nn.Linear(mlp_head_units[1], num_classes)
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (batch_size, channels, height, width)
            return_attention: if True, return attention weights
        Returns:
            logits: (batch_size, num_classes)
            attention_weights: optional attention maps
        """
        # Extract and encode patches
        patches = self.patch_extractor(x)
        encoded_patches = self.patch_encoder(patches)
        
        # Pass through transformer blocks
        attention_weights = []
        for block in self.transformer_blocks:
            if return_attention:
                # Store attention weights from last block
                x_norm = block.norm1(encoded_patches)
                _, attn_w = block.attn(x_norm, x_norm, x_norm)
                attention_weights.append(attn_w)
            encoded_patches = block(encoded_patches)
        
        # Classification head
        representation = self.norm(encoded_patches)
        representation = self.flatten(representation)
        representation = self.dropout(representation)
        features = self.mlp_head(representation)
        logits = self.classifier(features)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_maps(self, x):
        """Get attention maps for visualization."""
        logits, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


def create_model(config):
    """Factory function to create model."""
    model = VisionTransformer(config)
    return model


if __name__ == "__main__":
    # Test the model
    import yaml
    
    with open('E:/PROJECT/Vit_Classifier_CIFAR10/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model = create_model(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, config['data']['image_size'], config['data']['image_size'])
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")