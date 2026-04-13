import torch
from torch import nn
import torch.nn.functional as F
from utils import CANONICAL_PAIRS_TENSOR

class HelixCenterMaskedPriorGenerator(torch.nn.Module):
    """
    Based on HelixPairMaskedPriorGenerator, if (i, j) itself cannot form a valid base pair, then the corresponding stacking features are all set to zero.
    """
    def __init__(self, K=5, min_dist=3):
        super().__init__()
        self.K = K
        self.min_dist = min_dist
        # One-hot lookup table: 0-3 correspond to A, U, C, G; 4 corresponds to N ([0, 0, 0, 0])
        self.embedding = torch.nn.Embedding(5, 4)
        self.embedding.weight.data = torch.eye(5)[:4].T.contiguous()
        self.embedding.weight.requires_grad = False
        
        self.register_buffer('canonical_lookup', CANONICAL_PAIRS_TENSOR)

    def forward(self, seq_indices, legal_mask):
        B, L = seq_indices.shape
        device = seq_indices.device
        K = self.K
        
        # --- Part 1: Base Info ---
        one_hot = self.embedding(seq_indices) # (B, L, 4)
        rows = one_hot.unsqueeze(2).expand(B, L, L, 4)
        cols = one_hot.unsqueeze(1).expand(B, L, L, 4)
        base_info = torch.cat([rows, cols], dim=-1) # (B, L, L, 8)
        
        # --- Part 2: Stacking Potential ---
        # 1. Padding & Flatten
        padded_seq = torch.nn.functional.pad(seq_indices, (K, K), value=4)
        flat_padded = padded_seq.view(-1).contiguous()
        
        # 2. Index computation
        k_offsets = torch.arange(-K, K + 1, device=device).view(1, 1, 1, -1).contiguous()
        i_idx = torch.arange(L, device=device).view(1, L, 1, 1).contiguous()
        j_idx = torch.arange(L, device=device).view(1, 1, L, 1).contiguous()
        
        # Relative indices
        ik_rel = (i_idx + K) + k_offsets
        jk_rel = (j_idx + K) - k_offsets
        
        # Batch offset handling
        stride = L + 2 * K
        batch_offsets = (torch.arange(B, device=device) * stride).view(B, 1, 1, 1).contiguous()
        
        ik_global = ik_rel + batch_offsets
        jk_global = jk_rel + batch_offsets
        
        ik_global = ik_global.expand(-1, -1, L, -1)
        jk_global = jk_global.expand(-1, L, -1, -1)
        
        # 3. Extract bases
        bases_ik = flat_padded[ik_global]
        bases_jk = flat_padded[jk_global]
        
        # 4. Compute the mask
        # Condition 1: distance constraint
        dist_matrix = (j_idx - i_idx).squeeze(-1)
        mask_dist = dist_matrix.unsqueeze(-1) > (self.min_dist + 2 * k_offsets)
        
        # Condition 2: pairing constraint
        mask_canonical = self.canonical_lookup[bases_ik, bases_jk]
        
        # Use only distance and canonical-pair constraints
        final_mask = mask_dist & mask_canonical
        final_mask = final_mask.float()
        
        # 5. Extract features and apply mask
        feat_ik = self.embedding(bases_ik)
        feat_jk = self.embedding(bases_jk)
        
        feat_stack = torch.cat([feat_ik, feat_jk], dim=-1)
        feat_stack = feat_stack * final_mask.unsqueeze(-1)
        
        feat_stack = feat_stack.view(B, L, L, -1).contiguous()
        feat_stack = feat_stack * legal_mask.unsqueeze(-1)
        
        # --- Final concatenation ---
        result = torch.cat([base_info, feat_stack], dim=-1)
        
        return result
    
class PGAA(nn.Module):
    """
    Use the input logits as attention weights to perform axial feature aggregation on the 2D feature map.
    """
    def __init__(self, channels):
        super().__init__()
        self.row_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.col_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.projection = nn.Conv2d(channels * 2, channels, kernel_size=1)
    
    def forward(self, features, logits, mask):
        # 1. Apply the mask to the logits to prepare for softmax
        weights = F.sigmoid(logits)
        masked_weights = weights * mask.unsqueeze(1)
        eps = 1e-3

        # 2. Row-wise attention and feature aggregation
        row_value = self.row_proj(features)
        row_aggregated = torch.sum(masked_weights * row_value, dim=-1, keepdim=True) / (torch.sum(masked_weights, dim=-1, keepdim=True) + eps)
        row_broadcasted = row_aggregated.expand_as(features)

        # 3. Column-wise attention and feature aggregation
        col_value = self.col_proj(features)
        col_aggregated = torch.sum(masked_weights * col_value, dim=-2, keepdim=True) / (torch.sum(masked_weights, dim=-2, keepdim=True) + eps)
        col_broadcasted = col_aggregated.expand_as(features)

        # 4. Concatenate and fuse
        fused_features = torch.cat([row_broadcasted, col_broadcasted], dim=1)
        projected_features = self.projection(fused_features)
        
        return projected_features

class FeedForward(nn.Module):
    """
    A simple feed-forward network.
    """
    def __init__(self, channels, ff_kernel_size, ff_expansion, dropout_rate=0.1):
        super().__init__()
        hidden_dim = int(channels * ff_expansion)
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=ff_kernel_size, padding=ff_kernel_size//2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=ff_kernel_size, padding=ff_kernel_size//2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class StepFoldBlock(nn.Module):
    def __init__(self, channels, ff_kernel_size, ff_expansion, ffn_depth, is_first=False):
        super().__init__()
        self.is_first = is_first
        if not is_first:
            self.axial_attention = PGAA(channels=channels)
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            )
            
        self.ffn = nn.Sequential(
            *[FeedForward(channels, ff_kernel_size, ff_expansion) for _ in range(ffn_depth)]
        )

        self.output_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x_current, legal_masks, band_mask, padding_mask, prev_logits=None):
        
        if self.is_first:
            # The first block skips attention and directly processes the input features with a simple encoder.
            attn_output = x_current + self.encoder(x_current)
            attn_output = self.norm1(attn_output.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        else:
            # Pre-Norm + Attention + Residual
            attn_output = self.axial_attention(x_current, prev_logits, legal_masks * band_mask)
            attn_output = x_current + attn_output
            attn_output = self.norm1(attn_output.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        
        # Pre-Norm + FFN + Residual
        ffn_out = self.ffn(attn_output)
        ffn_out = attn_output + ffn_out
        x_out = self.norm2(ffn_out.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        
        new_logits = self.output_head(x_out)

        return x_out, new_logits

class StepFoldNet(nn.Module):
    def __init__(self, K_local=6, K_global=6, hidden_dim=48, helix_prior_K=5, ff_kernel_size=3, ff_expansion=2, ff_depth=2):
        super().__init__()
        
        self.K_total = K_local + K_global
        
        self.helix_prior_proj = nn.Conv2d((helix_prior_K * 2 + 2)*8, hidden_dim, kernel_size=1)
        
        self.step_blocks = nn.ModuleList([
            StepFoldBlock(hidden_dim, ff_kernel_size, ff_expansion, ff_depth, is_first=(i==0))
            for i in range(self.K_total)
        ])

    def forward(self, input):
        helix_prior, band_masks, legal_masks, padding_masks = input
        
        x_initial = self.helix_prior_proj(helix_prior.permute(0, 3, 1, 2).contiguous())
        
        x_current = x_initial
        prev_logits = None
        
        predictions = []
        for i in range(self.K_total):
            mask_k = band_masks[:, i-1, :, :]
            
            # Call the StepFoldBlock
            x_current, new_logits = self.step_blocks[i](
                x_current,
                legal_masks=legal_masks,
                band_mask=mask_k,
                padding_mask=padding_masks,
                prev_logits=prev_logits
            )

            prev_logits = new_logits

            # Apply symmetrization at the final step
            if i == self.K_total - 1:
                transposed_logits = new_logits.transpose(-2, -1).contiguous()
                final_logits = (new_logits + transposed_logits) / 2
                predictions.append(final_logits.squeeze(1))
            else:
                predictions.append(new_logits.squeeze(1))

        return predictions

    def inference(self, input):
        predictions = self.forward(input)
        return predictions[-1]

class LossFunc(nn.Module):
    def __init__(self, K_total, alpha=1.0, pos_weight=1.0):
        """
        Args:
            K_total (int): The number of blocks in the model.
            alpha (float): The exponential base used to compute the loss weights for different steps. alpha > 1 assigns larger weights to later steps.
            pos_weight (float): Weight for positive samples. A value greater than 1 increases the importance of positive samples in the loss.
        """
        super().__init__()

        self.K_total = K_total
        self.pos_weight = pos_weight
        
        weights = [(alpha**(i + 1)) for i in range(self.K_total)]
        
        total_weight = max(weights)
        self.loss_weights = [w / total_weight for w in weights]
        
        self.loss_weights_tensor = torch.tensor(self.loss_weights, dtype=torch.float)
        
        print("Loss Weights per Step:", self.loss_weights)

    def forward(self, predictions, contact_maps, band_masks):
        device = predictions[0].device
        
        pred_tensor = torch.stack(predictions, dim=1) # (B, K, L_max, L_max)
        target_tensor = contact_maps.unsqueeze(1).expand_as(pred_tensor) # (B, K, L_max, L_max)
                  
        pos_weight_tensor = torch.tensor(self.pos_weight, device=device)
        
        loss_per_element = F.binary_cross_entropy_with_logits(
            pred_tensor, 
            target_tensor,          
            reduction='none',
            pos_weight=pos_weight_tensor
        ) # (B, K+1, L_max, L_max)
        
        masked_loss = loss_per_element * band_masks
        
        step_loss_sum = masked_loss.sum(dim=(0, 2, 3)) # (K+1,)
        step_pixel_count = band_masks.sum(dim=(0, 2, 3))
        step_pixel_count = torch.clamp(step_pixel_count, min=1.0)
        
        step_mean_loss = step_loss_sum / step_pixel_count # (K+1,)
        
        weights = self.loss_weights_tensor.to(device)
        total_loss = (step_mean_loss * weights).sum()
            
        return total_loss