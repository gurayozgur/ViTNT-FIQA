import os
import sys

sys.path.append('../')
from FaceModel import FaceModel

import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backbones.vit.vit import VisionTransformer

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

class QualityModel(FaceModel):
    def __init__(self, model_path, model_name, gpu_id, backbone='vits',
                 scaling=8.0, ntfiq_use_attention_weights=False,
                 last_block_attention_only=True, blocks_to_use=None, num_blocks=None):
        """
        Initialize Quality Model.

        Args:
            model_path: Path to pretrained model weights
            model_name: Name of the model file (e.g., '10_vitb_w_pe_w_da_11-11_0.pt')
            gpu_id: GPU device ID
            backbone: Model size ('vits', 'vitb')
            scaling: Scaling factor for NT-FIQA
            ntfiq_use_attention_weights: Whether to use attention weights for patch aggregation
            last_block_attention_only: Use only last block's attention weights (for ntfiq)
            blocks_to_use: List of block indices to use (for ntfiq). If None or empty, uses all blocks.
            num_blocks: (Deprecated) Number of transformer blocks to use. Use blocks_to_use instead.
        """
        self.backbone = backbone
        self.ntfiq_use_attention_weights = ntfiq_use_attention_weights
        self.scaling = scaling
        self.last_block_attention_only = last_block_attention_only

        # Backward compatibility: convert num_blocks to blocks_to_use
        if num_blocks is not None and blocks_to_use is None:
            # If num_blocks is specified, use first num_blocks blocks
            self.blocks_to_use = list(range(num_blocks))
        else:
            self.blocks_to_use = blocks_to_use

        super(QualityModel, self).__init__(model_path, model_name, gpu_id, backbone)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        if backbone == 'vits':
            model = VisionTransformer(
                img_size=112,
                patch_size=8,
                num_classes=512,
                embed_dim=512,
                depth=12,
                mlp_ratio=5,
                num_heads=8,
                drop_path_rate=0.1,
                norm_layer="ln",
                mask_ratio=0.0
            )
        elif backbone == 'vitb':
            model = VisionTransformer(
                img_size=112,
                patch_size=8,
                num_classes=512,
                embed_dim=512,
                depth=24,
                mlp_ratio=3,
                num_heads=16,
                drop_path_rate=0.1,
                norm_layer="ln",
                mask_ratio=0.0
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Use 'small', or 'base'")

        checkpoint_path = os.path.join(prefix, epoch)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        dict_checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{ctx}")

        if 'state_dict' in dict_checkpoint:
            state_dict = dict_checkpoint['state_dict']
        elif 'model' in dict_checkpoint:
            state_dict = dict_checkpoint['model']
        else:
            state_dict = dict_checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            elif k.startswith('net.'):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(f"cuda:{ctx}")
        model.eval()

        return model

    def _getFeatureBlob(self, input_blob):
        """
        Extract features and quality scores from input images.

        Args:
            input_blob: Input images (batch_size, 3, 112, 112)

        Returns:
            features: Face embeddings
            quality_scores: Quality scores
        """
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)

        with torch.no_grad():
            qs = self.model.calculate_ntfiq(
                imgs,
                scaling=self.scaling,
                use_attention_weights=self.ntfiq_use_attention_weights,
                last_block_attention_only=self.last_block_attention_only,
                blocks_to_use=self.blocks_to_use
            )
        return None, qs.detach().cpu().numpy().reshape(-1, 1)

