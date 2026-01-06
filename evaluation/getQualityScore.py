import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from evaluation.QualityModel import QualityModel


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description='Extract quality scores using VIT-FIQA models')

    parser.add_argument('--data-dir', type=str,
                        default='../data/',
                        help='Root directory for evaluation dataset')
    parser.add_argument('--output-dir', type=str,
                        default='../results/extracted_quality_scores',
                        help='Directory to save quality scores')
    parser.add_argument('--datasets', type=str,
                        default='lfw,calfw,cplfw,agedb_30,cfp_fp,XQLFW',
                        help='Comma-separated list of datasets (e.g., XQLFW,lfw,calfw,agedb_30,cfp_fp,cplfw,IJBC)')
    parser.add_argument('--gpu-id', type=int,
                        default=0,
                        help='GPU device ID')
    parser.add_argument('--model-path', type=str,
                        default="../pretrained/",
                        help='Path to directory containing pretrained model')
    parser.add_argument('--model-name', type=str,
                        default="minchul_cvlface_adaface_vit_base_webface4m.pt",
                        help='Name of the pretrained model file')
    parser.add_argument('--backbone', type=str,
                        default="vitb",
                        choices=['vits', 'vitb', 'clip'],
                        help='Backbone architecture (vits, vitb, or clip)')
    parser.add_argument('--batch-size', type=int,
                        default=32,
                        help='Batch size for processing')
    parser.add_argument('--color-channel', type=str,
                        default="BGR",
                        choices=['BGR', 'RGB'],
                        help='Input image color channel format')
    parser.add_argument('--scaling', type=float,
                        default=8.0,
                        help='Scaling factor for NT-FIQA quality score computation')
    parser.add_argument('--ntfiq-use-attention-weights', type=str,
                        default='false',
                        choices=['true', 'false'],
                        help='Whether to use attention weights for patch aggregation (NT-FIQA only)')
    parser.add_argument('--last-block-attention-only', type=str,
                        default='true',
                        choices=['true', 'false'],
                        help='Use only last block attention weights (NT-FIQA only)')
    parser.add_argument('--blocks-to-use', type=str,
                        default=None,
                        help='Comma-separated list of block indices to use (NT-FIQA only), e.g., "0,1,2,3" or "20,21,22,23"')
    parser.add_argument('--num-blocks', type=int,
                        default=None,
                        help='(Deprecated) Number of transformer blocks to use. Use --blocks-to-use instead.')

    return parser.parse_args(argv)


def read_image_list(image_list_file, image_dir=''):
    """Read list of image paths from file."""
    image_lists = []
    with open(image_list_file) as f:
        absolute_list = f.readlines()
        for line in absolute_list:
            image_lists.append(os.path.join(image_dir, line.rstrip()))
    return image_lists, absolute_list


def main(param):
    datasets = param.datasets.split(',')

    # Parse blocks_to_use from command line
    blocks_to_use = None
    if param.blocks_to_use is not None:
        # Parse comma-separated block indices
        blocks_to_use = [int(x.strip()) for x in param.blocks_to_use.split(',')]
    elif param.num_blocks is not None:
        # Backward compatibility: convert num_blocks to blocks_to_use
        blocks_to_use = list(range(param.num_blocks))

    print(f"Initializing Quality Model...")
    print(f"  Model path: {param.model_path}")
    print(f"  Model name: {param.model_name}")
    print(f"  Backbone: {param.backbone}")
    print(f"  Scaling: {param.scaling}")
    print(f"  Use attention weights: {param.ntfiq_use_attention_weights}")
    print(f"  Last block attention only: {param.last_block_attention_only}")
    print(f"  Blocks to use: {blocks_to_use}")
    print(f"  GPU ID: {param.gpu_id}")

    # Initialize face quality model
    face_model = QualityModel(
        model_path=param.model_path,
        model_name=param.model_name,
        gpu_id=param.gpu_id,
        backbone=param.backbone,
        scaling=param.scaling,
        ntfiq_use_attention_weights=(param.ntfiq_use_attention_weights.lower() == 'true'),
        last_block_attention_only=(param.last_block_attention_only.lower() == 'true'),
        blocks_to_use=blocks_to_use
    )

    for dataset_name in datasets:
        print(f"\nProcessing dataset: {dataset_name}")

        # Construct paths
        root = param.data_dir
        image_list_file = os.path.join(param.data_dir, 'quality_data', dataset_name, 'image_path_list.txt')

        if not os.path.exists(image_list_file):
            print(f"  Warning: Image list file not found: {image_list_file}")
            print(f"  Skipping dataset {dataset_name}")
            continue

        # Read image paths
        image_list, absolute_list = read_image_list(image_list_file, root)
        print(f"  Found {len(image_list)} images")

        # Extract features and quality scores
        print(f"  Extracting quality getQualityScore...")
        _, quality = face_model.get_batch_feature(
            image_list,
            batch_size=param.batch_size,
            color=param.color_channel
        )

        # Create output directory
        output_dataset_dir = os.path.join(param.output_dir, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)

        # Set backbone label based on model_name
        if param.backbone == 'vitb':
            if '12m' in param.model_name.lower():
                backbone_label = 'VITB12M'
            elif '4m' in param.model_name.lower():
                backbone_label = 'VITB4M'
            else:
                backbone_label = 'VITB'
        elif param.backbone == 'vits':
            backbone_label = 'VITS'
        elif param.backbone == 'clip':
            backbone_label = 'CLIP'
        else:
            backbone_label = param.backbone.upper()

        # Ablation naming
        ablation_label = ""
        if blocks_to_use is not None:
            # Create a descriptive label based on blocks_to_use
            if len(blocks_to_use) > 0:
                # Check if it's a simple range from 0
                if blocks_to_use == list(range(len(blocks_to_use))):
                    # Simple case: first N blocks
                    ablation_label = f"BLOCKDEPTH{len(blocks_to_use)}"
                else:
                    # Custom block selection
                    blocks_str = '_'.join(map(str, blocks_to_use))
                    ablation_label = f"BLOCKS{blocks_str}"

        if param.ntfiq_use_attention_weights.lower() == 'true':
            if param.last_block_attention_only.lower() == 'true':
                if ablation_label:
                    ablation_label = f"ATTENTIONLASTBLOCK_{ablation_label}"
                else:
                    ablation_label = "ATTENTIONLASTBLOCK"
            else:
                if ablation_label:
                    ablation_label = f"ATTENTIONALLBLOCKS_{ablation_label}"
                else:
                    ablation_label = "ATTENTIONALLBLOCKS"

        if ablation_label:
            filename = f"NTFIQA_{backbone_label}_{ablation_label}_{dataset_name}.txt"
        else:
            filename = f"NTFIQA_{backbone_label}_{dataset_name}.txt"
        output_file = os.path.join(output_dataset_dir, filename)

        # Save quality scores
        with open(output_file, "w") as f:
            for i in range(len(quality)):
                f.write(f"{absolute_list[i].rstrip()} {quality[i][0]:.15f}\n")

        print(f"  Saved quality scores to: {output_file}")
        print(f"  Quality score statistics:")
        print(f"    Mean: {quality.mean():.15f}")
        print(f"    Std:  {quality.std():.15f}")
        print(f"    Min:  {quality.min():.15f}")
        print(f"    Max:  {quality.max():.15f}")


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
