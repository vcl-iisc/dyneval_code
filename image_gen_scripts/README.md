# T2 Train Generation Scripts

This directory contains standardized generation scripts for additional text-to-image models. Each script takes a consistent set of arguments and supports resumable generation.

## Available Models

- `flux1_dev.py` - FLUX.1-dev
- `omnigen.py` - OmniGen v1
- `omnigen2.py` - OmniGen2
- `incontext_lora.py` - In-Context LoRA (FLUX.1-dev + LoRA)

## Usage

All scripts follow the same command-line interface:

```bash
python <model_script>.py \
  --prompts_file <path_to_prompts.txt> \
  --start_line <start_line_number> \
  --finish_line <finish_line_number> \
  --output_dir <output_directory>
```

### Arguments

- `--prompts_file`: Path to text file containing prompts (one per line)
- `--start_line`: Starting line number (1-based indexing)
- `--finish_line`: Ending line number (1-based indexing)
- `--output_dir`: Directory where generated images will be saved

### Example

```bash
python flux1_dev.py \
  --prompts_file /path/to/prompts.txt \
  --start_line 1 \
  --finish_line 100 \
  --output_dir /path/to/output/
```

This will process lines 1-100 from the prompts file and save images as `1.png`, `2.png`, ..., `100.png` in the output directory.

## Features

### Resumable Generation
All scripts support resumable generation. If the script is interrupted, you can run the same command again and it will skip already generated images and continue from where it left off.

### Consistent Output Format
- Images are saved as `<line_number>.png` where line_number corresponds to the 1-based line number in the prompts file
- All images are generated at 1024x1024 resolution
- PNG format is used for all outputs

### Memory Optimization
- Each script includes CUDA optimization settings
- Memory-efficient attention mechanisms where supported
- Single GPU operation (no multi-GPU complexity)

## Model-Specific Notes

### FLUX.1-dev
Uses the standard FLUX.1-dev model with:
- 28 inference steps
- Guidance scale: 3.5
- bfloat16 precision

### OmniGen
Uses the OmniGen-v1 model with:
- 50 inference steps
- Guidance scale: 2.5
- float16 precision
- CPU offloading enabled for memory efficiency

### OmniGen2
Requires model path specification:
```bash
python omnigen2.py \
  --prompts_file /path/to/prompts.txt \
  --start_line 1 \
  --finish_line 100 \
  --output_dir /path/to/output/ \
  --model_path /path/to/OmniGen2-model-weight
```

Additional options:
- `--transformer_path`: Custom transformer weights (optional)
- `--dtype`: Model precision (fp32/fp16/bf16, default: bf16)
- `--num_inference_steps`: Number of steps (default: 30)
- `--text_guidance_scale`: Guidance scale (default: 5.0)

### In-Context LoRA
Uses FLUX.1-dev as base model with In-Context LoRA:
- 50 inference steps
- Automatic LoRA loading from HuggingFace
- VAE slicing and tiling for memory efficiency

Custom LoRA repository:
```bash
python incontext_lora.py \
  --prompts_file /path/to/prompts.txt \
  --start_line 1 \
  --finish_line 100 \
  --output_dir /path/to/output/ \
  --lora_repo_id custom/lora-repo
```

## Requirements

### General Dependencies
```
torch
diffusers
transformers
Pillow
tqdm
```

### Model-Specific Dependencies
- **FLUX.1-dev**: Standard diffusers installation
- **OmniGen**: Standard diffusers installation
- **OmniGen2**: Requires `omnigen2` package and custom model weights
- **In-Context LoRA**: Requires `diffusers` with FLUX and LoRA support

## Error Handling

- Scripts will print error messages for individual failed generations but continue processing remaining prompts
- Check console output for any error messages
- Failed generations will not create output files, so they can be retried

## Performance Tips

1. **GPU Memory**: If you encounter CUDA out of memory errors, try processing smaller batches by using smaller ranges of line numbers
2. **Disk Space**: Ensure sufficient disk space (approximately 2-5MB per 1024x1024 image)
3. **Model Loading**: First run may take longer due to model downloading/caching
4. **LoRA Models**: LoRA fusion can improve inference speed but may use more memory

## Output Structure

```
output_dir/
├── 1.png
├── 2.png
├── 3.png
└── ...
```

Where each number corresponds to the line number in the input prompts file.

## Model Comparison

| Model | Base Architecture | Inference Steps | Guidance Scale | Precision | Special Features |
|-------|------------------|-----------------|----------------|-----------|------------------|
| FLUX.1-dev | FLUX | 28 | 3.5 | bfloat16 | High quality, fast |
| OmniGen | Custom | 50 | 2.5 | float16 | CPU offloading |
| OmniGen2 | Custom | 30 | 5.0 | bfloat16 | Latest version, customizable |
| In-Context LoRA | FLUX + LoRA | 50 | - | bfloat16 | Style adaptation |