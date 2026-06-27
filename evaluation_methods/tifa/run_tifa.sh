srun --partition=ada --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=48:00:00 --pty conda run -n tifa --live-stream python run_tifa.py
