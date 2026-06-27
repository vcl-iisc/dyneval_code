#!/usr/bin/env python3
"""
Batch runner script for generating images with multiple models in t2_train_gen_scripts.
Allows running multiple models sequentially or managing batch jobs.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Map of model names to script files
MODEL_SCRIPTS = {
    "flux1.dev": "flux1_dev.py",
    "omnigen": "omnigen.py",
    "omnigen2": "omnigen2.py",
    "incontext-lora": "incontext_lora.py",
}


class BatchRunner:
    def __init__(self, config_file: Optional[str] = None):
        self.script_dir = Path(__file__).parent
        self.config = self.load_config(config_file) if config_file else {}

    def load_config(self, config_file: str) -> Dict:
        """Load configuration from JSON file"""
        with open(config_file, "r") as f:
            return json.load(f)

    def save_config(self, config: Dict, config_file: str):
        """Save configuration to JSON file"""
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def get_model_script(self, model_name: str) -> Path:
        """Get the script path for a model"""
        if model_name not in MODEL_SCRIPTS:
            raise ValueError(f"Unknown model: {model_name}")
        return self.script_dir / MODEL_SCRIPTS[model_name]

    def run_model(
        self,
        model_name: str,
        prompts_file: str,
        start_line: int,
        finish_line: int,
        output_dir: str,
        extra_args: Optional[Dict] = None,
    ) -> bool:
        """Run a single model generation"""
        script_path = self.get_model_script(model_name)

        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--prompts_file",
            prompts_file,
            "--start_line",
            str(start_line),
            "--finish_line",
            str(finish_line),
            "--output_dir",
            output_dir,
        ]

        # Add extra arguments for specific models
        if extra_args:
            for key, value in extra_args.items():
                cmd.extend([f"--{key}", str(value)])

        print(f"🚀 Starting {model_name}...")
        print(f"   Command: {' '.join(cmd)}")

        start_time = time.time()

        try:
            result = subprocess.run(cmd, check=True)
            end_time = time.time()
            duration = end_time - start_time
            print(f"✅ {model_name} completed successfully in {duration:.1f}s")
            return True

        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time
            print(
                f"❌ {model_name} failed after {duration:.1f}s (exit code {e.returncode})"
            )
            return False
        except KeyboardInterrupt:
            print(f"⚠️ {model_name} interrupted by user")
            return False

    def run_batch(
        self,
        models: List[str],
        prompts_file: str,
        start_line: int,
        finish_line: int,
        base_output_dir: str,
        stop_on_error: bool = False,
    ) -> Dict[str, bool]:
        """Run multiple models in sequence"""
        results = {}

        print(f"📋 Running batch job with {len(models)} models")
        print(f"   Prompts: {prompts_file} (lines {start_line}-{finish_line})")
        print(f"   Base output dir: {base_output_dir}")
        print()

        for i, model_name in enumerate(models, 1):
            print(f"[{i}/{len(models)}] Processing {model_name}")

            # Create model-specific output directory
            output_dir = os.path.join(base_output_dir, model_name)
            os.makedirs(output_dir, exist_ok=True)

            # Get extra args from config if available
            extra_args = self.config.get("models", {}).get(model_name, {})

            success = self.run_model(
                model_name=model_name,
                prompts_file=prompts_file,
                start_line=start_line,
                finish_line=finish_line,
                output_dir=output_dir,
                extra_args=extra_args,
            )

            results[model_name] = success

            if not success and stop_on_error:
                print(f"🛑 Stopping batch job due to error in {model_name}")
                break

            print()  # Empty line for readability

        return results

    def print_results(self, results: Dict[str, bool]):
        """Print batch job results summary"""
        successful = [model for model, success in results.items() if success]
        failed = [model for model, success in results.items() if not success]

        print("📊 Batch Job Results:")
        print(
            f"   ✅ Successful ({len(successful)}): {', '.join(successful) if successful else 'None'}"
        )
        print(
            f"   ❌ Failed ({len(failed)}): {', '.join(failed) if failed else 'None'}"
        )
        print(f"   📈 Success Rate: {len(successful) / len(results) * 100:.1f}%")


def create_sample_config():
    """Create a sample configuration file"""
    config = {
        "models": {
            "omnigen2": {
                "model_path": "/path/to/OmniGen2-model-weight",
                "transformer_path": "/path/to/custom/transformer",
                "dtype": "bf16",
                "num_inference_steps": 30,
                "text_guidance_scale": 5.0,
            },
            "incontext-lora": {
                "model_path": "black-forest-labs/FLUX.1-dev",
                "lora_repo_id": "ali-vilab/In-Context-LoRA",
            },
            "flux1.dev": {"model_path": "black-forest-labs/FLUX.1-dev"},
        }
    }
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for T2 text-to-image models"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        choices=list(MODEL_SCRIPTS.keys()),
        help="Models to run",
    )
    parser.add_argument(
        "--all-models", action="store_true", help="Run all available models"
    )
    parser.add_argument(
        "--prompts_file", type=str, required=True, help="Path to prompts file"
    )
    parser.add_argument(
        "--start_line", type=int, required=True, help="Start line number (1-based)"
    )
    parser.add_argument(
        "--finish_line", type=int, required=True, help="Finish line number (1-based)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Base output directory"
    )
    parser.add_argument(
        "--config", type=str, help="Configuration file for model-specific arguments"
    )
    parser.add_argument(
        "--stop_on_error", action="store_true", help="Stop batch job if any model fails"
    )
    parser.add_argument(
        "--list-models", action="store_true", help="List all available models"
    )
    parser.add_argument(
        "--create-config", type=str, help="Create sample configuration file"
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_models:
        print("Available models:")
        for model_name in sorted(MODEL_SCRIPTS.keys()):
            print(f"  - {model_name}")
        return

    if args.create_config:
        config = create_sample_config()
        runner = BatchRunner()
        runner.save_config(config, args.create_config)
        print(f"✅ Sample configuration saved to {args.create_config}")
        return

    # Determine models to run
    if args.all_models:
        models = list(MODEL_SCRIPTS.keys())
    elif args.models:
        models = args.models
    else:
        print("❌ Must specify either --models or --all-models")
        return 1

    # Run batch job
    runner = BatchRunner(args.config)
    results = runner.run_batch(
        models=models,
        prompts_file=args.prompts_file,
        start_line=args.start_line,
        finish_line=args.finish_line,
        base_output_dir=args.output_dir,
        stop_on_error=args.stop_on_error,
    )

    # Print results
    runner.print_results(results)

    # Return appropriate exit code
    failed_count = sum(1 for success in results.values() if not success)
    return min(failed_count, 1)  # Return 1 if any failures, 0 if all successful


if __name__ == "__main__":
    sys.exit(main())
