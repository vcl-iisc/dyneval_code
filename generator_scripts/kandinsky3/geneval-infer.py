import os
import torch
import multiprocessing as mp
from pathlib import Path
from diffusers import AutoPipelineForText2Image
from tqdm import tqdm

# =====================
# CONFIG
# =====================
MODEL_ID = "kandinsky-community/kandinsky-3"
PROMPTS_FILE = "/home/anirban/dheerajbaiju/dheeraj/dyneval-models/geneval-final-prompt-order.txt"   # <-- your .txt file (one prompt per line)
OUTDIR = "/home/anirban/dheerajbaiju/DynEVAL_Benchmark_output/kandinsky3-geneval-results"
IMAGES_PER_PROMPT = 4
NUM_STEPS = 25
SEED = 0
DTYPE = torch.float16

# Reduce CUDA fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# =====================
# UTILS
# =====================
def load_prompts(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_completed_prompt_indices(outdir, images_per_prompt):
    completed = set()
    counter = {}

    for f in Path(outdir).glob("prompt*_*.png"):
        try:
            pid = int(f.stem.split("_")[0].replace("prompt", ""))
            counter[pid] = counter.get(pid, 0) + 1
        except:
            pass

    for k, v in counter.items():
        if v >= images_per_prompt:
            completed.add(k)

    return completed


# =====================
# WORKER (1 GPU)
# =====================
def worker(gpu_id, prompts, indices):
    torch.cuda.set_device(gpu_id)

    pipe = AutoPipelineForText2Image.from_pretrained(
        MODEL_ID,
        variant="fp16",
        torch_dtype=DTYPE
    ).to(f"cuda:{gpu_id}")

    # Kandinsky-3 memory saver (works)
    pipe.enable_attention_slicing()

    # Speed-friendly flags
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    generator = torch.Generator(device=f"cuda:{gpu_id}").manual_seed(SEED + gpu_id)

    for idx in indices:
        prompt = prompts[idx]

        with torch.inference_mode():
            result = pipe(
                [prompt] * IMAGES_PER_PROMPT,
                num_inference_steps=NUM_STEPS,
                generator=generator
            )

        for img_idx, image in enumerate(result.images):
            out_path = os.path.join(OUTDIR, f"prompt{idx}_{img_idx}.png")
            if not os.path.exists(out_path):
                image.save(out_path)


# =====================
# MAIN
# =====================
def main():
    os.makedirs(OUTDIR, exist_ok=True)

    prompts = load_prompts(PROMPTS_FILE)
    completed = get_completed_prompt_indices(OUTDIR, IMAGES_PER_PROMPT)
    remaining = [i for i in range(len(prompts)) if i not in completed]

    print(f"📄 Total prompts     : {len(prompts)}")
    print(f"✅ Completed prompts : {len(completed)}")
    print(f"🕗 Remaining prompts : {len(remaining)}")

    if len(remaining) == 0:
        print("🎉 Nothing to do — all prompts already completed.")
        print("succ")
        return

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("❌ No GPUs detected")

    print(f"🖥️ Using {num_gpus} GPUs")

    # Split prompts across GPUs
    chunks = [remaining[i::num_gpus] for i in range(num_gpus)]

    procs = []
    for gpu_id in range(num_gpus):
        if len(chunks[gpu_id]) == 0:
            continue
        p = mp.Process(target=worker, args=(gpu_id, prompts, chunks[gpu_id]))
        p.start()
        procs.append(p)

    # Wait + fail loudly on crash
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"❌ Worker {p.pid} failed with exit code {p.exitcode}")

    print("succ")
    print("🎉 All generations complete!")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
