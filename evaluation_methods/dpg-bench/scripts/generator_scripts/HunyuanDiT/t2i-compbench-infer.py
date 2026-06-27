import os
import torch
import json
from diffusers import HunyuanDiTPipeline
from tqdm import tqdm
import re

# =====================================================
# CONFIGURATION
# =====================================================
PROMPT_DIR = "../T2I-CompBench/examples/dataset"  # folder containing .txt prompt files
OUTPUT_DIR = "./t2i-compbench-outputs/examples/samples"  # output folder (follows eval structure)
MODEL_PATH = "hunyuandit-model-path"

IMAGES_PER_PROMPT = 4
GUIDANCE_SCALE = 7.5
NUM_STEPS = 60
HEIGHT, WIDTH = 512, 512
NEGATIVE_PROMPT = "blurry, low resolution"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD MODEL
# =====================================================
print("🚀 Loading model...")
pipe = HunyuanDiTPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16
).to("cuda")

pipe.transformer.to(memory_format=torch.channels_last)
pipe.vae.to(memory_format=torch.channels_last)
print("✅ Model loaded successfully.\n")

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_filename(name: str) -> str:
    """Convert prompt text into a safe filename (used for saving images)."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)  # remove invalid characters
    name = re.sub(r"\s+", " ", name).strip()  # normalize spaces
    return name[:180]  # avoid filename too long for OS


def load_all_prompts(prompt_dir: str):
    """Load all prompts from all .txt files in the directory."""
    all_prompts = []
    txt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    for txt_file in txt_files:
        with open(os.path.join(prompt_dir, txt_file), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            all_prompts.extend(lines)
    return all_prompts


def get_existing_question_ids(output_dir: str):
    """Return a set of existing question IDs based on image filenames."""
    existing_ids = set()
    for fname in os.listdir(output_dir):
        match = re.search(r"_(\d{6})\.png$", fname)
        if match:
            existing_ids.add(int(match.group(1)))
    return existing_ids


# =====================================================
# LOAD PROMPTS
# =====================================================
prompts = load_all_prompts(PROMPT_DIR)
total_prompts = len(prompts)

if total_prompts == 0:
    print("⚠️ No prompts found in .txt files!")
    exit()

print(f"🧠 Loaded {total_prompts} total prompts across dataset.\n")

# =====================================================
# LOAD EXISTING STATE (RESUME SUPPORT)
# =====================================================
vqa_result_path = os.path.join(os.path.dirname(OUTPUT_DIR), "vqa_result.json")
existing_results = []

if os.path.exists(vqa_result_path):
    try:
        with open(vqa_result_path, "r", encoding="utf-8") as f:
            existing_results = json.load(f)
        print(f"🔁 Resuming from previous run — {len(existing_results)} entries found.")
    except Exception as e:
        print(f"⚠️ Failed to read existing JSON: {e}. Starting fresh.")
        existing_results = []

existing_ids = get_existing_question_ids(OUTPUT_DIR)
question_counter = max(existing_ids) + 1 if existing_ids else 0
results = existing_results.copy()

# =====================================================
# SKIP COMPLETED PROMPTS
# =====================================================
images_done = len(existing_ids)
prompts_done = images_done // IMAGES_PER_PROMPT

if prompts_done > 0:
    print(f"🔁 Detected {images_done} existing images ({prompts_done} prompts completed).")
    print(f"⏭️ Skipping first {prompts_done} prompts...\n")
    prompts = prompts[prompts_done:]
else:
    print("🚀 Starting fresh (no completed prompts found).\n")

print(f"Starting generation from question_id: {question_counter:06d}\n")

# =====================================================
# MAIN GENERATION LOOP
# =====================================================
print(f"🎨 Generating {IMAGES_PER_PROMPT} images per prompt...")

for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating all prompts", unit="prompt")):
    prompt_name = clean_filename(prompt)

    for _ in range(IMAGES_PER_PROMPT):
        image_filename = f"{prompt_name}_{question_counter:06d}.png"
        output_path = os.path.join(OUTPUT_DIR, image_filename)

        # Resume support — skip already generated
        if os.path.exists(output_path):
            tqdm.write(f"⏭️ Skipping {image_filename} (already exists)")
        else:
            tqdm.write(f"🖼️ Generating: {image_filename}")
            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=NUM_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    num_images_per_prompt=1
                )
                img = result.images[0]
                img.save(output_path)
                tqdm.write(f"✅ Saved: {output_path}")
            except Exception as e:
                tqdm.write(f"❌ Error generating {image_filename}: {e}")
                continue

        # Update JSON list and increment counter
        results.append({"question_id": question_counter, "answer": "0.0000"})
        question_counter += 1

        # 💾 Save progress every few images
        if question_counter % 20 == 0:
            with open(vqa_result_path, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=2)
            tqdm.write(f"💾 Auto-saved progress at {question_counter} images.")

# =====================================================
# FINAL SAVE
# =====================================================
with open(vqa_result_path, "w", encoding="utf-8") as jf:
    json.dump(results, jf, indent=2)

print("\n🎯 All prompts processed successfully!")
print(f"🧾 Saved evaluation JSON → {vqa_result_path}")
print(f"Total entries: {len(results)}")
import os
import torch
import json
from diffusers import HunyuanDiTPipeline
from tqdm import tqdm
import re
import torch.multiprocessing as mp

# =====================================================
# CONFIGURATION
# =====================================================
PROMPT_DIR = "../T2I-CompBench/examples/dataset"  # folder containing .txt prompt files
OUTPUT_DIR = "./t2i-compbench-outputs/examples/samples"  # output folder (follows eval structure)
MODEL_PATH = "hunyuandit-model-path"

IMAGES_PER_PROMPT = 4
GUIDANCE_SCALE = 7.5
NUM_STEPS = 60
HEIGHT, WIDTH = 512, 512
NEGATIVE_PROMPT = "blurry, low resolution"

os.makedirs(OUTPUT_DIR, exist_ok=True)
vqa_result_path = os.path.join(os.path.dirname(OUTPUT_DIR), "vqa_result.json")


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def clean_filename(name: str) -> str:
    """Convert prompt text into a safe filename (used for saving images)."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)  # remove invalid characters
    name = re.sub(r"\s+", " ", name).strip()  # normalize spaces
    return name[:180]  # avoid overly long filenames


def load_all_prompts(prompt_dir: str):
    """Load all prompts from .txt files."""
    all_prompts = []
    txt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    for txt_file in txt_files:
        with open(os.path.join(prompt_dir, txt_file), "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
            all_prompts.extend(lines)
    return all_prompts


def get_existing_question_ids(output_dir: str):
    """Return a set of existing question IDs based on image filenames."""
    existing_ids = set()
    for fname in os.listdir(output_dir):
        match = re.search(r"_(\d{6})\.png$", fname)
        if match:
            existing_ids.add(int(match.group(1)))
    return existing_ids


def save_results_threadsafe(results):
    """Save results JSON with file lock-like safety."""
    tmp_path = vqa_result_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)
    os.replace(tmp_path, vqa_result_path)


# =====================================================
# WORKER FUNCTION (RUNS PER GPU)
# =====================================================
def run_generation_on_gpu(gpu_id, prompts_subset, start_counter, existing_ids_shared):
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    print(f"[GPU {gpu_id}] 🚀 Loading model on {device}...")

    pipe = HunyuanDiTPipeline.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16
    ).to(device)

    pipe.transformer.to(memory_format=torch.channels_last)
    pipe.vae.to(memory_format=torch.channels_last)
    print(f"[GPU {gpu_id}] ✅ Model loaded successfully.")

    # Each GPU works independently
    local_results = []
    question_counter = start_counter

    for prompt_idx, prompt in enumerate(tqdm(prompts_subset, desc=f"[GPU {gpu_id}] Generating", unit="prompt")):
        prompt_name = clean_filename(prompt)

        for _ in range(IMAGES_PER_PROMPT):
            image_filename = f"{prompt_name}_{question_counter:06d}.png"
            output_path = os.path.join(OUTPUT_DIR, image_filename)

            if os.path.exists(output_path) or question_counter in existing_ids_shared:
                tqdm.write(f"[GPU {gpu_id}] ⏭️ Skipping {image_filename} (already exists)")
            else:
                tqdm.write(f"[GPU {gpu_id}] 🖼️ Generating: {image_filename}")
                try:
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        height=HEIGHT,
                        width=WIDTH,
                        num_inference_steps=NUM_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        num_images_per_prompt=1
                    )
                    img = result.images[0]
                    img.save(output_path)
                    tqdm.write(f"[GPU {gpu_id}] ✅ Saved: {output_path}")
                except Exception as e:
                    tqdm.write(f"[GPU {gpu_id}] ❌ Error generating {image_filename}: {e}")
                    continue

            local_results.append({"question_id": question_counter, "answer": "0.0000"})
            question_counter += 1

            # Periodically save to shared JSON
            if len(local_results) % 20 == 0:
                # Load current shared file, append, and save
                if os.path.exists(vqa_result_path):
                    try:
                        with open(vqa_result_path, "r", encoding="utf-8") as f:
                            shared_results = json.load(f)
                    except Exception:
                        shared_results = []
                else:
                    shared_results = []

                shared_results.extend(local_results)
                save_results_threadsafe(shared_results)
                local_results.clear()
                tqdm.write(f"[GPU {gpu_id}] 💾 Auto-saved progress at {question_counter} images.")

    # Final save
    if local_results:
        if os.path.exists(vqa_result_path):
            try:
                with open(vqa_result_path, "r", encoding="utf-8") as f:
                    shared_results = json.load(f)
            except Exception:
                shared_results = []
        else:
            shared_results = []

        shared_results.extend(local_results)
        save_results_threadsafe(shared_results)
        tqdm.write(f"[GPU {gpu_id}] 🧾 Final save completed.")


# =====================================================
# MAIN DRIVER
# =====================================================
if __name__ == "__main__":
    print("🔍 Loading prompts and existing progress...")

    prompts = load_all_prompts(PROMPT_DIR)
    total_prompts = len(prompts)

    if total_prompts == 0:
        print("⚠️ No prompts found in .txt files!")
        exit()

    print(f"🧠 Loaded {total_prompts} total prompts.\n")

    existing_results = []
    if os.path.exists(vqa_result_path):
        try:
            with open(vqa_result_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"🔁 Resuming from previous run — {len(existing_results)} entries found.")
        except Exception as e:
            print(f"⚠️ Failed to read existing JSON: {e}. Starting fresh.")
            existing_results = []

    existing_ids = get_existing_question_ids(OUTPUT_DIR)
    question_counter = max(existing_ids) + 1 if existing_ids else 0

    images_done = len(existing_ids)
    prompts_done = images_done // IMAGES_PER_PROMPT

    if prompts_done > 0:
        print(f"🔁 Detected {images_done} existing images ({prompts_done} prompts completed).")
        print(f"⏭️ Skipping first {prompts_done} prompts...\n")
        prompts = prompts[prompts_done:]
    else:
        print("🚀 Starting fresh (no completed prompts found).\n")

    print(f"Starting generation from question_id: {question_counter:06d}\n")

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("❌ No GPUs detected! Please run on a CUDA-enabled system.")

    print(f"⚡ Detected {num_gpus} GPU(s). Splitting workload...")

    # Split prompts equally across GPUs
    chunks = [prompts[i::num_gpus] for i in range(num_gpus)]

    # Shared set for skipping existing files
    manager = mp.Manager()
    existing_ids_shared = manager.list(existing_ids)

    # Spawn one process per GPU
    processes = []
    for gpu_id in range(num_gpus):
        start_counter = question_counter + gpu_id * len(chunks[gpu_id]) * IMAGES_PER_PROMPT
        p = mp.Process(target=run_generation_on_gpu, args=(gpu_id, chunks[gpu_id], start_counter, existing_ids_shared))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n🎯 All GPUs completed successfully!")
    print(f"🧾 Saved evaluation JSON → {vqa_result_path}")
