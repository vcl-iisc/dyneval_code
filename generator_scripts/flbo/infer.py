import json
import os

import torch
from diffusers import BriaFiboPipeline
from diffusers.modular_pipelines import ModularPipelineBlocks


# -------------------------------
# Load the VLM pipeline
# -------------------------------
torch.set_grad_enabled(False)
# Using Gemini API, requires GOOGLE_API_KEY environment variable

# Using local VLM, uncomment to run
vlm_pipe = ModularPipelineBlocks.from_pretrained("briaai/FIBO-VLM-prompt-to-JSON", trust_remote_code=True)
vlm_pipe = vlm_pipe.init_pipeline()

# Load the FIBO pipeline
pipe = BriaFiboPipeline.from_pretrained(
    "briaai/FIBO",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
# pipe.enable_model_cpu_offload() # uncomment if you're getting CUDA OOM errors

# -------------------------------
# Run Prompt to JSON
# -------------------------------

# Create a prompt to generate an initial image
output = vlm_pipe(
    prompt="A hyper-detailed, ultra-fluffy owl sitting in the trees at night, looking directly at the camera with wide, adorable, expressive eyes. Its feathers are soft and voluminous, catching the cool moonlight with subtle silver highlights. The owl's gaze is curious and full of charm, giving it a whimsical, storybook-like personality."
)
json_prompt_generate = output.values["json_prompt"]

def get_default_negative_prompt(existing_json: dict) -> str:
    negative_prompt = ""
    style_medium = existing_json.get("style_medium", "").lower()
    if style_medium in ["photograph", "photography", "photo"]:
        negative_prompt = """{'style_medium':'digital illustration','artistic_style':'non-realistic'}"""
    return negative_prompt


negative_prompt = get_default_negative_prompt(json.loads(json_prompt_generate))

# -------------------------------
# Run Image Generation
# -------------------------------
# Generate the image from the structured json prompt
results_generate = pipe(
    prompt=json_prompt_generate, num_inference_steps=50, guidance_scale=5, negative_prompt=negative_prompt
)
results_generate.images[0].save("image_generate.png")
with open("image_generate_json_prompt.json", "w") as f:
    f.write(json_prompt_generate)
