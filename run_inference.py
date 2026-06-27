from PIL import Image
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

repo_id = "vcl-iisc/DynEval-Evaluator"
subfolder = "DynEval-2B"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    repo_id,
    subfolder=subfolder,
    dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(repo_id, subfolder=subfolder)

image = Image.open("example.jpg").convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": "<|IQA|>\nEvaluate or answer the question for this image.",
            },
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt",
).to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

output = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)[0]

print(output)