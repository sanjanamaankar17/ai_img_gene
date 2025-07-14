# ai_image_gene

# SDXL LCM + Papercut Web Demo

This repo provides a simple web UI to run your provided Python code and generate images using SDXL + LCM + Papercut LoRAs.

## Features

- Enter a prompt to generate a 200x200 image using the specified pipeline.
- Requires a GPU with CUDA and the listed dependencies.


## Backend Python code (core logic)

```python
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers import AutoPipelineForText2Image

model_i = 'stabilityai/sdxl-turbo'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"
torch.cuda.empty_cache()
pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    variant="fp16",
    torch_dtype=torch.float16
).to("cuda")

# set scheduler
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# load LoRAs
pipe.load_lora_weights(adapter_id, adapter_name="lcm")
pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")

# Combine LoRAs
pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])

prompt = input()
generator = torch.manual_seed(0)
image = pipe(prompt, num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
image.resize((200,200))
```

## Notes

- This requires a CUDA-capable GPU and enough VRAM.
- The first request may take a while due to model loading.
- The UI and backend are basic and meant for demonstration.
