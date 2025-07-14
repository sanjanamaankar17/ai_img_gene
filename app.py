import io
import base64
from PIL import Image
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
from diffusers import DiffusionPipeline, LCMScheduler
from diffusers import AutoPipelineForText2Image

# NOTE: For local dev, install dependencies using:
# pip install --upgrade pip
# pip install --upgrade diffusers transformers accelerate peft fastapi uvicorn pillow

app = FastAPI()
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# Model/adapter IDs
model_i = 'stabilityai/sdxl-turbo'
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
adapter_id = "latent-consistency/lcm-lora-sdxl"

# Load pipeline globally
def load_pipeline():
    torch.cuda.empty_cache()
    pipe = AutoPipelineForText2Image.from_pretrained(
        model_id,
        variant="fp16",
        torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(adapter_id, adapter_name="lcm")
    pipe.load_lora_weights("TheLastBen/Papercut_SDXL", weight_name="papercut.safetensors", adapter_name="papercut")
    pipe.set_adapters(["lcm", "papercut"], adapter_weights=[1.0, 0.8])
    
    return pipe

pipe = load_pipeline()


'''@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)
    generator = torch.manual_seed(0)
    with torch.cuda.amp.autocast(), torch.inference_mode():
        image = pipe(prompt, num_inference_steps=4, guidance_scale=1, generator=generator).images[0]
        image = image.resize((200, 200))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image": img_b64}'''
@app.post("/generate")
async def generate(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")

    if not prompt:
        return JSONResponse({"error": "No prompt provided"}, status_code=400)

    generator = torch.manual_seed(0)

    # dummy image for illustration — replace with your pipeline
    image = Image.new("RGB", (256, 256), color="red").resize((200, 200))

    buf = io.BytesIO()
    image.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {"image": img_b64}
