import requests
import base64
from io import BytesIO
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import matplotlib.pyplot as plt
from PIL import Image



def get_sampler():
    url = "http://127.0.0.1:7860"
    url += "/sdapi/v1/samplers"
    opt = requests.get(url)
    print(opt.json())

def get_current_model():
    url = "http://127.0.0.1:7860"
    url += "/sdapi/v1/options"
    opt = requests.get(url)
    print(f'Model: {opt.json()["sd_model_checkpoint"]}')

def change_model(model_name):
    url = "http://127.0.0.1:7860"
    url += "/sdapi/v1/options"

    data = {
        "sd_model_checkpoint": model_name,
    }

    # 發送 POST 請求
    response = requests.post(url, json=data)

demo_prompt = "A beautiful sunset over the ocean."
demo_negative_prompt = "deform, ugly"

def txt2img(prompt=demo_prompt, negative_prompt=demo_negative_prompt, styles = [], steps=6, width=1024, height=1024, to_show=False):
    start_time = time.time()
    # 更換模型
    change_model("Stable Diffusion XL Base 1.0.safetensors [31e35c80fc]")
    get_current_model()
    
    # 定義 API URL
    url = "http://127.0.0.1:7860"
    url += "/sdapi/v1/txt2img"

    if not prompt.endswith("<lora:LCM_lora_sdxl:1>"):
        prompt += "<lora:LCM_lora_sdxl:1>"

    # 構造請求數據
    data = {
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "styles": styles,
    "sampler_name": "LCM",
    "batch_size": 1,
    "steps": steps,
    "cfg_scale": 1,
    "width": width,
    "height": height,
    "sampler_index": "LCM",
    }

    print(data)

    # 發送 POST 請求
    response = requests.post(url, json=data)

    # 檢查響應
    if response.status_code == 200:
        response_data = response.json()
        if "images" in response_data and response_data["images"]:
            # 解碼圖像數據
            image_data = response_data["images"][0]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))

            end_time = time.time()
            duration = end_time - start_time
            print(f"Time: {end_time - start_time:.2f}s")

            if not to_show: return image, duration
            # 使用 matplotlib 顯示圖像
            plt.imshow(image)
            plt.axis('off')  # 不顯示坐標軸
            plt.show()
            return image, duration
        else:
            print("No images returned.")
    else:
        print("Failed to generate image:", response.status_code, response.text)

        
app = FastAPI()

# 定義請求數據模型
class ImageRequest(BaseModel):
    prompt: str = "A beautiful sunset over the ocean."
    negative_prompt: str = "deform, ugly"
    styles: list = []
    steps: int = 6
    width: int = 1024
    height: int = 1024
    to_show: bool = False

# 定義 FastAPI 路徑操作
@app.post("/txt2img/")
def generate_image(request: ImageRequest):
    # 調用原有的 txt2img 函數
    image, duration = txt2img(prompt=request.prompt, negative_prompt=request.negative_prompt,
                              styles=request.styles, steps=request.steps, width=request.width,
                              height=request.height, to_show=request.to_show)
    if image:
        # 將 PIL 圖像轉換為 Base64 編碼的 JPEG 以透過 HTTP 傳送
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return {"image_base64": img_str, "duration": duration}
    else:
        raise HTTPException(status_code=500, detail="Image generation failed")



if __name__ == "__main__":
    # prompt = "A beautiful sunset over the ocean."
    # txt2img(prompt, to_show=True)
    # 使用 uvicorn 作為 ASGI 伺服器來運行應用程序
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
