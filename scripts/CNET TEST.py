import torch
from diffusers import ZImageControlNetPipeline
from diffusers import ZImageControlNetModel
from diffusers.utils import load_image
from huggingface_hub import hf_hub_download
controlnet = ZImageControlNetModel.from_single_file(
    hf_hub_download(
        "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        filename="Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
    ),
    torch_dtype=torch.float8_e4m3fn,
)

pipe = ZImageControlNetPipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", controlnet=controlnet, torch_dtype=torch.bfloat16
)
pipe.to("cuda")
control_image = load_image(
    "https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union/resolve/main/asset/pose.jpg?download=true"
)
prompt = "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"
image = pipe(
    prompt,
    control_image=control_image,
    controlnet_conditioning_scale=0.75,
    height=1728,
    width=992,
    num_inference_steps=8,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(43),
).images[0]
image.save("zimage-8step.png")