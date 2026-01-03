

The Goal: To enchance lora training by the use of a frozen controlnet.



For this particular project the plan is to use this model https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

Together with this controlnet https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1

They're already used together in this implementation https://github.com/aigc-apps/VideoX-Fun


We want to fully integrate this into the existing workflow of this OstrisAI-Toolkit

This means the UI will need to be modified at AI-Toolkit\ui\src\app\jobs to add the settings we will need.

This will include

* HF Style name for the Controlnet alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
* * We will want that to match the best version of the controlnet, which is Z-Image-Turbo-Fun-Controlnet-Tile-2.1-8steps.safetensors
* * We will need to be able to handle the fact that his repository only contains .safetensor files and no config file, so we will have to generate our own.

* An option toggle for wanting to do lora training with a controlnet


* Under the dataset options, we will need an option for which kind of control we want to use. Canny, Depth or Openpose. For the first implementation, just supporting Openpose is fine.


The code for loading datasets will then need to be modified such that after the dataset has been loaded into buckets but before it has been turned into latents by the VAE, the dataset needs to be copied into appropriate control samples. This should ensure a 1:1 match between samples and controls. It also needs to be ensured that samples and controls are tiled in the exact same way. It's important to keep in mind here that since Z-Image-Turbo is Flux2 based, it'll need a control context, not block samples.

The load adapter code will need to be modified to support the https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1. It's important here to thoroughly investigate how https://github.com/aigc-apps/VideoX-Fun loads the controlnet and all appropriate shims are implemented.