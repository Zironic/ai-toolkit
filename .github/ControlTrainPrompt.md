Your task is to implement lora training using a frozen controlnet.

Your reference implementation for the controlnet is https://github.com/aigc-apps/VideoX-Fun

The model used is https://huggingface.co/Tongyi-MAI/Z-Image-Turbo

Controlnet used is: https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1

Implementation plan is found in these three documents:
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan.md"
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan2.md"
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Plan3.md"

To measure how far along the work you have gotten, use this document:
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Implementation.md"

A document has been created to serve as the project memory:
"C:\GenAI\AI-Toolkit-Easy-Install\AI-Toolkit\ControlTrain-Reference.md"

When needing to find information about how something is implemented, consult the memory document first. Whenever you fail to easily find something and eventually find it, update the memory document with your findings to make it easier for you to find it again in the future. Document any issues with the implementation in the memory document.

You are to follow a fail-fast philosophy. It's critical that the training job is accurate and this requires every part to have proceeded correctly. Failing to attach the proper training data or failing to load the adapter would corrupt training and needs to raise runtime errors for debugging.

Use your tools to properly research the task at hand, don't be afraid to look up the https://github.com/aigc-apps/VideoX-Fun or use the HF api to research the models and how they're used. Ensure the implementation document is kept up to date so you know what has and has not been implemented. 

You don't have to stop and ask to implement required tests. You are fully empowered to implement this entire project.