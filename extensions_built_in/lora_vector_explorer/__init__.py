from toolkit.extension import Extension


class LoraVectorExplorerExtension(Extension):
    uid = "lora_vector_explore"
    name = "LoRA Vector Explorer"

    @classmethod
    def get_process(cls):
        from .LoraVectorExploreProcess import LoraVectorExploreProcess
        return LoraVectorExploreProcess


AI_TOOLKIT_EXTENSIONS = [
    LoraVectorExplorerExtension,
]