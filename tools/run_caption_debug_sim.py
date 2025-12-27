import torch
from tools.eval_dataset import make_compute_fn_for_sd

class DummySD:
    def __init__(self):
        self.device_torch = torch.device('cpu')
        self.device = 'cpu'
        class NS:
            config = type('C', (), {'num_train_timesteps': 10})
            def add_noise(self, latents, noise, timesteps):
                return latents + noise
        self.noise_scheduler = NS()
        self.vae_device_torch = torch.device('cpu')
        self.vae_torch_dtype = torch.float32

    def encode_prompt(self, prompts):
        # Return a simple object that resembles PromptEmbeds
        class PE:
            def __init__(self, prompts):
                # create a simple tensor embed per prompt
                self.text_embeds = torch.randn(len(prompts), 77, 768)
        return PE(prompts)

    def encode_images(self, images, device=None, dtype=None):
        b = len(images)
        return torch.randn(b, 4, 16, 16)

    def predict_noise(self, noisy, text_embeddings=None, timestep=None, batch=None, **kwargs):
        # returns zeros but we still want to see debug prints
        return torch.zeros_like(noisy)

    def get_loss_target(self, noise=None, batch=None, timesteps=None):
        return noise

# Build compute_fn with debug_captions True
sd = DummySD()
compute_fn = make_compute_fn_for_sd(sd, device='cpu', samples_per_image=2, fixed_noise_std=0.6, debug_captions=True)

# Build a fake batch
class FileItem:
    def __init__(self, path, caption):
        self.path = path
        self.raw_caption = caption
        self.dataset_config = type('D', (), {'dataset_path': 'jinx_references'})

class Batch:
    def __init__(self, file_items, latents):
        self.file_items = file_items
        self.latents = latents
        self.tensor = None
        self.prompt_embeds = None

files = [FileItem('a.png', 'a caption'), FileItem('b.png', 'b caption')]
lat = torch.randn(2, 4, 16, 16)
batch = Batch(files, lat)

print('Running compute_fn locally (should emit [EVAL-CAPTION-DEBUG] prints)')
entries = compute_fn(batch)
print('Entries:', entries)
