import React from 'react';
import { ConfigDoc } from '@/types';
import { IoFlaskSharp } from 'react-icons/io5';

const docs: { [key: string]: ConfigDoc } = {
  'config.name': {
    title: 'Training Name',
    description: (
      <>
        The name of the training job. This name will be used to identify the job in the system and will the the filename
        of the final model. It must be unique and can only contain alphanumeric characters, underscores, and dashes. No
        spaces or special characters are allowed.
      </>
    ),
  },
  gpuids: {
    title: 'GPU ID',
    description: (
      <>
        This is the GPU that will be used for training. Only one GPU can be used per job at a time via the UI currently.
        However, you can start multiple jobs in parallel, each using a different GPU.
      </>
    ),
  },
  'config.process[0].trigger_word': {
    title: 'Trigger Word',
    description: (
      <>
        Optional: This will be the word or token used to trigger your concept or character.
        <br />
        <br />
        When using a trigger word, If your captions do not contain the trigger word, it will be added automatically the
        beginning of the caption. If you do not have captions, the caption will become just the trigger word. If you
        want to have variable trigger words in your captions to put it in different spots, you can use the{' '}
        <code>{'[trigger]'}</code> placeholder in your captions. This will be automatically replaced with your trigger
        word.
        <br />
        <br />
        Trigger words will not automatically be added to your test prompts, so you will need to either add your trigger
        word manually or use the
        <code>{'[trigger]'}</code> placeholder in your test prompts as well.
      </>
    ),
  },
  'config.process[0].model.name_or_path': {
    title: 'Name or Path',
    description: (
      <>
        The name of a diffusers repo on Huggingface or the local path to the base model you want to train from. The
        folder needs to be in diffusers format for most models. For some models, such as SDXL and SD1, you can put the
        path to an all in one safetensors checkpoint here.
      </>
    ),
  },
  'datasets.control_path': {
    title: 'Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training. The control images will be
        resized to match the training images.
      </>
    ),
  },
  'datasets.multi_control_paths': {
    title: 'Multi Control Dataset',
    description: (
      <>
        The control dataset needs to have files that match the filenames of your training dataset. They should be
        matching file pairs. These images are fed as control/input images during training.
        <br />
        <br />
        For multi control datasets, the controls will all be applied in the order they are listed. If the model does not
        require the images to be the same aspect ratios, such as with Qwen/Qwen-Image-Edit-2509, then the control images
        do not need to match the aspect size or aspect ratio of the target image and they will be automatically resized
        to the ideal resolutions for the model / target images.
      </>
    ),
  },
  'datasets.num_frames': {
    title: 'Number of Frames',
    description: (
      <>
        This sets the number of frames to shrink videos to for a video dataset. If this dataset is images, set this to 1
        for one frame. If your dataset is only videos, frames will be extracted evenly spaced from the videos in the
        dataset.
        <br />
        <br />
        It is best to trim your videos to the proper length before training. Wan is 16 frames a second. Doing 81 frames
        will result in a 5 second video. So you would want all of your videos trimmed to around 5 seconds for best
        results.
        <br />
        <br />
        Example: Setting this to 81 and having 2 videos in your dataset, one is 2 seconds and one is 90 seconds long,
        will result in 81 evenly spaced frames for each video making the 2 second video appear slow and the 90second
        video appear very fast.
      </>
    ),
  },
  'datasets.do_i2v': {
    title: 'Do I2V',
    description: (
      <>
        For video models that can handle both I2V (Image to Video) and T2V (Text to Video), this option sets this
        dataset to be trained as an I2V dataset. This means that the first frame will be extracted from the video and
        used as the start image for the video. If this option is not set, the dataset will be treated as a T2V dataset.
      </>
    ),
  },
  'datasets.do_audio': {
    title: 'Do Audio',
    description: (
      <>
        For models that support audio with video, this option will load the audio from the video and resize it to match
        the video sequence. Since the video is automatically resized, the audio may drop or raise in pitch to match the
        new speed of the video. It is important to prep your dataset to have the proper length before training.
      </>
    ),
  },
  'datasets.audio_normalize': {
    title: 'Audio Normalize',
    description: (
      <>
        When loading audio, this will normalize the audio volume to the max peaks. Useful if your dataset has varying
        audio volumes. Warning, do not use if you have clips with full silence you want to keep, as it will raise the
        volume of those clips.
      </>
    ),
  },
  'datasets.audio_preserve_pitch': {
    title: 'Audio Preserve Pitch',
    description: (
      <>
        When loading audio to match the number of frames requested, this option will preserve the pitch of the audio if
        the length does not match training target. It is recommended to have a dataset that matches your target length,
        as this option can add sound distortions.
      </>
    ),
  },
  'datasets.flip': {
    title: 'Flip X and Flip Y',
    description: (
      <>
        You can augment your dataset on the fly by flipping the x (horizontal) and/or y (vertical) axis. Flipping a
        single axis will effectively double your dataset. It will result it training on normal images, and the flipped
        versions of the images. This can be very helpful, but keep in mind it can also be destructive. There is no
        reason to train people upside down, and flipping a face can confuse the model as a person's right side does not
        look identical to their left side. For text, obviously flipping text is not a good idea.
        <br />
        <br />
        Control images for a dataset will also be flipped to match the images, so they will always match on the pixel
        level.
      </>
    ),
  },
  'train.weight_noise': {
    title: 'Weight Noise',
    description: (
      <>
        Injects Gaussian noise directly into LoRA parameter values after each optimizer step. Biases training toward
        flatter loss minima and spreads learning across more singular directions of the LoRA factorization, which
        empirically improves subject likeness and helps the model resist memorization of source-image artifacts. Pairs
        well with small / single-image datasets where overfitting is the dominant failure mode.
      </>
    ),
  },
  'train.weight_noise.sigma': {
    title: 'Sigma',
    description: (
      <>
        Noise scale. In <code>relative</code> mode this is a multiplier on each tensor&apos;s weight RMS, so &quot;0.001
        = 0.1% per-tensor perturbation per step.&quot; Typical useful range is <strong>0.001 – 0.0017</strong>. Lower
        values barely do anything; higher values risk noise overpowering the gradient (loss flattens or training
        diverges). Watch the <code>weight_noise_norm</code> metric to verify a sensible magnitude relative to grad
        norm.
      </>
    ),
  },
  'train.unload_text_encoder': {
    title: 'Unload Text Encoder',
    description: (
      <>
        Unloading text encoder will cache the trigger word and the sample prompts and unload the text encoder from the
        GPU. Captions in for the dataset will be ignored
      </>
    ),
  },
  'train.cache_text_embeddings': {
    title: 'Cache Text Embeddings',
    description: (
      <>
        <small>(experimental)</small>
        <br />
        Caching text embeddings will process and cache all the text embeddings from the text encoder to the disk. The
        text encoder will be unloaded from the GPU. This does not work with things that dynamically change the prompt
        such as trigger words, caption dropout, etc.
      </>
    ),
  },
  'model.multistage': {
    title: 'Stages to Train',
    description: (
      <>
        Some models have multi stage networks that are trained and used separately in the denoising process. Most
        common, is to have 2 stages. One for high noise and one for low noise. You can choose to train both stages at
        once or train them separately. If trained at the same time, The trainer will alternate between training each
        model every so many steps and will output 2 different LoRAs. If you choose to train only one stage, the trainer
        will only train that stage and output a single LoRA.
      </>
    ),
  },
  'train.switch_boundary_every': {
    title: 'Switch Boundary Every',
    description: (
      <>
        When training a model with multiple stages, this setting controls how often the trainer will switch between
        training each stage.
        <br />
        <br />
        For low vram settings, the model not being trained will be unloaded from the gpu to save memory. This takes some
        time to do, so it is recommended to alternate less often when using low vram. A setting like 10 or 20 is
        recommended for low vram settings.
        <br />
        <br />
        The swap happens at the batch level, meaning it will swap between a gradient accumulation steps. To train both
        stages in a single step, set them to switch every 1 step and set gradient accumulation to 2.
      </>
    ),
  },
  'train.force_first_sample': {
    title: 'Force First Sample',
    description: (
      <>
        This option will force the trainer to generate samples when it starts. The trainer will normally only generate a
        first sample when nothing has been trained yet, but will not do a first sample when resuming from an existing
        checkpoint. This option forces a first sample every time the trainer is started. This can be useful if you have
        changed sample prompts and want to see the new prompts right away.
      </>
    ),
  },
  'model.layer_offloading': {
    title: (
      <>
        Layer Offloading{' '}
        <span className="text-yellow-500">
          ( <IoFlaskSharp className="inline text-yellow-500" name="Experimental" /> Experimental)
        </span>
      </>
    ),
    description: (
      <>
        This is an experimental feature based on{' '}
        <a className="text-blue-500" href="https://github.com/lodestone-rock/RamTorch" target="_blank">
          RamTorch
        </a>
        . This feature is early and will have many updates and changes, so be aware it may not work consistently from
        one update to the next. It will also only work with certain models.
        <br />
        <br />
        Layer Offloading uses the CPU RAM instead of the GPU ram to hold most of the model weights. This allows training
        a much larger model on a smaller GPU, assuming you have enough CPU RAM. This is slower than training on pure GPU
        RAM, but CPU RAM is cheaper and upgradeable. You will still need GPU RAM to hold the optimizer states and LoRA
        weights, so a larger card is usually still needed.
        <br />
        <br />
        You can also select the percentage of the layers to offload. It is generally best to offload as few as possible
        (close to 0%) for best performance, but you can offload more if you need the memory.
      </>
    ),
  },
  'model.layer_offloading_smart': {
    title: 'Smart Transformer Offloading',
    description: (
      <>
        Krea 2 experimental mode. Measures free VRAM and deterministically keeps as many transformer layers resident as
        fit after reserving memory for training and the shared transfer ring. This replaces the Transformer Offload %
        selection while enabled.
      </>
    ),
  },
  'model.layer_offloading_block_stream_only': {
    title: 'Block-Only Streaming',
    description: (
      <>
        Streams the transformer in whole-block units and keeps every non-block layer (embedders, the final projection,
        standalone norms and Linears) permanently resident. The prefetch worker then stages a block's weights in one
        batched fill instead of one request per Linear, cutting the CPU overhead of the offload path. Trades a little
        extra resident VRAM and a larger prefetch ring for far fewer small requests.
      </>
    ),
  },
  'model.layer_offloading_pinned_weight_gb': {
    title: 'Pinned Weights (GB)',
    description: (
      <>
        How many GB of offloaded weights to keep page-locked (pinned) in CPU RAM. The model already occupies this RAM
        while training — pinning just locks it so the OS can&apos;t page it out and fault it back in on every fetch,
        which is what makes the offload copies slow and the whole machine lag. Costs roughly no extra RAM.{' '}
        <b>-1 = auto</b>: pins the whole offloaded weight set (capped by free RAM) and lets already-pinned weights skip
        the copy workers entirely — the recommended default if you have the RAM to hold the model. 0 = pin nothing
        (workers copy on demand). A positive value pins up to that many GB. Lower it if the system runs short on RAM.
      </>
    ),
  },
  'model.layer_offloading_smart_working_reserve_gb': {
    title: 'Training VRAM Reserve',
    description: (
      <>
        Working reserve: VRAM kept free for our own transient working set — activations, dequant, gradients, optimizer
        state, and temporary kernels — during a training step. -1 = auto (the live controller learns and tunes it).
        Increase a fixed value if training runs out of memory; decrease it to keep more transformer layers resident.
      </>
    ),
  },
  'model.layer_offloading_smart_wddm_margin_gb': {
    title: 'Training WDDM Margin',
    description: (
      <>
        Driver-level free VRAM to preserve at the training peak. Use -1 for auto, which keeps max(10% of physical
        VRAM, 1 GiB) free. Increase this on Windows if WDDM paging or desktop pressure causes slowdowns.
      </>
    ),
  },
  'model.layer_offloading_wddm_spill_reserve_pct': {
    title: 'Shared Spill Reserve',
    description: (
      <>
        Fraction of the DXGI shared GPU memory budget kept free for pinned/shared-memory headroom and sampling
        transitions. The effective reserve is the larger of this percentage and the resolved Training WDDM Margin.
      </>
    ),
  },
  'model.layer_offloading_smart_wddm_hard_gb': {
    title: 'Training WDDM Hard Floor',
    description: (
      <>
        Emergency driver-level free VRAM floor for training. If realized free drops below this value, smart offload trims
        cache and demotes resident layers immediately. Keep it at or below the WDDM margin.
      </>
    ),
  },
  'model.layer_offloading_smart_sampling': {
    title: 'Smart Sampling Layout',
    description: (
      <>
        Temporarily replaces the training offload layout with a forward-only,
        byte-budgeted sampling layout, then restores training state. Disabled by default.
      </>
    ),
  },
  'model.layer_offloading_smart_sampling_working_reserve_gb': {
    title: 'Sampling VRAM Reserve',
    description: (
      <>
        VRAM kept free for sampling-only transient allocations. -1 = auto; use a fixed value to pin the sampling reserve
        independently from training.
      </>
    ),
  },
  'model.layer_offloading_smart_sampling_wddm_margin_gb': {
    title: 'Sampling WDDM Margin',
    description: (
      <>
        Driver-level free VRAM to preserve during sampling. Use -1 for auto, which keeps max(10% of physical VRAM,
        1 GiB) free. Increase this if validation images trigger WDDM paging or compete with desktop/browser memory.
      </>
    ),
  },
  'model.layer_offloading_smart_sampling_wddm_hard_gb': {
    title: 'Sampling WDDM Hard Floor',
    description: (
      <>
        Emergency driver-level free VRAM floor for sampling. The sampling guard demotes resident blocks before predicted
        peak free crosses this value. Keep it at or below the sampling WDDM margin.
      </>
    ),
  },
  'model.layer_offloading_fp8_sampling': {
    title: 'Native FP8 Sampling',
    description: (
      <>
        Uses compatible on-device FP8 weights directly during smart sampling.
        Requires an FP8-quantized model and supported CUDA hardware; otherwise it falls back safely.
      </>
    ),
  },
  'model.layer_offloading_profile': {
    title: 'Profile Layer Offloading',
    description: (
      <>
        Records pageable and pinned CPU submission stalls, GPU staging time, and compute-stream wait time for streamed
        layers. A report is printed after the first completed step and at each performance logging window. Profiling
        adds CUDA timing events, so leave it disabled for normal training.
      </>
    ),
  },
  'model.layer_offloading_prefetch': {
    title: 'Prefetch Offloaded Layers',
    description: (
      <>
        Experimental. Background CPU workers copy upcoming offloaded layers into pinned memory ahead of time, using the
        recorded per-step access order, so the training thread no longer blocks while staging pageable weights. This can
        sharply reduce step time when offloading is bottlenecked by pageable/pagefile stalls. Enables access tracing
        automatically and keeps a bounded pinned pool sized to available RAM.
      </>
    ),
  },
  'model.layer_offloading_prefetch_trace_capture': {
    title: 'Prefetch Trace Capture',
    description: (
      <>
        Optional JSONL filename or path for replaying real prefetch schedules and observed access streams with
        scripts/replay_prefetch_trace.py. Relative paths are written under the job output folder and archived into logs/ on the next run. Leave empty for normal training.
      </>
    ),
  },
  'model.layer_offloading_prefetch_trace_capture_steps': {
    title: 'Prefetch Capture Steps',
    description: (
      <>
        Maximum number of steps to write to the prefetch trace capture file. Set 0 for unlimited capture.
      </>
    ),
  },
  'model.layer_offloading_fp8_forward': {
    title: 'Native FP8 Training',
    description: (
      <>
        Uses the frozen base model&apos;s FP8 weights directly for streamed linear
        forwards. Checkpoint recompute leaves those weights resident for the
        following grad-input calculation, which also stays quantized through
        TorchAO. Experimental and only effective for compatible FP8 models and GPUs.
      </>
    ),
  },
  'model.layer_offloading_fp8_grad_input': {
    title: 'Native FP8 Grad Input',
    description: (
      <>
        Computes the backward grad-input with a native FP8 matmul (folding the
        per-row weight scales into the gradient and using <code>_scaled_mm</code>{' '}
        against the raw FP8 weight) instead of dequantizing the weight to bf16.
        Reduces backward compute and memory traffic. Experimental: a one-time
        self-check validates it against the bf16 result on first use and falls
        back automatically if it disagrees, so training stays correct.
      </>
    ),
  },
  'model.train_compile_blocks': {
    title: 'Compile Training Blocks',
    description: (
      <>
        Compiles the Krea 2 transformer block kernels used by the immutable runtime during training. Resident and
        streamed blocks share the same compiled kernel; residency changes do not recompile.
      </>
    ),
  },
  'model.layer_offloading_checkpoint_keep_last': {
    title: 'Uncheckpointed Trailing Blocks',
    description: (
      <>
        Number of final transformer blocks to leave uncheckpointed. Those blocks
        keep their activations instead of recomputing them in the backward pass,
        which removes recompute kernel launches once the step is launch-bound
        rather than transfer-bound. Each kept block costs some VRAM, so raise
        this only while peak memory stays under the card; 0 checkpoints every
        block (lowest VRAM, most recompute).
        <br />
        <br />
        Set to <code>-1</code> for auto: the trainer measures the real reserved
        memory and forward/backward time per resolution. It climbs while steps
        get faster, then returns to the fastest measured value on regression.
        The card&apos;s spill line remains a hard safety ceiling.
      </>
    ),
  },
  'model.qie.match_target_res': {
    title: 'Match Target Res',
    description: (
      <>
        This setting will make the control images match the resolution of the target image. The official inference
        example for Qwen-Image-Edit-2509 feeds the control image is at 1MP resolution, no matter what size you are
        generating. Doing this makes training at lower res difficult because 1MP control images are fed in despite how
        large your target image is. Match Target Res will match the resolution of your target to feed in the control
        images allowing you to use less VRAM when training with smaller resolutions. You can still use different aspect
        ratios, the image will just be resizes to match the amount of pixels in the target image.
      </>
    ),
  },
  'train.diff_output_preservation': {
    title: 'Differential Output Preservation',
    description: (
      <>
        Differential Output Preservation (DOP) is a technique to help preserve class of the trained concept during
        training. For this, you must have a trigger word set to differentiate your concept from its class. For instance,
        You may be training a woman named Alice. Your trigger word may be "Alice". The class is "woman", since Alice is
        a woman. We want to teach the model to remember what it knows about the class "woman" while teaching it what is
        different about Alice. During training, the trainer will make a prediction with your LoRA bypassed and your
        trigger word in the prompt replaced with the class word. Making "photo of Alice" become "photo of woman". This
        prediction is called the prior prediction. Each step, we will do the normal training step, but also do another
        step with this prior prediction and the class prompt in order to teach our LoRA to preserve the knowledge of the
        class. This should not only improve the performance of your trained concept, but also allow you to do things
        like "Alice standing next to a woman" and not make both of the people look like Alice.
      </>
    ),
  },
  'train.diff_output_preservation_resolution': {
    title: 'DOP Resolution (px long-side)',
    description: (
      <>
        Optional: run DOP preservation at a reduced spatial resolution (long-side in pixels), e.g. select from presets
        128 / 256 / 512 / 1024. This will downsample the latents for the preservation forward pass and compute
        preservation loss at the reduced size to reduce compute. Choose 'Full resolution' to run at full resolution.
      </>
    ),
  },
  'train.dop_single_backward': {
    title: 'Single Backward Pass',
    description: (
      <>
        By default DOP runs two backward passes per step (one for the main loss, one for the preservation loss),
        which keeps peak VRAM low because each forward graph is freed before the next is built. Enable this to instead
        sum the preservation loss into the main loss and run a single combined backward — slightly faster, but both
        forward graphs stay resident until the backward, so peak VRAM is higher by roughly the size of the (optionally
        downsampled) preservation graph. Pairs well with a reduced DOP Resolution, which keeps that extra graph small.
      </>
    ),
  },
  'train.dop_prior_cache': {
    title: 'Cache Prior Predictions',
    description: (
      <>
        The DOP prior is a prediction from the frozen base model with the LoRA disabled, so for a given
        noised latent and timestep it never changes during training. Enable this to persist a small per-image
        pool — warmed from live training batches — and reuse it, so later runs do not repeat that warmup and DOP
        step skips the extra frozen-model forward. Only active with a reduced DOP Resolution (it runs in the
        downsampled path) and standard LoRA training (disabled automatically when controlnet/adapter
        conditioning is present). Prompt, model, resolution, or scheduler changes select a new cache automatically.
      </>
    ),
  },
  'train.blank_prompt_preservation': {
    title: 'Blank Prompt Preservation',
    description: (
      <>
        Blank Prompt Preservation (BPP) is a technique to help preserve the current models knowledge when unprompted.
        This will not only help the model become more flexible, but will also help the quality of your concept during
        inference, especially when a model uses CFG (Classifier Free Guidance) on inference. At each step during
        training, a prior prediction is made with a blank prompt and with the LoRA disabled. This prediction is then
        used as a target on an additional training step with a blank prompt, to preserve the model's knowledge when no
        prompt is given. This helps the model to not overfit to the prompt and retain its generalization capabilities.
      </>
    ),
  },
  'train.blank_prompt_preservation_resolution': {
    title: 'BPP Resolution (px long-side)',
    description: (
      <>
        Optional: run Blank Prompt Preservation at a reduced spatial resolution (long-side in pixels), choose from
        presets 128 / 256 / 512 / 1024. When set, the BPP preservation pass and loss will operate at the smaller
        resolution. Choose 'Full resolution' to run at full resolution.
      </>
    ),
  },
  'train.do_differential_guidance': {
    title: 'Differential Guidance',
    description: (
      <>
        Differential Guidance will amplify the difference of the model prediction and the target during training to make
        a new target. Differential Guidance Scale will be the multiplier for the difference. This is still experimental,
        but in my tests, it makes the model train faster, and learns details better in every scenario I have tried with
        it.
        <br />
        <br />
        The idea is that normal training inches closer to the target but never actually gets there, because it is
        limited by the learning rate. With differential guidance, we amplify the difference for a new target beyond the
        actual target, this would make the model learn to hit or overshoot the target instead of falling short.
        <br />
        <br />
        <img src="/imgs/diff_guidance.png" alt="Differential Guidance Diagram" className="max-w-full mx-auto" />
      </>
    ),
  },
  'dataset.num_repeats': {
    title: 'Num Repeats',
    description: (
      <>
        Number of Repeats will allow you to repeate the items in a dataset multiple times. This is useful when you are
        using multiple datasets and want to balance the number of samples from each dataset. For instance, if you have a
        small dataset of 10 images and a large dataset of 100 images, you can set the small dataset to have 10 repeats
        to effectively make it 100 images, making the two datasets occour equally during training.
      </>
    ),
  },
  'train.audio_loss_multiplier': {
    title: 'Audio Loss Multiplier',
    description: (
      <>
        When training audio and video, sometimes the video loss is so great that it outweights the audio loss, causing
        the audio to become distorted. If you are noticing this happen, you can increase the audio loss multiplier to
        give more weight to the audio loss. You could try something like 2.0, 10.0 etc. Warning, setting this too high
        could overfit and damage the model.
      </>
    ),
  },
  'datasets.auto_frame_count': {
    title: 'Auto Frame Count',
    description: (
      <>
        This will automatically determine the number of frames to use for each video in your dataset instead of relying
        on a fixed num_frames. This allows you to include videos of different lengths in the dataset, and each video
        will be processed without speeding up or slowing down. Be careful about adding long videos into your dataset, as
        they use up more VRAM. This currently will not work with a batch size greater than 1.
      </>
    ),
  },
  'model.model_kwargs.kv_cache': {
    title: 'KV Cache',
    description: (
      <>
        This will enable KV Cache for control images in a model that supports it. LoRAs trained with this on
        need to also be inferenced with it, and vice versa. This does not speed up or slow down training, but on inference,
        the control images only need to be processed once for the entire generation, vs being processed for every step.
        Which leads to a significant speedup on inference.
      </>
    ),
  },
};

export const getDoc = (key: string | null | undefined): ConfigDoc | null => {
  if (key && key in docs) {
    return docs[key];
  }
  return null;
};

export default docs;
