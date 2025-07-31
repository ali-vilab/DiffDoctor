import torch
import torch.nn as nn
from PIL import Image
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(os.path.join(cwd, 'src'))
from tqdm import tqdm
import random
from collections import defaultdict
import src.prompts as prompts_file
import numpy as np
import wandb
import contextlib
import torchvision
import sys
from diffusers.loaders import AttnProcsLayers
from diffusers import UNet2DConditionModel
from diffusers import FluxPipeline
import datetime
from einops import rearrange, repeat
from src.flux.sampling import get_schedule
import torch.nn.functional as F
import hpsv2
from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from accelerate.logging import get_logger    
from peft import LoraConfig
from accelerate import Accelerator
from absl import flags
from ml_collections import config_flags
FLAGS = flags.FLAGS
try:
    config_flags.DEFINE_config_file(name="config", default="config/settings.py", help_string="Training configuration.")
except:
    print(f'Import config error. You should execute this file by using "accelerate launch train_diffusion_model.py --config config/settings.py:ad"')
    exit(1)
from accelerate.utils import set_seed, ProjectConfiguration
logger = get_logger(__name__)

def hps_loss_fn(config, inference_dtype=None, device=None):
    model_name = "ViT-H-14"
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        model_name,
        'laion2B-s32B-b79K',
        precision=inference_dtype,
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False
    )    
    
    tokenizer = get_tokenizer(model_name)
    
    checkpoint_path = config.hps_ckpt_path
    # force download of model via score
    hpsv2.score([], "")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device, dtype=inference_dtype)
    model.requires_grad_(False)
    model.eval()

    target_size =  224
    normalize = torchvision.transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                                std=[0.26862954, 0.26130258, 0.27577711])
        
    def loss_fn(im_pix, prompts):    
        # im_pix = ((im_pix / 2) + 0.5).clamp(0, 1) 
        im_pix = im_pix.clamp(0, 1) 
        x_var = torchvision.transforms.Resize(target_size)(im_pix)
        x_var = normalize(x_var).to(im_pix.dtype)        
        caption = tokenizer(prompts)
        caption = caption.to(device)
        outputs = model(x_var, caption)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        scores = torch.diagonal(logits)
        loss = 1.0 - scores
        return  loss, scores
    
    return loss_fn
    

def artifact_maps_to_loss(config, artifact_maps,method='max'):
    if len(artifact_maps.shape) == 3:
        artifact_maps = artifact_maps.unsqueeze(0)

    if method == 'max':
        loss = artifact_maps.amax(dim=(1,2,3))*config.ad_loss_scale
        scores = -loss
    elif method == 'sum':
        loss = artifact_maps.sum(dim=(1,2,3))*config.ad_loss_scale
        scores = -loss
    elif method == 'mean':
        loss = artifact_maps.mean(dim=(1,2,3))*config.ad_loss_scale
        scores = -loss
    elif method == 'weighted':
        threshold = 0.1
        non_zero_pixel = artifact_maps > threshold
        non_zero_pixel_cnt = non_zero_pixel.sum(dim=(1,2,3)) + 1e-5
        loss = ((artifact_maps * non_zero_pixel).sum(dim=(1,2,3))/ non_zero_pixel_cnt)*config.ad_loss_scale
        scores = - loss
    else:
        raise ValueError(f"method {method} is not supported")

    return loss, scores
    

def artifact_detector_loss(config, model, preprocessor=None, accelerator=None, method='max'):
    def _fn(images, prompts=None):
        only_one = False
        if isinstance(images, np.ndarray):
            images = torch.tensor(images)
            images = images.permute(-1, -3, -2)

        assert isinstance(images, torch.Tensor)
        if len(images.shape) == 3:
            only_one = True
            images = images.unsqueeze(0)
        images = images.float().clamp(0, 1) 
        # check the range of the image
        assert images.min() >= 0 and images.max() <= 1
        # replace preprocessor with derivable functions
        # resize -> normalize
        images = torchvision.transforms.Resize(512)(images)
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        images = normalize(images)

        if accelerator is not None:
            images = images.to(accelerator.device)
        model.eval()
        pred = model(images)
        pred = nn.functional.interpolate(
            pred.logits, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
        artifact_maps = torch.sigmoid(pred)
        
        if method == 'all':
            loss_max, rewards_max = artifact_maps_to_loss(config, artifact_maps,method='max')
            loss_mean, rewards_mean = artifact_maps_to_loss(config, artifact_maps,method='mean')
            return {
                'loss_max': loss_max,
                'rewards_max': rewards_max,
                'loss_mean': loss_mean,
                'rewards_mean': rewards_mean
            }
        else:
            loss, rewards = artifact_maps_to_loss(config, artifact_maps,method=method)
            # if only_one:
            #     scores = scores.squeeze(0)
            return loss, rewards
    return _fn

@torch.no_grad()
def evaluate(latent,train_neg_prompt_embeds,prompts, pipeline, accelerator, inference_dtype, config, loss_fn,latent_image_ids=None,generator=None):
    if 'flux' in config.pipe_type:
        height, width = 1024, 1024
        max_sequence_length = 256
        prompts = list(prompts)
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=accelerator.device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        num_inference_steps = config.eval_steps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latent.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipeline.scheduler.config.base_image_seq_len,
            pipeline.scheduler.config.max_image_seq_len,
            pipeline.scheduler.config.base_shift,
            pipeline.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            pipeline.scheduler,
            num_inference_steps,
            accelerator.device,
            None,
            sigmas,
            mu=mu,
        )
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc='evaluating'):
            timestep = t.expand(latent.shape[0]).to(latent.dtype).to(accelerator.device)
            guidance = None
            if config.pipe_type == 'fluxdev':
                if accelerator.unwrap_model(pipeline.transformer).config.guidance_embeds:
                    guidance = torch.tensor([3.5], device=accelerator.device)
                    guidance = guidance.expand(latent.shape[0])
                else:
                    guidance = None
            noise_pred = pipeline.transformer(
                hidden_states=latent,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]
            latent = pipeline.scheduler.step(noise_pred, t, latent, return_dict=False)[0]
        latent = pipeline._unpack_latents(latent, height, width, pipeline.vae_scale_factor)
        latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
        ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype)).sample
        ims = ims.to(inference_dtype)
        if "hps" in config.reward_fn:
            loss, rewards = loss_fn(ims, prompts)
        else:    
            # _, rewards = loss_fn(ims)
            rewards = loss_fn(ims)
        return ims, rewards
    elif 'sdxl' in config.pipe_type:
        prompts = list(prompts)
        ims = pipeline(
            prompt=prompts,
            num_inference_steps=config.eval_steps,
            generator=generator,
            output_type="pt").images
        ims = ims * 2 - 1 # from [0, 1] to [-1, 1]
        ims = ims.to(inference_dtype)
        if "hps" in config.reward_fn:
            loss, rewards = loss_fn(ims, prompts)
        else:
            rewards = loss_fn(ims)
        return ims, rewards
    else:
        raise NotImplementedError

    
    

def main():
    FLAGS(sys.argv)
    config = FLAGS.config
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id
    
    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )
        
    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )
    # wandb.login(key='YOURKEY')
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    
    
    
    if accelerator.is_main_process:
        wandb_args = {'name': config.run_name}
        if config.debug:
            wandb_args.update({'mode':"disabled"})       
        accelerator.init_trackers(
            project_name="DiffDoctor-rebuttal", config=config.to_dict(), init_kwargs={"wandb": wandb_args}
        )

        accelerator.project_configuration.project_dir = os.path.join(config.logdir, wandb.run.name)
        accelerator.project_configuration.logging_dir = os.path.join(config.logdir, wandb.run.name)    

    
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)
    
    # load scheduler, tokenizer and models.
    if config.pipe_type == 'sdxl':
        from diffusers import StableDiffusionXLPipeline
        pipeline = StableDiffusionXLPipeline.from_pretrained(config.pretrained.model)
    elif config.pipe_type == 'fluxSchnell' or config.pipe_type == 'fluxdev':
        pipeline = FluxPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    else:
        raise NotImplementedError
    
    # freeze parameters of models to save more memory

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    if hasattr(pipeline, 'text_encoder_2'):
        pipeline.text_encoder_2.requires_grad_(False)
    if hasattr(pipeline, 'transformer'):
        pipeline.transformer.requires_grad_(False)
    if hasattr(pipeline, 'unet'):
        pipeline.unet.requires_grad_(False)


    # disable safety checker
    pipeline.safety_checker = None    
    
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )    

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.    
    inference_dtype = torch.float32

    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16    

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    # pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if 'flux' in config.pipe_type:
        pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
        pipeline.transformer.to(accelerator.device, dtype=inference_dtype)
        if config.grad_checkpoint:
            pipeline.transformer.enable_gradient_checkpointing()
    elif 'sdxl' in config.pipe_type:
        pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
        pipeline.unet.to(accelerator.device, dtype=inference_dtype) 
        if config.grad_checkpoint:
            pipeline.unet.enable_gradient_checkpointing()
    else:
        raise NotImplementedError
    print('Number of parameters:')
    print(f'VAE: {sum(p.numel() for p in pipeline.vae.parameters())/1e6:.2f}M')
    print(f'Text Encoder: {sum(p.numel() for p in pipeline.text_encoder.parameters())/1e6:.2f}M')
    if 'flux' in config.pipe_type:
        print(f'Text Encoder 2: {sum(p.numel() for p in pipeline.text_encoder_2.parameters())/1e6:.2f}M')
        print(f'Transformer: {sum(p.numel() for p in pipeline.transformer.parameters())/1e6:.2f}M')
    elif 'sdxl' in config.pipe_type:
        print(f'Text Encoder 2: {sum(p.numel() for p in pipeline.text_encoder_2.parameters())/1e6:.2f}M')
        print(f'UNet: {sum(p.numel() for p in pipeline.unet.parameters())/1e6:.2f}M')


    # LoRA
    lora_config = LoraConfig(
        r = config.train.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    if 'sd' in config.pipe_type:
        pipeline.unet.add_adapter(lora_config)
        lora_layers = list(filter(lambda p: p.requires_grad, pipeline.unet.parameters()))
        model = pipeline.unet
    elif 'flux' in config.pipe_type:
        if config.load_from_lora is not None:
            assert os.path.exists(config.load_from_lora)
            pipeline.load_lora_weights(config.load_from_lora, adapter_name="default")
            print(f'Load LoRA from {config.load_from_lora}')
        else:
            pipeline.transformer.add_adapter(lora_config)
        lora_layers = list(filter(lambda p: p.requires_grad, pipeline.transformer.parameters()))
        assert len(lora_layers) > 0
        model = pipeline.transformer
    else:
        raise NotImplementedError
    print(f'LoRA: {sum(p.numel() for p in lora_layers)/1e6:.2f}M')

    def save_model_hook(models, weights, output_dir):
        if not accelerator.is_main_process:
            return
        output_splits = output_dir.split("/")
        output_splits[1] = wandb.run.name
        output_dir = "/".join(output_splits)

        if 'sd' in config.pipe_type:
            pipeline.unet.save_attn_procs(output_dir)
        elif 'flux' in config.pipe_type:
            from peft import get_peft_model_state_dict
            # only save lora_layers
            lora_state_dict = get_peft_model_state_dict(accelerator.unwrap_model(pipeline.transformer))
            torch.save(lora_state_dict, os.path.join(output_dir, "lora_model.bin"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
            
        while len(weights) > 0: # can be diffusion model, reward model...
            weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        if 'sd' in config.pipe_type:
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif 'flux' in config.pipe_type:
            from peft import set_peft_model_state_dict
            lora_state_dict = torch.load(os.path.join(input_dir, "lora_model.bin"), map_location=torch.device('cpu'))
            set_peft_model_state_dict(pipeline.transformer, lora_state_dict, adapter_name="default")
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)    

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        lora_layers,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    prompt_fn = getattr(prompts_file, config.prompt_fn)

    if config.eval_prompt_fn == '':
        eval_prompt_fn = prompt_fn
    else:
        eval_prompt_fn = getattr(prompts_file, config.eval_prompt_fn)

    # generate negative prompt embeddings
    train_neg_prompt_embeds = None

    autocast = contextlib.nullcontext
    
    # Prepare everything with our `accelerator`.
    model, optimizer = accelerator.prepare(model, optimizer)

    # prepare for the extra dataset for the diffusion loss
    if hasattr(config, 'diffusion_loss_step_freq') and config.diffusion_loss_step_freq > 0:
        # from datasets import load_dataset
        from src.dataset import DiffusionDataset
        from torch.utils.data import DataLoader
        # diff_dataset = load_dataset(config.diffusion_loss_data_path)
        diff_dataset = DiffusionDataset(config.diffusion_loss_data_path)
        diff_dataloader = DataLoader(diff_dataset, batch_size=config.diffusion_loss_batch_size, 
        shuffle=True, num_workers=32)

    
    if config.reward_fn=='hps':
        loss_fn = hps_loss_fn(config, inference_dtype, accelerator.device)
        eval_loss_fn = loss_fn
    elif config.reward_fn=='ad':
        from src.segformer import get_segformer
        seg_preprocessor, seg_model = get_segformer(config.segformer_path, out_channels=1)
        seg_model.load_state_dict(torch.load(config.resume_from_ad))
        seg_model = accelerator.prepare(seg_model)
        seg_model.requires_grad_(False)
        loss_fn = artifact_detector_loss(
            config, seg_model,seg_preprocessor,accelerator,
            method=config.ad_method
        )
        eval_loss_fn = artifact_detector_loss(
            config, seg_model,seg_preprocessor,accelerator,
            method='all'
        )
    else:
        raise NotImplementedError
    keep_input = True
    
    # prepare timesteps for sd
    if 'sd' in config.pipe_type:
        timesteps = pipeline.scheduler.timesteps
    
    eval_prompts, eval_prompt_metadata = zip(
        *[eval_prompt_fn() for _ in range(config.train.batch_size_per_gpu_available * config.max_vis_images)]
    )    

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 1
    global_step = 1

    #################### TRAINING #################### 
    if hasattr(config, 'diffusion_loss_step_freq') and config.diffusion_loss_step_freq > 0:
        diffusion_global_step = 0
        diffusion_global_epoch = 0
        diffusion_data_iter = iter(diff_dataloader)
    
    if config.vis_before_train:
        print('Visualizing before training')
        all_eval_images = []
        all_eval_rewards = []
        all_eval_max_rewards = []
        all_eval_average_rewards = []
        if 'sd' in config.pipe_type:
            if config.same_evaluation:
                generator = torch.cuda.manual_seed(config.seed)
                latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
            else:
                generator = None
                latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)
        elif 'flux' in config.pipe_type:
            if config.same_evaluation:
                generator = torch.cuda.manual_seed(config.seed)
            else:
                generator = None
            height, width = 1024, 1024
            num_channels_latents = pipeline.transformer.config.in_channels // 4
            latent, latent_image_ids = pipeline.prepare_latents(
                config.train.batch_size_per_gpu_available*config.max_vis_images,
                num_channels_latents,
                height,
                width,
                inference_dtype,
                accelerator.device,
                generator=generator,
            )
        else:
            raise NotImplementedError                                
        with torch.no_grad():
            for index in range(config.max_vis_images):
                if 'sd' in config.pipe_type:
                    latent_input = latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)]
                    latent_image_ids_input = None
                elif 'flux' in config.pipe_type:
                    latent_input = latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)]
                    if config.pipe_type == 'fluxSchnell':
                        latent_image_ids_input = latent_image_ids
                    elif config.pipe_type == 'fluxdev':
                        latent_image_ids_input = latent_image_ids
                else:
                    raise NotImplementedError
                ims, eval_result = evaluate(latent_input,train_neg_prompt_embeds, 
                eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], 
                pipeline, accelerator, inference_dtype,config, eval_loss_fn,latent_image_ids_input, generator)
                all_eval_images.append(ims)
                if type(eval_result) == tuple:
                    rewards = eval_result
                    all_eval_rewards.append(rewards)
                elif type(eval_result) == dict:
                    rewards_max = eval_result['rewards_max']
                    rewards_mean = eval_result['rewards_mean']
                    all_eval_max_rewards.append(rewards_max)
                    all_eval_average_rewards.append(rewards_mean)
        eval_rewards, eval_reward_max, eval_reward_average = None, None, None
        if len(all_eval_rewards) > 0: 
            eval_rewards = torch.cat(all_eval_rewards)
            eval_reward_mean = eval_rewards.mean()
            eval_reward_std = eval_rewards.std()
        if len(all_eval_max_rewards) > 0:
            eval_reward_max = torch.cat(all_eval_max_rewards)
            eval_reward_max_mean = eval_reward_max.mean()
            eval_reward_max_std = eval_reward_max.std()
            eval_reward_average = torch.cat(all_eval_average_rewards)
            eval_reward_average_mean = eval_reward_average.mean()
            eval_reward_average_std = eval_reward_average.std()
        eval_images = torch.cat(all_eval_images)
        eval_image_vis = []
        info = {}
        if accelerator.is_main_process:
            name_val = wandb.run.name
            log_dir = f"logs/{name_val}/eval_vis"
            os.makedirs(log_dir, exist_ok=True)
            for i, eval_image in enumerate(eval_images):
                eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                pil = Image.fromarray((eval_image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                prompt = eval_prompts[i]
                if len(prompt) > 200:
                    prompt = prompt[:200] + "..."
                pil.save(f"{log_dir}/{0:03d}_{0:03d}_{i:03d}_{prompt}.png")
                pil = pil.resize((256, 256))
                if eval_rewards is not None:
                    reward = eval_rewards[i]
                    eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                    info.update({"epoch": 0, "inner_epoch": 0, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
                elif eval_reward_max is not None:
                    reward_max = eval_reward_max[i]
                    reward_average = eval_reward_average[i]
                    eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward_max:.2f} | {reward_average:.2f}"))             
                    info.update({"epoch": 0, "inner_epoch": 0, "eval_rewards_max":eval_reward_max_mean,"eval_rewards_max_std":eval_reward_max_std, "eval_rewards_average":eval_reward_average_mean,"eval_rewards_average_std":eval_reward_average_std})
                info.update({"eval_images": eval_image_vis})
            accelerator.log(info,step=0)               

    for epoch in list(range(first_epoch, config.num_epochs)):
        model.train()
        info = defaultdict(list)
        info_vis = defaultdict(list)
        image_vis_list = []


        batch_cnt_inner_epoch = 0

        # print(config.train.gradient_accumulation_steps)
        for inner_iters in tqdm(list(range(config.train.data_loader_iterations)),position=0,disable=not accelerator.is_main_process, total=config.train.data_loader_iterations):
            # continue
            if accelerator.is_main_process:
                logger.info(f"{wandb.run.name} Epoch {epoch}.{inner_iters}: training")

            perform_optimization = True # for debug(False) by wyy

            batch_cnt_inner_epoch += 1

            if perform_optimization:
                # get the prompts ready
                prompts, prompt_metadata = zip(
                    *[prompt_fn() for _ in range(config.train.batch_size_per_gpu_available)]
                )
                prompts = list(prompts)
                if 'sdxl' in config.pipe_type:
                    # neg_prompts = [""]*(len(prompts))
                    with accelerator.accumulate(model):
                        with torch.enable_grad():
                            ims = pipeline(
                                prompt=prompts,
                                # negative_prompt=neg_prompts,
                                num_inference_steps=config.steps,
                                output_type="pt"
                                ).images
                            ims = 2*ims - 1
                            ims = ims.to(inference_dtype)

                            # from [-1,1] to [0,1]
                            ims = ((ims + 1)/2).clamp(0, 1)

                            # Pixel-aware loss
                            if "hps" in config.reward_fn:
                                loss, rewards = loss_fn(ims, prompts)
                            else:
                                loss, rewards = loss_fn(ims)
                            loss = loss.sum()
                            loss = loss/config.train.batch_size_per_gpu_available
                            loss = loss * config.train.loss_coeff

                    rewards_mean = rewards.mean()
                    rewards_std = rewards.std()
                    
                    if len(info_vis["image"]) < config.max_vis_images:
                        info_vis["image"].append(ims.clone().detach())
                        info_vis["rewards_img"].append(rewards.clone().detach())
                        info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                    
                    info["loss"].append(loss)
                    info["rewards"].append(rewards_mean)
                    info["rewards_std"].append(rewards_std)

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                else:   # FLUX
                    max_sequence_length = 256
                    (
                        prompt_embeds,
                        pooled_prompt_embeds,
                        text_ids,
                    ) = FluxPipeline.encode_prompt(
                        self=pipeline,
                        prompt=prompts,
                        prompt_2=None,
                        prompt_embeds=None,
                        pooled_prompt_embeds=None,
                        device=accelerator.device,
                        num_images_per_prompt=1,
                        max_sequence_length=max_sequence_length,
                        lora_scale=None,
                    )

                    height, width = 1024, 1024
                    num_channels_latents = pipeline.transformer.config.in_channels // 4
                    latent, latent_image_ids = pipeline.prepare_latents(
                        config.train.batch_size_per_gpu_available,
                        num_channels_latents,
                        height,
                        width,
                        inference_dtype,
                        accelerator.device,
                        generator=None,
                    )
                    
                    # prepare timesteps for flux
                    from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
                    num_inference_steps = config.steps
                    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
                    image_seq_len = latent.shape[1]
                    mu = calculate_shift(
                        image_seq_len,
                        pipeline.scheduler.config.base_image_seq_len,
                        pipeline.scheduler.config.max_image_seq_len,
                        pipeline.scheduler.config.base_shift,
                        pipeline.scheduler.config.max_shift,
                    )
                    timesteps, num_inference_steps = retrieve_timesteps(
                        pipeline.scheduler,
                        num_inference_steps,
                        accelerator.device,
                        None,
                        sigmas,
                        mu=mu,
                    )

                    with accelerator.accumulate(model):
                        with autocast():
                            with torch.enable_grad(): # important b/c don't have on by default in module                    
                                keep_input = True

                                # denoising loop
                                for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc='training'):
                                    timestep = t.expand(latent.shape[0]).to(latent.dtype)
                                    guidance = None
                                    if config.pipe_type == 'fluxdev':
                                        if accelerator.unwrap_model(pipeline.transformer).config.guidance_embeds:
                                            guidance = torch.tensor([3.5], device=accelerator.device)
                                            guidance = guidance.expand(latent.shape[0])
                                        else:
                                            guidance = None
                    
                                    noise_pred = model(
                                        hidden_states=latent,
                                        timestep=timestep / 1000,
                                        guidance=guidance,
                                        pooled_projections=pooled_prompt_embeds,
                                        encoder_hidden_states=prompt_embeds,
                                        txt_ids=text_ids,
                                        img_ids=latent_image_ids,
                                        joint_attention_kwargs=None,
                                        return_dict=False,
                                    )[0]
                                                                    
                                    if config.truncated_backprop:
                                        if config.truncated_backprop_rand:
                                            timestep = random.randint(config.truncated_backprop_minmax[0],config.truncated_backprop_minmax[1])
                                            if i < timestep:
                                                noise_pred = noise_pred.detach()
                                        else:
                                            if i > 0 and i < config.trunc_backprop_timestep:
                                                noise_pred = noise_pred.detach()
                                            # truncate backpropagation for all timesteps except the last one
                                            if i < 0 and i < len(timesteps) - 1:
                                                noise_pred.detach()
                                    latent = pipeline.scheduler.step(noise_pred, t, latent, return_dict=False)[0]

                                latent = pipeline._unpack_latents(latent, height, width, pipeline.vae_scale_factor)
                                latent = (latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                                ims = pipeline.vae.decode(latent.to(pipeline.vae.dtype)).sample # WYY: vae is buggy with mixed precision
                                ims = ims.to(inference_dtype)

                                # from [-1,1] to [0,1]
                                ims = ((ims + 1)/2).clamp(0, 1)
                                
                                if "hps" in config.reward_fn:
                                    loss, rewards = loss_fn(ims, prompts)
                                else:
                                    loss, rewards = loss_fn(ims)
                                loss = loss.sum()
                                loss = loss/config.train.batch_size_per_gpu_available
                                loss = loss * config.train.loss_coeff

                                rewards_mean = rewards.mean()
                                rewards_std = rewards.std()
                                
                                if len(info_vis["image"]) < config.max_vis_images:
                                    info_vis["image"].append(ims.clone().detach())
                                    info_vis["rewards_img"].append(rewards.clone().detach())
                                    info_vis["prompts"] = list(info_vis["prompts"]) + list(prompts)
                                
                                info["loss"].append(loss)
                                info["rewards"].append(rewards_mean)
                                info["rewards_std"].append(rewards_std)

                                # backward pass
                                accelerator.backward(loss)
                                if accelerator.sync_gradients:
                                    accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                                optimizer.step()
                                optimizer.zero_grad()
                                
                                ############ Perform diffusion regularization ##############
                                if hasattr(config, 'diffusion_loss_step_freq') and config.diffusion_loss_step_freq > 0 and batch_cnt_inner_epoch % config.diffusion_loss_step_freq == 0:
                                    try:
                                        diffusion_sample = next(diffusion_data_iter)
                                    except StopIteration:
                                        diffusion_data_iter = iter(diff_dataloader)
                                        diffusion_sample = next(diffusion_data_iter)
                                        diffusion_global_epoch += 1
                                    diffusion_global_step += 1
                                    diffusion_image = diffusion_sample["image"].to(accelerator.device)
                                    diffusion_prompt = diffusion_sample["prompt"]
                                    if 'flux' in config.pipe_type:
                                        timesteps = get_schedule(
                                            999,
                                            (1024 // 8) * (1024 // 8) // 4,
                                            shift=True,
                                        )
    
                                        # perform retified flow optimization
                                        with torch.no_grad():
                                            x_1 = pipeline.vae.encode(diffusion_image.to(torch.float32)).latent_dist.sample()
                                            x_1 = x_1.to(inference_dtype)

                                            # prepare pooled_prompt_embeds, prompt_embeds,text_ids, latent_image_ids
                                            # inp = prepare(t5=pipeline.text_encoder_2, clip=pipeline.text_encoder, img=x_1, prompt=diffusion_prompt) 
                                            bs, _, h, w = x_1.shape
                                            if bs == 1 and not isinstance(diffusion_prompt, str):
                                                bs = len(diffusion_prompt)

                                            img_ids = torch.zeros(h // 2, w // 2, 3)
                                            img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
                                            img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
                                            img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
                                            img_ids = img_ids.to(inference_dtype)

                                            max_sequence_length = 256
                                            diffusion_prompt = list(diffusion_prompt)
                                            (
                                                prompt_embeds, # t5 embeddings
                                                pooled_prompt_embeds, # clip pooled embeddings
                                                text_ids,
                                            ) = pipeline.encode_prompt(
                                                prompt=diffusion_prompt,
                                                prompt_2=None,
                                                prompt_embeds=None,
                                                pooled_prompt_embeds=None,
                                                device=accelerator.device,
                                                num_images_per_prompt=1,
                                                max_sequence_length=max_sequence_length,
                                                lora_scale=None,
                                            )
                                            prompt_embeds = prompt_embeds.to(x_1.device)
                                            text_ids = text_ids.to(x_1.device)
                                            pooled_prompt_embeds = pooled_prompt_embeds.to(x_1.device)
                                            img_ids = img_ids.to(x_1.device)
                                            
                                            
                                            # patchify x_1
                                            x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                                        bs = diffusion_image.shape[0]
                                        t = torch.tensor([timesteps[random.randint(0, 999)]]).to(accelerator.device).to(inference_dtype)
                                        x_0 = torch.randn_like(x_1).to(accelerator.device)
                                        x_t = (1 - t) * x_1 + t * x_0
                                        bsz = x_1.shape[0]
                                        # guidance_vec = torch.full((x_t.shape[0],), 1, device=x_t.device, dtype=x_t.dtype)
                                        guidance_vec = None
                                        if config.pipe_type == 'fluxdev':
                                            if accelerator.unwrap_model(pipeline.transformer).config.guidance_embeds:
                                                guidance_vec = torch.tensor([3.5], device=accelerator.device)
                                                guidance_vec = guidance.expand(latent.shape[0])
                                            else:
                                                guidance_vec = None

                                        model_pred = pipeline.transformer(
                                            hidden_states=x_t,
                                            img_ids=img_ids,
                                            encoder_hidden_states=prompt_embeds,
                                            txt_ids=text_ids,
                                            pooled_projections=pooled_prompt_embeds,
                                            timestep=t,
                                            guidance=guidance_vec,
                                            joint_attention_kwargs=None,
                                            return_dict=False,
                                        )[0]
                                        
                                        loss_diff = config.diffusion_loss_gamma * F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                                        info["diffusion_loss"].append(loss_diff)
                                        # backward pass
                                        accelerator.backward(loss_diff)
                                        if accelerator.sync_gradients:
                                            accelerator.clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
                                        optimizer.step()
                                        optimizer.zero_grad()
                                    else:
                                        print(f'{config.pipe_type} is not supported for diffusion regularization')
                                        exit(1)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients: 
                # print(epoch, inner_iters)
                if perform_optimization:
                    assert (
                        inner_iters + 1
                    ) % config.train.gradient_accumulation_steps == 0
                # log training and evaluation 
                if config.visualize_eval and ((global_step-1) % config.vis_freq ==0):
                    all_eval_images = []
                    all_eval_rewards = []
                    all_eval_max_rewards = []
                    all_eval_average_rewards = []
                    if 'sd' in config.pipe_type:
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype, generator=generator)    
                        else:
                            generator = None
                            latent = torch.randn((config.train.batch_size_per_gpu_available*config.max_vis_images, 4, 64, 64), device=accelerator.device, dtype=inference_dtype)   
                    elif 'flux' in config.pipe_type:
                        if config.same_evaluation:
                            generator = torch.cuda.manual_seed(config.seed)
                        else:
                            generator = None
                        height, width = 1024, 1024
                        num_channels_latents = pipeline.transformer.config.in_channels // 4
                        latent, latent_image_ids = pipeline.prepare_latents(
                            config.train.batch_size_per_gpu_available*config.max_vis_images,
                            num_channels_latents,
                            height,
                            width,
                            inference_dtype,
                            accelerator.device,
                            generator=generator,
                        )
                    else:
                        raise NotImplementedError                                
                    with torch.no_grad():
                        for index in range(config.max_vis_images):
                            if 'sd' in config.pipe_type:
                                latent_input = latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)]
                                latent_image_ids_input = None
                            elif 'flux' in config.pipe_type:
                                latent_input = latent[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)]
                                if config.pipe_type == 'fluxSchnell':
                                    latent_image_ids_input = latent_image_ids
                                elif config.pipe_type == 'fluxdev':
                                    latent_image_ids_input = latent_image_ids
                            else:
                                raise NotImplementedError
                            ims, eval_result = evaluate(latent_input,train_neg_prompt_embeds, 
                            eval_prompts[config.train.batch_size_per_gpu_available*index:config.train.batch_size_per_gpu_available *(index+1)], 
                            pipeline, accelerator, inference_dtype,config, eval_loss_fn,latent_image_ids_input,generator=generator)
                            all_eval_images.append(ims)
                            if type(eval_result) == tuple:
                                rewards = eval_result
                                all_eval_rewards.append(rewards)
                            elif type(eval_result) == dict:
                                rewards_max = eval_result['rewards_max']
                                rewards_mean = eval_result['rewards_mean']
                                all_eval_max_rewards.append(rewards_max)
                                all_eval_average_rewards.append(rewards_mean)
                    eval_rewards, eval_reward_max, eval_reward_average = None, None, None
                    if len(all_eval_rewards) > 0: 
                        eval_rewards = torch.cat(all_eval_rewards)
                        eval_reward_mean = eval_rewards.mean()
                        eval_reward_std = eval_rewards.std()
                    if len(all_eval_max_rewards) > 0:
                        eval_reward_max = torch.cat(all_eval_max_rewards)
                        eval_reward_max_mean = eval_reward_max.mean()
                        eval_reward_max_std = eval_reward_max.std()
                        eval_reward_average = torch.cat(all_eval_average_rewards)
                        eval_reward_average_mean = eval_reward_average.mean()
                        eval_reward_average_std = eval_reward_average.std()

                    eval_images = torch.cat(all_eval_images)
                    eval_image_vis = []

                    if accelerator.is_main_process:
                        name_val = wandb.run.name
                        log_dir = f"logs/{name_val}/eval_vis"
                        os.makedirs(log_dir, exist_ok=True)
                        for i, eval_image in enumerate(eval_images):
                            eval_image = (eval_image.clone().detach() / 2 + 0.5).clamp(0, 1)
                            pil = Image.fromarray((eval_image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                            prompt = eval_prompts[i]
                            if len(prompt) > 200:
                                prompt = prompt[:200] + "..."
                            pil.save(f"{log_dir}/{epoch:03d}_{inner_iters:03d}_{i:03d}_{prompt}.png")
                            pil = pil.resize((256, 256))
                            if eval_rewards is not None:
                                reward = eval_rewards[i]
                                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                            elif eval_reward_max is not None:
                                reward_max = eval_reward_max[i]
                                reward_average = eval_reward_average[i]
                                eval_image_vis.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward_max:.2f} | {reward_average:.2f}"))                  
                        accelerator.log({"eval_images": eval_image_vis},step=global_step)
                
                    logger.info("Logging")
                    
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = accelerator.reduce(info, reduction="mean")
                    logger.info(f"loss: {info['loss']}, rewards: {info['rewards']}")

                    if eval_rewards is not None:
                        info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards":eval_reward_mean,"eval_rewards_std":eval_reward_std})
                    elif eval_reward_max is not None:
                        info.update({"epoch": epoch, "inner_epoch": inner_iters, "eval_rewards_max":eval_reward_max_mean,"eval_rewards_max_std":eval_reward_max_std,
                        "eval_rewards_average":eval_reward_average_mean,"eval_rewards_average_std":eval_reward_average_std})
                    accelerator.log(info, step=global_step)

                if config.visualize_train:
                    ims = torch.cat(info_vis["image"])
                    rewards = torch.cat(info_vis["rewards_img"])
                    prompts = info_vis["prompts"]
                    images  = []
                    for i, image in enumerate(ims):
                        image = (image.clone().detach() / 2 + 0.5).clamp(0, 1)
                        pil = Image.fromarray((image.cpu().float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                        pil = pil.resize((256, 256))
                        prompt = prompts[i]
                        reward = rewards[i]
                        images.append(wandb.Image(pil, caption=f"{prompt:.25} | {reward:.2f}"))
                    
                    accelerator.log(
                        {"images": images},
                        step=global_step,
                    )

                global_step += 1
                info = defaultdict(list)

        # make sure we did an optimization step at the end of the inner epoch
        assert accelerator.sync_gradients
        
        if (epoch-1) % config.save_freq == 0:
            accelerator.save_state()

if __name__ == "__main__":
    main()
