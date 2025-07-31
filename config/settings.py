import ml_collections
import os



def general():
    config = ml_collections.ConfigDict()

    ###### General ######    
    config.vis_before_train = False
    config.eval_prompt_fn = ''
    config.save_freq = 4
    config.resume_from = ""
    config.resume_from_2 = ""
    config.vis_freq = 4
    config.max_vis_images = 2
    config.run_name = "test"
    
    config.debug =False
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision  = "bf16"
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 10
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    # config.run_name = ""
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # random seed for reproducibility.
    config.seed = 314159
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100    

    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True

    config.visualize_train = True
    config.visualize_eval = True

    config.truncated_backprop = True
    config.truncated_backprop_rand = False
    config.truncated_backprop_minmax = (35,45)
    config.trunc_backprop_timestep = -1
    
    config.grad_checkpoint = True
    config.same_evaluation = True
    
    
    ###### Training ######    
    config.train = train = ml_collections.ConfigDict()
    config.train.loss_coeff = 1.0
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8 
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0    
    train.lora_rank = 8

    config.grad_scale = 1
    config.sd_guidance_scale = 7.5

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    # pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.pipe_type = 'fluxdev'

    if config.pipe_type == 'fluxSchnell':
        pretrained.model = 'YOUR_PATH/FLUX.1-schnell'
        config.steps = 4 # inference step number for the diffusion model
        config.eval_steps = 4
    elif config.pipe_type == 'fluxdev':
        pretrained.model = 'YOUR_PATH/FLUX.1-dev'
        config.steps = 10
        config.eval_steps = 10
    elif config.pipe_type == 'sdxl':
        config.eval_steps = 30
        config.steps = 30
        pretrained.model = 'YOUR_PATH/stable-diffusion-xl-base-1.0'
    else:
        raise NotImplementedError(f"pipe_type {config.pipe_type} not implemented")
    
    config.load_from_lora = None
    
    # revision of the model to load.
    pretrained.revision = "main"
    return config



def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):
    #  Samples per epoch
    config.train.total_samples_per_epoch = total_samples_per_epoch  #(~~~~ this is desired ~~~~)
    config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    # config.train.num_gpus = 1
    
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #(~~~~ this is desired ~~~~)
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus
    config.train.batch_size_per_gpu_available = per_gpu_capacity    #(this quantity depends on the gpu used)
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available
    
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available    
    return config

def ad():
    config = general()
    config.run_name = 'test_run'

    # diffusion regularization term
    config.diffusion_loss_step_freq = 0 # <= 0 to disable
    config.diffusion_loss_data_path = 'YOUR_PATH/laion-600k-aesthetic-6.5plus-768'
    config.diffusion_loss_batch_size = 1
    config.diffusion_loss_gamma = 0.25

    # segformer for ad
    config.segformer_path = 'nvidia/mit-b5'
    config.resume_from_ad = 'checkpoints/ad_pytorch_model.bin'
    config.ad_loss_scale = 1.0
    # config.ad_reward_scare = 5
    config.ad_method = 'weighted'
    if config.ad_method == 'sum':
        config.ad_reward_scare = 1e-5
    elif config.ad_method == 'max':
        config.ad_reward_scare = 5
    elif config.ad_method == 'mean':
        config.ad_reward_scare = 5
    elif config.ad_method == 'weighted':
        config.ad_reward_scare = 5

    config.num_epochs = 100
    config.prompt_fn = "test_human_1"
    # config.prompt_fn = "humans_3000"
    # config.prompt_fn = "animals_train"
    # config.prompt_fn = "humans_500"
    # config.prompt_fn = "elon_train"

    # config.eval_prompt_fn = "elon_test"
    # config.eval_prompt_fn = "test_human_2"
    config.eval_prompt_fn = "test_human_2"
    # config.eval_prompt_fn = "animals_eval"

    config.reward_fn = 'ad' # CLIP or imagenet or .... or .. 
    config.train.max_grad_norm = 5.0    
    config.train.loss_coeff = 1.0
    config.train.learning_rate = 1e-4
    # config.max_vis_images = 100
    config.max_vis_images = 4
    config.train.adam_weight_decay = 0.1
    
    config.save_freq = 1
    config.num_checkpoint_limit = 50
    config.truncated_backprop_rand = False
    config.truncated_eanckprop_minmax = (0,50)
    config.trunc_backprop_timestep = -1
    config.truncated_backprop = True
    # config = set_config_batch(config,total_samples_per_epoch=128,total_batch_size=32, per_gpu_capacity=1)
    config = set_config_batch(config,total_samples_per_epoch=32,total_batch_size=4, per_gpu_capacity=1)
    return config




def hps():
    config = general()
    config.run_name = 'hps_v2_run'
    config.hps_ckpt_path = 'YOUR_PATH/HPS_v2_compressed.pt'
    config.num_epochs = 200
    config.vis_freq = 4
    # config.prompt_fn = "hps_v2_all"
    # config.eval_prompt_fn = 'eval_hps_v2_all'
    config.reward_fn = 'hps'
    # config.per_prompt_stat_tracking = { 
    #     "buffer_size": 32,
    #     "min_count": 16,
    # }
    config.prompt_fn = "humans_3000"
    config.eval_prompt_fn = "humans_100_2"
    config.max_vis_images = 4
    config.train.max_grad_norm = 5.0    
    # config.train.loss_coeff = 0.01
    config.train.loss_coeff = 0.1
    config.train.learning_rate = 1e-4
    config.train.adam_weight_decay = 0.1

    config.trunc_backprop_timestep = -1
    config.truncated_backprop = True
    config.truncated_backprop_rand = False
    # config.truncated_backprop_minmax = (0,50)    
    config = set_config_batch(config,total_samples_per_epoch=32,total_batch_size=4, per_gpu_capacity=1)
    return config


def get_config(name):
    return globals()[name]()