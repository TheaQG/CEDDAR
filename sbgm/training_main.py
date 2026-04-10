# sbgm/training_main.py
import os
import torch
import logging

from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt

from sbgm.training_utils import get_model_string, get_model, get_optimizer, get_dataloader, get_scheduler
from sbgm.plotting_utils import plot_sample
from sbgm.training import TrainingPipeline_general
from sbgm.score_unet import marginal_prob_std_fn, diffusion_coeff_fn


# Set up logging
logger = logging.getLogger(__name__)


def _runtime_from_cfg(cfg: dict) -> dict:
    runtime = cfg.get('runtime', {}) if isinstance(cfg, dict) else {}
    return {
        'distributed': bool(runtime.get('distributed', False)),
        'rank': int(runtime.get('rank', 0)),
        'local_rank': int(runtime.get('local_rank', 0)),
        'world_size': int(runtime.get('world_size', 1)),
        'device': runtime.get('device', cfg.get('training', {}).get('device', 'cpu')),
        'is_main_process': bool(runtime.get('is_main_process', runtime.get('rank', 0) == 0)),
    }


def _resolve_device(cfg: dict, runtime: dict) -> torch.device:
    runtime_device = str(runtime.get('device', cfg['training']['device']))

    if runtime_device.startswith('cuda'):
        if torch.cuda.is_available():
            return torch.device(runtime_device)
        logger.warning("Runtime requested CUDA device '%s', but CUDA is unavailable. Falling back to CPU.", runtime_device)
        return torch.device('cpu')

    if runtime_device == 'cuda':
        if torch.cuda.is_available():
            local_rank = int(runtime.get('local_rank', 0))
            return torch.device(f'cuda:{local_rank}')
        logger.warning("Config requested CUDA, but CUDA is unavailable. Falling back to CPU.")
        return torch.device('cpu')

    return torch.device('cpu')


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, 'module') else model

def train_main(cfg):
    """
    Main function to run the training process.
    
    Args:
        cfg (dict): Configuration dictionary containing all necessary parameters.
    """

    runtime = _runtime_from_cfg(cfg)
    is_main_process = runtime['is_main_process']

    logger.info("\n\n=== Starting SBGM_SD Training Pipeline ===")
    logger.info(f"          Experiment name: {cfg['experiment']['name']}")
    logger.info(
        "          Runtime: distributed=%s rank=%s local_rank=%s world_size=%s requested_device=%s",
        runtime['distributed'], runtime['rank'], runtime['local_rank'], runtime['world_size'], runtime['device'],
    )
    logger.info(
        "          Config path: %s | config basename: %s",
        cfg.get('config_path', 'N/A'),
        os.path.basename(str(cfg.get('config_path', 'N/A'))),
    )

    # Set path to figures, samples, losses
    save_str = get_model_string(cfg)
    path_samples = os.path.join(cfg['paths']['path_save'], 'samples', save_str)
    path_figures = os.path.join(path_samples, 'Figures')

    # Make sure figures directory exists (rank 0 only under DDP)
    if is_main_process:
        os.makedirs(path_figures, exist_ok=True)

    # Resolve device from distributed runtime first, then fall back to config.
    device = _resolve_device(cfg, runtime)

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        logger.info("          ▸ Using GPU device: %s", torch.cuda.get_device_name(device))
    else:
        logger.info("          ▸ Using CPU for training.")

    # Load data
    train_dataloader, val_dataloader, gen_dataloader = get_dataloader(cfg)

    # # ------------------------------------------------------------------------
    # # Quick data-loader throughput check: ~100 batches warm-up + timed 
    # # ------------------------------------------------------------------------
    # from time import perf_counter
    # start = perf_counter()
    # for i, _ in enumerate(train_dataloader):
    #     if i == 100:
    #         avg = (perf_counter() - start) / 100
    #         logger.info(f"          ▸ Dataloader average fetch time ~{avg:.3f} s / batch\n\n")



    # Examine sample from train dataloader (sample is full batch) on main process only.
    if is_main_process:
        sample = train_dataloader.dataset[0]
        for key, value in sample.items():
            try:
                logger.info(f'          {key}: {value.shape}')
                logger.info(f'              {key} device: {value.device}')
            except AttributeError:
                logger.info(f'          {key}: {value}')
            if key == 'classifier':
                logger.info(f'          ▸ Classifier: {value}')

        if cfg['visualization']['plot_initial_sample']:
            fig, _ = plot_sample(sample, cfg)
            if cfg['visualization']['show_figs']:
                plt.show()
            else:
                plt.close(fig)
            save_name = 'Initial_sample_plot.png'
            save_path = os.path.join(path_figures, save_name)
            fig.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"\n\n          ▸ Saved initial sample plot to {save_path}")
    
    
    #Setup checkpoint path
    checkpoint_dir = os.path.join(cfg['paths']['path_save'], cfg['paths']['checkpoint_dir'])

    checkpoint_name = save_str + '.pth.tar'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    # Define the seed for reproducibility.
    # IMPORTANT: model construction must use the same seed on all ranks before DDP wrap.
    base_seed = int(cfg['training']['seed'])
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
        torch.cuda.manual_seed_all(base_seed)
    np.random.seed(base_seed)

    # Set torch backend flags
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # Get the model
    # NOTE: get_model(cfg) already handles moving the model to the configured device.
    # Do not call model.to(device) again here.
    model, checkpoint_path, checkpoint_name = get_model(cfg)

    raw_model = _unwrap_model(model)
    n_params_pre_ddp = sum(p.numel() for p in raw_model.parameters())
    param_shapes_pre_ddp = [tuple(p.shape) for p in raw_model.parameters()]
    logger.info(
        "[DDP PRECHECK] rank=%s experiment=%s config_name=%s config_path=%s input_channels=%s output_channels=%s "
        "context_encoder=%s rain_gate=%s n_params=%s first_20_param_shapes=%s",
        runtime['rank'],
        cfg.get('experiment', {}).get('name'),
        cfg.get('config_name', 'N/A'),
        cfg.get('config_path', 'N/A'),
        getattr(raw_model, 'c_in', 'N/A'),
        getattr(raw_model, 'c_out', 'N/A'),
        cfg.get('model', {}).get('use_context_encoder', cfg.get('paper2', {}).get('spatial_context', {}).get('enabled', 'N/A')),
        cfg.get('rain_gate', {}).get('enabled', 'N/A'),
        n_params_pre_ddp,
        param_shapes_pre_ddp[:20],
    )
    logger.info(
        "[DDP CFG] rank=%s exp=%s lr_vars=%s use_spatial_context=%s main_lr_cond=%s cond_channels=%s",
        runtime['rank'],
        cfg.get('experiment', {}).get('name'),
        cfg.get('data', {}).get('lr_vars', cfg.get('data_handling', {}).get('lr_vars', 'N/A')),
        cfg.get('paper2', {}).get('spatial_context', {}).get('enabled', 'N/A'),
        cfg.get('data', {}).get('main_lr_cond', cfg.get('data_handling', {}).get('main_lr_cond', 'N/A')),
        getattr(raw_model, 'c_in', 'N/A'),
    )

    if runtime['distributed']:
        if device.type != 'cuda':
            raise RuntimeError('Distributed training is currently expected to run on CUDA devices.')
        model = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        logger.info("          ▸ Wrapped model in DistributedDataParallel on rank %s", runtime['rank'])

    # Get the optimizer
    optimizer = get_optimizer(cfg, _unwrap_model(model))

    # Get the learning rate scheduler (if applicable)
    lr_scheduler_type = cfg['training'].get('lr_scheduler', None)
    
    if lr_scheduler_type is not None:
        logger.info(f"          ▸ Using learning rate scheduler: {lr_scheduler_type}")
        scheduler = get_scheduler(cfg, optimizer)
    else:
        scheduler = None
        logger.info(f"          ▸ No learning rate scheduler specified, using default learning rate.")

    # Define the training pipeline
    pipeline = TrainingPipeline_general(model=model,
                                        marginal_prob_std_fn=marginal_prob_std_fn,
                                        diffusion_coeff_fn=diffusion_coeff_fn,
                                        optimizer=optimizer,
                                        device=device,
                                        lr_scheduler=scheduler,
                                        cfg=cfg
                                        )

    
    # Load checkpoint if it exists
    if cfg['training']['load_checkpoint'] and os.path.exists(checkpoint_path):
        logger.info(f"          ▸ Loading pretrained weights from checkpoint {checkpoint_path}")
        pipeline.load_checkpoint(checkpoint_path, load_ema=cfg['training']['load_ema'],)
    else:
        if is_main_process:
            logger.info(f"          ▸ No checkpoint found at {checkpoint_path}. Starting training from scratch.")

    
    raw_model = _unwrap_model(model)
    n_params = sum(p.numel() for p in raw_model.parameters())
    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    n_non_trainable = sum(p.numel() for p in raw_model.parameters() if not p.requires_grad)

    if device.type == 'cuda' and torch.cuda.is_available():
        logger.info("\n\n          ▸ Using GPU: %s", torch.cuda.get_device_name(device))
        logger.info("          ▸ Model is using %.2f GB of GPU memory.", torch.cuda.memory_allocated(device) / 1e9)
        logger.info("          ▸ Total GPU memory: %.2f GB", torch.cuda.get_device_properties(device).total_memory / 1e9)
        logger.info("\n          ▸ Number of parameters in model: %s", f"{n_params:,}")
        logger.info("          ▸ Number of trainable parameters in model: %s", f"{n_trainable:,}")
        logger.info("          ▸ Number of non-trainable parameters in model: %s", f"{n_non_trainable:,}")
        torch.cuda.empty_cache()
    else:
        logger.info("\n\n          ▸ Using CPU for training.")
        logger.info("          ▸ Number of parameters in model: %s", f"{n_params:,}")
        logger.info("          ▸ Number of trainable parameters in model: %s", f"{n_trainable:,}")
        logger.info("          ▸ Number of non-trainable parameters in model: %s", f"{n_non_trainable:,}")

    # Perform training
    logger.info(f"\n\n          === STARTING TRAINING MAIN LOOP ===\n")
    pipeline.train(train_dataloader,
                   val_dataloader,
                   gen_dataloader,
                   cfg,
                   epochs=cfg['training']['epochs'],
                   verbose=cfg['training']['verbose'],
                   use_mixed_precision=cfg['training']['use_mixed_precision'],
    )
    if is_main_process:
        logger.info("\n\n       === TRAINING COMPLETE ===\n\n")


















