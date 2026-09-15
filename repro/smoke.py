"""One-date inference check; synthetic inputs/random weights are NOT scientific results."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from sbgm.runtime import SOURCE_ROOT, external_output, setup_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=SOURCE_ROOT / 'repro/01_small_test/small_test_config.yaml')
    parser.add_argument('--data-root', type=Path, help='Real data; omitted means an external synthetic fixture')
    parser.add_argument('--checkpoint', type=Path, help='Trusted trained checkpoint matching --config')
    parser.add_argument('--stats-root', type=Path, help='Training statistics belonging to the checkpoint')
    parser.add_argument('--require-real', action='store_true', help='Require data, checkpoint and statistics; never use fixtures')
    parser.add_argument('--preflight-only', action='store_true', help='Check inputs, weight compatibility and inverse transforms without sampling')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--split', choices=['train', 'valid', 'test'], default='test')
    parser.add_argument('--output', type=Path, help='New external directory; existing directories are rejected')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Smoke checks require assertions; unset PYTHONOPTIMIZE and do not use python -O')
    if args.steps < 2 or args.threads < 1:
        parser.error('--steps must be >=2 and --threads >=1')
    if args.require_real and not all((args.data_root, args.checkpoint, args.stats_root)):
        parser.error('--require-real needs --data-root, --checkpoint and --stats-root')
    # Fail before constructing fixtures/model or writing a run directory.
    for path, is_dir in ((args.config, False), (args.data_root, True),
                         (args.checkpoint, False), (args.stats_root, True)):
        if path is not None and not (path.expanduser().is_dir() if is_dir else path.expanduser().is_file()):
            parser.error(f'Missing {"directory" if is_dir else "file"}: {path}')
    if args.stats_root:
        os.environ['STATS_LOAD_DIR'] = str(args.stats_root.expanduser().resolve())
    setup_environment()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    out = external_output(args.output or Path(os.environ['CEDDAR_RUNS']) / 'smoke' / stamp)
    out.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import torch
    import zarr
    from sbgm.utils import build_data_path, extract_samples, load_config
    from sbgm.training_utils import get_final_gen_dataloader, get_model
    from sbgm.generate.generation import GenerationConfig, GenerationRunner
    from sbgm.special_transforms import get_transforms_from_stats
    from sbgm.evaluate.data_resolver import EvalDataResolver
    from sbgm.evaluate.evaluate_prcp.eval_sigma_star.metrics_sigma_control import crps_ensemble_local
    from sbgm.provenance import checkpoint_info, write_provenance

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    torch.manual_seed(504)
    np.random.seed(504)
    cfg = load_config(args.config.expanduser())
    if not cfg.get('edm', {}).get('enabled', False) or cfg.highres.variable != 'prcp':
        raise ValueError('This smoke test supports EDM precipitation configurations only')
    if args.stats_root:
        cfg.paths.stats_load_dir = str(args.stats_root.expanduser().resolve())
    cfg.training.device = args.device
    cfg.edm.sampling_steps = args.steps
    cfg.sampler.n_timesteps = args.steps
    cfg.full_gen_eval.update({'ensemble_size': 2, 'max_dates': 1, 'split': args.split, 'seed': 504})
    # Supplied artifacts are not automatically authentic or trained; record origin only.
    cfg.smoke = {'data_source': 'synthetic_fixture' if args.data_root is None else 'supplied',
                 'weights_source': 'random_initialization' if args.checkpoint is None else 'supplied_checkpoint'}
    for key in ('checkpoint_dir', 'sample_dir', 'evaluation_dir', 'log_dir', 'path_save'):
        cfg.paths[key] = str(out)
    cfg.full_gen_eval.gen_dir = str(out / 'generation')
    cfg.full_gen_eval.eval_dir = str(out)
    if 'diagnostics' in cfg:
        cfg.diagnostics.histogram_path = str(out / 'histograms')
    if args.data_root is not None:
        root = args.data_root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        # Paths were resolved by load_config; rebase static inputs explicitly.
        old_root = Path(cfg.paths.data_dir)
        for key in ('lsm_path', 'topo_path', 'slope_path'):
            if Path(cfg.paths[key]).is_relative_to(old_root):
                cfg.paths[key] = str(root / Path(cfg.paths[key]).relative_to(old_root))
    else:
        if cfg.highres.variable != 'prcp' or list(cfg.lowres.condition_variables) != ['prcp']:
            raise ValueError('Synthetic fixture supports precipitation-only configs; use --data-root for others')
        root = out / 'synthetic_data'
        dims = tuple(cfg.highres.full_domain_dims)
        yy, xx = np.indices(dims, dtype=np.float32)
        physical = 2 + 0.5 * np.sin(xx / 11) + 0.5 * np.cos(yy / 17)
        for model, values in ((cfg.highres.model, physical), (cfg.lowres.model, physical / 1000)):
            path = build_data_path(root, model, 'prcp', dims, args.split)
            group = zarr.open_group(path, mode='w').create_group('prcp_20190101')
            group.create_dataset('tp', data=values, chunks=(128, 128))
        for key, values in (('lsm_path', (xx > dims[1] / 2).astype('float32')),
                            ('topo_path', 20 + xx / 20), ('slope_path', np.zeros(dims, dtype='float32'))):
            path = root / f'{key}.npz'
            np.savez_compressed(path, data=values)
            cfg.paths[key] = str(path)
        # Smaller spatial crop only for setup; keep the configured network architecture.
        for section in ('highres', 'lowres'):
            cfg[section].data_size = [32, 32]
        cfg.evaluation.stationary_cutout.hr_bounds = [200, 232, 380, 412]
        cfg.evaluation.stationary_cutout.lr_bounds = [200, 232, 380, 412]
    cfg.paths.data_dir = str(root)
    cfg.data_handling.split = args.split
    print(f'Data: {root}\nStatistics: {cfg.paths.stats_load_dir}\nOutput: {out}', flush=True)
    print('Loading one date and constructing the configured model...', flush=True)
    loader = get_final_gen_dataloader(cfg, split=args.split, verbose=False)
    batch = next(iter(loader), None)
    if batch is None:
        raise ValueError(f'No paired dates available for split {args.split}')
    tensors = extract_samples(batch, args.device)
    x = tensors[0]
    assert x is not None, 'Missing HR input'
    for tensor in tensors[:7]:
        if torch.is_tensor(tensor):
            assert torch.isfinite(tensor).all(), 'Nonfinite input/conditioning after loading/scaling'
    model, _, _ = get_model(cfg)
    model = model.to(args.device).eval()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else None
    if checkpoint is None:
        checkpoint = out / 'random_weights.pth.tar'
        torch.save({'network_params': model.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location=args.device)
    key = 'ema_network_params' if cfg.training.get('eval_use_ema', False) and 'ema_network_params' in saved else 'network_params'
    model.load_state_dict(saved[key])
    del saved
    model._ceddar_checkpoint = checkpoint_info(checkpoint, key)
    cfg.paths.checkpoint_name = checkpoint.name
    # The input checkpoint is distinct from the output checkpoint directory.
    cfg.smoke.checkpoint_path = str(checkpoint)
    print(f'Loaded {key}: {checkpoint}', flush=True)
    generation_root = out / 'generation'
    runner = GenerationRunner(model, cfg, args.device, generation_root,
                              GenerationConfig(str(generation_root), ensemble_size=2, max_dates=1, save_space='both'))
    hr = cfg.highres
    forward = get_transforms_from_stats(
        hr.variable, hr.model, 'x'.join(map(str, hr.full_domain_dims)),
        '_'.join(map(str, hr.cutout_domains)), cfg.transforms.get('scaling_split', 'train'), hr.scaling_method, hr.buffer_frac,
        stats_file_path=cfg.paths.stats_load_dir, eps=cfg.transforms.prcp_eps)
    physical_probe = torch.tensor([[1., 2.], [3., 5.]])
    torch.testing.assert_close(runner.bt_hr(forward(physical_probe)), physical_probe, rtol=1e-5, atol=1e-5)
    if args.preflight_only:
        summary = {'preflight_passed': True, 'sampling_run': False, **dict(cfg.smoke),
                   'date': str(batch['date'][0]), 'device': args.device, 'input_shape': list(x.shape)}
        write_provenance(out, cfg, stage='preflight_completed', device=args.device,
                         checkpoint=model._ceddar_checkpoint, preflight_result=summary)
        (out / 'preflight_result.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('Preflight passed; inference/evaluation have not run.', flush=True)
        return
    print(f'Sampling two members with {args.steps} EDM steps...', flush=True)
    runner.run([batch], save=True)
    resolver = EvalDataResolver(gen_root=generation_root, eval_land_only=False, lr_phys_key='lr')
    dates = resolver.list_dates()
    assert len(dates) == 1, f'Expected one saved date, got {dates}'
    obs, ens = resolver.load_obs(dates[0]), resolver.load_ens(dates[0])
    for array in (obs, ens, resolver.load_pmm(dates[0]), resolver.load_lr(dates[0])):
        assert array is not None and torch.isfinite(array).all(), 'Missing/nonfinite saved physical array'
    assert ens.shape == (2, *obs.shape), ens.shape
    torch.testing.assert_close(obs, runner.bt_hr(x).detach().cpu().squeeze(), rtol=1e-5, atol=1e-5)
    crps = crps_ensemble_local(obs, ens).item()
    assert np.isfinite(crps) and crps >= 0
    assert crps_ensemble_local(obs, obs.unsqueeze(0).repeat(2, 1, 1)).item() == 0
    summary = {'passed': True, **dict(cfg.smoke), 'date': dates[0], 'device': args.device,
               'shape': list(ens.shape), 'steps': args.steps, 'crps': crps,
               'seconds': time.perf_counter() - started}
    write_provenance(out, cfg, stage='smoke_completed', device=args.device,
                     checkpoint=model._ceddar_checkpoint, smoke_result=summary)
    (out / 'smoke_result.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'Smoke artifacts: {out}')


if __name__ == '__main__':
    main()
