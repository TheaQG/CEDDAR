"""Read-only first-date diagnosis; no model/checkpoint loading or inference.

Run alongside recovery.py using diagnose_inputs.sh and the existing settings.
Only a diagnostic JSON and temporary caches/directories are written. No hash
checks are relaxed and this script does not authorize recovery.
"""
import datetime
import json
import os
from pathlib import Path
import platform
import tempfile
import traceback

import recovery as r


def tensor_summary(value):
    import torch
    result = dict(shape=list(value.shape), dtype=str(value.dtype))
    if value.numel():
        flat = value.detach().cpu().double().flatten()
        result.update(finite=bool(torch.isfinite(flat).all()),
                      min=float(flat.min()), max=float(flat.max()), mean=float(flat.mean()))
        if value.numel() <= 64:
            result['values'] = value.detach().cpu().tolist()
    return result


def differences(saved, current):
    return {key: dict(saved=saved.get(key), current=current.get(key),
                      matches=key in saved and key in current and saved[key] == current[key])
            for key in sorted(set(saved) | set(current))}


def array_comparison(saved, current):
    import numpy as np
    if hasattr(current, 'detach'):
        current = current.detach().cpu().numpy()
    current = np.asarray(current)
    result = dict(saved_shape=list(saved.shape), current_shape=list(current.shape),
                  saved_dtype=str(saved.dtype), current_dtype=str(current.dtype))
    if saved.shape != current.shape:
        result['shape_matches'] = False
        return result
    a, b = saved.astype('float64'), current.astype('float64')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    result.update(shape_matches=True, finite=finite, exact=bool(np.array_equal(saved, current)))
    if finite and a.size:
        result.update(max_absolute_difference=float(np.max(np.abs(a-b))),
                      rmse=float(np.sqrt(np.mean((a-b)**2))),
                      differing_values=int(np.count_nonzero(a != b)), values=int(a.size))
    return result


def inspect_first_date(result, work):
    import numpy as np
    import torch
    import yaml
    from omegaconf import OmegaConf
    from sbgm.training_utils import get_final_gen_dataloader
    from sbgm.utils import extract_samples
    from sbgm.sampling_noise import tensor_sha256
    from sbgm.generate.generation import GenerationRunner
    from sbgm.generate.generation_sigma_grid_main import _build_generation_config

    inspection = r.read_json(os.environ['INSPECTION_REPORT'])
    r.require(Path(inspection['campaign']).resolve() == r.OLD, 'Wrong inspection campaign')
    state = r.read_json(r.OLD/'campaign.json')
    result['source_matches'] = {name: r.digest(r.REPO/name) == expected
                                for name, expected in state['source_sha256'].items()}
    row = inspection['tasks'][0]
    output = Path(row['output']).resolve()
    r.require(r.OLD in output.parents, 'Saved output outside campaign')
    date = row['complete_candidates'][0]
    saved = r.read_json(output/'meta/noise'/f'{date}.json')
    result['saved_date'] = date
    result['saved_noise_record'] = saved
    result['original_provenance_environment'] = []
    for path in sorted((output/'meta').glob('*_generation_*.yaml')):
        doc = yaml.safe_load(path.read_text())
        result['original_provenance_environment'].append(
            {key: doc.get(key) for key in ('torch', 'environment', 'platform', 'data')})

    cfg = r.configuration(r.OLD/'combined/resolved_config.yaml')
    result['configuration'] = {key: OmegaConf.to_container(cfg[key], resolve=True)
                                for key in ('highres', 'lowres', 'evaluation',
                                            'stationary_conditions', 'transforms', 'data_handling')}
    result['paths'] = OmegaConf.to_container(cfg.paths, resolve=True)
    result['split'] = cfg.full_gen_eval.split
    result['max_dates'] = cfg.full_gen_eval.max_dates
    cfg.data_handling.update(split=cfg.full_gen_eval.split, shuffle=False, drop_last=False)
    print('Loading the first dataset date; no model inference.', flush=True)
    loader = get_final_gen_dataloader(cfg, split=cfg.full_gen_eval.split)
    batch = next(iter(loader))
    actual_date, actual = r.batch_record(batch)
    result['current_date'] = actual_date
    result['batch_tensors'] = {key: tensor_summary(value) for key, value in batch.items()
                               if torch.is_tensor(value)}
    result['batch_conditioning_order'] = [key for key in batch if key.endswith('_lr')]
    result['input_hashes'] = differences(saved['inputs_sha256'], actual)
    print(f'Saved date: {date}; current date: {actual_date}', flush=True)
    for key, item in result['input_hashes'].items():
        print(f"  {key}: {'MATCH' if item['matches'] else 'DIFFERENT'}", flush=True)
    if actual_date != date:
        result['note'] = 'Different first date: comparisons of physical references were skipped.'
        return

    second_date, second = r.batch_record(next(iter(loader)))
    result['repeat_load'] = dict(date=second_date, same_date=second_date == actual_date,
                                 hashes=differences(actual, second))

    # Exercise the real runner's preparation path, intercepting BEFORE sampling.
    # save=False and quicklook=False prevent provenance writes in runner.run.
    # All constructor directories are within this temporary directory.
    run = work/'runner'
    gen_cfg = _build_generation_config(cfg, run)
    runner = GenerationRunner(model=torch.nn.Identity(), cfg=cfg, device='cpu',
                              out_root=run, gen_config=gen_cfg)
    captured = {}

    class BeforeInference(Exception):
        pass

    def capture(**kwargs):
        captured.update({key: value.detach().clone() for key, value in kwargs.items()
                         if torch.is_tensor(value)})
        raise BeforeInference()

    runner._sampler_fn = capture
    try:
        runner.run([batch], save=False)
    except BeforeInference:
        pass
    else:
        raise RuntimeError('Runner did not reach the sampler interception')
    x, y, cond, lsm_hr, lsm, sdf, topo, hr_points, lr_points = extract_samples(batch, 'cpu')
    native = {key: tensor_sha256(value) for key, value in captured.items()}
    native['hr_reference'] = tensor_sha256(x[:1]) if x is not None else None
    result['checker_vs_native_runner'] = differences(actual, native)
    result['native_input_tensors'] = {key: tensor_summary(value) for key, value in captured.items()}
    equal = actual == native
    print(f'Checker agrees with native runner before sampling: {equal}', flush=True)

    # Physical references help distinguish substantial input changes from very
    # small discrepancies. Their inverse transforms are lossy, so even a close
    # match cannot substitute for the required model-space hash check.
    refs = {}
    if x is not None and callable(runner.bt_hr):
        refs['hr'] = runner.bt_hr(x[:1]).float()
    channel = 0
    for variable in runner.lr_vars:
        dual = bool(cfg.lowres.get('dual_lr', False)) and variable == runner.hr_var
        if variable == runner.hr_var and cond is not None:
            kind = 'lrspace' if str(cfg.lowres.get('lr_main_var_scale', 'LR')).upper() == 'LR' else 'hrspace'
            selected = {kind: cond[:1, channel:channel+1]}
            if dual:
                selected['lrspace'] = cond[:1, channel+1:channel+2]
                # Native code assigns the main channel last if both are LR-space.
                if kind == 'lrspace':
                    selected['lrspace'] = cond[:1, channel:channel+1]
            for space, value in selected.items():
                transform = getattr(runner, 'bt_lr_'+space)
                if callable(transform):
                    refs['lr_'+space] = transform(value).float()
        channel += 2 if dual else 1
    if 'lr_lrspace' in refs:
        refs['lr'] = refs['lr_lrspace']
    with np.load(output/'lr_hr_phys'/f'{date}.npz', allow_pickle=False) as original:
        result['physical_references'] = {key: array_comparison(original[key], value)
                                         for key, value in refs.items() if key in original}
    if lsm_hr is not None:
        mask = (lsm_hr.detach().cpu() > 0.5)
        mask = mask[0, 0] if mask.dim() == 4 else mask.squeeze()
        with np.load(output/'lsm'/f'{date}.npz', allow_pickle=False) as original:
            result['physical_land_mask'] = array_comparison(original['lsm_hr'], mask)
    print('Physical-reference comparisons:', json.dumps(result['physical_references'], indent=2), flush=True)
    result['interpretation_limit'] = ('No inference was run. Native comparison is before sampling; '
        'saved hashes were recorded after sampling. Physical reference agreement does not establish '
        'identical model-space inputs. This report does not authorize recovery.')


def main():
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    destination = Path.cwd()/f'recovery-input-diagnostic-{stamp}.json'
    r.require(r.OLD != destination.parent and r.OLD not in destination.parents,
              'Run this diagnostic outside the original campaign')
    r.require(r.ROOT != destination.parent and r.ROOT not in destination.parents,
              'Run this diagnostic outside the recovery output directory')
    result = dict(original_campaign=str(r.OLD), host=platform.node(), platform=platform.platform(),
                  python=platform.python_version(), writes_original_campaign=False, inference_run=False)
    failed = False
    with tempfile.TemporaryDirectory(prefix='ceddar-input-diagnostic-') as cache:
        r.isolate(Path(cache))
        try:
            torch = r.libraries()
            import numpy as np
            result['runtime'] = dict(torch=str(torch.__version__), numpy=np.__version__,
                threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                torch_build=torch.__config__.show(),
                environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                             'ATEN_CPU_CAPABILITY', 'MKL_CBWR')})
            if hasattr(torch.backends.cpu, 'get_cpu_capability'):
                result['runtime']['cpu_capability'] = torch.backends.cpu.get_cpu_capability()
            inspect_first_date(result, Path(cache))
        except Exception:
            failed = True
            result['diagnostic_error'] = traceback.format_exc()
            print(result['diagnostic_error'], flush=True)
        finally:
            r.save_json(destination, result)
            print(f'DIAGNOSTIC REPORT: {destination}', flush=True)
            print('Original outputs untouched. No recovery plan or inference was started.', flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
