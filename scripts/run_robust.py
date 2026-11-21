"""Run B0/B1/M on generated robustness conditions and publish atomically."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from qwen_vlm import DATA_DIR, PROJECT_ROOT
from qwen_vlm.data.cv import FIXED_CV3_TEST_GROUPS
from qwen_vlm.data.io import load_jsonl, resolve_image_path, sha256_file, write_jsonl
from qwen_vlm.data.robust import (
    CONDITION_NAMES as CONDITIONS,
    pair_group_id_for_condition as pair_id,
    sample_id_for_condition as sample_id,
)
from qwen_vlm.training.config import load_train_config
MODELS = ('B0', 'B1', 'M')
N, IMAGES, STEPS, MAX_PIXELS, MIN_PIXELS = (100, 50, 50, 512 * 512, 256 * 256)
MUTABLE = {'sample_id', 'image_path', 'pair_group_id', 'condition_tags'}
BASE_MANIFEST = PROJECT_ROOT / 'artifacts/base/manifest.json'

def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path

def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(f'{path} must be inside {root} for portable provenance') from exc

def _need(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)

def _fingerprint(payload: object) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    os.replace(temporary, path)

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    write_jsonl(temporary, rows)
    os.replace(temporary, path)

def _index(rows: list[dict], label: str) -> dict[str, dict]:
    ids = [row.get('sample_id') for row in rows]
    _need(all((isinstance(key, str) and key for key in ids)) and len(ids) == len(set(ids)), f'{label}: missing or duplicate sample_id')
    return {key: row for key, row in zip(ids, rows)}

def _expect_ids(path: Path, expected: set[str]) -> None:
    actual = set(_index(load_jsonl(path), str(path)))
    _need(actual == expected, f'{path}: expected {len(expected)} IDs, got {len(actual)}; missing={sorted(expected - actual)[:5]} extra={sorted(actual - expected)[:5]}')

def _run(script: str, *args: object) -> None:
    command = [sys.executable, '-u', str(PROJECT_ROOT / 'scripts' / script), *map(str, args)]
    print('[run]', ' '.join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)

def _load_gold(path: Path) -> dict[str, dict]:
    gold = _index(load_jsonl(path), str(path))
    scenes = {row['scene_group_id'] for row in gold.values()}
    fixed = {scene for values in FIXED_CV3_TEST_GROUPS.values() for scene in values}
    _need(len(gold) == N and scenes == fixed, f'{path}: expected {N} rows across the fixed CV3 scenes')
    return gold

def _load_conditions(root: Path, data_dir: Path, gold_path: Path, gold: dict[str, dict], names: list[str]) -> dict:
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    _need(manifest.get('version') == 'robust-v1' and manifest.get('training_eligible') is False, f'{manifest_path}: not an evaluation-only robust-v1 manifest')
    _need(isinstance(manifest.get('seed'), int), f'{manifest_path}: missing integer seed')
    _need(manifest.get('source_jsonl_sha256') == sha256_file(gold_path), f'{manifest_path}: stale source_jsonl_sha256')
    entries = manifest.get('conditions')
    _need(isinstance(entries, dict) and set(entries) == set(CONDITIONS), f'{manifest_path}: conditions differ from generator convention')
    loaded = {}
    for name in names:
        entry = entries[name]
        required = ('kind', 'parameters', 'id_suffix', 'samples_path', 'samples_sha256', 'sample_count', 'image_count', 'images')
        _need(isinstance(entry, dict) and all((key in entry for key in required)), f'{manifest_path}: missing {name} metadata')
        suffix = entry['id_suffix']
        _need(isinstance(suffix, str), f'{manifest_path}: invalid {name} id_suffix')
        sample_path = resolve_image_path(entry['samples_path'], data_dir)
        _need(sample_path == root / name / 'samples.jsonl', f'{manifest_path}: {name} samples_path is not canonical')
        _need(sha256_file(sample_path) == entry.get('samples_sha256'), f'{sample_path}: sample hash differs from manifest')
        rows = load_jsonl(sample_path)
        by_id = _index(rows, str(sample_path))
        expected = {sample_id(base, suffix): base for base in gold}
        _need((len(rows), entry.get('sample_count')) == (N, N) and set(by_id) == set(expected), f'{sample_path}: condition does not cover gold exactly once')
        images = entry.get('images')
        _need(isinstance(images, list) and entry.get('image_count') == IMAGES, f'{manifest_path}: {name} has invalid images/image_count')
        declared = {item['image_path']: item['sha256'] for item in images}
        _need({row['image_path'] for row in rows} == set(declared), f'{sample_path}: image set differs from manifest')
        stale = [path for path, digest in declared.items() if sha256_file(resolve_image_path(path, data_dir)) != digest]
        _need(not stale, f'{name}: stale image hashes: {stale[:5]}')
        base_to_robust = {base: robust for robust, base in expected.items()}
        for robust, base in expected.items():
            row, original = (by_id[robust], gold[base])
            changed = [field for field, value in original.items() if field not in MUTABLE and row.get(field) != value]
            _need(not changed, f'{robust}: changed labels/group fields {changed}')
            tags = list(original.get('condition_tags') or [])
            if f'robust_{name}' not in tags:
                tags.append(f'robust_{name}')
            _need(row.get('condition_tags') == tags, f'{robust}: invalid condition_tags')
            _need(row.get('pair_group_id') == pair_id(original.get('pair_group_id'), suffix), f'{robust}: invalid pair_group_id')
        loaded[name] = {
            'path': sample_path, 'rows': rows, 'by_id': by_id,
            'base_to_robust': base_to_robust, 'suffix': suffix,
            'samples_path': entry['samples_path'], 'sample_sha256': entry['samples_sha256'],
            'kind': entry['kind'], 'parameters': entry['parameters'],
            'images': declared, 'seed': manifest['seed'],
        }
    return loaded

def _fold_segments(model: str, condition: dict, gold: dict, gold_path: Path, splits: Path, data_dir: Path, model_dir: Path) -> list[dict]:
    config_path = PROJECT_ROOT / 'configs/train' / f'{model}.yaml'
    config = load_train_config(config_path)
    config['optim']['max_steps'] = STEPS
    _need(_resolve(config['model_dir']).resolve() == model_dir.resolve(), f'{config_path}: model_dir differs from --model-dir')
    revision = json.loads(BASE_MANIFEST.read_text())['revision']
    train_name = 'train_real.jsonl' if model == 'B1' else 'train.jsonl'
    segments = []
    for fold, scenes in FIXED_CV3_TEST_GROUPS.items():
        folder = splits / f'fold{fold}'
        split_path = folder / 'manifest.json'
        split = json.loads(split_path.read_text())
        train_path, test_path = (folder / train_name, folder / 'test.jsonl')
        _need(split.get('source_sha256') == sha256_file(gold_path), f'{split_path}: stale source_sha256')
        _need(tuple(split.get('test_scene_groups') or ()) == scenes, f'{split_path}: test groups differ from FIXED_CV3_TEST_GROUPS')
        for path, key in ((train_path, Path(train_name).stem), (test_path, 'test')):
            _need(sha256_file(path) == split['files'][key]['sha256'], f'{path}: hash differs from split manifest')
        train_rows = load_jsonl(train_path)
        test_rows = _index(load_jsonl(test_path), str(test_path))
        expected_test = {sid for sid, row in gold.items() if row['scene_group_id'] in scenes}
        _need(set(test_rows) == expected_test and all((test_rows[sid] == gold[sid] for sid in expected_test)), f'{test_path}: test rows differ from real gold')
        _need({row['scene_group_id'] for row in train_rows}.isdisjoint(scenes), f'{train_path}: train/test scene groups overlap')
        run_dir = PROJECT_ROOT / 'runs/cv3_folds' / f'{model}_fold{fold}'
        run_path, adapter = (run_dir / 'run_manifest.json', run_dir / 'adapter')
        run = json.loads(run_path.read_text())
        _need(run.get('experiment_id') == model and run.get('config') == config, f'{run_path}: config differs from {config_path}')
        _need(run.get('steps_completed') == STEPS and run.get('max_steps') == STEPS and (not run.get('error')), f'{run_path}: training did not complete cleanly')
        _need(run.get('base_model_revision') == revision, f'{run_path}: base model revision is not pinned')
        _need(_resolve(run['base_model_dir']).resolve() == model_dir.resolve(), f'{run_path}: adapter used a different base model')
        data = run.get('data') or {}
        _need(data.get('val_jsonl') is None, f'{run_path}: validation set was not disabled')
        _need(data.get('train_jsonl_sha256') == sha256_file(train_path), f'{run_path}: train JSONL hash is stale')
        _need(data.get('split_manifest_sha256') == sha256_file(split_path), f'{run_path}: split manifest hash is stale')
        adapter_files = ('adapter_config.json', 'adapter_model.safetensors')
        _need(all(((adapter / filename).is_file() for filename in adapter_files)), f'{adapter}: missing adapter files')
        rows = [row for row in condition['rows'] if row['scene_group_id'] in scenes]
        test_ids = set(test_rows)
        expected = {condition['base_to_robust'][sid] for sid in test_ids}
        _need({row['sample_id'] for row in rows} == expected, f'fold{fold}: robust rows differ from held-out test group')
        segments.append({
            'name': f'fold{fold}', 'rows': rows, 'expected': expected,
            'adapter': adapter, 'fold': fold, 'test_ids': sorted(test_ids),
            'split_manifest': _relative(split_path, data_dir),
            'adapter_rel': _relative(adapter, PROJECT_ROOT),
            'split_sha256': sha256_file(split_path),
            'run_manifest_sha256': sha256_file(run_path),
            'adapter_files': {
                name: sha256_file(adapter / name) for name in adapter_files
            },
        })
    return segments

def _predict(run_id: str, segment: dict, work: Path, model_dir: Path, data_dir: Path) -> None:
    output = work / f"{segment['name']}.predictions.jsonl"
    if output.exists():
        _expect_ids(output, segment['expected'])
        return
    samples, temporary = (work / f"{segment['name']}.jsonl", work / f"{segment['name']}.next.jsonl")
    _write_jsonl(samples, segment['rows'])
    args = [
        '--jsonl', samples, '--experiment-id', f"{run_id}_{segment['name']}",
        '--model-dir', model_dir, '--data-root', data_dir, '--output', temporary,
        '--max-pixels', MAX_PIXELS, '--min-pixels', MIN_PIXELS,
    ]
    if segment.get('adapter'):
        args += ['--adapter', segment['adapter']]
    _run('predict_transformers.py', *args)
    _expect_ids(temporary, segment['expected'])
    os.replace(temporary, output)

def _merge(condition: dict, segments: list[dict], work: Path) -> Path:
    merged = {}
    for segment in segments:
        for row in load_jsonl(work / f"{segment['name']}.predictions.jsonl"):
            _need(row['sample_id'] not in merged, f"duplicate merged prediction: {row['sample_id']}")
            merged[row['sample_id']] = row
    _need(set(merged) == set(condition['by_id']), 'merged predictions do not cover the condition exactly once')
    output = work / 'predictions.jsonl'
    _write_jsonl(output, [merged[row['sample_id']] for row in condition['rows']])
    _expect_ids(output, set(condition['by_id']))
    return output

def _evaluate(run_id: str, condition: dict, predictions: Path, work: Path) -> Path:
    output = work / 'eval'
    metrics = output / 'metrics.json'
    if metrics.is_file():
        return output
    temporary = Path(tempfile.mkdtemp(prefix='eval.', dir=work))
    _run('eval_predictions.py', '--gold', condition['path'], '--pred', predictions, '--out', temporary, '--experiment-id', run_id, '--pixel-budget', '512')
    result = json.loads((temporary / 'metrics.json').read_text())
    if result.get('experiment_id') != run_id or result.get('n_records') != N:
        raise RuntimeError(f'{temporary}: evaluator produced unexpected metrics')
    os.replace(temporary, output)
    return output

def _reusable(target: Path, fingerprint: str, expected: set[str]) -> bool:
    if not target.exists():
        return False
    provenance = target / 'provenance.json'
    _need(provenance.is_file(), f'{target}: existing output has no provenance')
    _need(json.loads(provenance.read_text()).get('input_fingerprint') == fingerprint, f'{target}: existing output used different inputs')
    _expect_ids(target / 'predictions.jsonl', expected)
    _need((target / 'eval/metrics.json').is_file(), f'{target}: existing output has no metrics')
    return True

def _publish(target: Path, work: Path, profile: dict, fingerprint: str) -> None:
    staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
    try:
        shutil.copy2(work / 'predictions.jsonl', staging / 'predictions.jsonl')
        shutil.copytree(work / 'eval', staging / 'eval')
        _write_json(staging / 'provenance.json', {**profile, 'input_fingerprint': fingerprint, 'completed_at': datetime.now(timezone.utc).isoformat(), 'predictions': N})
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

def _run_one(model: str, name: str, condition: dict, gold: dict, gold_path: Path,
             splits: Path, data_dir: Path, model_dir: Path, robust_dir: Path) -> None:
    segments = (
        [{'name': 'all', 'rows': condition['rows'],
          'expected': set(condition['by_id']), 'adapter': None}]
        if model == 'B0'
        else _fold_segments(model, condition, gold, gold_path, splits, data_dir, model_dir)
    )
    inputs = {
        'model': model, 'condition_name': name,
        'gold_path': _relative(gold_path, data_dir),
        'gold_sha256': sha256_file(gold_path),
        'condition': {
            'kind': condition['kind'], 'parameters': condition['parameters'],
            'id_suffix': condition['suffix'], 'samples_path': condition['samples_path'],
            'samples_sha256': condition['sample_sha256'],
            'images': condition['images'], 'seed': condition['seed'],
        },
        'model_dir': _relative(model_dir, PROJECT_ROOT),
        'runner_sha256': sha256_file(Path(__file__)),
        'base_manifest_sha256': sha256_file(BASE_MANIFEST),
        'base_revision': json.loads(BASE_MANIFEST.read_text())['revision'],
        'settings': {
            'max_pixels': MAX_PIXELS, 'min_pixels': MIN_PIXELS, 'train_steps': STEPS,
        },
        'folds': [
            {key: value for key, value in segment.items()
             if key not in {'rows', 'expected', 'adapter'}}
            for segment in segments
        ],
    }
    fingerprint, run_id = (_fingerprint(inputs), f'{model}_{name}')
    target, expected = (robust_dir / run_id, set(condition['by_id']))
    if _reusable(target, fingerprint, expected):
        print(f'[reuse] {run_id}', flush=True)
        return
    work = robust_dir / '.work' / run_id
    work.mkdir(parents=True, exist_ok=True)
    record = work / 'provenance.json'
    if record.exists():
        _need(json.loads(record.read_text()).get('input_fingerprint') == fingerprint, f'{work}: stale work directory; use a new --runs-root')
    else:
        _write_json(record, {'input_fingerprint': fingerprint, 'inputs': inputs})
    for segment in segments:
        print(f"[{run_id}] {segment['name']}: {len(segment['expected'])} samples", flush=True)
        _predict(run_id, segment, work, model_dir, data_dir)
    predictions = _merge(condition, segments, work)
    _evaluate(run_id, condition, predictions, work)
    _publish(target, work, {'run_id': run_id, 'inputs': inputs}, fingerprint)
    print(f'[published] {target}', flush=True)

@contextmanager
def _lock(path: Path):
    with path.open('a+') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'another robustness runner holds {path}') from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, type=str.upper, default=list(MODELS))
    parser.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR)
    parser.add_argument('--gold', type=Path)
    parser.add_argument('--robust-root', type=Path)
    parser.add_argument('--splits-root', type=Path)
    parser.add_argument('--runs-root', type=Path, default=PROJECT_ROOT / 'runs')
    parser.add_argument('--model-dir', type=Path, default=PROJECT_ROOT / 'artifacts/base/Qwen3-VL-2B-Instruct')
    args = parser.parse_args()
    data_dir = _resolve(args.data_dir).resolve()
    gold_path = (_resolve(args.gold) if args.gold else data_dir / 'annotations/real_v1.jsonl').resolve()
    root = (_resolve(args.robust_root) if args.robust_root else data_dir / 'robust/v1').resolve()
    splits = (_resolve(args.splits_root) if args.splits_root else data_dir / 'splits/cv3').resolve()
    robust_dir = _resolve(args.runs_root) / 'robust'
    model_dir = _resolve(args.model_dir).resolve()
    names = [name for name in CONDITIONS if name in set(args.conditions)]
    models = [model for model in MODELS if model in set(args.models)]
    if not names or not models:
        parser.error('select at least one model and condition')
    robust_dir.mkdir(parents=True, exist_ok=True)
    with _lock(robust_dir / '.lock'):
        gold = _load_gold(gold_path)
        conditions = _load_conditions(root, data_dir, gold_path, gold, names)
        for model in models:
            for name in names:
                _run_one(model, name, conditions[name], gold, gold_path, splits, data_dir, model_dir, robust_dir)
if __name__ == '__main__':
    main()
