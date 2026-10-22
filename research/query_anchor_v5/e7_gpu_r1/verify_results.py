from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile

import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import torch


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--run-root', type=Path, default=ROOT / 'results' / 'kaggle-v1')
parser.add_argument('--local-source', type=Path, default=ROOT)
parser.add_argument('--receipt', type=Path, default=ROOT / 'RETAINED_EVIDENCE_VERIFICATION.json')
args = parser.parse_args()
RUN_ROOT = args.run_root.resolve()
OUTPUT = RUN_ROOT / 'epu-e7-writer-disjoint-gpu-r1'
ARCHIVE = RUN_ROOT / 'epu-e7-writer-disjoint-gpu-r1-evidence.tar.gz'
LOG = RUN_ROOT / 'epu-e7-writer-disjoint-gpu-r1-20261021.log'
LOCAL = args.local_source.resolve()
DOWNLOADED_SOURCE = RUN_ROOT / 'epu-e7-source' / 'epu_e7'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, atol=1e-12):
    assert np.allclose(a, b, rtol=0.0, atol=atol), (a, b)


spec = importlib.util.spec_from_file_location('e7_study', LOCAL / 'study.py')
study = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(study)

manifest = json.loads((OUTPUT / 'MANIFEST.json').read_text())
protocol = json.loads((OUTPUT / 'PROTOCOL.json').read_text())
hypotheses = json.loads((OUTPUT / 'HYPOTHESES.json').read_text())
dataset = json.loads((OUTPUT / 'DATASET.json').read_text())
environment = json.loads((OUTPUT / 'ENVIRONMENT.json').read_text())
self_test = json.loads((OUTPUT / 'SELF_TEST.json').read_text())

assert manifest['status'] == 'COMPLETE'
assert manifest['version'] == protocol['version'] == hypotheses['version'] == 'EPU-REALDATA-E7-WRITER-DISJOINT-GPU-R1'
assert self_test['status'] == 'PASS_SELF_TEST'
assert self_test['protocol_sha256'] == manifest['source']['PROTOCOL.json']
assert manifest['records'] == 20 * 1797 * 3 == 107820
assert manifest['gpu_prediction_batches'] == hypotheses['gpu_prediction_batches'] == 260
assert manifest['inference_group_count'] == hypotheses['inference_group_count'] == 1797

for name, expected in manifest['files'].items():
    assert sha(OUTPUT / name) == expected, name
for name in ('PROTOCOL.json', 'study.py', 'README.md', 'SOURCE_RECEIPT.json',
             'optdigits.names', 'optdigits.tra', 'optdigits.tes'):
    assert (DOWNLOADED_SOURCE / name).read_bytes() == (LOCAL / name).read_bytes(), name
source_receipt = json.loads((LOCAL / 'SOURCE_RECEIPT.json').read_text())
for name, expected in source_receipt['files'].items():
    assert sha(LOCAL / name) == expected, name
assert source_receipt['archive_sha256'] == protocol['dataset']['archive_sha256']
assert sha(LOCAL / 'PROTOCOL.json') == manifest['source']['PROTOCOL.json']
assert sha(LOCAL / 'study.py') == manifest['source']['study.py']
assert json.loads((LOCAL / 'PROTOCOL.json').read_text()) == protocol

with tarfile.open(ARCHIVE, 'r:gz') as archive:
    members = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
for path in OUTPUT.iterdir():
    if path.is_file():
        expected = path.read_bytes()
        candidates = [v for k, v in members.items() if k.endswith('/' + path.name)]
        assert candidates == [expected], path.name

log_text = LOG.read_text(errors='replace')
assert 'PASS_EXACT_SOURCE' in log_text
assert 'PASS_SELF_TEST' in log_text
assert 'COMPLETE' in log_text
assert 'Traceback (most recent call last)' not in log_text

hardware = environment['hardware']
assert hardware['device'] == 'cuda:0'
assert hardware['total_memory_bytes'] >= 8 * 1024**3
assert hardware['name'] == 'Tesla T4'
assert hardware['benchmark']['estimated_tflops'] > 0

train_file = LOCAL / 'optdigits.tra'
test_file = LOCAL / 'optdigits.tes'
train_raw = np.loadtxt(train_file, delimiter=',', dtype=np.int64)
test_raw = np.loadtxt(test_file, delimiter=',', dtype=np.int64)
X_train = np.asarray(train_raw[:, :64], dtype=np.float64) / 16.0
y_train = np.asarray(train_raw[:, 64], dtype=np.int64)
X_test = np.asarray(test_raw[:, :64], dtype=np.float64) / 16.0
y_test = np.asarray(test_raw[:, 64], dtype=np.int64)
assert X_train.shape == (3823, 64) and X_test.shape == (1797, 64)
data_digest = hashlib.sha256(train_file.read_bytes() + test_file.read_bytes()).hexdigest()
assert data_digest == dataset['canonical_files_sha256']
assert sha(train_file) == dataset['train_file_sha256']
assert sha(test_file) == dataset['test_file_sha256']
groups = study.duplicate_groups(test_raw[:, :64])
assert len(np.unique(groups)) == dataset['test_duplicate_groups'] == 1797

repeats = protocol['repeats']
noises = protocol['noise_standard_deviations']
methods = protocol['methods']
correct = {(noise, method): np.full((repeats, len(y_test)), np.nan) for noise in noises for method in methods}
displacement = {(noise, method): np.full((repeats, len(y_test)), np.nan)
                for noise in noises for method in ('direct_multi', 'original_topk', 'anchor_topk')}
seen = set()
raw_records = 0
raw_groups = {}
with gzip.open(OUTPUT / 'RAW.jsonl.gz', 'rt', encoding='utf-8') as handle:
    for line in handle:
        row = json.loads(line)
        raw_records += 1
        key = (row['repeat'], row['row'], float(row['noise']))
        assert key not in seen
        seen.add(key)
        repeat, index, noise = row['repeat'], row['row'], float(row['noise'])
        fold = 0
        assert row['partition'] == 'official_writer_independent_test'
        assert row['label'] == int(y_test[index])
        assert set(row['predictions']) == set(methods)
        assert set(row['displacement']) == {'direct_multi', 'original_topk', 'anchor_topk'}
        for method in methods:
            correct[(noise, method)][repeat, index] = float(row['predictions'][method] == row['label'])
        for method, value in row['displacement'].items():
            displacement[(noise, method)][repeat, index] = float(value)
        raw_groups.setdefault((repeat, fold, noise), []).append(row)

assert raw_records == manifest['records']
assert len(seen) == raw_records
for values in list(correct.values()) + list(displacement.values()):
    assert np.all(np.isfinite(values))

fold_rows = list(csv.DictReader((OUTPUT / 'FOLD_METRICS.csv').open()))
assert len(fold_rows) == repeats * len(noises) * len(methods) == 300
indexed_metrics = {(int(r['repeat']), int(r['fold']), float(r['noise']), r['method']): r for r in fold_rows}
assert len(indexed_metrics) == len(fold_rows)
for (repeat, fold, noise), records in raw_groups.items():
    truth = np.asarray([r['label'] for r in records])
    for method in methods:
        pred = np.asarray([r['predictions'][method] for r in records])
        item = indexed_metrics[(repeat, fold, noise, method)]
        assert int(item['n']) == len(records)
        close(float(item['accuracy']), accuracy_score(truth, pred))
        close(float(item['balanced_accuracy']), balanced_accuracy_score(truth, pred))
        close(float(item['macro_f1']), f1_score(truth, pred, average='macro'))
        assert float(item['elapsed_seconds']) >= 0
        if method in ('direct_multi', 'original_topk', 'anchor_topk'):
            observed = np.mean([r['displacement'][method] for r in records])
            close(float(item['mean_displacement']), observed)
        else:
            assert item['mean_displacement'] == ''

unique_groups = np.unique(groups)
def group_contrast(left, right, selected_noises, use_displacement=False):
    source = displacement if use_displacement else correct
    values = []
    for group in unique_groups:
        rows = np.flatnonzero(groups == group)
        lv = np.concatenate([source[(noise, left)][:, rows].reshape(-1) for noise in selected_noises])
        rv = np.concatenate([source[(noise, right)][:, rows].reshape(-1) for noise in selected_noises])
        values.append(float(lv.mean() - rv.mean()))
    return values

effects = {
    'H-E7-UTILITY': group_contrast('anchor_topk', 'direct_multi', noises),
    'H-E7-ROBUSTNESS': group_contrast('anchor_topk', 'direct_multi', [0.3]),
    'H-E7-NONINFERIOR-LOGREG': group_contrast('anchor_topk', 'logreg', noises),
    'H-E7-STABILITY': group_contrast('original_topk', 'anchor_topk', noises, True),
}
accuracy_keys = ('H-E7-UTILITY', 'H-E7-ROBUSTNESS', 'H-E7-NONINFERIOR-LOGREG')
raw_p = {key: study.permutation_p(effects[key], 100000, 7000 + i,
                                  null=-0.01 if key == 'H-E7-NONINFERIOR-LOGREG' else 0.0)
         for i, key in enumerate(accuracy_keys)}
holm = study.holm_adjust(raw_p)
reported = {item['id']: item for item in hypotheses['outcomes']}
decisions = {}
for index, hypothesis in enumerate(protocol['hypotheses']):
    key = hypothesis['id']
    item = reported[key]
    effect = float(np.mean(effects[key]))
    interval = study.bootstrap_ci(effects[key], 10000, 9000 + index)
    close(item['group_effects'], effects[key])
    close(item['effect'], effect)
    close(item['bootstrap_95'], interval)
    if key in accuracy_keys:
        close(item['raw_permutation_p'], raw_p[key])
        close(item['holm_p'], holm[key])
    if key in ('H-E7-UTILITY', 'H-E7-ROBUSTNESS'):
        passed = effect >= hypothesis['minimum_effect'] and interval[0] > 0 and holm[key] < 0.05
    elif key == 'H-E7-NONINFERIOR-LOGREG':
        passed = effect >= hypothesis['margin'] and interval[0] > hypothesis['margin'] and holm[key] < 0.05
    else:
        passed = effect >= hypothesis['minimum_effect'] and interval[0] > hypothesis['minimum_effect']
    decision = 'PASS' if passed else 'FAIL'
    assert item['status'] == decision
    decisions[key] = {
        'status': decision,
        'effect': effect,
        'bootstrap_95': list(interval),
        'raw_permutation_p': raw_p.get(key),
        'holm_p': holm.get(key),
    }

control = hypotheses['negative_control']
close(control['mean_accuracy'], float(np.mean(control['accuracies'])))
assert control['status'] == ('PASS' if control['mean_accuracy'] <= control['threshold'] else 'FAIL')
control_replay = []
for permutation in range(20):
    shuffled = y_train.copy()
    np.random.default_rng(50000 + permutation).shuffle(shuffled)
    p, pl = study.fit_prototypes(X_train, shuffled)
    pred, _, _ = study.prototype_predict(X_test, p, pl, 'anchor_topk', torch.device('cpu'), protocol['model'])
    control_replay.append(float(accuracy_score(y_test, pred)))
control_cpu_exact = bool(np.allclose(control_replay, control['accuracies'], rtol=0.0, atol=1e-12))
control_cpu_mean = float(np.mean(control_replay))
control_cpu_status = 'PASS' if control_cpu_mean <= control['threshold'] else 'FAIL'

aggregate_accuracy = {
    str(noise): {method: float(correct[(noise, method)].mean()) for method in methods}
    for noise in noises
}
verification = {
    'status': ('PASS_E7_R1_RETAINED_EVIDENCE' if control_cpu_exact else
               'VERIFIED_E7_R1_RETAINED_EVIDENCE_WITH_CROSS_RUNTIME_CONTROL_MISMATCH'),
    'version': manifest['version'],
    'kaggle_url': 'https://www.kaggle.com/code/ryangomez010/epu-e7-writer-disjoint-gpu-r1-20261021',
    'archive_sha256': sha(ARCHIVE),
    'manifest_sha256': sha(OUTPUT / 'MANIFEST.json'),
    'raw_sha256': sha(OUTPUT / 'RAW.jsonl.gz'),
    'raw_records': raw_records,
    'fold_metric_rows': len(fold_rows),
    'source': manifest['source'],
    'hardware': hardware,
    'aggregate_accuracy': aggregate_accuracy,
    'outcomes': decisions,
    'negative_control': {
        'gpu_status': control['status'],
        'gpu_mean_accuracy': control['mean_accuracy'],
        'threshold': control['threshold'],
        'cpu_status': control_cpu_status,
        'cpu_mean_accuracy': control_cpu_mean,
        'cpu_replay_exact': control_cpu_exact,
        'maximum_absolute_accuracy_difference': float(np.max(np.abs(np.asarray(control_replay) - np.asarray(control['accuracies'])))),
    },
    'failures': ([] if control_cpu_exact else [
        'Shuffled-label control does not reproduce exactly on CPU; GPU and CPU decisions are recorded separately. Exact independent GPU reproduction remains required.'
    ]),
    'checks': [
        'manifest file hashes', 'archive-directory byte equality', 'downloaded-local source identity',
        'GPU admission', 'dataset digest', 'raw uniqueness and completeness', 'fold metric replay',
        'group effects', 'bootstrap intervals', 'permutation p-values', 'Holm adjustment',
        'hypothesis decisions', 'negative-control retained arithmetic', 'log completion',
    ],
    'claim_boundary': manifest['claim_boundary'],
}
args.receipt.write_text(
    json.dumps(verification, indent=2, sort_keys=True, allow_nan=False) + '\n'
)
print(json.dumps(verification, indent=2, sort_keys=True, allow_nan=False))
