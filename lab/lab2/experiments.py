import argparse
import csv
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT / 'lab/lab2'
LAMBDAS = [0.0, 0.0001, 0.001, 0.01]
SEEDS = [7, 11, 42, 53, 101]
RAW_SHA = 'd0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9'

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def write_csv(path, names, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        writer.writerow(names)
        writer.writerows(rows)

def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

def arrays(path):
    rows = read_csv(path)
    dim = len(rows[0]) - 1
    return np.array([[float(r[f'x{j}']) for j in range(dim)] for r in rows]), np.array([int(r['label']) for r in rows])

def save_partition(directory, train, test):
    write_csv(directory / 'train_indices.csv', ['index'], [[int(i)] for i in train])
    write_csv(directory / 'test_indices.csv', ['index'], [[int(i)] for i in test])

def partition(labels, seed):
    rng = np.random.default_rng(seed)
    train, test = [], []
    for label in [0, 1]:
        group = rng.permutation(np.flatnonzero(labels == label))
        count = int(len(group) * 0.2)
        test.extend(group[:count])
        train.extend(group[count:])
    return rng.permutation(train), rng.permutation(test)

def verify_run(run, source):
    x, y = arrays(source)
    indices = {name: np.array([int(r['index']) for r in read_csv(run / f'{name}_indices.csv')]) for name in ['train', 'test']}
    flat = np.concatenate(list(indices.values()))
    if not np.array_equal(np.sort(flat), np.arange(len(y))):
        raise RuntimeError('split does not partition source data')
    config = read_csv(run / 'config.csv')[0]
    scaling = read_csv(run / 'standardizer.csv')
    mean = np.array([float(r['mean']) for r in scaling])
    scale = np.array([float(r['scale']) for r in scaling])
    if int(config['standardize']):
        expected_scale = x[indices['train']].std(axis=0, ddof=1)
        expected_scale = np.where(expected_scale > 1e-12, expected_scale, 1.0)
        if not np.allclose(mean, x[indices['train']].mean(axis=0), atol=1e-12, rtol=1e-12) or not np.allclose(scale, expected_scale, atol=1e-12, rtol=1e-12):
            raise RuntimeError('standardizer differs from training-only statistics')
    recorded = {r['split']: r for r in read_csv(run / 'metrics.csv')}
    for name, order in indices.items():
        xx = (x[order] - mean) / scale
        yy = y[order]
        if (run / 'weights.csv').exists():
            w = np.array([float(r['weight']) for r in read_csv(run / 'weights.csv')])
            z = w[0] + xx @ w[1:]
        else:
            score = []
            for row in read_csv(run / 'nb_params.csv'):
                mu = np.array([float(row[f'mean{j}']) for j in range(xx.shape[1])])
                var = np.array([float(row[f'var{j}']) for j in range(xx.shape[1])])
                score.append(math.log(float(row['prior'])) - 0.5 * np.sum(np.log(2*np.pi*var) + (xx-mu)**2/var, axis=1))
            z = score[1] - score[0]
        pred = z >= 0
        accuracy = float(np.mean(pred == yy))
        loss = float(np.mean(np.logaddexp(0, z) - yy*z))
        row = recorded[name]
        if abs(accuracy-float(row['accuracy'])) > 1e-12 or abs(loss-float(row['log_loss'])) > 1e-9:
            raise RuntimeError(f'{name} metrics cannot be reproduced')
        observed_x, observed_y = arrays(run / f'{name}_samples.csv')
        if not np.array_equal(yy, observed_y) or not np.allclose(x[order], observed_x, atol=0, rtol=0):
            raise RuntimeError('saved samples do not match original row indices')
    return {name: {key: float(value) for key, value in row.items() if key != 'split'} for name, row in recorded.items()}

def true_bayes(data, run):
    x, y = arrays(run / 'test_samples.csv')
    centers = read_csv(data / 'centers.csv')
    directions = read_csv(data / 'directions.csv')
    cfg = read_csv(data / 'config.csv')[0]
    sigma, elong = float(cfg['sigma']), float(cfg['elong'])
    scores, theoretical = [], []
    for center, direction in zip(centers, directions):
        mu = np.array([float(center[f'x{j}']) for j in range(x.shape[1])])
        u = np.array([float(direction[f'u{j}']) for j in range(x.shape[1])])
        cov = sigma**2 * (np.eye(len(u)) + (elong**2-1)*np.outer(u,u))
        shift = x-mu
        score = -0.5*(np.einsum('ni,ij,nj->n',shift,np.linalg.inv(cov),shift) + np.linalg.slogdet(cov)[1] + len(u)*np.log(2*np.pi))
        scores.append(score)
        theoretical.append(float(cov[0,1] / np.sqrt(cov[0,0]*cov[1,1])))
    pred = scores[1] >= scores[0]
    train_x, train_y = arrays(run / 'train_samples.csv')
    empirical = [float(np.corrcoef(train_x[train_y==label].T)[0,1]) for label in [0,1]]
    return float(np.mean(pred==y)), empirical, theoretical

def fetch_banknote():
    folder = LAB / 'data/uci_banknote'
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / 'data_banknote_authentication.txt'
    url = 'https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip'
    if not raw.exists():
        with urllib.request.urlopen(url, timeout=40) as response:
            archive = response.read()
        (folder / 'banknote_authentication.zip').write_bytes(archive)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            name = next(name for name in zipped.namelist() if name.endswith('data_banknote_authentication.txt'))
            raw.write_bytes(zipped.read(name))
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if digest != RAW_SHA:
        raise RuntimeError('UCI raw file differs from verified reference hash')
    subprocess.run([sys.executable, str(LAB/'prepare_uci.py'), str(raw), str(folder/'samples.csv')], check=True)
    metadata_path = folder / 'uci_metadata.json'
    if not metadata_path.exists():
        with urllib.request.urlopen('https://archive.ics.uci.edu/api/dataset?id=267', timeout=40) as response:
            metadata_path.write_bytes(response.read())
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))['data']
    x,y = arrays(folder/'samples.csv')
    if x.shape != (1372,4) or np.bincount(y).tolist() != [762,610] or not np.isfinite(x).all():
        raise RuntimeError('UCI shape, labels, or missing-data check failed')
    provenance = {'name':metadata['name'], 'doi':metadata['dataset_doi'], 'download_url':url, 'repository_url':metadata['repository_url'], 'raw_sha256':digest, 'samples':len(y), 'features':x.shape[1], 'class_counts':np.bincount(y).tolist()}
    (folder/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
    return folder/'samples.csv', provenance

class Suite:
    def __init__(self, binaries, output):
        self.binaries = binaries
        self.output = output
        self.records = []
        self.control = []
        self.cv = []
        self.suffix = '.exe' if os.name == 'nt' else ''

    def execute(self, name, tag, arguments):
        command = [str(self.binaries/(name+self.suffix)), *map(str,arguments)]
        result = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,check=True)
        (self.output/(tag+'.txt')).write_text(' '.join(command)+'\n'+result.stdout+result.stderr,encoding='utf-8')
        path = next(line[len('CSV output: '):] for line in result.stdout.splitlines() if line.startswith('CSV output: '))
        return Path(path)

    def generate(self, tag, dim=2, samples=80, sigma=0.4, elong=1, low=-2, high=2, seed=42, covariance='class-specific'):
        return self.execute('lab2_gen',tag,['--dim',dim,'--samples',samples,'--sigma',sigma,'--elong',elong,'--min',low,'--max',high,'--seed',seed,'--covariance',covariance])

    def train(self, tag, source, optimizer='newton', decay=0.01, seed=42, epochs=80, rate=0.03, standardize=0, split=None, stage='synthetic'):
        source = Path(source)
        arguments = ['--data',source,'--seed',seed,'--standardize',standardize,'--output-root',self.output/'runs','--test-ratio',0.2]
        if split is not None: arguments += ['--split-dir',split]
        if optimizer != 'nb':
            arguments += ['--optimizer',optimizer,'--weight-decay',decay,'--epochs',epochs,'--tolerance',1e-8,'--save-every',1]
            if optimizer == 'sgd': arguments += ['--learning-rate',rate]
        run = self.execute('lab2_nb' if optimizer=='nb' else 'lab2_logistic',tag,arguments)
        metrics = verify_run(run,source)
        summary = read_csv(run/'training_summary.csv')[0]
        if summary['status'] == 'line_search_failed': raise RuntimeError(f'line search failed for {tag}')
        record = {'tag':tag,'stage':stage,'optimizer':optimizer,'lambda':decay if optimizer!='nb' else None,'seed':seed,'source':relative(source),'run':relative(run),'metrics':metrics,'training':summary}
        self.records.append(record)
        print(f'{tag}: test={metrics["test"]["accuracy"]:.4%}, loss={metrics["test"]["log_loss"]:.6g}, status={summary["status"]}',flush=True)
        return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bin-dir',type=Path,default=LAB/'build')
    parser.add_argument('--output-dir',type=Path,default=None)
    args = parser.parse_args()
    output = args.output_dir.resolve() if args.output_dir else LAB/'output'/('suite_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True,exist_ok=False)
    suite = Suite(args.bin_dir.resolve(),output)
    subprocess.run([str(suite.binaries/('lab2_verify'+suite.suffix))],check=True)
    print('1/5 synthetic optimizer comparison',flush=True)
    round_data = suite.generate('generate_round')
    round_runs = {opt:suite.train('round_'+opt,round_data/'samples.csv',opt,epochs=600 if opt=='sgd' else 150) for opt in ['sgd','cg','newton']}
    for decay in LAMBDAS:
        suite.train('round_lambda_'+str(decay),round_data/'samples.csv','newton',decay,epochs=40,stage='synthetic_regularization')
    dim3 = suite.generate('generate_3d',dim=3,elong=2,seed=11)
    suite.train('three_dimensional',dim3/'samples.csv','newton',seed=11,standardize=1,stage='3d')
    print('2/5 paired independent/correlated comparisons',flush=True)
    for covariance in ['shared','class-specific']:
        for seed in SEEDS:
            for elong in [1,5]:
                tag = f'control_{covariance}_{seed}_elong{elong}'
                data = suite.generate('generate_'+tag,samples=300,sigma=0.3,elong=elong,low=-1,high=1,seed=seed,covariance=covariance)
                lr = suite.train(tag+'_lr',data/'samples.csv','newton',0.001,seed=seed,standardize=1,stage='correlation')
                nb = suite.train(tag+'_nb',data/'samples.csv','nb',seed=seed,standardize=1,split=lr,stage='correlation')
                if (lr/'train_indices.csv').read_bytes() != (nb/'train_indices.csv').read_bytes() or (lr/'test_indices.csv').read_bytes() != (nb/'test_indices.csv').read_bytes():
                    raise RuntimeError('NB and LR do not share identical split indices')
                bayes,empirical,theoretical = true_bayes(data,lr)
                suite.control.append({'seed':seed,'elong':elong,'covariance':covariance,'lr_run':relative(lr),'nb_run':relative(nb),'data':relative(data),'bayes_accuracy':bayes,'class_correlations':empirical,'theoretical_correlations':theoretical})
    print('3/5 UCI source validation and training-only five-fold CV',flush=True)
    banknote,provenance = fetch_banknote()
    x,y = arrays(banknote)
    outer_train,outer_test = partition(y,2026)
    outer = output/'outer_split'
    save_partition(outer,outer_train,outer_test)
    training_source = output/'outer_train_samples.csv'
    write_csv(training_source,['label','x0','x1','x2','x3'],[[int(y[i]),*x[i]] for i in outer_train])
    inner_y = y[outer_train]
    rng = np.random.default_rng(2027)
    groups = [np.array_split(rng.permutation(np.flatnonzero(inner_y==label)),5) for label in [0,1]]
    for fold in range(5):
        valid = np.concatenate([g[fold] for g in groups])
        train = np.setdiff1d(np.arange(len(inner_y)),valid)
        directory = output/f'cv_fold_{fold}'
        save_partition(directory,train,valid)
        for decay in LAMBDAS:
            suite.train(f'cv_{fold}_lambda_{decay}',training_source,'newton',decay,standardize=1,split=directory,stage='cv')
    for decay in LAMBDAS:
        records = [r for r in suite.records if r['stage']=='cv' and r['lambda']==decay]
        losses = np.array([r['metrics']['test']['log_loss'] for r in records])
        sizes = np.array([r['metrics']['test']['samples'] for r in records])
        suite.cv.append({'lambda':decay,'validation_loss':float(np.average(losses,weights=sizes)),'loss_std':float(losses.std(ddof=1)),'accuracy_mean':float(np.mean([r['metrics']['test']['accuracy'] for r in records]))})
    selected = min(suite.cv,key=lambda row:row['validation_loss'])['lambda']
    (output/'cv_selection.json').write_text(json.dumps({'selected_lambda':selected,'criterion':'training-only five-fold validation mean log loss','cv':suite.cv},indent=2),encoding='utf-8')
    print(f'4/5 final UCI evaluation; selected lambda={selected}',flush=True)
    for opt in ['sgd','cg','newton','nb']:
        suite.train('uci_final_'+opt,banknote,opt,selected,seed=2026,epochs=800 if opt=='sgd' else 250,standardize=1,split=outer,stage='uci_final')
    for decay in LAMBDAS:
        suite.train('uci_lambda_'+str(decay),banknote,'newton',decay,seed=2026,standardize=1,split=outer,stage='uci_regularization')
    manifest = {'created':datetime.now().isoformat(),'platform':sys.platform,'python':sys.version,'seeds':SEEDS,'lambdas':LAMBDAS,'selected_lambda':selected,'uci':provenance,'records':suite.records,'controls':suite.control,'cv':suite.cv,'round_data':relative(round_data),'suite':relative(output)}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print('5/5 plots and verified report',flush=True)
    subprocess.run([sys.executable,str(LAB/'write_report.py'),'--manifest',str(output/'manifest.json')],cwd=ROOT,check=True)
    print('LAB2 SUITE:',output,flush=True)

if __name__ == '__main__':
    main()
