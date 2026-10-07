"""Small-sample meter forecasting from raw Excel, with train-only model selection.

Forecast rounds 4--6 using rounds 1--3. Dates are retained as labels, not
converted to unverified accumulated stress hours. No synthetic observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from openpyxl import load_workbook
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def read_excel(path):
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book['基本误差']
    rows = list(sheet.iter_rows(max_col=32, values_only=True))
    headers = [i for i, row in enumerate(rows) if row[1] == '表号']
    assert len(headers) == 6
    rounds, labels = [], []
    channels = [str(rows[headers[0] + j][1]) for j in range(1, 11)]
    for h in headers:
        labels.append(str(rows[h][0]))
        assert [str(rows[h + j][1]) for j in range(1, 11)] == channels
        ids = [str(rows[h][j]).strip() for j in range(2, 32)]
        assert len(set(ids)) == 30 and 'None' not in ids
        rounds.append({meter: np.array([float(rows[h + k][j + 2])
                                       for k in range(1, 11)]) for j, meter in enumerate(ids)})
    ids = sorted(set.intersection(*(set(r) for r in rounds)))
    y = np.stack([[r[meter] for r in rounds] for meter in ids])
    assert np.isfinite(y).all()
    book.close()
    return np.array(ids), y, labels, channels


def features(prefix):
    """Only historical observations; no meter ID or future summary features."""
    if prefix.ndim != 3 or prefix.shape[1:] != (3, 10):
        raise ValueError('Expected exactly three observed rounds and ten channels')
    return np.concatenate([prefix[:, 2], prefix[:, 1] - prefix[:, 0],
                           prefix[:, 2] - prefix[:, 1]], axis=1)


def basis():
    common = np.ones(10)
    pf = np.tile([1., -1.], 5)
    current = np.repeat([-2., -1., 0., 1., 2.], 2)
    return np.linalg.qr(np.stack([common, pf, current], axis=1))[0].astype('float32')


class MeterResidualNet(nn.Module):
    """Anchor + train-population increment + nonlinear individual correction.

    Three structured channel factors and a penalized free-channel branch avoid
    forcing signed measurement errors to be monotone or into [0, 1].
    """
    def __init__(self, width=16, structured=True):
        super().__init__()
        self.structured = structured
        self.encoder = nn.Sequential(nn.Linear(30, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.head = nn.Linear(width, 9 if structured else 30)
        self.free = nn.Linear(width, 30) if structured else None
        self.register_buffer('channel_basis', torch.tensor(basis()))
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        if self.free is not None:
            nn.init.zeros_(self.free.weight)
            nn.init.zeros_(self.free.bias)

    def forward(self, x):
        z = self.encoder(x)
        if not self.structured:
            return self.head(z)
        structured = self.head(z).reshape(-1, 3, 3) @ self.channel_basis.T
        return structured.flatten(1) + 0.1 * self.free(z)


def neural_fit(x, y, config, seed, validation=None, epochs=500):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), np.maximum(x.std(0), 0.02)
    # Offset is estimated from this training fold only.
    offset = np.median(y, axis=0)
    def tx(a):
        return torch.tensor(np.clip((a - mean) / scale, -6, 6), dtype=torch.float32)
    xx = tx(x)
    yy = torch.tensor((y - offset) / 0.1, dtype=torch.float32)
    model = MeterResidualNet(config['width'], config['structured'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.03)
    best, stale, best_epoch, state, history = float('inf'), 0, 0, None, []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(xx)
        loss = nn.functional.smooth_l1_loss(pred, yy, beta=0.3)
        loss = loss + config['shrinkage'] * pred.square().mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mae = float((model(xx) - yy).abs().mean() * 0.1)
            score = train_mae
            if validation is not None:
                vx, vy = validation
                vp = model(tx(vx)).numpy() * 0.1 + offset
                score = float(np.abs(vp - vy).mean())
        history.append({'epoch': epoch, 'train_mae': train_mae,
                        'validation_mae': score if validation is not None else None})
        if validation is not None:
            if score < best - 1e-7:
                best, best_epoch, stale = score, epoch, 0
                state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= 60:
                break
    if state is not None:
        model.load_state_dict(state)
    else:
        best_epoch = epochs
    return model, mean, scale, offset, best_epoch, history


def neural_predict(fit, x):
    model, mean, scale, offset = fit[:4]
    with torch.no_grad():
        z = torch.tensor(np.clip((x - mean) / scale, -6, 6), dtype=torch.float32)
        return model(z).numpy() * 0.1 + offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/meter_neural_v1')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    source = next((ROOT / 'data/raw').glob('单相表*.xlsx'))
    ids, full, dates, channels = read_excel(source)
    order = np.random.default_rng(20260927).permutation(len(ids))
    nt = max(6, round(0.3 * len(ids)))
    test, train = order[:nt], order[nt:]
    assert set(ids[test]).isdisjoint(ids[train])
    x = features(full[:, :3])
    anchor = np.tile(full[:, 2], (1, 3))
    truth = full[:, 3:].reshape(len(ids), 30)
    delta = truth - anchor
    folds = list(KFold(4, shuffle=True, random_state=1928).split(train))
    configs = [dict(width=w, structured=s, shrinkage=r)
               for w, s, r in [(16, True, .1), (16, True, 1.),
                                (32, True, .1), (16, False, .1)]]
    selection = []
    for ci, config in enumerate(configs):
        scores, epochs = [], []
        for fi, (a, b) in enumerate(folds):
            tr, va = train[a], train[b]
            fit = neural_fit(x[tr], delta[tr], config, 170 + fi,
                             (x[va], delta[va]))
            scores.append(float(np.abs(neural_predict(fit, x[va]) - delta[va]).mean()))
            epochs.append(fit[4])
        selection.append(dict(config=config, mae=float(np.mean(scores)), folds=scores, epochs=epochs))
        print('neural CV', ci, selection[-1], flush=True)
    chosen = min(selection, key=lambda row: row['mae'])
    # Freeze settings and refit full development pool, no outer-test early stopping.
    budget = max(1, int(np.median(chosen['epochs'])))
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    predictions = {'persistence': anchor[test],
                   'population_increment': anchor[test] + np.median(delta[train], axis=0)}
    neural_preds, histories = [], {}
    for seed in (17, 42, 73, 101, 137):
        fit = neural_fit(x[train], delta[train], chosen['config'], seed, epochs=budget)
        neural_preds.append(anchor[test] + neural_predict(fit, x[test]))
        histories[str(seed)] = fit[5]
        torch.save({'state_dict': fit[0].state_dict(), 'mean': fit[1], 'scale': fit[2],
                    'offset': fit[3], 'config': chosen['config'], 'epochs': budget,
                    'train_ids': ids[train].tolist(), 'channels': channels},
                   args.output / f'meter_residual_seed{seed}.pt')
    predictions['meter_residual_ensemble'] = np.mean(neural_preds, axis=0)
    for seed, pred in zip((17, 42, 73, 101, 137), neural_preds):
        predictions[f'meter_residual_seed{seed}'] = pred
    # Fair random-forest and ridge comparisons use exactly the same histories,
    # residual targets, inner folds and outer holdout.
    baseline_selection = {}
    for name in ('random_forest', 'ridge'):
        options = (1, 2, 4, 6) if name == 'random_forest' else (1., 10., 100., 1000.)
        def fit_baseline(tr, option):
            mean, scale = x[tr].mean(0), np.maximum(x[tr].std(0), .02)
            model = (RandomForestRegressor(n_estimators=400, min_samples_leaf=option,
                                           random_state=42, n_jobs=1)
                     if name == 'random_forest' else Ridge(alpha=option))
            model.fit((x[tr] - mean) / scale, delta[tr])
            return model, mean, scale
        scores = []
        for option in options:
            errors = []
            for a, b in folds:
                tr, va = train[a], train[b]
                m, mu, sd = fit_baseline(tr, option)
                errors.append(float(np.abs(m.predict((x[va] - mu) / sd) - delta[va]).mean()))
            scores.append((float(np.mean(errors)), option))
        _, option = min(scores)
        m, mu, sd = fit_baseline(train, option)
        predictions[name] = anchor[test] + m.predict((x[test] - mu) / sd)
        baseline_selection[name] = {'candidates': scores, 'chosen': option}
    # Ordinary absolute-output MLP: same CV treatment, one fixed architecture.
    config = dict(width=32, structured=False, shrinkage=.0)
    plain_epochs = []
    for fi, (a, b) in enumerate(folds):
        fit = neural_fit(x[train[a]], truth[train[a]], config, 170 + fi,
                         (x[train[b]], truth[train[b]]))
        plain_epochs.append(fit[4])
    plain = [neural_predict(neural_fit(x[train], truth[train], config, seed,
                                     epochs=max(1, int(np.median(plain_epochs)))), x[test])
             for seed in (17, 42, 73, 101, 137)]
    predictions['absolute_mlp_ensemble'] = np.mean(plain, axis=0)
    rows, metric_rows = [], []
    for method, prediction in predictions.items():
        error = prediction - truth[test]
        metric_rows.append(dict(method=method, mae=float(abs(error).mean()),
                                rmse=float(np.sqrt((error ** 2).mean()))))
        for i, meter in enumerate(ids[test]):
            for j in range(30):
                rows.append(dict(method=method, meter=meter, round=j // 10 + 4,
                                 channel=channels[j % 10], target=float(truth[test[i], j]),
                                 prediction=float(prediction[i, j])))
    pd.DataFrame(rows).to_csv(args.output / 'predictions.csv', index=False)
    pd.DataFrame(metric_rows).to_csv(args.output / 'metrics.csv', index=False)
    # Paired bootstrap resamples meters, not correlated channels/time points.
    neural_error = abs(predictions['meter_residual_ensemble'] - truth[test]).mean(1)
    rf_error = abs(predictions['random_forest'] - truth[test]).mean(1)
    rng = np.random.default_rng(381)
    bootstrap = (neural_error - rf_error)[rng.integers(nt, size=(10000, nt))].mean(1)
    provenance = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      train_ids=ids[train].tolist(), test_ids=ids[test].tolist(), dates=dates,
                      input_rounds=[1, 2, 3], target_rounds=[4, 5, 6],
                      selected=chosen, final_epochs=budget, baseline_selection=baseline_selection,
                      plain_mlp_epochs=plain_epochs, synthetic_data=False,
                      paired_neural_minus_rf_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                      limitation='Previously inspected holdout; exploratory, not a new confirmatory test. '
                      'Single stress setting, 29 complete meters. No verified accumulated stress hours.')
    (args.output / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'training_history.json').write_text(json.dumps(histories), encoding='utf-8')
    print(json.dumps(metric_rows, indent=2), flush=True)
    print('paired neural-minus-RF interval', provenance['paired_neural_minus_rf_ci95'], flush=True)


if __name__ == '__main__':
    main()
