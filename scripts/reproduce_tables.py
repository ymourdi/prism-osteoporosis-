#!/usr/bin/env python
"""Recomputes the main results of the paper from the published out-of-fold predictions, without any GPU.

Labels and T-scores are read from the official LUMOS spreadsheet (they are not redistributed here):
    python scripts/reproduce_tables.py --lumos-xlsx /path/to/LUMOS/lumos_clinical_data.xlsx

Outputs are written to results/reproduced/ and a summary is printed next to the values reported in the paper.
Bootstrap CIs and p-values use the same seeds and number of resamples as the notebook, so they match exactly.
Only the 'image + clinical linear probe' baseline cannot be recomputed (it needs the ResNet-50 image features);
its values are available in results/reference_tables/table2_model_comparison.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from prism.metrics import metrics3, expected_calibration_error, net_benefit, operating_points  # noqa: E402
from prism.stats import bootstrap_ci, delong_test, paired_qwk_pvalue, holm  # noqa: E402

CLIN = ['age', 'sex_f', 'height', 'weight', 'BMI']
FINAL = 'raddino__full__keep__frozen__full__ct-raddino'


def load_labels(xlsx):
    df = pd.read_excel(xlsx)
    df.columns = df.columns.astype(str).str.strip()
    df['sex_f'] = (df['gender'] == 'Female').astype(float)
    folds = pd.read_csv(REPO / 'data' / 'folds.csv')
    return folds.merge(df, on='patient_id', how='left', validate='one_to_one')


def load_run(run_id):
    return pd.read_csv(REPO / 'results' / 'predictions' / f'{run_id}.csv')


def probs(r, suffix=''):
    return r[[f'p_normal{suffix}', f'p_osteopenia{suffix}', f'p_osteoporosis{suffix}']].values


def baselines(df, Y):
    folds = sorted(df.fold.unique())
    oof_clin, oof_ost = np.zeros((len(df), 3)), np.zeros((len(df), 3))
    ost = (0.2 * (df['weight'] - df['age'])).to_frame('OST')
    for k in folds:
        tr, te = (df.fold != k).values, (df.fold == k).values
        sc = StandardScaler().fit(df.loc[tr, CLIN])
        lr = LogisticRegression(max_iter=3000).fit(sc.transform(df.loc[tr, CLIN]), Y[tr])
        oof_clin[te] = lr.predict_proba(sc.transform(df.loc[te, CLIN]))
        oof_ost[te] = LogisticRegression(max_iter=3000).fit(ost[tr], Y[tr]).predict_proba(ost[te])
    return oof_clin, oof_ost


def row(Y, T, p, t=None):
    m = metrics3(Y, p, T if t is not None else None, t)
    ci = bootstrap_ci(Y, p)
    return {'QWK': m['qwk'], 'QWK 95% CI': f"{ci['qwk_ci'][0]:.3f}-{ci['qwk_ci'][1]:.3f}", 'Macro F1': m['f1_macro'],
            'Balanced acc.': m['bal_acc'], 'Macro AUC': m['auc_macro'], 'Osteoporosis AUC': m['auc_osteoporosis'],
            'Osteo. AUC 95% CI': f"{ci['auc_osteo_ci'][0]:.3f}-{ci['auc_osteo_ci'][1]:.3f}",
            'Osteo. sens.': m['sens_osteoporosis'], 'Osteo. spec.': m['spec_osteoporosis'], 'T-score MAE': m.get('t_mae', np.nan)}


def nested_selection(runs, Y, folds_col):
    """In each outer fold, keeps the Experiment 1 configuration with the best mean inner-validation QWK."""
    vq = pd.read_csv(REPO / 'results' / 'inner_validation_qwk.csv')
    cand = runs[runs.experiment == 'exp1_design_grid'].run_id.tolist()
    mean_vq = vq[vq.run_id.isin(cand)].groupby(['run_id', 'fold']).inner_val_qwk.mean().unstack()
    p, t, chosen = np.zeros((len(Y), 3)), np.zeros(len(Y)), []
    for k in sorted(mean_vq.columns):
        best = mean_vq[k].idxmax()
        r = load_run(best)
        te = (folds_col == k).values
        p[te], t[te] = probs(r)[te], r['t_pred'].values[te]
        chosen.append(dict(fold=int(k), configuration=best, inner_val_qwk=mean_vq.loc[best, k]))
    return p, t, pd.DataFrame(chosen)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lumos-xlsx', required=True, type=Path)
    args = ap.parse_args()
    out = REPO / 'results' / 'reproduced'
    out.mkdir(exist_ok=True)

    df = load_labels(args.lumos_xlsx)
    Y, T = df['Osteoporosis'].values.astype(int), df['t_value'].values.astype(float)
    runs = pd.read_csv(REPO / 'results' / 'runs.csv')
    R = {rid: load_run(rid) for rid in runs.run_id}
    for rid, r in R.items():
        assert (r.patient_id.values == df.patient_id.values).all(), rid
    oof_clin, oof_ost = baselines(df, Y)
    fin = R[FINAL]
    p_nest, t_nest, chosen = nested_selection(runs, Y, df.fold)
    chosen.to_csv(out / 'nested_selection.csv', index=False)

    # Comparison with baselines (Table 'Comparison of PRISM with clinical and imaging baselines')
    cands = {'OST score (LR)': (oof_ost, None), 'Clinical variables (LR)': (oof_clin, None),
             'PRISM, ResNet-50': (probs(R['resnet50__full__keep__frozen__full__ct-raddino']),
                                  R['resnet50__full__keep__frozen__full__ct-raddino'].t_pred.values),
             'PRISM, RAD-DINO (final)': (probs(fin), fin.t_pred.values), 'PRISM, nested selection': (p_nest, t_nest),
             'Final PRISM, without CT': (probs(fin, '_noct'), fin.t_pred_noct.values)}
    tab, pdl, pq = {}, [], []
    for name, (p, t) in cands.items():
        tab[name] = row(Y, T, p, t)
        is_ref = p is oof_clin
        pdl.append(np.nan if is_ref else delong_test(Y == 2, p[:, 2], oof_clin[:, 2])[2])
        pq.append(np.nan if is_ref else paired_qwk_pvalue(Y, p, oof_clin))
    tab = pd.DataFrame(tab).T
    tab['p DeLong vs clinical (Holm)'], tab['p QWK vs clinical (Holm)'] = holm(pdl), holm(pq)
    tab.to_csv(out / 'comparison_baselines.csv')

    # Experiment 1 grid and Experiment 2 encoder benchmark
    grid = []
    for _, m in runs[runs.experiment.isin(['exp1_design_grid', 'exp2_encoder_benchmark'])].iterrows():
        r = R[m.run_id]
        grid.append(dict(run_id=m.run_id, experiment=m.experiment, **row(Y, T, probs(r), r.t_pred.values)))
    pd.DataFrame(grid).sort_values('QWK', ascending=False).to_csv(out / 'design_grid_and_encoders.csv', index=False)
    enc = runs[(runs.region == 'full') & (runs.implants == 'keep') & ~runs.finetuned & (runs.variant == 'full')
               & (runs.ct_encoder == 'raddino')]
    ref = probs(fin)
    bench = {m.backbone: row(Y, T, probs(R[m.run_id]), R[m.run_id].t_pred.values) for _, m in enc.iterrows()}
    bench = pd.DataFrame(bench).T.sort_values('QWK')
    others = [b for b in bench.index if b != 'raddino']
    bench.loc[others, 'p QWK vs raddino (Holm)'] = holm([paired_qwk_pvalue(Y, probs(R[enc.set_index('backbone').loc[b, 'run_id']]), ref) for b in others])
    bench.loc[others, 'p DeLong vs raddino (Holm)'] = holm([delong_test(Y == 2, probs(R[enc.set_index('backbone').loc[b, 'run_id']])[:, 2], ref[:, 2])[2] for b in others])
    bench.to_csv(out / 'encoder_benchmark.csv')

    # Experiments 3 to 5: ablations, clinical encoding, CT encoder
    for exp, fname in [('exp3_ablation', 'component_ablation.csv'), ('exp4_clinical_encoding', 'clinical_encoding.csv'),
                       ('exp5_ct_encoder', 'ct_encoder.csv')]:
        sub = runs[runs.experiment == exp]
        t = {'PRISM (full, final)': row(Y, T, ref, fin.t_pred.values)}
        pv = [np.nan]
        for _, m in sub.iterrows():
            name = m.variant if exp != 'exp5_ct_encoder' else f'CT encoder: {m.ct_encoder}'
            t[name] = row(Y, T, probs(R[m.run_id]), R[m.run_id].t_pred.values)
            pv.append(paired_qwk_pvalue(Y, probs(R[m.run_id]), ref))
        t = pd.DataFrame(t).T
        t['Delta QWK vs final'] = t['QWK'] - t.loc['PRISM (full, final)', 'QWK']
        t['p QWK vs final (Holm)'] = holm(pv)
        t.to_csv(out / fname)

    # Calibration, decision curve and screening thresholds of the final model
    y2 = (Y == 2).astype(int)
    cal = pd.DataFrame({'ECE PRISM': [expected_calibration_error(y2, ref[:, 2])],
                        'ECE clinical': [expected_calibration_error(y2, oof_clin[:, 2])],
                        'Net benefit PRISM @0.20': [net_benefit(y2, ref[:, 2], 0.20)],
                        'Net benefit clinical @0.20': [net_benefit(y2, oof_clin[:, 2], 0.20)]})
    cal.to_csv(out / 'calibration_decision_curve.csv', index=False)
    pd.DataFrame(operating_points(y2, ref[:, 2])).to_csv(out / 'screening_thresholds.csv', index=False)
    diff = fin.t_pred.values - T
    ba = dict(bias=diff.mean(), loa_low=diff.mean() - 1.96 * diff.std(ddof=1), loa_high=diff.mean() + 1.96 * diff.std(ddof=1),
              pearson_r=np.corrcoef(T, fin.t_pred.values)[0, 1])

    # Summary next to the values reported in the paper
    paper = [('Final PRISM, QWK', tab.loc['PRISM, RAD-DINO (final)', 'QWK'], 0.701),
             ('Final PRISM, osteoporosis AUC', tab.loc['PRISM, RAD-DINO (final)', 'Osteoporosis AUC'], 0.887),
             ('Nested selection, QWK', tab.loc['PRISM, nested selection', 'QWK'], 0.660),
             ('Nested selection, osteoporosis AUC', tab.loc['PRISM, nested selection', 'Osteoporosis AUC'], 0.861),
             ('Final PRISM without CT, QWK', tab.loc['Final PRISM, without CT', 'QWK'], 0.673),
             ('Clinical LR, QWK', tab.loc['Clinical variables (LR)', 'QWK'], 0.410),
             ('OST score, QWK', tab.loc['OST score (LR)', 'QWK'], 0.354),
             ('DINOv2 encoder, QWK', bench.loc['dinov2', 'QWK'], 0.667),
             ('ECE (osteoporosis)', cal['ECE PRISM'][0], 0.028),
             ('T-score MAE', tab.loc['PRISM, RAD-DINO (final)', 'T-score MAE'], 0.805),
             ('Bland-Altman bias', ba['bias'], -0.10)]
    print(f"{'Quantity':38s} {'recomputed':>11s} {'paper':>8s}")
    for name, v, ref_v in paper:
        print(f'{name:38s} {v:11.3f} {ref_v:8.3f}')
    print(f'\nNested selection: {chosen.configuration.tolist()}')
    print(f'Tables written to {out}')


if __name__ == '__main__':
    main()
