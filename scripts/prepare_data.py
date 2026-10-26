#!/usr/bin/env python
"""Builds `lumos_master_dataset.csv`, the patient table used by the notebook.

Input (downloaded from https://doi.org/10.5281/zenodo.18173664):
    <lumos_dir>/lumos_clinical_data.xlsx
    <lumos_dir>/lumos_npy/lumos_npy/lumos_x/lumos_x_XXX_k.npy     (radiographs, from lumos_npy.zip)
    <lumos_dir>/lumos_npy/lumos_npy/lumos_ct/lumos_ct_XXX.npy     (CT volumes in HU, from lumos_npy.zip)

Output:
    <lumos_dir>/lumos_master_dataset.csv  (clinical/DXA variables, image paths and the 5-fold assignment)

Patients without any usable image are excluded (803 -> 796). By default the fold of each patient is read from
`data/folds.csv`, which guarantees exactly the folds used in the paper. With `--regenerate-folds`, the folds are
recomputed as in the original study: StratifiedKFold(5, shuffle=True, random_state=42) stratified on
class x CT availability, in the order of the clinical spreadsheet.

Usage:
    python scripts/prepare_data.py --lumos-dir /path/to/LUMOS
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def first_number(name):
    m = re.findall(r'\d+', name)
    return int(m[0]) if m else None


def index_files(root, sub):
    out = {}
    for f in sorted((root / sub).glob('**/*.npy')):
        pid = first_number(f.name)
        if pid is not None:
            out.setdefault(pid, []).append(f.relative_to(root.parent.parent).as_posix())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lumos-dir', required=True, type=Path, help='folder containing lumos_clinical_data.xlsx and lumos_npy/')
    ap.add_argument('--folds', type=Path, default=REPO / 'data' / 'folds.csv', help='fold assignment (patient_id, fold)')
    ap.add_argument('--regenerate-folds', action='store_true', help='recompute the folds instead of reading --folds')
    args = ap.parse_args()

    root = args.lumos_dir
    npy = root / 'lumos_npy' / 'lumos_npy'
    if not npy.exists():
        raise SystemExit(f'{npy} not found: unzip lumos_npy.zip inside {root}')
    df = pd.read_excel(root / 'lumos_clinical_data.xlsx')
    df.columns = df.columns.astype(str).str.strip()
    df['target_t_score'] = df['t_value'] if 't_value' in df.columns else df['T_L1-L4']

    rx, ct = index_files(npy, 'lumos_x'), index_files(npy, 'lumos_ct')
    pid = df['patient_id'].astype(int)
    df['has_xray'] = pid.map(lambda p: p in rx)
    df['xray_count'] = pid.map(lambda p: len(rx.get(p, [])))
    df['xray_paths'] = pid.map(lambda p: ';'.join(rx.get(p, [])))
    df['has_ct'] = pid.map(lambda p: p in ct)
    df['ct_count'] = pid.map(lambda p: len(ct.get(p, [])))
    df['ct_paths'] = pid.map(lambda p: ';'.join(ct.get(p, [])))

    df = df[df['has_xray'] | df['has_ct']].copy().reset_index(drop=True)
    if args.regenerate_folds:
        from sklearn.model_selection import StratifiedKFold
        key = df['Osteoporosis'].astype(str) + '_' + df['has_ct'].astype(str)
        df['fold'] = -1
        for k, (_, te) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(df, key)):
            df.loc[te, 'fold'] = k
    else:
        folds = pd.read_csv(args.folds)
        df = df.merge(folds, on='patient_id', how='inner', validate='one_to_one')
        if len(df) != len(folds):
            raise SystemExit(f'{len(folds) - len(df)} patients of {args.folds} were not found in the LUMOS files')

    out = root / 'lumos_master_dataset.csv'
    df.to_csv(out, index=False)
    print(f'{len(df)} patients | radiographs: {df.xray_count.sum()} | CT: {int(df.has_ct.sum())} -> {out}')
    print(pd.crosstab(df['fold'], df['Osteoporosis']))


if __name__ == '__main__':
    main()
