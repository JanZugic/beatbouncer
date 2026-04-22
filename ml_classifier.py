"""ML classifier: learn club-ness from tag-labeled tracks, apply to untagged ones.

Uses audio+spotify numeric features. Trains a gradient-boosted tree on tracks
where MB tags clearly say 'club-ish' or 'non-club', then predicts probability
of 'club' for every track. The prediction becomes an additional signal in the
scoring pipeline (high weight when it's confident, low when not).

This replaces hand-crafted artist blacklists — patterns are learned from data.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from classifier import CLUB_TAGS, NON_CLUB_TAGS, Library


# Audio feature keys used by the ML model (fixed order!)
AUDIO_FEATS = [
    'tempo_au', 'beat_regularity', 'rms', 'dyn_range',
    'spectral_centroid', 'spectral_flatness', 'zcr', 'onset_strength',
    'sub_bass_frac', 'bass_frac', 'mid_frac', 'high_mid_frac', 'air_frac',
    'percussive_frac',
]
SPOTIFY_FEATS = [
    'danceability', 'energy', 'tempo', 'valence', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
]


def featurize(feats: dict) -> Optional[np.ndarray]:
    au = feats.get('audio') or {}
    if not au or 'tempo_au' not in au:
        return None
    sp = feats.get('spotify') or {}
    row = []
    for k in AUDIO_FEATS:
        v = au.get(k)
        row.append(float(v) if v is not None else np.nan)
    for k in SPOTIFY_FEATS:
        v = sp.get(k)
        row.append(float(v) if v is not None else np.nan)
    # Spotify-available flag (so model knows when imputations apply)
    row.append(1.0 if sp else 0.0)
    return np.array(row, dtype=np.float32)


FEATURE_NAMES = AUDIO_FEATS + SPOTIFY_FEATS + ['has_spotify']


def label_from_tags(tags: list[str]) -> Optional[int]:
    """+1 = club-leaning, 0 = non-club, None = ambiguous/no label."""
    if not tags: return None
    low = [t.lower() for t in tags]
    club_hits = sum(1 for t in low if any(ct in t for ct in CLUB_TAGS))
    non_hits = sum(1 for t in low if any(nt in t for nt in NON_CLUB_TAGS))
    if club_hits >= 2 and non_hits == 0: return 1
    if non_hits >= 2 and club_hits == 0: return 0
    if club_hits >= 1 and non_hits == 0: return 1
    if non_hits >= 1 and club_hits == 0: return 0
    return None  # mixed tags → unreliable label


def build_dataset(lib: Library):
    X, y, paths = [], [], []
    for p, _s, f in lib.records():
        tags = f.get('tags')
        if not tags: continue
        label = label_from_tags(tags)
        if label is None: continue
        row = featurize(f)
        if row is None: continue
        X.append(row); y.append(label); paths.append(p)
    return np.array(X), np.array(y), paths


def train(lib: Library, model_path: Path):
    X, y, paths = build_dataset(lib)
    print(f'training examples: {len(y)}  club={int(y.sum())}  non_club={int(len(y)-y.sum())}')
    # Impute NaN -> 0 (GradientBoostingClassifier handles it poorly otherwise)
    # Use column median to preserve scale
    medians = np.nanmedian(X, axis=0)
    mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[mask[:, j], j] = medians[j]
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                     random_state=42)
    scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print(f'5-fold CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}')
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print('In-sample:')
    print(classification_report(y, y_pred, target_names=['non_club','club'], digits=3))
    # Feature importance
    print('\nTop feature importance:')
    imp = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1])
    for name, v in imp[:10]:
        print(f'  {v:.3f}  {name}')
    with open(model_path, 'wb') as f:
        pickle.dump({'model': clf, 'medians': medians, 'features': FEATURE_NAMES}, f)
    return clf, medians


def load_model(model_path: Path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_club_prob(clf, medians, feats_dict: dict) -> Optional[float]:
    row = featurize(feats_dict)
    if row is None: return None
    mask = np.isnan(row)
    row[mask] = medians[mask]
    return float(clf.predict_proba(row.reshape(1, -1))[0, 1])


def apply_to_library(lib: Library, model_path: Path):
    """Attach ML predictions to every record in the cache."""
    bundle = load_model(model_path)
    clf, medians = bundle['model'], bundle['medians']
    cache = lib._load()
    files = cache['files']
    hit = miss = 0
    for k, rec in files.items():
        if k.startswith('_'): continue
        feats = rec.get('features') or {}
        p = predict_club_prob(clf, medians, feats)
        if p is None:
            feats.pop('ml_prob', None)
            miss += 1
        else:
            feats['ml_prob'] = p
            hit += 1
    from classifier import _save
    _save(lib.cache_file, cache)
    print(f'scored {hit}  skipped {miss}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=r'C:\Users\jz\Tools\club_classifier\cache.json')
    ap.add_argument('--model', default=r'C:\Users\jz\Tools\club_classifier\ml_model.pkl')
    ap.add_argument('--train', action='store_true')
    ap.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    lib = Library(cache_file=args.cache)
    if args.train:
        train(lib, Path(args.model))
    if args.apply:
        apply_to_library(lib, Path(args.model))


if __name__ == '__main__':
    main()
