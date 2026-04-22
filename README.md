# beatbouncer

The bouncer for your DJ library — decides which tracks are club-worthy and
which shouldn't make it through the door. Multi-signal classifier that scores
MP3 tracks on club-worthiness and deletes the ones that don't belong. Combines Spotify audio features,
librosa audio analysis (HPSS, sub-bass fraction, spectral bands), MusicBrainz
artist tags, ID3 metadata, filename keywords, and a gradient-boosted tree
trained on tracks with unambiguous genre tags.

The classifier produces a `club_score ∈ [0, 1]` plus a `confidence` value for
every track, cached incrementally so re-runs are cheap.

## Why

Curating a DJ set from a few thousand tracks, one-off scripts get messy fast.
Filename greps miss rock tracks that score well on tempo, and manual artist
blacklists don't generalize across genres. A trained classifier over timbre +
tag features catches what filename heuristics and hand-written rules miss.

## Install

```bash
uv sync
```

Requires Python 3.12+. Uses librosa (audio analysis), mutagen (ID3),
scikit-learn (ML), numpy, requests.

## Signal sources

| Signal | Weight | What it sees |
|---|---|---|
| **ML classifier** | 0.25–0.35 | gradient-boosted tree over all numeric features; learned on MB-tagged tracks |
| **Spotify CSV** | 0.45 | danceability, energy, acousticness, liveness, instrumentalness, valence from a Spotify playlist export |
| **Audio (librosa)** | 0.35 | tempo + beat regularity, RMS loudness, dynamic range, **HPSS percussive fraction**, **sub-bass fraction <60Hz**, mid-band concentration, spectral flatness |
| **MusicBrainz tags** | 0.18 | cached artist genre tags |
| **Filename keywords** | 0.12–0.18 | "Extended Mix" / "Club Edit" +, "Acoustic/Slowed/Live at/Interlude" − |
| **ID3 genre** | 0.08 | fallback |

The ML classifier's top learned features are `sub_bass_frac` (~32%) and
`onset_strength` (~19%) — modern club production is distinguished by sustained
sub-bass and consistent onset energy.

## CLI usage

```bash
# Incremental scan: analyzes new tracks, caches expensive work
uv run python classifier.py scan \
    --music-dir /path/to/library \
    --spotify-csv playlist_A.csv \
    --spotify-csv playlist_B.csv

# List below-threshold tracks (from cache, fast)
uv run python classifier.py list --threshold 0.40 --only-below

# Delete below threshold (writes manifest to deletions.log)
uv run python classifier.py delete --threshold 0.40 --min-confidence 0.5 \
    --reason "April cleanup" --yes

# Train the ML classifier (once you have a scan with MB tags)
uv run python ml_classifier.py --train

# Apply the ML predictions to every track in the cache
uv run python ml_classifier.py --apply
```

## Python API

```python
from classifier import Library
lib = Library(music_dir='/path/to/library')
lib.scan()                                # incremental
lib.top(20)                               # peak bangers
lib.below(threshold=0.4, min_conf=0.6)    # non-club candidates
lib.by_tempo(120, 135)                    # house-tempo slice
lib.by_artist('Kukus')                    # substring match
lib.stats()                               # coverage summary

# Delete with manifest written to deletions.log
entries, freed = lib.delete_below(
    threshold=0.35, min_conf=0.5, dry_run=False,
    spare=['Taylor Swift - Shake', 'Steve Lacy - Bad Habit'],
    reason='club-only cut, April 2026',
)

# Audit past deletions
for e in lib.deletion_history(limit=50):
    print(e['score']['club_score'], e['name'])
```

## Deletion logging

`delete_below` writes a JSONL manifest to `deletions.log` alongside the cache:

```
# 2026-04-22T21:45:12  reason='club-only cut'  count=237  freed=2261.8MB
{"path":"D:\\songs\\Plavi Orkestar - ...","name":"Plavi Orkestar - ...","size":6834123,"score":{"club_score":0.28,"confidence":0.69,...},"tempo":117.0,"ml_prob":0.04,"tags":[]}
...
```

## Design principles

- **No hand-curated artist blacklists.** When tempted to write `if artist in [...]`, improve the features instead. Audio timbre (HPSS, frequency bands) and learned ML classifiers generalize; lists rot.
- **Confidence-aware.** Every score carries a `confidence` based on how many signals contributed. Deletion respects `--min-confidence` so tracks with one weak signal don't get axed.
- **Incremental everywhere.** File features are keyed by mtime; MB artist tags are cached; the ML model is saved to disk. A rescan of an unchanged library is seconds.
- **Auditable deletions.** Every destructive action writes a JSONL manifest. `lib.deletion_history()` replays what was removed and why.

## Known quirks

- Spotify `acousticness` is noisy on heavy tracks (Stormzy "Shut Up" tagged 0.81). Scoring caps that penalty when energy is also high.
- Librosa's `beat_track` sometimes doubles tempo. Scoring folds tempos >170 BPM to half when computing the tempo-suitability band.
- MusicBrainz tag coverage for Balkan artists is sparse. Audio features and the ML classifier carry those tracks.
