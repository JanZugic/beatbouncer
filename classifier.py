"""
club_classifier — score DJ libraries on club-worthiness.

Signals (weighted):
  - Spotify audio features (CSV playlist export) — strongest signal
  - Librosa audio analysis — tempo, beat regularity, loudness, dynamic range
  - MusicBrainz artist tags — genre reputation (cached, rate-limited)
  - ID3 tags — genre, BPM
  - Filename keyword heuristics — "Extended Mix" / "Acoustic" / "Slowed"

Produces club_score in [0,1] and a confidence score.
Caches per-file features in a JSON sidecar so re-runs are incremental.
"""

from __future__ import annotations

import sys as _sys
try: _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
try: _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass

import argparse
import csv
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from mutagen.id3 import ID3, ID3NoHeaderError
from mutagen.mp3 import MP3

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ─── 1. Knowledge bases ────────────────────────────────────────────────────

CLUB_POSITIVE_KW = [
    'extended mix', 'extended', 'club mix', 'club edit', 'dub mix',
    'dance mix', 'rework', 'bootleg', 'original mix', 'radio edit',
    'tech mix', 'house mix', 'techno mix', 'dnb mix',
]
CLUB_NEGATIVE_KW = [
    'acoustic', 'unplugged', 'piano version', 'stripped back', 'stripped',
    'live at', 'live from', 'live session', 'live acoustic', 'live performance',
    'slowed', '+ reverb', 'slowed + reverb', 'nightcore', 'lofi', 'lo-fi',
    'chopped and screwed', 'lullaby', 'ballad version', 'symphonic',
    'orchestral version', 'instrumental version', 'karaoke',
    '(intro)', 'interlude', '(outro)', 'skit',
    'demo version', 'cover by', 'mtv unplugged',
]

# Genre tag vocabulary (lowercased substrings matched against MB tags & ID3 genre)
CLUB_TAGS = {
    'house','techno','trance','dance','edm','electronic','electro',
    'disco','funk','dance-pop','club','drum and bass','drum-and-bass',
    'dnb','dubstep','jersey club','uk garage','garage','breakbeat',
    'trap','hip hop','hip-hop','rap','grime','drill','reggaeton',
    'afrobeats','afrobeat','moombahton','hardstyle','progressive house',
    'deep house','tech house','future bass','tropical house','latin',
    'baile funk','bass','jungle','donk','rave','phonk',
}
NON_CLUB_TAGS = {
    'ballad','acoustic','folk','country','classical','jazz','blues','gospel',
    'opera','ambient','post-rock','singer-songwriter','chamber',
    'metal','hard rock','doom','death metal','black metal','power metal',
    'symphonic metal','thrash','prog','progressive rock','prog rock',
    'punk rock','hardcore punk','shoegaze','drone','soundtrack',
    'orchestral','choral','lullaby','easy listening','adult contemporary',
    'yugoslavian rock','yu rock','sevdah','turbo-folk','folk rock','worship',
    'christian rock','new age','vaporwave','slowcore',
}


# ─── 2. Filename parsing ───────────────────────────────────────────────────

_ID_BRACKET_RE = re.compile(r'\[[a-zA-Z0-9_-]{11}\]')
_PAREN_NOISE_RE = re.compile(
    r'\([^)]*(?:official|lyric|audio|visualizer|video|reupload|prod|feat|ft\.?)[^)]*\)',
    re.I,
)
_FEAT_RE = re.compile(r'\b(?:feat|ft)\.?\b.*', re.I)
_NONALPHANUM_RE = re.compile(r'[^a-z0-9]+', re.I)
_SPLIT_DASH_RE = re.compile(r'\s+-\s+')


def norm(s: str) -> str:
    s = _ID_BRACKET_RE.sub('', s)
    s = _PAREN_NOISE_RE.sub('', s)
    s = _FEAT_RE.sub('', s)
    s = _NONALPHANUM_RE.sub(' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def parse_artist_title(stem: str) -> tuple[str, str]:
    stem = _ID_BRACKET_RE.sub('', stem).strip()
    parts = [p.strip() for p in _SPLIT_DASH_RE.split(stem) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return '', stem


def filename_features(stem: str) -> dict:
    low = stem.lower()
    return {
        'fn_pos': sum(1 for k in CLUB_POSITIVE_KW if k in low),
        'fn_neg': sum(1 for k in CLUB_NEGATIVE_KW if k in low),
    }


# ─── 3. ID3 metadata ───────────────────────────────────────────────────────

def id3_features(path: Path) -> dict:
    try:
        m = MP3(path)
        dur = float(m.info.length)
    except Exception:
        return {}
    bpm = genre = None
    try:
        tags = ID3(path)
        bpms = tags.getall('TBPM')
        if bpms:
            try: bpm = float(bpms[0].text[0])
            except (ValueError, IndexError): pass
        gens = tags.getall('TCON')
        if gens:
            genre = str(gens[0].text[0])
    except (ID3NoHeaderError, Exception):
        pass
    return {'duration': dur, 'bpm_id3': bpm, 'genre': genre}


# ─── 4. Spotify CSV lookup ─────────────────────────────────────────────────

SPOTIFY_FIELDS = ['Danceability','Energy','Tempo','Valence','Loudness',
                  'Speechiness','Acousticness','Instrumentalness','Liveness']


def load_spotify_csvs(paths: list[str]) -> dict:
    lookup = {}
    for path in paths:
        p = Path(path)
        if not p.exists(): continue
        try:
            with p.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    name = (row.get('Track Name') or '').strip()
                    artists = (row.get('Artist Name(s)') or '').strip()
                    if not name or not artists: continue
                    try:
                        feats = {k.lower(): float(row[k])
                                 for k in SPOTIFY_FIELDS if row.get(k)}
                    except (ValueError, TypeError):
                        continue
                    tk = norm(name)
                    for a in artists.split(';'):
                        lookup[(norm(a), tk)] = feats
        except Exception as e:
            print(f'  warn: cannot read {path}: {e}')
    return lookup


def spotify_features(artist: str, title: str, lookup: dict) -> Optional[dict]:
    if not lookup: return None
    a, t = norm(artist), norm(title)
    if (a, t) in lookup: return lookup[(a, t)]
    # Fuzzy: same artist, title contains (handles "Song (Official Video)" vs "Song")
    for (ka, kt), v in lookup.items():
        if ka == a and kt and (kt in t or t in kt):
            return v
    return None


# ─── 5. MusicBrainz artist tag lookup ──────────────────────────────────────

_MB_UA = {'User-Agent': 'club-classifier/1.0 (personal DJ tool)'}


def mb_artist_tags(artist: str, cache: dict, sleep: float = 1.1) -> list[str]:
    if not artist: return []
    key = norm(artist)
    if key in cache:
        return cache[key]
    try:
        q = urllib.parse.quote(f'"{artist}"')
        url1 = f'https://musicbrainz.org/ws/2/artist/?query={q}&fmt=json&limit=1'
        with urllib.request.urlopen(
            urllib.request.Request(url1, headers=_MB_UA), timeout=15
        ) as r:
            data = json.loads(r.read())
        artists = data.get('artists', [])
        if not artists or artists[0].get('score', 0) < 70:
            cache[key] = []
            return []
        mbid = artists[0]['id']
        time.sleep(sleep)
        url2 = f'https://musicbrainz.org/ws/2/artist/{mbid}?inc=tags&fmt=json'
        with urllib.request.urlopen(
            urllib.request.Request(url2, headers=_MB_UA), timeout=15
        ) as r:
            data = json.loads(r.read())
        tags = [t['name'].lower() for t in data.get('tags', [])
                if t.get('count', 0) > 0]
        # Also include genres array for newer MB schema
        tags += [g['name'].lower() for g in data.get('genres', [])
                 if g.get('count', 0) > 0]
        cache[key] = sorted(set(tags))
        time.sleep(sleep)
        return cache[key]
    except Exception:
        cache[key] = []
        return []


# ─── 6. Audio analysis (librosa) ───────────────────────────────────────────

def audio_features(path: Path) -> dict:
    """Extract audio features. Loads 60s from 30s offset to focus on the drop."""
    if not HAS_LIBROSA:
        return {}
    try:
        y, sr = librosa.load(str(path), sr=22050, mono=True,
                             duration=60.0, offset=30.0)
        if len(y) < sr * 8:
            # try from start if track < 90s
            y, sr = librosa.load(str(path), sr=22050, mono=True, duration=60.0)
        if len(y) < sr * 5:
            return {'audio_err': 'too short'}

        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo)[0])

        rms_series = librosa.feature.rms(y=y)
        rms_mean = float(np.mean(rms_series))
        dyn = float(np.percentile(rms_series, 95) - np.percentile(rms_series, 5))

        sc = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

        onset = librosa.onset.onset_strength(y=y, sr=sr)
        onset_mean = float(np.mean(onset))

        # Beat regularity: low IBI variance = steady beat
        if len(beats) >= 4:
            ibi = np.diff(librosa.frames_to_time(beats, sr=sr))
            beat_reg = float(1.0 / (1.0 + np.std(ibi) / (np.mean(ibi) + 1e-6)))
        else:
            beat_reg = 0.0

        # Frequency band energy fractions
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        total = S.sum() + 1e-9
        sub_bass_frac  = float(S[freqs < 60].sum() / total)
        bass_frac      = float(S[(freqs >= 60) & (freqs < 250)].sum() / total)
        mid_frac       = float(S[(freqs >= 250) & (freqs < 2000)].sum() / total)
        high_mid_frac  = float(S[(freqs >= 2000) & (freqs < 6000)].sum() / total)
        air_frac       = float(S[freqs >= 6000].sum() / total)
        low_energy_frac = sub_bass_frac + bass_frac  # backward compat

        # Harmonic/Percussive Source Separation — rock is harmonic-heavy (guitars),
        # club music is percussive-heavy (drum machines).
        try:
            y_h, y_p = librosa.effects.hpss(y, margin=3.0)
            e_h = float(np.sqrt(np.mean(y_h**2)))
            e_p = float(np.sqrt(np.mean(y_p**2)))
            percussive_frac = e_p / (e_h + e_p + 1e-9)
        except Exception:
            percussive_frac = None

        return {
            'tempo_au': tempo,
            'rms': rms_mean,
            'dyn_range': dyn,
            'spectral_centroid': sc,
            'spectral_flatness': flat,
            'zcr': zcr,
            'onset_strength': onset_mean,
            'beat_regularity': beat_reg,
            'low_energy_frac': low_energy_frac,
            'sub_bass_frac': sub_bass_frac,
            'bass_frac': bass_frac,
            'mid_frac': mid_frac,
            'high_mid_frac': high_mid_frac,
            'air_frac': air_frac,
            'percussive_frac': percussive_frac,
        }
    except Exception as e:
        return {'audio_err': str(e)[:200]}


# ─── 7. Scoring ────────────────────────────────────────────────────────────

@dataclass
class Score:
    club_score: float
    confidence: float
    sources: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_track(feats: dict) -> Score:
    contribs: list[tuple[float, float, str]] = []  # (weight, score_0_1, label)
    reasons: list[str] = []

    # 0. ML prediction — data-driven, overrides audio when confident
    ml_p = feats.get('ml_prob')
    if ml_p is not None:
        # Confidence is higher when prob is far from 0.5
        ml_conf = abs(ml_p - 0.5) * 2
        w = 0.35 if ml_conf > 0.6 else 0.25
        contribs.append((w, ml_p, 'ml'))
        reasons.append(f'ml_prob={ml_p:.2f}')

    # 1. Spotify features — strongest
    sp = feats.get('spotify')
    if sp:
        dance = sp.get('danceability', 0.5)
        energy = sp.get('energy', 0.5)
        acoust = sp.get('acousticness', 0.0)
        live = sp.get('liveness', 0.0)
        instr = sp.get('instrumentalness', 0.0)
        valence = sp.get('valence', 0.5)
        # Spotify acousticness is noisy (misclassifies heavy tracks); cap its penalty.
        # Only strong penalty when BOTH energy low AND acousticness high.
        base = 0.55 * dance + 0.35 * energy + 0.10 * valence
        acoust_pen = 0.35 * acoust if energy < 0.5 else 0.12 * acoust
        penalty = acoust_pen + 0.25 * live + 0.15 * instr
        sp_score = _clip(base - penalty + 0.10)
        contribs.append((0.45, sp_score, 'spotify'))
        reasons.append(f'spotify d={dance:.2f} e={energy:.2f} ac={acoust:.2f} lv={live:.2f}')

    # 2. Audio analysis
    au = feats.get('audio') or {}
    if 'tempo_au' in au:
        tempo = au['tempo_au']
        # Librosa often doubles tempo (e.g. reports 198 for a 99 BPM track).
        # Also sometimes halves. Score using the BEST match over {t, t/2, t*2}.
        # Only fold DOWN fast mis-detections (librosa commonly doubles fast trap).
        # Don't fold UP slow tempos — that rescues ballads incorrectly.
        candidates = [tempo]
        if tempo > 170: candidates.append(tempo / 2)
        def _tempo_band(t):
            if   100 <= t <= 140: return 1.0
            elif 140 <  t <= 180: return 0.90   # DnB / hardstyle
            elif  85 <= t <  100: return 0.80   # hip hop / trap
            elif  70 <= t <   85: return 0.55
            elif  60 <= t <   70: return 0.25
            else:                 return 0.10
        tempo_s = max(_tempo_band(c) for c in candidates)
        beat_reg = au.get('beat_regularity', 0.0)
        rms = au.get('rms', 0.0)
        loud_s = _clip(rms * 10)           # rms ~0.1 is loud
        dyn = au.get('dyn_range', 0.0)
        compr_s = _clip(1.0 - dyn * 8)      # low dynamic range = mastered-loud, club-friendly

        # Timbre / instrumentation signals
        perc = au.get('percussive_frac')
        sub  = au.get('sub_bass_frac', 0.0)
        mid  = au.get('mid_frac', 0.0)
        # percussive score: club music sits ~0.45-0.60; rock sits ~0.20-0.35
        if perc is not None:
            perc_s = _clip((perc - 0.25) / 0.30)   # 0 at perc=0.25, 1 at perc=0.55+
        else:
            perc_s = 0.5
        # sub-bass presence (typical club track has sub_bass_frac ~0.03-0.10, rock ~0.005-0.02)
        subbass_s = _clip((sub - 0.015) / 0.05)
        # mid-band dominance penalizes rock (guitars/vocals at 300-2k Hz)
        # typical club track mid_frac ~0.25-0.45, rock ~0.50-0.75
        mid_penalty = _clip((mid - 0.45) / 0.20)   # 0 if mid<=0.45, 1 if mid>=0.65

        # Combined timbre score
        timbre_s = _clip(0.45 * perc_s + 0.35 * subbass_s + 0.20 * (1 - mid_penalty))

        # Peak-time energy — catches "passive" chill tracks that otherwise look club-ish.
        # Combines onset strength (kick/hit intensity), RMS (overall loudness),
        # and Spotify energy if we have it.
        onset = au.get('onset_strength', 0)
        # Normalize: onset ~1 is sparse, ~3 is solid dance, ~5+ is peak-time
        onset_s = _clip((onset - 1.5) / 3.0)
        rms_s = _clip((rms - 0.04) / 0.08)
        sp = feats.get('spotify') or {}
        sp_energy = sp.get('energy')
        if sp_energy is not None:
            energy_s = _clip(0.4 * onset_s + 0.3 * rms_s + 0.3 * sp_energy)
        else:
            energy_s = _clip(0.6 * onset_s + 0.4 * rms_s)

        au_score = (0.22 * tempo_s + 0.12 * beat_reg + 0.08 * loud_s
                    + 0.03 * compr_s + 0.35 * timbre_s + 0.20 * energy_s)
        contribs.append((0.35, au_score, 'audio'))
        reasons.append(
            f'audio t={tempo:.0f} perc={perc or 0:.2f} sub={sub:.3f} '
            f'onset={onset:.1f} energy={energy_s:.2f}'
        )

        # Hard passive flag: multiple weak signals stack up. Gated by ML — if the
        # learned model strongly says 'club' (DnB, house with sparse production),
        # skip this penalty. It only fires when both audio AND the ML agree
        # the track isn't energetic.
        passive_hits = 0
        if onset < 2.0: passive_hits += 1
        if rms < 0.055: passive_hits += 1
        if sp_energy is not None and sp_energy < 0.55: passive_hits += 1
        if sp.get('danceability') is not None and sp['danceability'] < 0.60: passive_hits += 1
        if tempo < 95 and onset < 3.0: passive_hits += 1   # slow AND sparse
        ml_p_check = feats.get('ml_prob')
        ml_endorses = ml_p_check is not None and ml_p_check > 0.80
        if passive_hits >= 3 and not ml_endorses:
            contribs.append((0.20, 0.15, 'passive'))
            reasons.append(f'passive signals: {passive_hits}')
    elif au.get('audio_err'):
        reasons.append(f"audio_err: {au['audio_err']}")

    # 3. Filename keywords
    fn_pos = feats.get('fn_pos', 0)
    fn_neg = feats.get('fn_neg', 0)
    if fn_pos or fn_neg:
        fn_score = _clip(0.5 + 0.30 * fn_pos - 0.55 * fn_neg)
        w = 0.18 if fn_neg else 0.12   # negatives more diagnostic
        contribs.append((w, fn_score, 'filename'))
        reasons.append(f'filename +{fn_pos}/-{fn_neg}')

    # 4. MusicBrainz / genre tags
    tags = feats.get('tags') or []
    if tags:
        club_hits = sum(1 for t in tags if any(ct in t for ct in CLUB_TAGS))
        non_hits = sum(1 for t in tags if any(nt in t for nt in NON_CLUB_TAGS))
        if club_hits or non_hits:
            gt_score = _clip(0.5 + 0.22 * (club_hits - non_hits))
            contribs.append((0.18, gt_score, 'mb_tags'))
            reasons.append(f'mb_tags +{club_hits}/-{non_hits} ({", ".join(tags[:5])})')
        else:
            reasons.append(f'mb_tags neutral ({", ".join(tags[:5])})')

    # 5. ID3 genre (last, usually empty from YT downloads)
    genre = (feats.get('genre') or '').lower()
    if genre:
        club_m = any(ct in genre for ct in CLUB_TAGS)
        non_m = any(nt in genre for nt in NON_CLUB_TAGS)
        if club_m and not non_m:
            contribs.append((0.08, 0.85, 'id3_genre')); reasons.append(f'id3 genre=+{genre}')
        elif non_m and not club_m:
            contribs.append((0.08, 0.15, 'id3_genre')); reasons.append(f'id3 genre=-{genre}')

    # Hard flags
    dur = feats.get('duration') or 0
    if 0 < dur < 60:
        contribs.append((0.25, 0.05, 'short'))
        reasons.append(f'<60s ({int(dur)}s)')

    if not contribs:
        return Score(0.5, 0.0, [], ['no features'])

    total_w = sum(w for w, _, _ in contribs)
    score = sum(w * v for w, v, _ in contribs) / total_w
    confidence = _clip(total_w / 0.7)
    sources = [lbl for _, _, lbl in contribs]
    return Score(round(score, 3), round(confidence, 2), sources, reasons)


# ─── 8. Pipeline ───────────────────────────────────────────────────────────

def process_library(
    music_dir: Path,
    spotify_csvs: list[str],
    cache_file: Path,
    use_audio: bool = True,
    use_mb: bool = True,
    mb_limit_per_run: int = 400,
    audio_workers: int = 3,
    reclassify: bool = False,
) -> dict:
    cache: dict = {}
    if cache_file.exists():
        try: cache = json.loads(cache_file.read_text(encoding='utf-8'))
        except Exception: cache = {}

    spotify = load_spotify_csvs(spotify_csvs)
    print(f'spotify lookup: {len(spotify)} rows')

    mb_cache = cache.setdefault('_mb_cache', {})
    file_cache = cache.setdefault('files', {})

    songs = sorted(music_dir.glob('*.mp3'))
    print(f'library: {len(songs)} tracks')

    # Stage 1: metadata + filename + spotify (fast, synchronous)
    todo_audio: list[tuple[Path, str]] = []
    todo_mb: list[tuple[Path, str]] = []

    def key_of(p: Path) -> str:
        return str(p)

    for p in songs:
        k = key_of(p)
        mtime = p.stat().st_mtime
        rec = file_cache.get(k, {})
        if not reclassify and rec.get('mtime') == mtime and rec.get('complete'):
            continue

        artist, title = parse_artist_title(p.stem)
        feats: dict = {'artist': artist, 'title': title}
        feats.update(filename_features(p.stem))
        feats.update(id3_features(p))
        sp = spotify_features(artist, title, spotify)
        if sp: feats['spotify'] = sp

        # carry over cached expensive fields if mtime unchanged
        cached_feats = rec.get('features', {}) if rec.get('mtime') == mtime else {}
        if cached_feats.get('audio'): feats['audio'] = cached_feats['audio']
        if cached_feats.get('tags'):  feats['tags'] = cached_feats['tags']

        file_cache[k] = {'mtime': mtime, 'features': feats, 'complete': False}

        if use_audio and 'audio' not in feats:
            todo_audio.append((p, k))
        if use_mb and artist and 'tags' not in feats:
            todo_mb.append((p, k))

    print(f'need audio analysis: {len(todo_audio)}')
    print(f'need MB lookups:     {len(todo_mb)} (limit this run: {mb_limit_per_run})')

    # Stage 2: parallel audio analysis
    if todo_audio and use_audio and HAS_LIBROSA:
        done = 0
        with ThreadPoolExecutor(max_workers=audio_workers) as pool:
            futs = {pool.submit(audio_features, p): (p, k) for p, k in todo_audio}
            for fut in as_completed(futs):
                p, k = futs[fut]
                try:
                    af = fut.result()
                except Exception as e:
                    af = {'audio_err': str(e)[:120]}
                file_cache[k]['features']['audio'] = af
                done += 1
                if done % 25 == 0:
                    print(f'  audio {done}/{len(todo_audio)}')
                    _save(cache_file, cache)
        _save(cache_file, cache)

    # Stage 3: MB lookups (rate-limited; capped per run)
    if todo_mb and use_mb:
        seen = set()
        done = 0
        for p, k in todo_mb:
            if done >= mb_limit_per_run: break
            artist = file_cache[k]['features'].get('artist') or ''
            akey = norm(artist)
            if not akey or akey in seen:
                # fill from cache if we already looked up this artist
                if akey in mb_cache:
                    file_cache[k]['features']['tags'] = mb_cache[akey]
                continue
            seen.add(akey)
            tags = mb_artist_tags(artist, mb_cache)
            file_cache[k]['features']['tags'] = tags
            done += 1
            if done % 10 == 0:
                print(f'  mb {done}/{min(len(todo_mb), mb_limit_per_run)}: {artist} -> {tags[:3]}')
                _save(cache_file, cache)
        # Fill the rest from cache (same artist as already looked up)
        for p, k in todo_mb:
            artist = file_cache[k]['features'].get('artist') or ''
            akey = norm(artist)
            if akey in mb_cache and 'tags' not in file_cache[k]['features']:
                file_cache[k]['features']['tags'] = mb_cache[akey]
        _save(cache_file, cache)

    # Stage 4: score every record
    for k, rec in file_cache.items():
        if k.startswith('_'): continue
        feats = rec.get('features', {})
        s = score_track(feats)
        rec['score'] = asdict(s)
        rec['complete'] = True

    _save(cache_file, cache)
    return cache


def _save(path: Path, data: dict) -> None:
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data, default=str), encoding='utf-8')
    tmp.replace(path)


# ─── 9. CLI ────────────────────────────────────────────────────────────────

# ─── 9b. Programmatic API (for inline use from other scripts / Claude sessions) ──

DEFAULT_CACHE = Path(r'C:\Users\jz\Tools\club_classifier\cache.json')
DEFAULT_MUSIC = Path(r'D:\songs')
DEFAULT_SPOTIFY = [
    r'C:\Users\jz\Downloads\UK_NIGHTCLUB_.csv',
    'C:\\Users\\jz\\Downloads\\Nightclub_Anthems_\U0001f4bf.csv',
]


class Library:
    """High-level interface for scoring and querying a music library.

    Usage (from another script or `uv run python -c`):

        from classifier import Library
        lib = Library()
        lib.scan()                          # incremental, uses default dirs
        hits = lib.below(threshold=0.4, min_conf=0.6)
        bangers = lib.top(20)
        stats = lib.stats()
    """

    def __init__(self, music_dir=None, cache_file=None, spotify_csvs=None):
        self.music_dir = Path(music_dir or DEFAULT_MUSIC)
        self.cache_file = Path(cache_file or DEFAULT_CACHE)
        self.spotify_csvs = spotify_csvs if spotify_csvs is not None else DEFAULT_SPOTIFY
        self._cache = None

    def _load(self):
        if self._cache is None:
            if self.cache_file.exists():
                try: self._cache = json.loads(self.cache_file.read_text(encoding='utf-8'))
                except Exception: self._cache = {'files': {}, '_mb_cache': {}}
            else:
                self._cache = {'files': {}, '_mb_cache': {}}
        return self._cache

    def scan(self, *, audio=True, mb=True, mb_limit=400, workers=3, reclassify=False):
        """Run/continue the scan. Incremental by default."""
        self._cache = process_library(
            self.music_dir, self.spotify_csvs, self.cache_file,
            use_audio=audio, use_mb=mb, mb_limit_per_run=mb_limit,
            audio_workers=workers, reclassify=reclassify,
        )
        return self

    def records(self):
        """Return [(path, score_dict, features_dict)] for all scored tracks."""
        files = self._load().get('files', {})
        out = []
        for p, r in files.items():
            if 'score' not in r: continue
            out.append((p, r['score'], r.get('features', {})))
        return out

    def below(self, threshold=0.4, min_conf=0.5):
        """Tracks the classifier thinks are NOT club-worthy."""
        return [(p, s, f) for p, s, f in self.records()
                if s['club_score'] < threshold and s['confidence'] >= min_conf]

    def above(self, threshold=0.6, min_conf=0.3):
        return [(p, s, f) for p, s, f in self.records()
                if s['club_score'] >= threshold and s['confidence'] >= min_conf]

    def top(self, n=20):
        return sorted(self.records(), key=lambda x: -x[1]['club_score'])[:n]

    def bottom(self, n=20):
        return sorted(self.records(), key=lambda x: x[1]['club_score'])[:n]

    def by_tempo(self, lo=None, hi=None):
        """Filter by audio-detected tempo (if audio analysis ran)."""
        out = []
        for p, s, f in self.records():
            tempo = (f.get('audio') or {}).get('tempo_au')
            if tempo is None: continue
            if lo is not None and tempo < lo: continue
            if hi is not None and tempo > hi: continue
            out.append((p, s, f))
        return out

    def by_artist(self, artist_query):
        """Case-insensitive substring match on filename artist."""
        q = artist_query.lower()
        return [(p, s, f) for p, s, f in self.records()
                if q in Path(p).stem.lower()]

    def stats(self):
        recs = self.records()
        scores = [s['club_score'] for _, s, _ in recs]
        confs = [s['confidence'] for _, s, _ in recs]
        sources = {}
        for _, s, _ in recs:
            for src in s.get('sources', []):
                sources[src] = sources.get(src, 0) + 1
        return {
            'total': len(recs),
            'mean_score': round(sum(scores)/len(scores), 3) if scores else 0,
            'mean_conf': round(sum(confs)/len(confs), 3) if confs else 0,
            'below_0.4': sum(1 for s in scores if s < 0.4),
            'above_0.6': sum(1 for s in scores if s >= 0.6),
            'source_coverage': sources,
        }

    def delete(self, victims, *, reason='', dry_run=True):
        """Delete a specific list of [(path, score, feats)] entries.
        Appends a JSONL manifest to deletions.log for future reference.
        """
        import datetime
        freed = 0
        entries = []
        for p, s, f in victims:
            pp = Path(p)
            if not pp.exists(): continue
            sz = pp.stat().st_size
            entry = {
                'path': p,
                'name': pp.name,
                'size': sz,
                'score': s,
                'tempo': (f.get('audio') or {}).get('tempo_au'),
                'ml_prob': f.get('ml_prob'),
                'tags': f.get('tags') or [],
            }
            entries.append(entry)
            if not dry_run:
                pp.unlink()
                self._cache['files'].pop(p, None)
                freed += sz
        if not dry_run and entries:
            log_path = self.cache_file.parent / 'deletions.log'
            ts = datetime.datetime.now().isoformat(timespec='seconds')
            with log_path.open('a', encoding='utf-8') as lg:
                lg.write(f'\n# {ts}  reason={reason!r}  count={len(entries)}  freed={freed/1024/1024:.1f}MB\n')
                for e in entries:
                    lg.write(json.dumps(e, ensure_ascii=False) + '\n')
            _save(self.cache_file, self._cache)
        return entries, freed

    def delete_below(self, threshold=0.35, min_conf=0.5, dry_run=True,
                     spare=None, reason=None):
        """Delete all tracks below threshold except those matching any 'spare' substring."""
        spare = spare or []
        victims = self.below(threshold=threshold, min_conf=min_conf)
        victims.sort(key=lambda x: x[1]['club_score'])
        if spare:
            victims = [(p, s, f) for p, s, f in victims
                       if not any(sp.lower() in Path(p).name.lower() for sp in spare)]
        reason = reason or f'below-{threshold}-conf-{min_conf}'
        return self.delete(victims, reason=reason, dry_run=dry_run)

    def deletion_history(self, limit=None):
        """Parse deletions.log and return recent deletion entries."""
        log_path = self.cache_file.parent / 'deletions.log'
        if not log_path.exists(): return []
        entries = []
        for line in log_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            try: entries.append(json.loads(line))
            except json.JSONDecodeError: pass
        return entries[-limit:] if limit else entries


def cmd_scan(args):
    cache = process_library(
        Path(args.music_dir),
        args.spotify_csv,
        Path(args.cache),
        use_audio=not args.no_audio,
        use_mb=not args.no_mb,
        mb_limit_per_run=args.mb_limit,
        audio_workers=args.workers,
        reclassify=args.reclassify,
    )
    _print_summary(cache, args.threshold)


def cmd_list(args):
    cache = json.loads(Path(args.cache).read_text(encoding='utf-8'))
    _print_summary(cache, args.threshold, show_top=args.top, only_below=args.only_below)


def cmd_delete(args):
    lib = Library(cache_file=args.cache)
    entries, freed = lib.delete_below(
        threshold=args.threshold,
        min_conf=args.min_confidence,
        dry_run=True,
    )
    print(f'will delete {len(entries)} files below {args.threshold} (conf ≥ {args.min_confidence})')
    for e in entries[:40]:
        print(f"  {e['score']['club_score']:.2f}  {e['name']}")
    if len(entries) > 40:
        print(f'  ... and {len(entries)-40} more')
    if not args.yes:
        ans = input('\nProceed? (y/N): ').strip().lower()
        if ans != 'y':
            print('aborted'); return
    entries, freed = lib.delete_below(
        threshold=args.threshold,
        min_conf=args.min_confidence,
        dry_run=False,
        reason=args.reason,
    )
    print(f'deleted {len(entries)} files, freed {freed/1024/1024:.1f} MB')
    print(f'manifest written to {Path(args.cache).parent / "deletions.log"}')


def _print_summary(cache, threshold, show_top=10, only_below=False):
    files = cache.get('files', {})
    ranked = sorted(
        [(p, r) for p, r in files.items() if 'score' in r],
        key=lambda x: x[1]['score']['club_score'],
    )
    below = [x for x in ranked if x[1]['score']['club_score'] < threshold]
    if not only_below:
        print(f'\nTOP CLUB BANGERS:')
        for p, r in ranked[-show_top:][::-1]:
            s = r['score']
            print(f"  {s['club_score']:.2f} [{','.join(s['sources'])[:30]}]  {Path(p).name}")
    print(f'\n{len(below)} / {len(ranked)} tracks below threshold {threshold}:')
    for p, r in below[:60]:
        s = r['score']
        print(f"  {s['club_score']:.2f} conf={s['confidence']:.2f}  {Path(p).name}")
        for rs in s['reasons'][:2]:
            print(f"      · {rs}")
    if len(below) > 60:
        print(f'  ... and {len(below)-60} more (use `list --top 200` to see all)')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', default=r'C:\Users\jz\Tools\club_classifier\cache.json')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sp_scan = sub.add_parser('scan', help='ingest a library (incremental)')
    sp_scan.add_argument('--music-dir', default=r'D:\songs')
    sp_scan.add_argument('--spotify-csv', action='append', default=[])
    sp_scan.add_argument('--no-audio', action='store_true')
    sp_scan.add_argument('--no-mb', action='store_true')
    sp_scan.add_argument('--mb-limit', type=int, default=400)
    sp_scan.add_argument('--workers', type=int, default=3)
    sp_scan.add_argument('--reclassify', action='store_true')
    sp_scan.add_argument('--threshold', type=float, default=0.40)
    sp_scan.set_defaults(func=cmd_scan)

    sp_list = sub.add_parser('list', help='print flagged tracks from cache')
    sp_list.add_argument('--threshold', type=float, default=0.40)
    sp_list.add_argument('--top', type=int, default=10)
    sp_list.add_argument('--only-below', action='store_true')
    sp_list.set_defaults(func=cmd_list)

    sp_del = sub.add_parser('delete', help='delete tracks below threshold')
    sp_del.add_argument('--threshold', type=float, default=0.35)
    sp_del.add_argument('--min-confidence', type=float, default=0.5)
    sp_del.add_argument('--reason', default='')
    sp_del.add_argument('--yes', action='store_true')
    sp_del.set_defaults(func=cmd_delete)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
