"""
picker.py — Phase Picker
==========================================
Salton Sea Geothermal Region

Runs SAR phase picking on continuous MiniSEED data.
All config embedded — no external config.py needed.

Supports both local data directories and Pelican OSDF URLs:
    data_dir  = 'Data_Salton'                              # local
    data_dir  = 'osdf:///ndp/public/ucr_seis/Data_Salton'  # Pelican

Supports both local checkpoint paths and Pelican OSDF URLs:
    ckpt_path = 'models/8700_17-319.ckpt'                           # local
    ckpt_path = 'osdf:///ndp/public/ucr_seis/models/8700_17-319.ckpt'  # Pelican

Usage (from notebook):
    from picker import SAR_Picker, run_picking
    run_picking(
        data_dir  = 'osdf:///ndp/public/ucr_seis/Data_Salton',
        sta_file  = 'config/salton_sea.sta',
        ckpt_path = 'osdf:///ndp/public/ucr_seis/models/8700_17-319.ckpt',
        out_dir   = 'picks',
        time_range= '20120823-20120831'
    )
"""

import os
import glob
import time
import warnings
from io import BytesIO
import numpy as np
import torch
import torch.nn.functional as F
from obspy import UTCDateTime, read, Stream
from models import SAR

warnings.filterwarnings('ignore')

# ── Config (Salton Sea) ───────────────────────────────────────────────────────
SAMP_RATE       = 100
WIN_LEN         = 30        # seconds
WIN_STRIDE      = 15        # seconds — 50% overlap
NUM_CHN         = 3
FREQ_BAND       = [2, 45]   # Hz
STEP_LEN        = 0.5       # seconds
STEP_STRIDE     = 0.1       # seconds
NUM_STEPS       = int((WIN_LEN - STEP_LEN) / STEP_STRIDE) + 1   # 296
INPUT_SIZE      = int(STEP_LEN * SAMP_RATE) * NUM_CHN            # 150
TRIG_THRES      = 0.3
PICKER_BATCH    = 20
TP_DEV          = 1.5
TS_DEV          = 1.5
AMP_WIN         = [1, 6]
RM_GLITCH       = True
WIN_PEAK        = 1
AMP_RATIO_THRES = [5, 8, 3]
GLOBAL_MAX_NORM = False

# Derived
WIN_LEN_NPTS     = int(WIN_LEN     * SAMP_RATE)
WIN_STRIDE_NPTS  = int(WIN_STRIDE  * SAMP_RATE)
STEP_LEN_NPTS    = int(STEP_LEN    * SAMP_RATE)
STEP_STRIDE_NPTS = int(STEP_STRIDE * SAMP_RATE)
AMP_WIN_NPTS     = int(sum(AMP_WIN) * SAMP_RATE)
WIN_PEAK_NPTS    = int(WIN_PEAK    * SAMP_RATE)


# ── Pelican helpers ───────────────────────────────────────────────────────────

def _is_remote(path):
    """Return True if path is a Pelican/S3 URL."""
    return str(path).startswith('osdf://') or str(path).startswith('s3://')


def _get_fs(data_dir):
    """Return (fs, base_path) — fsspec filesystem + base path."""
    if _is_remote(data_dir):
        import fsspec
        fs = fsspec.filesystem('osdf')
        return fs, str(data_dir)
    return None, str(data_dir)


# ── Station file reader ───────────────────────────────────────────────────────

def get_sta_dict(sta_file):
    """Read station file → dict {NET.STA: [lat, lon, ele, gain]}"""
    sta_dict = {}
    with open(sta_file) as f:
        lines = f.readlines()
    for line in lines:
        codes = line.strip().split(',')
        if len(codes) < 4:
            continue
        net_sta = codes[0]
        lat, lon, ele = float(codes[1]), float(codes[2]), float(codes[3])
        if len(codes[4:]) == 1:
            gain = float(codes[4])
        elif len(codes[4:]) == 3:
            gain = [float(c) for c in codes[4:7]]
        else:
            gain = 1.0
        sta_dict[net_sta] = [lat, lon, ele, gain]
    print(f'Loaded {len(sta_dict)} stations from {sta_file}')
    return sta_dict


# ── Data reader ───────────────────────────────────────────────────────────────

def get_data_dict(date, data_dir, fs=None):
    """
    Get mseed file paths (or remote paths) for a given date.
    Expects: data_dir/YYYYMMDD/NET.STA.LOC.CHA.mseed
    Returns: dict {NET.STA: [path1, path2, path3]}
    """
    data_dict = {}
    date_code = '{:04d}{:02d}{:02d}'.format(date.year, date.month, date.day)
    day_dir   = os.path.join(data_dir, date_code) if fs is None \
                else f'{data_dir}/{date_code}'

    if fs is not None:
        # Remote — list via fsspec
        try:
            entries = fs.ls(day_dir)
            all_paths = []
            for e in entries:
                raw = e['name'] if isinstance(e, dict) else e
                # fs.ls() strips the osdf:// scheme — add it back
                if not raw.startswith('osdf://'):
                    raw = 'osdf://' + raw
                if raw.endswith('.mseed'):
                    all_paths.append(raw)
        except Exception as e:
            print(f'  Cannot list {day_dir}: {e}')
            return {}
    else:
        # Local
        all_paths = sorted(glob.glob(os.path.join(day_dir, '*.mseed')))

    for path in all_paths:
        fname   = path.rstrip('/').split('/')[-1]
        net_sta = '.'.join(fname.split('.')[:2])
        if net_sta in data_dict:
            data_dict[net_sta].append(path)
        else:
            data_dict[net_sta] = [path]

    # Keep only stations with exactly 3 components
    todel = [k for k, v in data_dict.items() if len(v) != 3]
    for k in todel:
        data_dict.pop(k)
    return data_dict


def read_data(st_paths, sta_dict, fs=None):
    """Read 3-component MiniSEED and apply gain correction."""
    try:
        st = Stream()
        for path in st_paths[:3]:
            if fs is not None:
                with fs.open(path, 'rb') as f:
                    st += read(BytesIO(f.read()))
            else:
                st += read(path)
    except Exception as e:
        print(f'  Bad data: {e}')
        return []

    fname   = st_paths[0].rstrip('/').split('/')[-1]
    net, sta = fname.split('.')[:2]
    net_sta  = f'{net}.{sta}'
    gain     = sta_dict[net_sta][3]

    start_time = max(tr.stats.starttime for tr in st)
    end_time   = min(tr.stats.endtime   for tr in st)
    st_time    = start_time + (end_time - start_time) / 2

    for ii in range(3):
        st[ii].stats.network = net
        st[ii].stats.station = sta

    # Apply gain
    if isinstance(gain, float):
        for ii in range(3):
            st[ii].data = st[ii].data / gain
    elif isinstance(gain, list) and isinstance(gain[0], float):
        for ii in range(3):
            st[ii].data = st[ii].data / gain[ii]
    elif isinstance(gain, list) and isinstance(gain[0], list):
        for [ge, gn, gz, t0, t1] in gain:
            if t0 < st_time < t1:
                break
        for ii in range(3):
            st[ii].data = st[ii].data / [ge, gn, gz][ii]
    return st


# ── SAR Picker ────────────────────────────────────────────────────────────────

class SAR_Picker:
    """
    SAR phase picker — runs on continuous MiniSEED data.

    Args:
        ckpt_path : path to .ckpt checkpoint file.
                    Accepts local paths or Pelican OSDF URLs
                    (e.g. 'osdf:///ndp/public/ucr_seis/models/8700_17-319.ckpt').
        gpu_idx   : GPU index (0, 1, ...) or -1 for CPU
    """

    def __init__(self, ckpt_path, gpu_idx=-1):
        # Auto-detect device
        if gpu_idx >= 0 and torch.cuda.is_available():
            self.device = torch.device(f'cuda:{gpu_idx}')
            print(f'Device: CUDA:{gpu_idx}')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print('Device: Apple MPS')
        else:
            self.device = torch.device('cpu')
            print('Device: CPU')

        # Load model — supports local paths and Pelican OSDF / S3 URLs
        self.model = SAR()
        ckpt_str   = str(ckpt_path)
        if _is_remote(ckpt_str):
            import fsspec
            with fsspec.open(ckpt_str, 'rb') as f:
                state_dict = torch.load(f, map_location=self.device)
        else:
            state_dict = torch.load(ckpt_str, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        print(f'Checkpoint loaded: {ckpt_path}')

    def pick(self, stream, fout=None):
        """Pick P/S arrivals from a 3-component ObsPy stream."""
        t0 = time.time()

        # 1. Preprocess
        stream, st_raw = self._preprocess(stream)
        if len(stream) != NUM_CHN:
            return []

        start_time = stream[0].stats.starttime + WIN_STRIDE
        end_time   = stream[0].stats.endtime
        if end_time < start_time + WIN_LEN:
            return []

        stream  = stream.slice(start_time, end_time)
        st_raw  = st_raw.slice(start_time, end_time)
        net_sta = f'{stream[0].stats.network}.{stream[0].stats.station}'
        num_win = int((end_time - start_time - WIN_LEN) / WIN_STRIDE) + 1

        st_len  = min(len(tr) for tr in stream)
        st_data = np.array([tr.data[:st_len] for tr in stream], dtype=np.float32)
        st_cuda = torch.from_numpy(st_data).to(self.device)

        # Miss channel detection
        st_raw_npts  = min(len(tr) for tr in st_raw)
        st_raw_data  = np.array([tr.data[:st_raw_npts] for tr in st_raw])
        raw_stride   = int(st_raw[0].stats.sampling_rate * WIN_STRIDE)
        raw_win_npts = int(st_raw[0].stats.sampling_rate * WIN_LEN)
        miss_chn     = np.array([
            np.sum(st_raw_data[:, i*raw_stride: i*raw_stride+raw_win_npts] == 0,
                   axis=1) > WIN_LEN_NPTS / 4
            for i in range(num_win)])

        # 2. Run SAR
        picks_raw = self._run_sar(st_cuda, start_time, num_win, miss_chn)

        # 3. Merge overlapping picks from sliding windows
        to_drop = []
        for ii in range(len(picks_raw)):
            is_nbr = ((np.abs(picks_raw['tp'] - picks_raw['tp'][ii]) < TP_DEV) &
                      (np.abs(picks_raw['ts'] - picks_raw['ts'][ii]) < TS_DEV))
            if sum(is_nbr) == 1:
                continue
            prob_max = np.amax(picks_raw[is_nbr]['p_prob'] +
                               picks_raw[is_nbr]['s_prob'])
            if picks_raw[ii]['p_prob'] + picks_raw[ii]['s_prob'] != prob_max:
                to_drop.append(ii)
        picks_raw = np.delete(picks_raw, to_drop)

        # 4. Get amplitude & glitch removal
        picks = []
        for tp, ts, p_prob, s_prob in picks_raw:
            seg   = stream.slice(tp - AMP_WIN[0], ts + AMP_WIN[1]).copy()
            amp_d = np.array([tr.data[:AMP_WIN_NPTS] for tr in seg])
            s_amp = self._get_s_amp(amp_d)
            if RM_GLITCH and self._remove_glitch(stream, tp, ts):
                continue
            picks.append([net_sta, tp, ts, s_amp, p_prob, s_prob])
            if fout:
                fout.write(f'{net_sta},{tp},{ts},{s_amp},{p_prob:.2f},{s_prob:.2f}\n')

        print(f'  {net_sta}: {len(picks)} picks | {time.time()-t0:.1f}s')
        return picks

    def _run_sar(self, st_cuda, start_time, num_win, miss_chn):
        """Run SAR model over sliding windows."""
        dtype     = [('tp','O'), ('ts','O'), ('p_prob','O'), ('s_prob','O')]
        picks_raw = []
        num_batch = int(np.ceil(num_win / PICKER_BATCH))

        for batch_idx in range(num_batch):
            n_win = PICKER_BATCH if batch_idx < num_batch - 1 else num_win % PICKER_BATCH
            if n_win == 0:
                n_win = PICKER_BATCH
            win_idx_list = [nn + batch_idx * PICKER_BATCH for nn in range(n_win)]
            data_seq     = self._st2seq(st_cuda, win_idx_list, miss_chn)

            with torch.no_grad():
                pred_logits = self.model(data_seq)
            pred_probs = F.softmax(pred_logits, dim=-1).detach().cpu().numpy()

            for nn, pred_prob in enumerate(pred_probs):
                win_idx = nn + batch_idx * PICKER_BATCH
                t0      = start_time + win_idx * WIN_STRIDE
                if sum(miss_chn[win_idx]) == 3:
                    continue
                pp = pred_prob[:, 1].copy()
                ps = pred_prob[:, 2].copy()
                pp[np.isnan(pp)] = 0
                ps[np.isnan(ps)] = 0
                if min(np.amax(pp), np.amax(ps)) < TRIG_THRES:
                    continue
                p_idxs = np.where(pp >= TRIG_THRES)[0]
                s_idxs = np.where(ps >= TRIG_THRES)[0]
                p_dets = np.split(p_idxs, np.where(np.diff(p_idxs) != 1)[0] + 1)
                s_dets = np.split(s_idxs, np.where(np.diff(s_idxs) != 1)[0] + 1)
                p_probs = [np.amax(pp[d]) for d in p_dets]
                s_probs = [np.amax(ps[d]) for d in s_dets]
                p_idxs  = [np.median(d) for d in p_dets]
                s_idxs  = [np.median(d) for d in s_dets]
                for ii, p_idx in enumerate(p_idxs):
                    tp = t0 + STEP_LEN / 2 + STEP_STRIDE * p_idx
                    for jj, s_idx in enumerate(s_idxs):
                        if s_idx <= p_idx:
                            continue
                        ts = t0 + STEP_LEN / 2 + STEP_STRIDE * s_idx
                        picks_raw.append((tp, ts, p_probs[ii], s_probs[jj]))

        return np.array(picks_raw, dtype=dtype)

    def _st2seq(self, st_cuda, win_idx_list, miss_chn):
        """Convert stream windows to input sequences for the model."""
        num_win  = len(win_idx_list)
        data_seq = torch.zeros(
            (num_win, NUM_STEPS, NUM_CHN * STEP_LEN_NPTS),
            dtype=torch.float32, device=self.device)
        for i, win_idx in enumerate(win_idx_list):
            s        = win_idx * WIN_STRIDE_NPTS
            win_data = st_cuda[:, s: s + WIN_LEN_NPTS].clone()
            win_data = self._preprocess_cuda(win_data, miss_chn[win_idx])
            win_data = (win_data
                        .unfold(1, STEP_LEN_NPTS, STEP_STRIDE_NPTS)
                        .permute(1, 0, 2))
            data_seq[i] = win_data.reshape(win_data.size(0), -1)
        return data_seq

    def _preprocess(self, st, max_gap=5.):
        """Preprocess stream: align, resample, filter."""
        if len(st) != NUM_CHN:
            return [], []
        start_time = max(tr.stats.starttime for tr in st)
        end_time   = min(tr.stats.endtime   for tr in st)
        if end_time < start_time + WIN_LEN:
            return [], []
        st = st.slice(start_time, end_time, nearest_sample=True)
        if len(st) != NUM_CHN:
            return [], []
        for tr in st:
            tr.data[np.isnan(tr.data)] = 0
            tr.data[np.isinf(tr.data)] = 0
        if max(st.max()) == 0:
            return [], []
        st_raw = st.copy()
        # Fill data gaps
        max_gap_npts = int(max_gap * SAMP_RATE)
        for tr in st:
            npts     = len(tr.data)
            gap_idx  = np.where(np.diff(tr.data) == 0)[0]
            gap_list = np.split(gap_idx, np.where(np.diff(gap_idx) != 1)[0] + 1)
            gap_list = [g for g in gap_list if len(g) >= 10]
            num_gap  = len(gap_list)
            for ii, gap in enumerate(gap_list):
                idx0 = max(0, gap[0] - 1)
                idx1 = min(npts - 1, gap[-1] + 1)
                if ii < num_gap - 1:
                    idx2 = min(idx1 + (idx1 - idx0),
                               idx1 + max_gap_npts,
                               gap_list[ii + 1][0])
                else:
                    idx2 = min(idx1 + (idx1 - idx0),
                               idx1 + max_gap_npts, npts - 1)
                if idx1 == idx2:
                    continue
                if idx2 == idx1 + (idx1 - idx0):
                    tr.data[idx0:idx1] = tr.data[idx1:idx2]
                else:
                    num_tile = int(np.ceil((idx1 - idx0) / (idx2 - idx1)))
                    tr.data[idx0:idx1] = np.tile(
                        tr.data[idx1:idx2], num_tile)[:idx1 - idx0]
        # Resample + filter
        st = (st.detrend('demean')
               .detrend('linear')
               .taper(max_percentage=0.05, max_length=5.))
        if st[0].stats.sampling_rate != SAMP_RATE:
            st.resample(SAMP_RATE)
        fmin, fmax = FREQ_BAND
        st = st.filter('bandpass', freqmin=fmin, freqmax=fmax)
        return st, st_raw

    def _preprocess_cuda(self, data, is_miss):
        """Normalize window data on GPU."""
        if 0 < sum(is_miss) < 3:
            data[is_miss] = data[~is_miss][-1]
        data -= torch.mean(data, dim=1, keepdim=True)
        if GLOBAL_MAX_NORM:
            data /= torch.max(torch.abs(data))
        else:
            data /= torch.max(torch.abs(data), dim=1).values.view(NUM_CHN, 1)
        return data

    def _get_s_amp(self, velo):
        """Compute S-wave amplitude from velocity trace."""
        velo = velo - np.mean(velo, axis=1, keepdims=True)
        disp = np.cumsum(velo, axis=1) / SAMP_RATE
        return float(np.amax(np.sum(disp ** 2, axis=0)) ** 0.5)

    def _remove_glitch(self, stream, tp, ts):
        """Glitch removal based on amplitude ratios."""
        p_ratio = self._calc_peak_amp_ratio(stream.slice(tp, tp + WIN_PEAK * 3))
        if np.amin(p_ratio) > AMP_RATIO_THRES[0]:
            return True
        s_ratio = self._calc_peak_amp_ratio(stream.slice(ts, ts + WIN_PEAK * 3))
        if np.amin(s_ratio) > AMP_RATIO_THRES[0]:
            return True
        half = (ts - tp) / 2
        A1 = np.array([np.ptp(tr.data) for tr in stream.slice(tp, tp + half)])
        A2 = np.array([np.ptp(tr.data) for tr in stream.slice(tp + half, ts)])
        A3 = np.array([np.ptp(tr.data) for tr in stream.slice(ts, ts + half)])
        A12 = min(A1[ii] / A2[ii] for ii in range(3))
        A13 = min(A1[ii] / A3[ii] for ii in range(3))
        return not (A12 < AMP_RATIO_THRES[1] and A13 < AMP_RATIO_THRES[2])

    def _calc_peak_amp_ratio(self, st):
        peak_data = np.array([np.abs(tr.data[:WIN_PEAK_NPTS]) for tr in st])
        chn_idx   = np.unravel_index(np.argmax(peak_data), peak_data.shape)[0]
        idx0      = np.argmax(np.abs(st[chn_idx].data[:WIN_PEAK_NPTS]))
        idx1      = idx0 + self._find_first_peak(st[chn_idx].data[idx0:])
        idx0     -= self._find_second_peak(st[chn_idx].data[:idx0][::-1])
        idx1     += self._find_second_peak(st[chn_idx].data[idx1:]) + 1
        idx0      = max(0, idx0)
        ratios = []
        for tr in st:
            amp_peak = np.ptp(tr.data[idx0:idx1])
            amp_tail = np.ptp(tr.data[idx1: 2 * idx1 - idx0])
            ratios.append(amp_peak / (amp_tail + 1e-9))
        return ratios

    def _find_first_peak(self, data):
        npts = len(data)
        if npts < 2:
            return 0
        dd = np.diff(data)
        if dd.min() >= 0 or dd.max() <= 0:
            return 0
        return max(np.where(dd < 0)[0][0], np.where(dd >= 0)[0][0])

    def _find_second_peak(self, data):
        npts = len(data)
        if npts < 2:
            return 0
        dd = np.diff(data)
        if dd.min() >= 0 or dd.max() <= 0:
            return 0
        neg = np.where(dd < 0)[0]
        pos = np.where(dd >= 0)[0]
        if len(neg) == 0 or len(pos) == 0:
            return 0
        first = max(neg[0], pos[0])
        neg2  = neg[neg > first]
        pos2  = pos[pos > first]
        if len(neg2) == 0 or len(pos2) == 0:
            return first
        return max(neg2[0], pos2[0])


# ── Main picking function (called from notebook) ──────────────────────────────

def run_picking(data_dir, sta_file, ckpt_path, out_dir,
                time_range, gpu_idx=-1, num_workers=1):
    """
    Run SAR phase picking on continuous MiniSEED data.

    Args:
        data_dir   : path to MiniSEED data (YYYYMMDD/NET.STA.LOC.CHA.mseed)
                     Accepts local path or Pelican OSDF URL.
        sta_file   : path to station file (.sta)
        ckpt_path  : path to model checkpoint (.ckpt) — local or Pelican OSDF URL
        out_dir    : output directory for .pick files
        time_range : 'YYYYMMDD-YYYYMMDD' e.g. '20120823-20120831'
        gpu_idx    : GPU index or -1 for CPU
        num_workers: number of parallel workers (1 = serial)

    Output:
        out_dir/YYYY-MM-DD.pick — one file per day
        Each line: NET.STA, tp, ts, s_amp, p_prob, s_prob
    """
    os.makedirs(out_dir, exist_ok=True)

    # Setup filesystem (None = local, fsspec fs = remote)
    fs, base_dir = _get_fs(data_dir)

    # Setup picker and stations
    picker   = SAR_Picker(ckpt_path, gpu_idx=gpu_idx)
    sta_dict = get_sta_dict(sta_file)

    # Date range
    start_str, end_str = time_range.split('-')
    start_time = UTCDateTime(start_str)
    end_time   = UTCDateTime(end_str)
    num_days   = int((end_time - start_time) / 86400)
    date_list  = [start_time + 86400 * i for i in range(num_days)]

    src_label = 'Pelican OSDF' if fs is not None else base_dir
    print(f'\n{"="*60}')
    print(f'SAR Phase Picking — Salton Sea')
    print(f'Days      : {num_days} ({start_str} -> {end_str})')
    print(f'Stations  : {len(sta_dict)}')
    print(f'Data src  : {src_label}')
    print(f'Output    : {out_dir}')
    print(f'{"="*60}\n')

    total_picks  = 0
    daily_counts = {}

    for date in date_list:
        date_str  = str(date.date)
        pick_file = os.path.join(out_dir, f'{date_str}.pick')

        # Skip if already done
        if os.path.exists(pick_file) and os.path.getsize(pick_file) > 0:
            with open(pick_file) as f_in:
                n = sum(1 for line in f_in if line.strip())
            print(f'{date_str}: {n} picks (already done — skipping)')
            daily_counts[date_str] = n
            total_picks += n
            continue

        print(f'\n── {date_str} ──────────────────────────────')
        data_dict = get_data_dict(date, base_dir, fs=fs)
        day_picks = 0

        with open(pick_file, 'w') as fout:
            for net_sta, data_paths in data_dict.items():
                if net_sta not in sta_dict:
                    continue
                st = read_data(data_paths, sta_dict, fs=fs)
                if len(st) == 0:
                    continue
                picks = picker.pick(st, fout)
                day_picks += len(picks)

        daily_counts[date_str] = day_picks
        total_picks += day_picks
        print(f'{date_str}: {day_picks} picks total')

    # Summary
    print(f'\n{"="*60}')
    print(f'PICKING COMPLETE')
    print(f'{"="*60}')
    print(f'{"Date":<15} {"Picks":>8}')
    print(f'{"-"*25}')
    for date_str, count in daily_counts.items():
        print(f'{date_str:<15} {count:>8,}')
    print(f'{"-"*25}')
    print(f'{"TOTAL":<15} {total_picks:>8,}')
    print(f'{"="*60}')
    print(f'Pick files saved to: {out_dir}')

    return daily_counts, total_picks


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print('Usage: python picker.py <data_dir> <sta_file> <ckpt_path> <out_dir> <time_range>')
        sys.exit(1)
    run_picking(
        data_dir   = sys.argv[1],
        sta_file   = sys.argv[2],
        ckpt_path  = sys.argv[3],
        out_dir    = sys.argv[4],
        time_range = sys.argv[5] if len(sys.argv) > 5 else '20120826-20120827'
    )