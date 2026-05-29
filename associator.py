"""
associator.py — PAL Phase Association (Standalone)
====================================================
Salton Sea Geothermal Region
AI-PAL: Zhou et al. (2025), JGR Solid Earth

Associates P/S picks from multiple stations into located earthquake events.
All config embedded — no external config.py needed.

Usage (from notebook):
    from associator import run_association
    run_association(
        pick_dir   = 'picks',
        sta_file   = 'config/salton_sea.sta',
        out_dir    = 'output',
        time_range = '20120823-20120831'
    )
"""

import os
import warnings
import numpy as np
from obspy import UTCDateTime

warnings.filterwarnings('ignore')

# ── Config (Salton Sea) ───────────────────────────────────────────────────────
VP        = 5.92
VS        = 3.40
MIN_STA   = 3
OT_DEV    = 3.0
MAX_RES   = 1.5
MAX_DROP  = 1
XY_MARGIN = 0.1
XY_GRID   = 0.02
Z_GRIDS   = np.arange(1, 26, 1)


# ── Station file reader ───────────────────────────────────────────────────────

def get_sta_dict(sta_file):
    sta_dict = {}
    with open(sta_file) as f:
        for line in f:
            codes = line.strip().split(',')
            if len(codes) < 4:
                continue
            net_sta = codes[0]
            lat = float(codes[1])
            lon = float(codes[2])
            ele = float(codes[3])
            if len(codes[4:]) == 1:
                gain = float(codes[4])
            elif len(codes[4:]) == 3:
                gain = [float(c) for c in codes[4:7]]
            else:
                gain = 1.0
            sta_dict[net_sta] = [lat, lon, ele, gain]
    return sta_dict


# ── Pick file reader ──────────────────────────────────────────────────────────

def calc_ot(tp, ts, vp=VP, vs=VS):
    dist = (ts - tp) / (1/vs - 1/vp)
    return tp - dist / vp


def get_picks(date, pick_dir):
    dtype = [('net_sta', 'O'), ('sta_ot', 'O'),
             ('tp', 'O'), ('ts', 'O'), ('s_amp', 'O')]
    pick_path = os.path.join(pick_dir, str(date.date) + '.pick')
    if not os.path.exists(pick_path):
        return np.array([], dtype=dtype)
    picks = []
    with open(pick_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            codes   = line.split(',')
            net_sta = codes[0]
            tp      = UTCDateTime(codes[1])
            ts      = UTCDateTime(codes[2])
            s_amp   = float(codes[3])
            picks.append((net_sta, calc_ot(tp, ts), tp, ts, s_amp))
    return np.array(picks, dtype=dtype)


# ── PAL Associator ────────────────────────────────────────────────────────────

class PS_Pair_Assoc:

    def __init__(self, sta_dict,
                 vp=VP, ot_dev=OT_DEV, max_res=MAX_RES,
                 max_drop=MAX_DROP, min_sta=MIN_STA,
                 xy_margin=XY_MARGIN, xy_grid=XY_GRID,
                 z_grids=Z_GRIDS):
        self.sta_dict  = sta_dict
        self.vp        = vp
        self.ot_dev    = ot_dev
        self.max_res   = max_res
        self.max_drop  = max_drop
        self.min_sta   = min_sta
        self.xy_margin = xy_margin
        self.xy_grid   = xy_grid
        self.z_grids   = z_grids
        self.tt_dict   = self._calc_tt()

    def _calc_tt(self):
        lats = [v[0] for v in self.sta_dict.values()]
        lons = [v[1] for v in self.sta_dict.values()]
        lat_min = min(lats) - self.xy_margin * (max(lats) - min(lats))
        lat_max = max(lats) + self.xy_margin * (max(lats) - min(lats))
        lon_min = min(lons) - self.xy_margin * (max(lons) - min(lons))
        lon_max = max(lons) + self.xy_margin * (max(lons) - min(lons))
        lat_grids = np.arange(lat_min, lat_max, self.xy_grid)
        lon_grids = np.arange(lon_min, lon_max, self.xy_grid)
        tt_dict = {}
        for net_sta, (lat, lon, ele, _) in self.sta_dict.items():
            tt_dict[net_sta] = {}
            for z in self.z_grids:
                dep_km = z + ele / 1000.
                tt_mat = np.zeros((len(lat_grids), len(lon_grids)))
                for ii, glat in enumerate(lat_grids):
                    for jj, glon in enumerate(lon_grids):
                        dist_km = ((glat-lat)**2 + (glon-lon)**2)**0.5 * 111.
                        tt_mat[ii, jj] = (dist_km**2 + dep_km**2)**0.5 / self.vp
                tt_dict[net_sta][z] = tt_mat
        self.lat_grids = lat_grids
        self.lon_grids = lon_grids
        return tt_dict

    def associate(self, picks, out_ctlg=None, out_pha=None):
        events_loc, events_pick = [], []
        if len(picks) == 0:
            return events_loc, events_pick

        picks    = np.sort(picks, order='sta_ot')
        num_picks = len(picks)
        num_nbr  = np.zeros(num_picks)
        num_drop = np.zeros(num_picks)

        for ii in range(num_picks):
            num_nbr[ii] = sum(
                np.abs(picks['sta_ot'] - picks['sta_ot'][ii]) < self.ot_dev)

        for _ in range(num_picks):
            if np.amax(num_nbr) < self.min_sta:
                break
            ots    = picks['sta_ot']
            ot_i   = ots[np.argmax(num_nbr)]
            to_idx = np.where(np.abs(ots - ot_i) < self.ot_dev)[0]

            event_loc, event_pick, assoc_idx, drop_idx = self._assoc_loc(
                picks[to_idx])

            if len(event_loc) > 0:
                event_loc_mag = self._calc_mag(event_pick, event_loc)
                ot  = event_loc_mag['evt_ot']
                lat = event_loc_mag['evt_lat']
                lon = event_loc_mag['evt_lon']
                dep = event_loc_mag['evt_dep']
                mag = event_loc_mag['mag']
                res = event_loc_mag['res']
                print(f'{ot}  {lat:.3f}  {lon:.3f}  {dep:>5.1f}km  M{mag:.2f}  res={res:.2f}s')
                if out_ctlg:
                    self._write_catalog(event_loc_mag, out_ctlg)
                if out_pha:
                    self._write_phase(event_loc_mag, event_pick, out_pha)
                events_loc.append(event_loc_mag)
                events_pick.append(event_pick)

            drop_idx  = np.array(drop_idx,  dtype=np.int32) + to_idx[0]
            assoc_idx = np.array(assoc_idx, dtype=np.int32) + to_idx[0]
            num_drop[drop_idx] += 1
            to_del = np.unique(np.concatenate([
                assoc_idx,
                np.where(num_drop > self.max_drop)[0]]))
            to_renew = np.where(np.abs(ots - ot_i) < 2 * self.ot_dev)[0]
            for idx in to_renew:
                num_nbr[idx] -= sum(
                    np.abs(ots[to_del] - ots[idx]) < self.ot_dev)
            picks    = np.delete(picks,    to_del)
            num_nbr  = np.delete(num_nbr,  to_del)
            num_drop = np.delete(num_drop, to_del)

        return events_loc, events_pick

    def _assoc_loc(self, picks):
        ot = picks['sta_ot'][len(picks) // 2]
        n_lat = len(self.lat_grids)
        n_lon = len(self.lon_grids)
        n_dep = len(self.z_grids)
        count_3d = np.zeros((n_lat, n_lon, n_dep), dtype=np.float32)

        det_dict = {}
        for ii, pick in enumerate(picks):
            net_sta = pick['net_sta']
            if net_sta not in self.tt_dict:
                continue
            det_dict[f'{ii}_{net_sta}'] = {
                'tp': pick['tp'], 'ts': pick['ts'],
                's_amp': pick['s_amp'], 'net_sta': net_sta}
            for kk, z in enumerate(self.z_grids):
                tt_p = self.tt_dict[net_sta][z]
                res  = np.abs(pick['tp'] - ot - tt_p)
                count_3d[:, :, kk] += (res < self.max_res).astype(np.float32)

        if np.amax(count_3d) < self.min_sta:
            return [], [], [], []

        best_idx = np.unravel_index(np.argmax(count_3d), count_3d.shape)
        best_lat = self.lat_grids[best_idx[0]]
        best_lon = self.lon_grids[best_idx[1]]
        best_dep = self.z_grids[best_idx[2]]

        assoc_picks, assoc_idx, drop_idx = [], [], []
        for ii, (key, info) in enumerate(det_dict.items()):
            net_sta = info['net_sta']
            if net_sta not in self.tt_dict:
                continue
            tt_p = self.tt_dict[net_sta][best_dep][best_idx[0], best_idx[1]]
            res  = abs(float(info['tp']) - float(ot) - tt_p)
            if res < self.max_res:
                assoc_picks.append(info)
                assoc_idx.append(ii)
            else:
                drop_idx.append(ii)

        if len(assoc_picks) < self.min_sta:
            return [], [], [], drop_idx

        ot_refined = UTCDateTime(
            np.median([float(calc_ot(p['tp'], p['ts'])) for p in assoc_picks]))

        res_vals = []
        for p in assoc_picks:
            tt_p = self.tt_dict[p['net_sta']][best_dep][best_idx[0], best_idx[1]]
            res_vals.append(abs(float(p['tp']) - float(ot_refined) - tt_p))

        event_loc = {
            'evt_ot' : ot_refined,
            'evt_lat': best_lat,
            'evt_lon': best_lon,
            'evt_dep': best_dep,
            'res'    : float(np.mean(res_vals)) if res_vals else 0.0,
        }
        return event_loc, assoc_picks, assoc_idx, drop_idx

    def _calc_mag(self, event_pick, event_loc):
        mags = []
        for pick in event_pick:
            net_sta = pick['net_sta']
            if net_sta not in self.sta_dict:
                continue
            lat, lon, ele, _ = self.sta_dict[net_sta]
            dist_km = (((event_loc['evt_lat'] - lat)**2 +
                        (event_loc['evt_lon'] - lon)**2)**0.5 * 111)**2
            dist_km = (dist_km + event_loc['evt_dep']**2)**0.5
            s_amp   = float(pick['s_amp'])
            if s_amp > 0 and dist_km > 0:
                s_amp_mm = s_amp * 1000.0
                mags.append(np.log10(s_amp_mm)
                             + 1.11 * np.log10(dist_km)
                             + 0.00189 * dist_km
                             - 2.09)
        event_loc_mag        = dict(event_loc)
        event_loc_mag['mag'] = round(float(np.median(mags)), 2) if mags else -9.9
        return event_loc_mag

    def _write_catalog(self, e, out_ctlg):
        out_ctlg.write(
            f"{e['evt_ot']},{e['evt_lat']:.4f},{e['evt_lon']:.4f},"
            f"{e['evt_dep']},{e['mag']:.2f}\n")

    def _write_phase(self, e, event_pick, out_pha):
        out_pha.write(
            f"{e['evt_ot']},{e['evt_lat']:.4f},{e['evt_lon']:.4f},"
            f"{e['evt_dep']},{e['mag']:.2f}\n")
        for pick in event_pick:
            out_pha.write(
                f"{pick['net_sta']},{pick['tp']},{pick['ts']},{pick['s_amp']}\n")


# ── Main function ─────────────────────────────────────────────────────────────

def run_association(pick_dir, sta_file, out_dir, time_range,
                    min_sta=MIN_STA, vp=VP):
    os.makedirs(out_dir, exist_ok=True)

    sta_dict   = get_sta_dict(sta_file)
    associator = PS_Pair_Assoc(
        sta_dict=sta_dict, vp=vp, min_sta=min_sta,
        ot_dev=OT_DEV, max_res=MAX_RES, max_drop=MAX_DROP,
        xy_margin=XY_MARGIN, xy_grid=XY_GRID, z_grids=Z_GRIDS)

    start_str, end_str = time_range.split('-')
    start_time = UTCDateTime(start_str)
    end_time   = UTCDateTime(end_str)
    num_days   = int((end_time - start_time) / 86400)

    out_ctlg_path = os.path.join(out_dir, 'catalog.csv')
    out_pha_path  = os.path.join(out_dir, 'phase.dat')

    total_events = 0
    with open(out_ctlg_path, 'w') as out_ctlg, \
         open(out_pha_path,  'w') as out_pha:
        out_ctlg.write('ot,lat,lon,dep,mag\n')
        for day_idx in range(num_days):
            date  = start_time + day_idx * 86400
            picks = get_picks(date, pick_dir)
            if len(picks) == 0:
                continue
            picks = picks[np.array([p in sta_dict for p in picks['net_sta']])]
            events_loc, _ = associator.associate(picks, out_ctlg, out_pha)
            total_events += len(events_loc)

    return out_ctlg_path, out_pha_path, total_events


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 5:
        print('Usage: python associator.py <pick_dir> <sta_file> <out_dir> <time_range>')
        sys.exit(1)
    run_association(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])