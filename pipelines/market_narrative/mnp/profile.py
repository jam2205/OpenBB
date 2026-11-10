"""Where a pair stands inside its day and its week, and how often that state has held.

Pure functions over one pair's hourly bars (time, high, low, close); no network, no models.

What this is: conditional frequencies from the pair's own history, each printed beside the
rate it must beat. "The day's low is in" is judged like this: on past days that, at the same
hour, had used a similar share of their normal range and whose low was about as old, how often
was that low the final low? The comparison is the same hour with no conditions.

Two comparisons are printed, and they answer different questions:
  * the plain rate at the same hour, no conditions: is this state better than usual?
  * the same state on a volatility-matched random walk (`null`): is the MARKET doing anything
    a coin would not? An old extreme with most of the range used holds often on a pure random
    walk too (a test on synthetic noise gave 89% against a 73% plain rate), so a high rate is a
    usable probability but not, by itself, evidence of structure. Only a rate outside the
    random-walk band is worth a research ticket.

The random walk is built from the pair's own bars: each hourly bar is mirrored around its open
with probability one half, which keeps every hour's volatility and removes any directional
pattern. The same day/hour/condition statistic is then recomputed on each mirrored history.

A second walk (`drift`) mirrors each bar around the pair's average hourly move instead, so the
long-run drift is kept and only the asymmetry between up and down moves is removed. A rate
outside the plain band but inside the drift band is the pair's trend showing; outside both, it
is skew or reversal. First seen on real data 2026-11-10/11: lows on pairs quoted against JPY
and CHF held 3-6 points more often than the plain walk at three different hours.

What this is not: a tested claim. One observation per day, counted in days and in distinct
weeks, read only from bars that had closed. Nothing here has been through a registered run
with a control. It says nothing about direction beyond which extreme is more likely to stand.

Conventions: the trading day rolls at 17:00 New York (the FX close); hour 0 is the bar that
opens at the roll. A day counts as complete with at least 20 hourly bars. ADR20 is the mean
range of the 20 complete days BEFORE the day it is attached to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLL_TZ, ROLL_HOUR = "America/New_York", 17
MIN_BARS = 20
USED_EDGES = [0.0, 0.4, 0.7, 1.0, np.inf]           # share of ADR20 already travelled
USED_LABELS = ["under 40%", "40-70%", "70-100%", "over 100%"]
AGE_EDGES = [-1, 1, 5, 99]                           # hours since the extreme was made
AGE_LABELS = ["0-1h old", "2-5h old", "6h+ old"]
SESSIONS = {"Asia (17:00-02:00 NY)": range(0, 9), "London (02:00-08:00 NY)": range(9, 15),
            "New York (08:00-17:00 NY)": range(15, 24)}
MIN_CELL_DAYS = 60
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def _prep(bars: pd.DataFrame) -> pd.DataFrame:
    """Add the trading day (labelled by the date it ends) and the hour index since the roll."""
    b = bars.dropna(subset=["high", "low"]).copy()
    t = pd.to_datetime(b["time"], utc=True)
    wall = t.dt.tz_convert(ROLL_TZ).dt.tz_localize(None)
    b["time"] = t
    b["day"] = (wall + pd.Timedelta(hours=24 - ROLL_HOUR)).dt.normalize()
    b["k"] = (wall.dt.hour - ROLL_HOUR) % 24
    b = b[b["day"].dt.weekday < 5]                    # a stray weekend bar is not a session
    return b.sort_values("time").drop_duplicates(["day", "k"], keep="last")


def _grid(b: pd.DataFrame):
    """(days, row index of each bar, hour index of each bar)."""
    days, di = np.unique(b["day"].to_numpy(), return_inverse=True)
    return pd.DatetimeIndex(days), di, b["k"].to_numpy()


def _matrices(n_days: int, di: np.ndarray, kk: np.ndarray, high: np.ndarray, low: np.ndarray):
    """days x 24 running high / low and the hours since each was last extended."""
    hi, lo = np.full((n_days, 24), np.nan), np.full((n_days, 24), np.nan)
    hi[di, kk], lo[di, kk] = high, low
    H = np.fmax.accumulate(hi, axis=1)
    L = np.fmin.accumulate(lo, axis=1)
    k = np.arange(24)

    def age(run, better):
        prev = np.concatenate([np.full((len(run), 1), np.nan), run[:, :-1]], axis=1)
        new = np.isnan(prev) | better(run, prev)
        last = np.maximum.accumulate(np.where(new, k, -1), axis=1)
        return k - last

    return (~np.isnan(hi)).sum(axis=1), H, L, age(H, np.greater), age(L, np.less)


def _adr(rng: np.ndarray, complete: np.ndarray, days) -> np.ndarray:
    """Mean range of the 20 complete days before each day (the day itself never included)."""
    return pd.Series(np.where(complete, rng, np.nan), index=days).dropna().rolling(20).mean().reindex(days).ffill().shift(1).to_numpy()


def _cells(H, L, ageH, ageL, adr, ok, k_now: int, ub: int, ab_low: int | None, ab_high: int | None):
    """Boolean day masks for "same range-used bucket (and same age bucket) at hour k_now"."""
    used = (H[:, k_now] - L[:, k_now]) / adr
    in_u = ok & (np.searchsorted(USED_EDGES, used, side="right") - 1 == ub)
    low = in_u if ab_low is None else in_u & (np.searchsorted(AGE_EDGES, ageL[:, k_now], side="left") - 1 == ab_low)
    high = in_u if ab_high is None else in_u & (np.searchsorted(AGE_EDGES, ageH[:, k_now], side="left") - 1 == ab_high)
    return in_u, low, high


def _mirrored(b: pd.DataFrame, rng: np.random.Generator, keep_drift: bool = False):
    """(high, low) of a history in which each bar is mirrored half the time.

    Plain: mirrored around its open, so up and down become equally likely and the pair's
    long-run drift is removed. `keep_drift`: mirrored around the pair's average hourly move, so
    the drift survives exactly and only the asymmetry between up and down moves (skew, reversal)
    is removed. Comparing the two says whether a tilt is drift or something else."""
    c, h, l = np.log(b["close"].to_numpy(float)), np.log(b["high"].to_numpy(float)), np.log(b["low"].to_numpy(float))
    o = np.log(b["open"].to_numpy(float)) if "open" in b else np.r_[c[0], c[:-1]]
    gap = np.r_[0.0, o[1:] - c[:-1]]
    flip = rng.random(len(c)) < 0.5
    m2 = 2 * float(np.mean(c - o)) if keep_drift else 0.0
    rc = np.where(flip, m2 - (c - o), c - o)
    rh = np.where(flip, m2 - (l - o), h - o)
    rl = np.where(flip, m2 - (h - o), l - o)
    rh, rl = np.maximum(rh, np.maximum(rc, 0.0)), np.minimum(rl, np.minimum(rc, 0.0))   # a bar still contains its open and close
    close = o[0] + np.cumsum(gap + rc)
    open_ = np.r_[o[0], close[:-1] + gap[1:]]
    return np.exp(open_ + rh), np.exp(open_ + rl)


def _null_band(b, days, di, kk, today: int, k_now: int, ub: int, ab_low, ab_high, n: int, seed: int,
               keep_drift: bool = False, key: str = "null") -> dict:
    """Hold rates of the same state on `n` volatility-matched random walks: mean and 2.5-97.5% band."""
    if n <= 0 or "close" not in b:
        return {}
    rng, lows, highs = np.random.default_rng(seed), [], []
    for _ in range(n):
        nh, nl = _mirrored(b, rng, keep_drift)
        nbars, H, L, ageH, ageL = _matrices(len(days), di, kk, nh, nl)
        complete = nbars >= MIN_BARS
        hist = complete.copy()
        hist[today] = False
        adr = _adr(H[:, -1] - L[:, -1], complete, days)
        ok = hist & ~np.isnan(adr) & ~np.isnan(H[:, k_now]) & ~np.isnan(L[:, k_now])
        _, cl, ch = _cells(H, L, ageH, ageL, adr, ok, k_now, ub, ab_low, ab_high)
        if cl.sum() >= 20:
            lows.append(float((L[cl, k_now] <= L[cl, -1]).mean()))
        if ch.sum() >= 20:
            highs.append(float((H[ch, k_now] >= H[ch, -1]).mean()))

    def band(x):
        return {key: float(np.mean(x)), f"{key}_low": float(np.quantile(x, 0.025)), f"{key}_high": float(np.quantile(x, 0.975))} if len(x) >= 5 else {}

    return {"low": band(lows), "high": band(highs)}


def _wilson(p: float, n: int) -> tuple[float, float]:
    if not n or np.isnan(p):
        return (np.nan, np.nan)
    z = 1.96
    d = 1 + z * z / n
    c, h = p + z * z / (2 * n), z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max((c - h) / d, 0.0), min((c + h) / d, 1.0))


def _bucket(x: float, edges, labels, side: str = "right") -> tuple[int, str]:
    i = int(np.clip(np.searchsorted(edges, x, side=side) - 1, 0, len(labels) - 1))
    return i, labels[i]


def _hold(final_ok: np.ndarray, cond: np.ndarray, weeks: np.ndarray, base_mask: np.ndarray) -> dict:
    """Rate at which the running extreme was final, in the matched cell and for all days."""
    n, nw = int(cond.sum()), int(len(np.unique(weeks[cond])))
    p = float(final_ok[cond].mean()) if n else np.nan
    p0 = float(final_ok[base_mask].mean()) if base_mask.any() else np.nan
    lo, hi = _wilson(p, nw)                           # distinct weeks: days in a week are not independent
    return {"p": p, "baseline": p0, "lift_pp": (p - p0) * 100, "n_days": n, "n_weeks": nw, "ci_low": lo, "ci_high": hi}


def day_profile(bars: pd.DataFrame, now: pd.Timestamp | None = None, ref_range: float | None = None,
                null_runs: int = 40, seed: int = 0) -> dict:
    """State of the current trading day and how often each extreme held from here.

    `now` defaults to the close of the last bar; bars that open at or after it are ignored, so
    nothing unfinished is read. `ref_range` is today's expected range in price units (a range
    forecast); ADR20 is used when it is missing. Returns {} when there is too little history.
    """
    b = _prep(bars)
    if now is not None:
        b = b[b["time"] + pd.Timedelta(hours=1) <= pd.Timestamp(now).tz_convert("UTC")]
    if b.empty:
        return {}
    days, di, kk = _grid(b)
    nbars, H, L, ageH, ageL = _matrices(len(days), di, kk, b["high"].to_numpy(float), b["low"].to_numpy(float))
    rng = H[:, -1] - L[:, -1]
    complete = nbars >= MIN_BARS
    today = len(days) - 1
    k_now = int(b.loc[b["day"] == days[today], "k"].max())
    hist = complete.copy()
    hist[today] = False                               # today is never its own evidence
    if hist.sum() < 120:
        return {}
    adr = _adr(rng, complete, days)
    weeks = (days - pd.to_timedelta(days.weekday, unit="D")).to_numpy()
    ok = hist & ~np.isnan(adr) & ~np.isnan(H[:, k_now]) & ~np.isnan(L[:, k_now])
    lows_final = L[:, k_now] <= L[:, -1]
    highs_final = H[:, k_now] >= H[:, -1]

    u_now = float((H[today, k_now] - L[today, k_now]) / adr[today])
    ub, ulabel = _bucket(u_now, USED_EDGES, USED_LABELS)
    aL, aH = int(ageL[today, k_now]), int(ageH[today, k_now])
    (abL, lblL), (abH, lblH) = _bucket(aL, AGE_EDGES, AGE_LABELS, "left"), _bucket(aH, AGE_EDGES, AGE_LABELS, "left")
    in_u, cl, ch = _cells(H, L, ageH, ageL, adr, ok, k_now, ub, abL, abH)
    notes = {}
    for name, cell, lbl in (("low", cl, lblL), ("high", ch, lblH)):
        notes[name] = f"range used {ulabel}, extreme {lbl}"
        if cell.sum() < MIN_CELL_DAYS:                # too few days: drop the age condition and say so
            notes[name] = f"range used {ulabel} (age of the extreme dropped: too few days)"
            if name == "low":
                cl, abL = in_u, None
            else:
                ch, abH = in_u, None
    null = _null_band(b, days, di, kk, today, k_now, ub, abL, abH, null_runs, seed)
    drift = _null_band(b, days, di, kk, today, k_now, ub, abL, abH, null_runs, seed + 1, keep_drift=True, key="drift")

    def against(r, key, label):
        return (f"no {label} comparison" if key not in r else f"above a {label}" if r["p"] > r[f"{key}_high"] else
                f"below a {label}" if r["p"] < r[f"{key}_low"] else f"same as a {label}")

    def side(final, cell, age_h, name):
        r = {**_hold(final, cell, weeks, ok), "age_h": age_h, "matched_on": notes[name], **null.get(name, {}), **drift.get(name, {})}
        r["vs_null"], r["vs_drift"] = against(r, "null", "random walk"), against(r, "drift", "drifting walk")
        off, off_d = not r["vs_null"].startswith(("same", "no")), not r["vs_drift"].startswith(("same", "no"))
        r["tilt"] = ("none" if not off else "explained by the pair's drift" if not off_d else
                     "not explained by drift (skew or reversal in the moves)")
        return r

    low, high = side(lows_final, cl, aL, "low"), side(highs_final, ch, aH, "high")
    low_now, high_now = float(L[today, k_now]), float(H[today, k_now])
    adr_now = float(adr[today]) if not np.isnan(adr[today]) else np.nan
    ref = float(ref_range) if ref_range and ref_range > 0 else adr_now
    sess = {}
    kl = np.argmax(L[hist] <= L[hist][:, -1][:, None], axis=1)     # first hour the final low was in place
    kh = np.argmax(H[hist] >= H[hist][:, -1][:, None], axis=1)
    for name, hours in SESSIONS.items():
        sess[name] = {"low_set_%": float(np.isin(kl, list(hours)).mean() * 100), "high_set_%": float(np.isin(kh, list(hours)).mean() * 100)}
    by_hour = pd.DataFrame({"low_in_%": [np.nanmean(np.where(ok, L[:, k] <= L[:, -1], np.nan)) * 100 for k in range(24)],
                            "high_in_%": [np.nanmean(np.where(ok, H[:, k] >= H[:, -1], np.nan)) * 100 for k in range(24)]},
                           index=pd.Index(range(1, 25), name="hours_into_day"))
    return {
        "asof": b["time"].iloc[-1] + pd.Timedelta(hours=1), "day": pd.Timestamp(days[today]).date(),
        "hours_elapsed": k_now + 1, "hours_left": 23 - k_now, "last": float(b["close"].iloc[-1]) if "close" in b else np.nan,
        "low": low_now, "high": high_now, "range_so_far": high_now - low_now, "adr20": adr_now,
        "expected_range": ref, "expected_from": "range forecast" if ref_range and ref_range > 0 else "ADR20",
        "used_of_adr": u_now, "used_of_expected": (high_now - low_now) / ref if ref else np.nan,
        "room_left": max(ref - (high_now - low_now), 0.0) if ref else np.nan,
        "high_if_low_holds": low_now + ref if ref else np.nan, "low_if_high_holds": high_now - ref if ref else np.nan,
        "low_hold": low, "high_hold": high, "history_days": int(ok.sum()),
        "sessions": pd.DataFrame(sess).T.round(1), "by_hour": by_hour.round(1),
    }


def week_profile(bars: pd.DataFrame, now: pd.Timestamp | None = None) -> dict:
    """Which session day sets the weekly high and low, and how often the week's running extreme
    held from this point. Weeks need all five days complete to count as history."""
    b = _prep(bars)
    if now is not None:
        b = b[b["time"] + pd.Timedelta(hours=1) <= pd.Timestamp(now).tz_convert("UTC")]
    if b.empty:
        return {}
    g = b.groupby("day").agg(high=("high", "max"), low=("low", "min"), n=("k", "size"))
    last_day, last_k = g.index[-1], int(b.loc[b["day"] == b["day"].iloc[-1], "k"].max())
    g["week"] = g.index - pd.to_timedelta(g.index.weekday, unit="D")
    g["wd"] = g.index.weekday
    cur_week = g.loc[last_day, "week"]
    full = g[(g["n"] >= MIN_BARS) & (g["week"] != cur_week)]
    full = full[full.groupby("week")["wd"].transform("nunique") == 5]
    if full["week"].nunique() < 60:
        return {}
    hi = full.pivot(index="week", columns="wd", values="high").to_numpy(float)
    lo = full.pivot(index="week", columns="wd", values="low").to_numpy(float)
    day_hi, day_lo = hi.argmax(axis=1), lo.argmin(axis=1)
    table = pd.DataFrame({"sets_week_high_%": [float((day_hi == d).mean() * 100) for d in range(5)],
                          "sets_week_low_%": [float((day_lo == d).mean() * 100) for d in range(5)]}, index=WEEKDAYS).round(1)
    out = {"table": table, "weeks": int(len(hi)), "week_of": pd.Timestamp(cur_week).date()}
    cur = g[g["week"] == cur_week]
    done = cur[(cur.index < last_day) | (last_k >= 23)]          # only finished days count as "so far"
    d = len(done)
    out["days_done"] = d
    if 1 <= d <= 4 and done["wd"].tolist() == list(range(d)):
        j_hi, j_lo = int(done["high"].to_numpy().argmax()), int(done["low"].to_numpy().argmin())
        run_hi, run_lo = hi[:, :d].max(axis=1), lo[:, :d].min(axis=1)
        held_hi, held_lo = run_hi >= hi.max(axis=1), run_lo <= lo.min(axis=1)
        same_hi, same_lo = hi[:, :d].argmax(axis=1) == j_hi, lo[:, :d].argmin(axis=1) == j_lo

        def rate(held, same):
            n = int(same.sum())
            p, p0 = (float(held[same].mean()) if n else np.nan), float(held.mean())
            lo_, hi_ = _wilson(p, n)
            return {"p": p, "baseline": p0, "lift_pp": (p - p0) * 100, "n_weeks": n, "ci_low": lo_, "ci_high": hi_}

        out["high"] = {**rate(held_hi, same_hi), "level": float(done["high"].max()), "set_on": WEEKDAYS[j_hi]}
        out["low"] = {**rate(held_lo, same_lo), "level": float(done["low"].min()), "set_on": WEEKDAYS[j_lo]}
        out["through"] = WEEKDAYS[d - 1]
    return out


def verdict(side: dict, min_p: float = 0.70, min_lift_pp: float = 5.0, min_weeks: int = 60) -> str:
    """Plain reading of one hold rate. "likely in" needs a high rate, a clear margin over the
    unconditional rate at the same point, and enough distinct weeks behind it."""
    if not side or np.isnan(side.get("p", np.nan)):
        return "no reading"
    if side["n_weeks"] < min_weeks:
        return "too few past cases to say"
    if side["p"] >= min_p and side["lift_pp"] >= min_lift_pp and side.get("ci_low", 0) >= 0.5:
        return "likely in"
    if side["p"] <= 1 - min_p:
        return "likely to be broken"
    return "not established"
