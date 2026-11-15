"""
Walk-forward backtest on RESOLVED Polymarket tennis markets.

For every finished match we use:
  * the market price ~30 min before the start (Polymarket price history),
  * an Elo model built ONLY from results older than (start - lag_days)  -> no look-ahead,
  * the real result (market resolution).
Then the strategy rules are replayed bet by bet.

  python backtest.py fetch --max-events 1500    # slow once; resumable; saves data/backtest_dataset.csv
  python backtest.py run --min-edge 0.05 --mode kelly
  python backtest.py diag                       # does the model beat the market at all?
  python backtest.py sweep                      # parameter grid, train vs. unseen test split
"""
import argparse, os, time

import numpy as np
import pandas as pd
import requests

import tennis_core as tc

DS = os.path.join(tc.DATA_DIR, "backtest_dataset.csv")
COLS = ["event_id", "market_id", "start", "title", "surface", "p1", "p2", "price1", "model_p1",
        "form1", "form2", "n1", "n2", "winner", "volume"]


# ------------------------------------------------------------------ data collection
def fetch_closed_events(max_events):
    return tc.fetch_events(max_events, closed=True)


def price_before(token, ts, max_stale_h=24):
    """Last traded/mid price of `token` at or before unix time ts (None if no recent data)."""
    end, start = int(ts), int(ts - 3 * 86400)
    hist = []
    for fid in (15, 60, 360):
        try:
            r = requests.get(f"{tc.CLOB}/prices-history", timeout=20,
                             params={"market": token, "startTs": start, "endTs": end, "fidelity": fid})
            hist = r.json().get("history", []) if r.status_code == 200 else []
        except (requests.RequestException, ValueError):
            hist = []
        if hist:
            break
    if not hist:
        try:
            r = requests.get(f"{tc.CLOB}/prices-history", timeout=30,
                             params={"market": token, "interval": "max", "fidelity": 60})
            hist = r.json().get("history", []) if r.status_code == 200 else []
        except (requests.RequestException, ValueError):
            hist = []
    h = [x for x in hist if x["t"] <= end]
    if not h or end - h[-1]["t"] > max_stale_h * 3600:
        return None
    return float(h[-1]["p"])


def fetch_dataset(max_events=1500, entry_min_before=30, lag_days=14, progress=None):
    this = pd.Timestamp.now().year
    sack = tc.load_matches(years=list(range(this - 5, this + 1)))
    sack["ts"] = pd.to_datetime(sack.tourney_date.astype(str), format="%Y%m%d")
    S = list(zip(sack.ts, sack.winner_name, sack.loser_name, sack.surface))

    items = []
    for ev in fetch_closed_events(max_events):
        m = tc.pick_moneyline(ev)
        if not m:
            continue
        try:
            outs = tc._j(m.get("outcomes")); px = [float(x) for x in tc._j(m.get("outcomePrices"))]
            toks = tc._j(m.get("clobTokenIds"))
        except (TypeError, ValueError):
            continue
        if len(outs) != 2 or len(px) != 2 or len(toks) != 2 or max(px) < 0.99:
            continue                                              # unresolved / void
        start = pd.to_datetime(m.get("gameStartTime") or ev.get("startTime"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        items.append((start, ev, m, outs, px, toks))
    items.sort(key=lambda x: x[0])

    done = set(pd.read_csv(DS).event_id.astype(str)) if os.path.exists(DS) else set()
    R, ptr, known, buf, added = tc.Ratings(), 0, 0, [], 0
    for k, (start, ev, m, outs, px, toks) in enumerate(items):
        cutoff = start.tz_convert(None).normalize() - pd.Timedelta(days=lag_days)
        while ptr < len(S) and S[ptr][0] < cutoff:
            R.update(S[ptr][1], S[ptr][2], S[ptr][3]); ptr += 1
        if len(R.o) != known:
            R.finalize(); known = len(R.o)
        if progress:
            progress(k + 1, len(items))
        if str(ev.get("id")) in done:
            continue
        pr = price_before(toks[0], start.timestamp() - entry_min_before * 60)
        time.sleep(0.05)
        if pr is None:
            continue
        surf = tc.detect_surface(ev)
        a, b = R.find(outs[0]), R.find(outs[1])
        mp = f1 = f2 = n1 = n2 = np.nan
        if a and b and a != b:
            mp, f1, f2 = R.predict(a, b, surf); n1, n2 = R.n[a], R.n[b]
        buf.append(dict(event_id=ev.get("id"), market_id=m.get("id"), start=start.isoformat(),
                        title=ev.get("title"), surface=surf, p1=outs[0], p2=outs[1], price1=pr,
                        model_p1=mp, form1=f1, form2=f2, n1=n1, n2=n2,
                        winner=0 if px[0] >= 0.99 else 1, volume=float(ev.get("volume") or 0)))
        added += 1
        if len(buf) >= 25:
            _flush(buf); buf = []
    _flush(buf)
    return added


def _flush(buf):
    if buf:
        pd.DataFrame(buf)[COLS].to_csv(DS, mode="a", header=not os.path.exists(DS), index=False)


def load_dataset():
    ds = pd.read_csv(DS)
    ds["start"] = pd.to_datetime(ds.start, utc=True)
    return ds.drop_duplicates("event_id").sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------ strategy replay
DEFAULTS = dict(w_model=0.5, min_edge=0.05, min_ev=0.15, min_prob=0.25, max_price=0.60, slip=0.01,
                fee=0.0, min_games=15, mode="flat", stake=20.0, bankroll=1000.0, kelly_frac=0.25, cap=0.05)


def run_backtest(ds, **kw):
    P = {**DEFAULTS, **kw}; bank = P["bankroll"]; bets = []
    for r in ds.sort_values("start").itertuples():
        if pd.isna(r.model_p1) or r.n1 < P["min_games"] or r.n2 < P["min_games"]:
            continue
        mk = [r.price1, 1 - r.price1]; mod = [r.model_p1, 1 - r.model_p1]
        if min(mk) < 0.03:
            continue
        best = None
        for i in (0, 1):
            pb = P["w_model"] * mod[i] + (1 - P["w_model"]) * mk[i]
            ask = min(mk[i] + P["slip"], 0.99)
            if best is None or pb - ask > best[0]:
                best = (pb - ask, i, pb, ask)
        edge, i, pb, ask = best
        if not (edge >= P["min_edge"] and pb / ask - 1 >= P["min_ev"] and pb >= P["min_prob"] and ask <= P["max_price"]):
            continue
        stake = P["stake"] if P["mode"] == "flat" else bank * min((pb - ask) / (1 - ask) * P["kelly_frac"], P["cap"])
        win = r.winner == i
        pnl = stake * (1 / ask - 1) if win else -stake
        if win:
            pnl -= P["fee"] * pnl
        bank += pnl
        bets.append(dict(start=r.start, match=f"{r.p1} vs {r.p2}", surface=r.surface, pick=[r.p1, r.p2][i],
                         ask=ask, pb=pb, edge=edge, stake=stake, won=win, pnl=pnl, bank=bank))
    return pd.DataFrame(bets)


def summarize(bets, bankroll=1000.0):
    if bets.empty:
        return {"bets": 0}
    st, pl = bets.stake.values, bets.pnl.values
    eq = bankroll + np.cumsum(pl)
    idx = np.random.default_rng(0).integers(0, len(bets), (2000, len(bets)))
    roi_s = pl[idx].sum(1) / st[idx].sum(1)
    return {"bets": len(bets), "hit_rate": float(bets.won.mean()), "avg_price(break-even)": float(bets.ask.mean()),
            "avg_model_p": float(bets.pb.mean()), "avg_payout_x": float((1 / bets.ask).mean()),
            "staked": float(st.sum()), "pnl": float(pl.sum()), "roi": float(pl.sum() / st.sum()),
            "roi_95ci": (float(np.percentile(roi_s, 2.5)), float(np.percentile(roi_s, 97.5))),
            "final_equity": float(eq[-1]), "max_drawdown": float((eq / np.maximum.accumulate(eq) - 1).min())}


def baseline(ds, lo, hi, slip=0.01, stake=20.0):
    """Dumb benchmark: bet EVERY side priced in [lo, hi]. If your model can't beat this, it adds nothing."""
    bets = []
    for r in ds.itertuples():
        for i, mk in enumerate((r.price1, 1 - r.price1)):
            if lo <= mk <= hi:
                ask = mk + slip; win = r.winner == i
                bets.append(dict(ask=ask, pb=mk, stake=stake, won=win, pnl=stake * (1 / ask - 1) if win else -stake))
    return summarize(pd.DataFrame(bets)) if bets else {"bets": 0}


def diagnostics(ds):
    d = ds.dropna(subset=["model_p1"])
    d = d[(d.n1 >= 15) & (d.n2 >= 15) & (d.price1.between(0.03, 0.97))]
    y = (d.winner == 0).astype(float).values
    ll = lambda p: float(-np.mean(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1))))
    out = {"matches": len(d), "logloss_market": ll(d.price1.values), "logloss_model": ll(d.model_p1.values),
           "logloss_blend50": ll(0.5 * d.model_p1.values + 0.5 * d.price1.values),
           "accuracy_market": float(((d.price1 > .5) == (y == 1)).mean()),
           "accuracy_model": float(((d.model_p1 > .5) == (y == 1)).mean())}
    s = pd.concat([pd.DataFrame({"model": d.model_p1, "market": d.price1, "won": y}),
                   pd.DataFrame({"model": 1 - d.model_p1, "market": 1 - d.price1, "won": 1 - y})])
    s["bucket"] = pd.cut(s.market, [0, .2, .3, .4, .5, .6, .7, .8, 1])
    cal = s.groupby("bucket", observed=True).agg(n=("won", "size"), market_price=("market", "mean"),
                                                 model_p=("model", "mean"), actual_win=("won", "mean")).reset_index()
    return out, cal


def sweep(ds, split=0.6, **base):
    cut = ds.start.iloc[int(len(ds) * split)]
    tr, te = ds[ds.start < cut], ds[ds.start >= cut]
    rows = []
    for me in (0.03, 0.05, 0.08, 0.12):
        for w in (0.3, 0.5, 0.7, 1.0):
            for mp in (0.5, 0.6, 0.8):
                kw = {**base, "min_edge": me, "w_model": w, "max_price": mp}
                a, b = summarize(run_backtest(tr, **kw)), summarize(run_backtest(te, **kw))
                rows.append(dict(min_edge=me, w_model=w, max_price=mp, train_bets=a["bets"], train_roi=a.get("roi"),
                                 test_bets=b["bets"], test_roi=b.get("roi")))
    t = pd.DataFrame(rows)
    return t[t.train_bets >= 30].sort_values("train_roi", ascending=False), cut


# ------------------------------------------------------------------ CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "run", "diag", "sweep"])
    ap.add_argument("--max-events", type=int, default=1500)
    ap.add_argument("--lag-days", type=int, default=14)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    a = ap.parse_args()
    pd.set_option("display.width", 200, "display.max_columns", 30)
    kw = {k: getattr(a, k) for k in DEFAULTS}
    if a.cmd == "fetch":
        n = fetch_dataset(a.max_events, lag_days=a.lag_days,
                          progress=lambda i, t: print(f"\r{i}/{t}", end="", flush=True) if i % 20 == 0 else None)
        print(f"\nAdded {n} matches -> {DS}")
    else:
        ds = load_dataset(); print(f"{len(ds)} matches, {ds.start.min():%Y-%m-%d} .. {ds.start.max():%Y-%m-%d}")
        if a.cmd == "run":
            b = run_backtest(ds, **kw)
            for k, v in summarize(b, a.bankroll).items():
                print(f"  {k:24s} {v}")
            print("\nBenchmarks (bet every side in band):")
            print("  all underdogs 0.20-0.45:", baseline(ds, .20, .45, a.slip))
            print("  all favourites 0.60-0.85:", baseline(ds, .60, .85, a.slip))
        elif a.cmd == "diag":
            o, cal = diagnostics(ds)
            print(o); print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        else:
            t, cut = sweep(ds, **{k: kw[k] for k in ("mode", "stake", "slip", "fee", "min_games")})
            print(f"train < {cut:%Y-%m-%d} <= test (unseen). Trust test_roi, not train_roi.")
            print(t.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
