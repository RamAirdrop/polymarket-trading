"""
Backtest engine for Polymarket tennis model.
"""
import os, json, math, time
import pandas as pd
import requests
import tennis_core as tc

DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "backtest.csv")
COLS = ["event_id", "market_id", "match", "surface", "start", "p1", "p2", "model_p1", "model_p2", "winner", "n1", "n2"]


def fetch_dataset(max_events=1500, progress=None):
    sack = tc.load_matches()
    sack["ts"] = pd.to_datetime(sack.tourney_date.astype(str), format="%Y%m%d")
    evs = tc.fetch_events(max_events=max_events, closed=True)
    existing = set(pd.read_csv(DS).event_id) if os.path.exists(DS) else set()
    buf, total = [], len(evs)

    for idx, ev in enumerate(evs):
        if progress:
            progress(idx + 1, total)
        eid = str(ev.get("id"))
        if eid in existing:
            continue
        m = tc.pick_moneyline(ev)
        if not m:
            continue
        start = pd.to_datetime(m.get("gameStartTime") or ev.get("startTime") or ev.get("endDate"), utc=True, errors="coerce")
        if pd.isna(start):
            continue
        outs = tc._j(m.get("outcomes"))
        px = [float(x) for x in tc._j(m.get("outcomePrices")) or [0, 0]]
        if len(outs) != 2 or len(px) != 2 or max(px) < 0.99:
            continue
        w_idx = 0 if px[0] >= 0.99 else 1
        winner = outs[w_idx]
        surf = tc.detect_surface(ev)
        
        # Build historical ratings up to start - 14 days to prevent look-ahead bias
        cutoff_date = int(start.strftime("%Y%m%d")) - 1400  # rough 14-day lag in Sackmann date int format
        past_df = sack[sack.tourney_date <= cutoff_date]
        if len(past_df) < 1000:
            continue
        R = tc.build_ratings(past_df)
        
        a, b = R.find(outs[0]), R.find(outs[1])
        if not a or not b or a == b:
            continue
        
        p0, _, _, _ = R.predict_parts(a, b, surf, best_of=tc.guess_best_of(ev))
        buf.append(dict(
            event_id=eid, market_id=m.get("id"), match=f"{outs[0]} vs {outs[1]}",
            surface=surf, start=start.isoformat(), p1=px[0], p2=px[1],
            model_p1=p0, model_p2=1 - p0, winner=winner,
            n1=R.n.get(a, 0), n2=R.n.get(b, 0)
        ))
        if len(buf) >= 50:
            pd.DataFrame(buf)[COLS].to_csv(DS, mode="a", header=not os.path.exists(DS), index=False)
            buf = []
    if buf:
        pd.DataFrame(buf)[COLS].to_csv(DS, mode="a", header=not os.path.exists(DS), index=False)
    return len(evs)


def load_dataset():
    ds = pd.read_csv(DS)
    ds["start"] = pd.to_datetime(ds.start, utc=True, errors="coerce")
    ds = ds.dropna(subset=["start"])
    return ds.drop_duplicates("event_id").sort_values("start").reset_index(drop=True)


# ------------------------------------------------------------------ strategy replay
DEFAULTS = dict(w_model=0.5, min_edge=0.05, min_ev=0.15, min_prob=0.25, max_price=0.60, slip=0.01,
                fee=0.0, min_games=15, mode="flat", stake=20.0, bankroll=1000.0, kelly_frac=0.25, cap=0.05)


def run_backtest(ds, **kw):
    P = {**DEFAULTS, **kw}; bank = P["bankroll"]; bets = []
    for r in ds.sort_values("start").itertuples():
        if pd.isna(r.model_p1) or r.n1 < P["min_games"] or r.n2 < P["min_games"]:
            continue
        outs = r.match.split(" vs ")
        if len(outs) != 2:
            continue
        for i, (p, model_p, ask) in enumerate([(outs[0], r.model_p1, r.p1), (outs[1], r.model_p2, r.p2)]):
            if model_p < P["min_prob"] or ask > P["max_price"]:
                continue
            edge = model_p - (ask + P["slip"])
            ev = model_p / (ask + P["slip"]) - 1
            if edge < P["min_edge"] or ev < P["min_ev"]:
                continue
            
            effective_ask = min(ask + P["slip"], 0.99)
            stake = P["stake"]
            if P["mode"] == "kelly":
                b_odds = (1 / effective_ask) - 1
                f = (model_p * b_odds - (1 - model_p)) / b_odds
                stake = max(0.0, bank * P["kelly_frac"] * f)
                stake = min(stake, bank * P["cap"])
            if stake <= 0:
                continue
            
            won = (r.winner.strip().lower() == p.strip().lower())
            shares = stake / effective_ask
            payout = shares if won else 0.0
            fee_amt = (payout - stake) * P["fee"] if (won and P["fee"] > 0) else 0.0
            pnl = payout - stake - fee_amt
            bank += pnl
            bets.append(dict(start=r.start, match=r.match, pick=p, ask=effective_ask, model_p=model_p,
                             won=won, pnl=pnl, bankroll=bank, surface=r.surface))
            break  # Only bet one side per match
    return pd.DataFrame(bets) if bets else pd.DataFrame(columns=["start", "match", "pick", "ask", "model_p", "won", "pnl", "bankroll", "surface"])


def summarize(bets):
    if bets.empty:
        return dict(bets=0, hit_rate=0, roi=0, pnl=0, max_drawdown=0, roi_95ci=(0, 0), avg_break_even=0)
    n = len(bets)
    hit = bets.won.mean()
    pnl = bets.pnl.sum()
    staked = (bets.ask * (bets.pnl / (1 - bets.ask))).abs().sum() if "ask" in bets else n * 100
    roi = pnl / (n * 100)
    
    # 95% Confidence Interval for ROI
    import numpy as np
    returns = bets.pnl.values / 100.0
    se = np.std(returns) / np.sqrt(n) if n > 1 else 0
    ci = (roi - 1.96 * se, roi + 1.96 * se)
    
    peak = bets.bankroll.cummax()
    dd = ((bets.bankroll - peak) / peak).min()
    
    return dict(bets=n, hit_rate=hit, roi=roi, pnl=pnl, max_drawdown=dd, roi_95ci=ci,
                **{"avg_price(break-even)": bets.ask.mean()})


def diagnostics(ds):
    from sklearn.metrics import log_loss
    valid = ds.dropna(subset=["model_p1", "p1", "winner"])
    if valid.empty:
        return {}, pd.DataFrame()
    
    y = [1 if w.strip().lower() == m.split(" vs ")[0].strip().lower() else 0 for w, m in zip(valid.winner, valid.match)]
    model_preds = valid.model_p1.clip(1e-6, 1 - 1e-6)
    market_preds = valid.p1.clip(1e-6, 1 - 1e-6)
    
    metrics = {
        "model_log_loss": log_loss(y, model_preds),
        "market_log_loss": log_loss(y, market_preds),
        "coin_flip_log_loss": math.log(2)
    }
    
    valid["bucket"] = pd.cut(valid.model_p1, bins=[0, .3, .4, .5, .6, .7, .8, 1])
    cal = valid.groupby("bucket", observed=True).agg(
        matches=("winner", "size"),
        model_prob=("model_p1", "mean"),
        actual_win=("winner", lambda s: sum(1 for w, m in zip(s, valid.loc[s.index, "match"]) if w.strip().lower() == m.split(" vs ")[0].strip().lower()) / len(s))
    ).reset_index()
    return metrics, cal


def baseline(ds, lo, hi, slip):
    valid = ds[(ds.p1 + slip >= lo) & (ds.p1 + slip <= hi)]
    if valid.empty:
        return {"bets": 0, "roi": 0}
    pnl = 0
    for r in valid.itertuples():
        ask = r.p1 + slip
        won = r.winner.strip().lower() == r.match.split(" vs ")[0].strip().lower()
        pnl += (1 / ask - 1) if won else -1
    return {"bets": len(valid), "roi": pnl / len(valid)}


def sweep(ds, mode="flat", stake=100.0, slip=0.01, fee=0.0):
    cut = ds.start.iloc[int(len(ds) * 0.7)]
    train, test = ds[ds.start < cut], ds[ds.start >= cut]
    results = []
    for min_edge in [0.0, 0.02, 0.05, 0.10]:
        for min_prob in [0.30, 0.40, 0.50, 0.60]:
            kw = dict(min_edge=min_edge, min_prob=min_prob, mode=mode, stake=stake, slip=slip, fee=fee)
            tr_bets = run_backtest(train, **kw)
            te_bets = run_backtest(test, **kw)
            results.append(dict(
                min_edge=min_edge, min_prob=min_prob,
                train_bets=len(tr_bets), train_roi=tr_bets.pnl.sum() / (len(tr_bets) * stake) if len(tr_bets) else 0,
                test_bets=len(te_bets), test_roi=te_bets.pnl.sum() / (len(te_bets) * stake) if len(te_bets) else 0
            ))
    return pd.DataFrame(results).sort_values("test_roi", ascending=False), cut


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", type=int, default=1500)
    a = ap.parse_args()
    fetch_dataset(a.fetch)
    ds = load_dataset()
    print(f"{len(ds)} matches, {ds.start.min():%Y-%m-%d} .. {ds.start.max():%Y-%m-%d}")