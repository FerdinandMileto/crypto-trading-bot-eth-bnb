#!/usr/bin/env python3
"""
Super Sniper Monitor v4 — Menos ruido, ensemble Ridge + RandomForest.
- Requiere: requests, numpy, sklearn (opcional para RF).
- Ejecutar: python super_sniper_ml_v4.py
"""

import requests, time, os, math
from collections import deque
from statistics import median
from time import time as now

# try import heavy libs
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("Este script requiere numpy. Instálalo: pip install numpy") from e

# sklearn optional
SKLEARN_AVAILABLE = True
try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import mean_squared_error
except Exception:
    SKLEARN_AVAILABLE = False

# -------------------- CONFIG --------------------
SAMPLE_INTERVAL = 5          # seconds between samples
WINDOW_SIZE = 12             # window size (12 * 5s = 60s)
MIN_SAMPLES_TO_TRAIN = 10    # need at least this many samples to train
RANDOM_STATE = 42

# -------------------- COLORS --------------------
class C:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"

# -------------------- UTIL API --------------------
def buscar_token_dexscreener(mint):
    url = f"https://api.dexscreener.com/latest/dex/search?q={mint}"
    try:
        r = requests.get(url, timeout=4)
        if r.status_code == 200:
            data = r.json()
            if data.get("pairs"):
                return data["pairs"][0]
    except Exception:
        return None
    return None

def safe_usd(x):
    if isinstance(x, dict):
        return float(x.get("usd", 0) or 0)
    try:
        return float(x or 0)
    except:
        return 0.0

def estimate_supply_and_mc(token_data, price):
    base = token_data.get("baseToken", {}) or {}
    for k in ("totalSupply","supply","total_supply","circulatingSupply","circulating_supply"):
        if k in base:
            try:
                s = float(base[k])
                if s>0:
                    return s, s*price
            except:
                pass
    fdv = token_data.get("fdv", None)
    if fdv:
        try:
            fdv_f = float(fdv)
            if price>0:
                supply = fdv_f / price
                return supply, supply*price
        except:
            pass
    return None, None

# -------------------- FEATURE ENGINEERING --------------------
def build_features(token_data, last_price, prices_win):
    price = float(token_data.get("priceUsd", 0.0))
    txns = token_data.get("txns", {}).get("m5", {}) or {}
    buys = float(txns.get("buys", 0))
    sells = float(txns.get("sells", 0))
    total_trades = buys + sells + 1e-9
    buy_ratio = buys / total_trades

    volume = safe_usd(token_data.get("volume", {}) or 0)
    liquidity = safe_usd(token_data.get("liquidity", {}) or 0)
    holders = int(token_data.get("baseToken", {}).get("holders", 0) or 0)

    # EMA short (EWMA) over the price window (fast alpha by default)
    if prices_win:
        alpha = 2/(len(prices_win)+1)
        ema = prices_win[0]
        for p in prices_win[1:]:
            ema = alpha * p + (1-alpha) * ema
    else:
        ema = price

    momentum = (price - (last_price or price))
    pct_change = (price - (last_price or price)) / ((last_price or price) + 1e-9)

    # EWMA volume: use simple decaying average of last n volumes (approx using price window length)
    # but since we may not have per-sample volume, keep volume raw.

    # features chosen for robustness (avoid extreme scale mismatch)
    features = [
        price,
        ema,
        momentum,
        pct_change,
        buy_ratio,
        buys,
        sells,
        volume,
        liquidity,
        holders
    ]
    return np.array(features, dtype=float), price, buys, sells, volume, liquidity, buy_ratio, ema

# -------------------- PREPROCESSING: winsorize and robust scale --------------------
def winsorize_array(X, lower_q=0.05, upper_q=0.95):
    # X: numpy array (n_samples, n_features)
    Xw = X.copy()
    for j in range(X.shape[1]):
        col = X[:,j]
        low = np.quantile(col, lower_q)
        high = np.quantile(col, upper_q)
        col_clipped = np.clip(col, low, high)
        Xw[:,j] = col_clipped
    return Xw

def robust_scale_fit_transform(X):
    # fit median and IQR
    med = np.median(X, axis=0)
    q1 = np.percentile(X,25, axis=0)
    q3 = np.percentile(X,75, axis=0)
    iqr = q3 - q1
    iqr[iqr==0] = 1.0
    Xs = (X - med) / iqr
    return Xs, med, iqr

def robust_scale_transform(X, med, iqr):
    iqr[iqr==0] = 1.0
    return (X - med) / iqr

# -------------------- MODEL HELPERS --------------------
def train_models(X_raw, y_raw):
    """
    X_raw: list of feature vectors (numpy)
    y_raw: list of targets (float)
    returns: dict with trained models and scaler info
    """
    X = np.vstack(X_raw)
    y = np.array(y_raw, dtype=float)

    # winsorize
    Xw = winsorize_array(X)

    # robust scaling
    Xs, med, iqr = robust_scale_fit_transform(Xw)

    models = {}
    # Ridge (sklearn if available)
    if SKLEARN_AVAILABLE:
        ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        ridge.fit(Xs, y)
        models['ridge'] = ridge
        # RandomForest for nonlinearity (use small tree count to keep latency)
        rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=RANDOM_STATE, n_jobs=1)
        rf.fit(Xs, y)
        models['rf'] = rf
    else:
        # fallback: closed-form ridge (normal equation)
        # w = (X'X + lambda I)^-1 X'y
        lam = 1.0
        XtX = Xs.T.dot(Xs)
        n = XtX.shape[0]
        A = XtX + lam * np.eye(n)
        try:
            w = np.linalg.inv(A).dot(Xs.T).dot(y)
            b = 0.0
            models['ridge_cf'] = (w, b)
        except Exception:
            models['ridge_cf'] = None

    return models, med, iqr

def predict_ensemble(models, x_raw, med, iqr):
    """
    x_raw: single raw feature vector (numpy)
    returns: ensemble preds list
    """
    x_scaled = robust_scale_transform(x_raw.reshape(1,-1), med, iqr)
    preds = []
    if SKLEARN_AVAILABLE:
        preds.append(float(models['ridge'].predict(x_scaled)[0]))
        preds.append(float(models['rf'].predict(x_scaled)[0]))
    else:
        if models.get('ridge_cf') is not None:
            w,b = models['ridge_cf']
            pred = float(np.dot(w, x_scaled.flatten()) + b)
            preds.append(pred)
    return preds

# -------------------- UTILS --------------------
def rmse(y_true, y_pred):
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    return float(np.sqrt(np.mean((y_t - y_p)**2))) if len(y_t)>0 else float('nan')

# -------------------- DASHBOARD LOOP --------------------
def monitor_loop(mint):
    last_price = None
    prices_q = deque(maxlen=WINDOW_SIZE)
    feats_q = deque(maxlen=WINDOW_SIZE)
    targets_q = deque(maxlen=WINDOW_SIZE)

    trained = None
    med = None
    iqr = None

    while True:
        t0 = now()
        os.system('cls' if os.name == 'nt' else 'clear')
        token = buscar_token_dexscreener(mint)
        ts = time.strftime("%H:%M:%S")
        if not token:
            print(f"{C.RED}No data from DexScreener for {mint}{C.RESET}")
            time.sleep(SAMPLE_INTERVAL)
            continue

        feat, price_now, buys, sells, volume, liquidity, buy_ratio, ema = build_features(token, last_price, list(prices_q))
        prices_q.append(price_now)
        feats_q.append(feat)
        targets_q.append(price_now)

        # basic analytics
        holders = int(token.get("baseToken",{}).get("holders",0) or 0)
        trend = "🔍"
        if last_price:
            if price_now > last_price * 1.005:
                trend = "🚀 Subiendo"
            elif price_now < last_price * 0.995:
                trend = "📉 Bajando"
            else:
                trend = "⏸️ Rango"

        activity = "📈 Orgánica"
        if holders < 50 or liquidity == 0 or (volume / (liquidity+1e-9)) > 1.2:  # tighter threshold
            activity = "⚠️ Posible Artificial"
        if max(buys, sells) > 10 * (min(buys, sells)+1):
            activity = "⚠️ Posible Pump/Bot"

        # train models if enough data (we train to predict next-step price)
        pred_text = "N/A"
        pred_mc_text = "N/A"
        conf_text = "N/A"
        ensemble_preds = []
        model_rmse = None
        if len(feats_q) >= MIN_SAMPLES_TO_TRAIN:
            # X = feats[0..n-2], y = prices[1..n-1]
            X_raw = list(feats_q)[:-1]
            y_raw = list(targets_q)[1:]
            try:
                models, med, iqr = train_models(X_raw, y_raw)
                # predict for last feature vector (to predict price next step)
                latest_raw = np.array(list(feats_q)[-1])
                preds = predict_ensemble(models, latest_raw, med, iqr)
                # ensemble: median of preds (robust)
                if preds:
                    ensemble_preds = preds
                    pred_val = float(np.median(preds))
                    pred_text = f"${pred_val:.8f}"
                    # compute train predictions to estimate RMSE
                    # create scaled X for computing preds
                    Xw = winsorize_wrapper(np.vstack(X_raw))
                    Xs = robust_scale_transform(Xw, med.copy(), iqr.copy())
                    train_preds = []
                    if SKLEARN_AVAILABLE:
                        train_preds.append(models['ridge'].predict(Xs))
                        train_preds.append(models['rf'].predict(Xs))
                    else:
                        if models.get('ridge_cf') is not None:
                            w,b = models['ridge_cf']
                            train_preds.append(np.dot(Xs, w) + b)
                    # average ensemble train preds
                    train_pred_mean = np.mean(np.vstack(train_preds), axis=0) if train_preds else None
                    if train_pred_mean is not None:
                        model_rmse = rmse(y_raw, train_pred_mean)
                        conf_text = f"RMSE(train): {model_rmse:.6f}"
                    # estimate MC
                    supply_est, mc_est = estimate_supply_and_mc(token, pred_val)
                    pred_mc_text = f"${mc_est:,.2f}" if mc_est else "N/A"
            except Exception as e:
                # safe fallback
                pred_text = "ERR"
                conf_text = str(e)

        # pico detection (robust zscore on prices)
        pico = detect_peaks_safe(list(prices_q))

        # display panel
        print(f"{C.CYAN}Super Sniper ML v4 — {ts}{C.RESET}")
        print(f"Token: {mint}")
        print(f"Price: ${price_now:.8f}    EMA(short): {ema:.8f}    Trend: {trend}")
        print(f"Liquidity: ${liquidity:,.2f}    Volume(24h): ${volume:,.2f}    Holders: {holders}")
        print(f"Buys(m5): {buys}    Sells(m5): {sells}    BuyRatio: {buy_ratio:.3f}")
        print("-"*60)
        print(f"Predicted next price: {pred_text}    Predicted MC next: {pred_mc_text}    {conf_text}")
        if ensemble_preds:
            var = float(np.var(ensemble_preds))
            print(f"Ensemble preds: {ensemble_preds}    Var: {var:.6f}")
        if pico:
            print(f"{C.RED}ALERTA PICO: {pico}{C.RESET}")
        color_act = C.GREEN if activity.startswith("📈") else C.RED
        print(f"Activity: {color_act}{activity}{C.RESET}")
        print("-"*60)
        # housekeeping
        last_price = price_now
        # sleep remaining time to maintain SAMPLE_INTERVAL
        elapsed = now() - t0
        to_sleep = max(0, SAMPLE_INTERVAL - elapsed)
        time.sleep(to_sleep)

# -------------------- HELPERS MISSING (implementations used above) --------------------
def winsorize_wrapper(X):
    # same simple winsorize used earlier
    Xw = X.copy()
    for j in range(X.shape[1]):
        col = X[:,j]
        low = np.quantile(col, 0.05)
        high = np.quantile(col, 0.95)
        Xw[:,j] = np.clip(col, low, high)
    return Xw

def detect_peaks_safe(prices):
    if len(prices) < 3:
        return None
    if len(set(prices)) < 2:
        return None
    mu = float(np.mean(prices))
    try:
        sigma = float(np.std(prices, ddof=1))
    except:
        sigma = 1e-6
    if sigma <= 0:
        return None
    z = (prices[-1] - mu) / sigma
    if z > 2.2:
        return "🚀 Pico alto detectado"
    if z < -2.2:
        return "📉 Pico bajo detectado"
    return None

# -------------------- RUN --------------------
if __name__ == "__main__":
    mint = input("Introduce el contrato (mint) del token): ").strip()
    print("Iniciando monitor... (asegúrate de tener requests, numpy; sklearn opcional para RandomForest)")
    monitor_loop(mint)
