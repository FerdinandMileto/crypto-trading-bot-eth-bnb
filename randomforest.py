import requests
import time
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ================================
# CONFIGURACIONES
# ================================
JUPITER_URL = "https://price.jup.ag/v4/price"
DEX_URL = "https://api.dexscreener.io/latest/dex/tokens/"
TOKEN_MINT = input("Introduce el contrato (mint) del token: ")
INTERVALO = 10  # segundos
HISTORIAL_MAX = 100
WINDOW_EMA = 5  # EMA corta

# ================================
# HISTORIAL
# ================================
precios_hist = deque(maxlen=HISTORIAL_MAX)
ema_hist = deque(maxlen=HISTORIAL_MAX)
liquidez_hist = deque(maxlen=HISTORIAL_MAX)
buy_ratio_hist = deque(maxlen=HISTORIAL_MAX)
price_window = []
volume_window = []
liquidez_inicial = None

# ================================
# FUNCIONES
# ================================
def ema(values, window):
    if len(values) < window:
        return np.mean(values)
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return np.convolve(values, weights, mode='valid')[-1]

def detectar_tendencia(precios):
    if len(precios) < 5:
        return "⏸️ Rango"
    dif = precios[-1] - precios[0]
    if dif > 0.01 * precios[0]:
        return "🚀 Alcista"
    elif dif < -0.01 * precios[0]:
        return "📉 Bajista"
    return "⏸️ Rango"

def detectar_picos(precios):
    if len(precios) < 5:
        return None
    media = np.mean(precios)
    std = np.std(precios)
    if std == 0:
        return None
    z_score = (precios[-1] - media) / std
    if z_score > 2:
        return "🔥 Pico alcista"
    elif z_score < -2:
        return "⚠️ Pico bajista"
    return None

def obtener_precio_jupiter(mint):
    try:
        r = requests.get(f"{JUPITER_URL}?ids={mint}", timeout=5)
        data = r.json()
        return float(data["data"][mint]["price"])
    except:
        return None

def obtener_datos_dex(mint):
    global liquidez_inicial
    try:
        r = requests.get(f"{DEX_URL}{mint}", timeout=5)
        data = r.json()
        if "pairs" in data and data["pairs"]:
            p = data["pairs"][0]
            price = float(p.get("priceUsd", 0))
            liquidity = float(p.get("liquidity", {}).get("usd", 0))
            buys = float(p.get("txns", {}).get("m5", {}).get("buys", 0))
            sells = float(p.get("txns", {}).get("m5", {}).get("sells", 0))
            total_supply = float(p.get("token", {}).get("supply", 1_000_000_000))
            if liquidez_inicial is None:
                liquidez_inicial = liquidity if liquidity > 0 else 1
            return {"price": price, "liquidity": liquidity, "buys_m5": buys, "sells_m5": sells, "supply": total_supply}
        return {"price": 0, "liquidity": 0, "buys_m5": 0, "sells_m5": 0, "supply": 1_000_000_000}
    except:
        return {"price": 0, "liquidity": 0, "buys_m5": 0, "sells_m5": 0, "supply": 1_000_000_000}

def obtener_precio(mint):
    precio = obtener_precio_jupiter(mint)
    if precio is not None:
        return precio
    dex = obtener_datos_dex(mint)
    if dex["price"] > 0:
        return dex["price"]
    return None

# ================================
# DETECCIÓN DE CAMBIOS BRUSCOS
# ================================
def detectar_movimiento_bruso(precio_actual, volumen_actual, umbral_precio=0.05, umbral_volumen=3):
    """Detecta cambios bruscos y calcula momentum inmediato"""
    price_window.append(precio_actual)
    volume_window.append(volumen_actual)

    if len(price_window) < 2:
        return False, 0.0, 0.0

    # Momentum inmediato
    momentum = (price_window[-1] - price_window[-2]) / price_window[-2]

    # Z-score volumen
    vol_mean = np.mean(volume_window)
    vol_std = np.std(volume_window) if np.std(volume_window) > 0 else 1e-6
    z_score_vol = (volumen_actual - vol_mean) / vol_std

    alerta = abs(momentum) > umbral_precio or abs(z_score_vol) > umbral_volumen
    return alerta, momentum, z_score_vol

# ================================
# MODELOS
# ================================
def entrenar_modelos():
    if len(precios_hist) < 20:
        return None, None
    X, y_precio, y_marketcap = [], [], []
    precios_list = list(precios_hist)
    ema_list = list(ema_hist)
    liquidez_list = list(liquidez_hist)
    buy_ratio_list = list(buy_ratio_hist)
    for i in range(len(precios_list)-1):
        features = [precios_list[i], ema_list[i], liquidez_list[i], buy_ratio_list[i]]
        X.append(features)
        y_precio.append(precios_list[i+1])
        mc = precios_list[i+1] * liquidez_list[i] / precios_list[i]  # Ajuste inicial
        y_marketcap.append(mc)
    X = np.array(X)
    sample_weight = np.linspace(1, 3, len(X))
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y_precio, sample_weight=sample_weight)
    gb.fit(X, y_marketcap)
    return rf, gb

def predecir_modelos(rf, gb, precio, ema_val, liquidez, buy_ratio):
    features = np.array([[precio, ema_val, liquidez, buy_ratio]])
    precio_pred = rf.predict(features)[0] if rf else None
    marketcap_pred = gb.predict(features)[0] if gb else None
    return precio_pred, marketcap_pred

# ================================
# MONITOREO
# ================================
def monitorear(mint):
    global WINDOW_EMA
    while True:
        dex = obtener_datos_dex(mint)
        precio = dex["price"]
        if precio == 0:
            print("❌ No se pudo obtener precio.")
            time.sleep(INTERVALO)
            continue

        precios_hist.append(precio)
        # Momentum inmediato para ajustar EMA
        if len(precios_hist) > 1:
            momentum = (precios_hist[-1] - precios_hist[-2]) / precios_hist[-2]
            WINDOW_EMA = 3 if abs(momentum) > 0.02 else 5

        ema_val = ema(list(precios_hist), window=WINDOW_EMA)
        ema_hist.append(ema_val)
        liquidez_hist.append(dex["liquidity"])
        buys, sells = dex["buys_m5"], dex["sells_m5"]
        buy_ratio = buys / (buys + sells) if (buys + sells) > 0 else 0
        buy_ratio_hist.append(buy_ratio)

        alerta, momentum, z_score_vol = detectar_movimiento_bruso(precio, buys + sells)

        # Reentrenar modelos
        rf_model, gb_model = entrenar_modelos()
        precio_pred, marketcap_pred = predecir_modelos(rf_model, gb_model, precio, ema_val, dex["liquidity"], buy_ratio)

        # Ajustar Market Cap con supply real
        if marketcap_pred and dex["supply"]:
            marketcap_pred = precio_pred * dex["supply"]
            marketcap_pred *= (dex["liquidity"] / liquidez_inicial)

        tendencia = detectar_tendencia(list(precios_hist))
        pico = detectar_picos(list(precios_hist))

        # =====================
        # SALIDA CONSOLA
        # =====================
        print("\n============================")
        print(f"Super Sniper ML v9 — {time.strftime('%H:%M:%S')}")
        print(f"Token: {mint}")
        print(f"Precio: ${precio:.8f}")
        print(f"EMA Corta: {ema_val:.8f}")
        print(f"Tendencia: {tendencia}")
        print(f"Liquidez: ${dex['liquidity']:.2f}")
        print(f"Buys(m5): {buys} | Sells(m5): {sells} | BuyRatio: {buy_ratio:.2f}")
        print(f"Momentum inmediato: {momentum:.4f}")
        if alerta:
            print(f"⚠️ ALERTA: Cambio brusco detectado | Volumen Z: {z_score_vol:.2f}")
        if precio_pred:
            print(f"Predicción precio próximo min: ${precio_pred:.8f}")
        if marketcap_pred:
            print(f"Estimación Market Cap próximo min: ${marketcap_pred:,.2f}")
        if pico:
            print(f"⚠️ ALERTA: {pico}")
        print("============================")
        time.sleep(INTERVALO)

if __name__ == "__main__":
    monitorear(TOKEN_MINT)
