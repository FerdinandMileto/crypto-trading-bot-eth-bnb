import time
import requests

# Configuración
MIN_MC = 20000       # Market Cap mínimo
MIN_LIQ = 5000       # Liquidez mínima
PUMP_5M = 50         # Pump mínimo en 5m (%)
VOL_5M = 5000        # Volumen mínimo en 5m ($)

BIRDEYE_API = "https://public-api.birdeye.so/public/tokenlist"
DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/tokens/solana"
HEADERS = {"x-chain": "solana"}

def get_tokens_birdeye():
    """Obtiene tokens desde Birdeye"""
    try:
        r = requests.get(f"{BIRDEYE_API}?sort_by=mc&sort_type=desc&offset=0&limit=50", headers=HEADERS)
        data = r.json()
        return data.get("data", {}).get("tokens", [])
    except Exception as e:
        print(f"Error Birdeye: {e}")
        return []

def get_tokens_dexscreener():
    """Obtiene tokens desde DexScreener"""
    try:
        r = requests.get(DEXSCREENER_API)
        data = r.json()
        return data.get("pairs", []) or []  # 🔥 Devuelve [] si no hay "pairs"
    except Exception as e:
        print(f"Error DexScreener: {e}")
        return []

def analyze_token(symbol, address, mc, liq, price_change_5m, vol_5m, source):
    """Analiza y muestra tokens que cumplen criterios degen"""
    if mc > MIN_MC and liq > MIN_LIQ and (price_change_5m > PUMP_5M or vol_5m > VOL_5M):
        print("\n🚨 PUMP DETECTADO 🚨")
        print(f"Fuente: {source}")
        print(f"Token: {symbol}")
        print(f"Contrato: {address}")
        print(f"MC: ${mc:,.0f}")
        print(f"Liquidez: ${liq:,.0f}")
        print(f"Cambio 5m: {price_change_5m}%")
        print(f"Volumen 5m: ${vol_5m:,.0f}")
        print("-" * 40)

def degen_sniper():
    while True:
        # 🔹 Birdeye
        tokens_birdeye = get_tokens_birdeye()
        for t in tokens_birdeye:
            try:
                analyze_token(
                    symbol=t.get("symbol"),
                    address=t.get("address"),
                    mc=float(t.get("mc", 0)),
                    liq=float(t.get("liquidity", 0)),
                    price_change_5m=float(t.get("priceChange5m", 0)),
                    vol_5m=float(t.get("v5mUsd", 0)),
                    source="Birdeye"
                )
            except Exception as e:
                print(f"Error token Birdeye: {e}")

        # 🔹 DexScreener
        tokens_dex = get_tokens_dexscreener()
        for t in tokens_dex:
            try:
                symbol = t.get("baseToken", {}).get("symbol")
                address = t.get("baseToken", {}).get("address")
                mc = float(t.get("fdv", 0))
                liq = float(t.get("liquidity", {}).get("usd", 0))
                price_change_5m = float(t.get("priceChange", {}).get("m5", 0))
                vol_5m = float(t.get("volume", {}).get("m5", 0))

                analyze_token(symbol, address, mc, liq, price_change_5m, vol_5m, "DexScreener")
            except Exception as e:
                print(f"Error token DexScreener: {e}")

        time.sleep(5)

if __name__ == "__main__":
    degen_sniper()

