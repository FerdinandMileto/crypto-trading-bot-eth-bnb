import requests
import time
import os

# ------------------- FUNCIONES BASE -------------------

def buscar_token_dexscreener(mint):
    """Busca información del token en DexScreener por dirección."""
    url = f"https://api.dexscreener.com/latest/dex/search?q={mint}"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("pairs"):
                return data["pairs"][0]
    except:
        return None
    return None

def analizar_actividad(token_data):
    """
    Devuelve True si la actividad parece orgánica, False si sospechosa.
    Basado en holders, balance compras/ventas y volumen vs liquidez
    """
    holders = token_data.get("baseToken", {}).get("holders", 0)
    
    # Corregimos volumen y liquidez
    volumen_dict = token_data.get("volume", {})
    volume = float(volumen_dict.get("usd", 0)) if isinstance(volumen_dict, dict) else float(volumen_dict)

    liquidity_dict = token_data.get("liquidity", {})
    liquidity = float(liquidity_dict.get("usd", 1)) if isinstance(liquidity_dict, dict) else float(liquidity_dict)

    txns = token_data.get("txns", {}).get("m5", {})
    buys = float(txns.get("buys", 0))
    sells = float(txns.get("sells", 0))

    # Condiciones sospechosas
    if holders < 50:
        return False
    if max(buys, sells) > 10 * (min(buys, sells)+1):
        return False
    if liquidity == 0:
        return False
    if volume / liquidity > 1.5:
        return False

    return True

def calcular_tendencia(price, last_price):
    """Detecta tendencia simple según último precio."""
    if not last_price:
        return "🔍 Analizando..."
    if price > last_price * 1.005:
        return "🚀 Subiendo"
    elif price < last_price * 0.995:
        return "📉 Bajando"
    else:
        return "⏸️ Rango"

# ------------------- SCRIPT PRINCIPAL -------------------

def mostrar_info(mint):
    last_price = None
    while True:
        os.system('cls' if os.name == 'nt' else 'clear')  # Limpiar consola
        token_data = buscar_token_dexscreener(mint)

        if token_data:
            # Corregimos volumen y liquidez
            volumen_dict = token_data.get("volume", {})
            volume = float(volumen_dict.get("usd", 0)) if isinstance(volumen_dict, dict) else float(volumen_dict)

            liquidity_dict = token_data.get("liquidity", {})
            liquidity = float(liquidity_dict.get("usd", 0)) if isinstance(liquidity_dict, dict) else float(liquidity_dict)

            price = float(token_data.get("priceUsd", 0))
            dex = token_data.get("dexId", "N/A")

            tendencia = calcular_tendencia(price, last_price)
            actividad = "📈 Orgánica" if analizar_actividad(token_data) else "⚠️ Artificial"

            # Mostrar datos
            print(f"""
=============================
📊 Token: {mint}
💰 Precio: ${price:.6f}
💧 Liquidez: ${liquidity}
📊 Volumen 24h: ${volume}
🏦 Dex: {dex}
=============================
""")

            # Alertas al final
            print(f"📈 Tendencia: {tendencia}")
            print(f"⚡ Actividad: {actividad}")

            last_price = price
        else:
            print(f"❌ No se encontraron datos del token: {mint}")

        time.sleep(5)

if __name__ == "__main__":
    mint = input("Introduce el contrato (mint) del token: ").strip()
    mostrar_info(mint)
