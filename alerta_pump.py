import requests
import time
from plyer import notification

DEXSCREENER_API = "https://api.dexscreener.com/latest/dex/search?q=solana"
REFRESH_INTERVAL = 30  # segundos
MIN_LIQUIDITY = 10000   # Liquidez mínima
MAX_AGE_HOURS = 24      # Tokens creados hace menos de 24h
MIN_TRENDING_SCORE = 10 # Ajusta según tu preferencia

def send_alert(pair):
    """Muestra notificación local"""
    name = pair.get('baseToken', {}).get('name', 'Unknown')
    symbol = pair.get('baseToken', {}).get('symbol', '')
    liquidity = pair.get('liquidity', {}).get('usd', 0)
    score = pair.get('trendingScoreH6', 0)
    url = pair.get('url', '')

    notification.notify(
        title=f"🔥 {name} ({symbol})",
        message=f"Liquidity: ${liquidity:,.0f}\nScore: {score}\n{url}",
        timeout=7
    )

def fetch_trending_pairs():
    """Consulta DexScreener y filtra por score y antigüedad"""
    try:
        response = requests.get(DEXSCREENER_API, timeout=10)
        response.raise_for_status()
        pairs = response.json().get('pairs', [])

        trending = []
        for p in pairs:
            liquidity = p.get('liquidity', {}).get('usd', 0)
            score = p.get('trendingScoreH6', 0)
            created_at = p.get('pairCreatedAt', 0) / 1000
            age_hours = (time.time() - created_at) / 3600 if created_at > 0 else None

            if liquidity >= MIN_LIQUIDITY and score >= MIN_TRENDING_SCORE and age_hours is not None and age_hours <= MAX_AGE_HOURS:
                trending.append(p)

        # Ordenar por trendingScoreH6 descendente
        trending.sort(key=lambda x: x.get('trendingScoreH6', 0), reverse=True)
        return trending

    except Exception as e:
        print(f"❌ Error al consultar DexScreener: {e}")
        return []

def main():
    print("🚀 Monitoreando tokens nuevos en Solana (DexScreener con trending score)...\n")
    while True:
        pairs = fetch_trending_pairs()
        if pairs:
            for token in pairs[:5]:  # Solo mostramos top 5
                send_alert(token)
                print(f"🔥 ALERTA: {token.get('baseToken', {}).get('name')} "
                      f"| Score: {token.get('trendingScoreH6', 0)} "
                      f"| Liquidez: ${token.get('liquidity', {}).get('usd', 0):,.0f} "
                      f"| URL: {token.get('url')}")
        else:
            print("⏳ Sin tokens destacados ahora mismo.")
        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Script detenido manualmente.")




