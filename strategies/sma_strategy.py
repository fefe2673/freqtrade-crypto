"""
Stratégie 1 : SMA + Analyse du carnet d'ordres + Kill Switch
=============================================================
Ce module fournit les fonctions utilitaires essentielles pour un bot de trading
algorithmique utilisant la Simple Moving Average (SMA) couplée à une analyse
du carnet d'ordres pour optimiser les entrées et sorties de position.

Fonctionnalités :
- Récupération des prix ask/bid via CCXT
- Calcul de la SMA et génération de signaux buy/sell
- Récupération des positions ouvertes
- Fermeture élégante des positions (kill switch)
- Protection contre le sur-trading (sleep on close)
- Analyse du volume du carnet d'ordres
- Gestion du P&L avec take-profit et stop-loss automatiques
"""

import time

import ccxt
import pandas as pd

# ============================================================
# CONFIGURATION PAR DÉFAUT — À remplacer par config/config.py
# ============================================================
try:
    from config.config import (
        API_KEY,
        API_SECRET,
        DEFAULT_SYMBOL,
        DEFAULT_TIMEFRAME,
        DEFAULT_SMA_PERIOD,
        MAX_LOSS_PCT,
        PAUSE_TIME_MINUTES,
        TARGET_PROFIT_PCT,
    )
except ImportError:
    # Valeurs par défaut si le fichier de config n'est pas présent
    API_KEY = ""
    API_SECRET = ""
    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_TIMEFRAME = "15m"
    DEFAULT_SMA_PERIOD = 20
    TARGET_PROFIT_PCT = 9.0
    MAX_LOSS_PCT = 8.0
    PAUSE_TIME_MINUTES = 10

# Paramètres d'ordre
POSITION_SIZE = 40       # Taille totale de la position (divisée en 2 ordres)
BUY_OFFSET = 200         # Décalage en $ pour les ordres d'achat (sous le bid)
SELL_OFFSET = 200        # Décalage en $ pour les ordres de vente (au-dessus du ask)
VOL_DECIMAL = 0.4        # Seuil de ratio bid/ask pour le maintien de position
PARAMS = {"postOnly": True}  # Maker uniquement (réduit les frais)

# ============================================================
# CONNEXION À L'EXCHANGE
# ============================================================
exchange = ccxt.phemex({
    "enableRateLimit": True,
    "apiKey": API_KEY,
    "secret": API_SECRET,
})


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def ask_bid(symbol: str) -> tuple[float, float]:
    """
    Récupère le meilleur prix ask et bid actuels pour un symbole donné.

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :return: Tuple (ask, bid) — prix de vente et d'achat les plus proches
    """
    try:
        order_book = exchange.fetch_order_book(symbol)
        ask = order_book["asks"][0][0]
        bid = order_book["bids"][0][0]
        print(f"  📊 Ask: {ask} | Bid: {bid} pour {symbol}")
        return ask, bid
    except Exception as e:
        print(f"  ⚠️ Erreur ask_bid ({symbol}): {e}")
        raise


def get_sma(symbol: str, timeframe: str, limit: int, sma_period: int) -> tuple[pd.DataFrame, str]:
    """
    Récupère les données OHLCV et calcule la SMA (Simple Moving Average) rolling.

    Génère un signal buy/sell :
    - Si le bid actuel est AU-DESSUS de la SMA → signal 'buy' (tendance haussière)
    - Si le bid actuel est EN DESSOUS de la SMA → signal 'sell' (tendance baissière)

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :param timeframe: Le timeframe OHLCV (ex: '15m', '1h', '4h')
    :param limit: Nombre de bougies à récupérer
    :param sma_period: Période pour le calcul de la SMA
    :return: Tuple (DataFrame avec SMA et signal, nom de la colonne SMA)
    """
    try:
        # Récupération des données OHLCV depuis l'exchange
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

        # Calcul de la SMA rolling (moyenne mobile simple)
        sma_col = f"SMA_{sma_period}_{timeframe}"
        df[sma_col] = df["close"].rolling(window=sma_period).mean()

        # Récupération du prix actuel pour générer le signal
        current_ask, current_bid = ask_bid(symbol)

        # Signal : buy si le bid est au-dessus de la SMA, sell sinon
        last_sma = df[sma_col].iloc[-1]
        if last_sma is not None and not pd.isna(last_sma):
            signal = "buy" if current_bid > last_sma else "sell"
        else:
            signal = None
        df["signal"] = signal

        print(f"  📈 SMA {sma_period} ({timeframe}) dernière valeur : {last_sma:.2f}")
        print(f"  🚦 Signal : {signal}")
        return df, sma_col

    except Exception as e:
        print(f"  ⚠️ Erreur get_sma ({symbol}): {e}")
        raise


def get_open_positions(symbol: str) -> tuple[bool, int, bool | None]:
    """
    Récupère les informations sur les positions ouvertes pour un symbole.

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :return: Tuple (in_position, size, is_long)
             - in_position (bool) : True si une position est ouverte
             - size (int) : Taille de la position en contrats
             - is_long (bool|None) : True si long, False si short, None si pas de position
    """
    try:
        positions = exchange.fetch_positions([symbol])
        for pos in positions:
            if pos["symbol"] == symbol and float(pos.get("contracts", 0) or 0) > 0:
                size = abs(float(pos["contracts"]))
                is_long = pos["side"] == "long"
                direction = "LONG" if is_long else "SHORT"
                print(f"  📍 Position ouverte : {direction} | Taille : {size}")
                return True, int(size), is_long
        print("  📍 Aucune position ouverte")
        return False, 0, None
    except Exception as e:
        print(f"  ⚠️ Erreur get_open_positions ({symbol}): {e}")
        return False, 0, None


def kill_switch(symbol: str) -> None:
    """
    Ferme élégamment une position ouverte via des ordres limites.

    Boucle jusqu'à la fermeture complète :
    1. Annule tous les ordres ouverts
    2. Place un ordre limite au meilleur prix disponible
    3. Attend 30 secondes
    4. Répète jusqu'à fermeture complète

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    """
    print("  🔴 KILL SWITCH ACTIVÉ")
    in_position, size, is_long = get_open_positions(symbol)

    while in_position and size > 0:
        try:
            # Annuler tous les ordres existants pour repartir proprement
            exchange.cancel_all_orders(symbol)
            current_ask, current_bid = ask_bid(symbol)

            if is_long:
                # Fermer un long → ordre de vente au prix ask
                exchange.create_limit_sell_order(symbol, size, current_ask, PARAMS)
                print(f"  ➡️ Ordre de vente (fermeture long) placé à {current_ask}")
            else:
                # Fermer un short → ordre d'achat au prix bid
                exchange.create_limit_buy_order(symbol, size, current_bid, PARAMS)
                print(f"  ➡️ Ordre d'achat (fermeture short) placé à {current_bid}")

            # Attendre que l'ordre soit exécuté
            time.sleep(30)
            in_position, size, is_long = get_open_positions(symbol)

        except Exception as e:
            print(f"  ⚠️ Erreur kill_switch ({symbol}): {e}")
            time.sleep(10)

    print("  ✅ Kill switch terminé — Position fermée")


def sleep_on_close(symbol: str, pause_time_minutes: float) -> bool:
    """
    Protection contre le sur-trading : vérifie le dernier trade fermé.

    Si un trade a été fermé il y a moins de `pause_time_minutes` minutes,
    le bot dort le temps restant avant de reprendre.

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :param pause_time_minutes: Durée de pause minimale entre les trades (en minutes)
    :return: True si une pause a été effectuée, False sinon
    """
    try:
        closed_orders = exchange.fetch_closed_orders(symbol, limit=5)
        for order in closed_orders:
            if order["status"] == "closed" and order.get("filled", 0) > 0:
                trade_time = order["timestamp"]
                now = exchange.milliseconds()
                minutes_since = (now - trade_time) / 60000

                if minutes_since < pause_time_minutes:
                    sleep_seconds = (pause_time_minutes - minutes_since) * 60
                    print(
                        f"  😴 Dernier trade il y a {minutes_since:.1f}min "
                        f"— Pause de {sleep_seconds:.0f}s pour éviter le sur-trading"
                    )
                    time.sleep(max(sleep_seconds, 0))
                    return True
        return False
    except Exception as e:
        print(f"  ⚠️ Erreur sleep_on_close ({symbol}): {e}")
        return False


def order_book_analysis(symbol: str, vol_repeat: int = 12, vol_time: int = 5) -> tuple[bool, float]:
    """
    Analyse le volume du carnet d'ordres sur une durée configurable.

    Accumule le volume bid et ask pendant (vol_repeat × vol_time) secondes,
    puis détermine si les acheteurs (bulls) ou vendeurs (bears) sont en contrôle.

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :param vol_repeat: Nombre de répétitions de la collecte (défaut : 12)
    :param vol_time: Durée d'attente entre chaque collecte en secondes (défaut : 5)
    :return: Tuple (bulls_in_control, control_ratio)
             - bulls_in_control (bool) : True si les acheteurs dominent
             - control_ratio (float) : Ratio bid_vol / ask_vol
    """
    duree_totale = vol_repeat * vol_time
    print(f"  📖 Analyse du carnet d'ordres sur {duree_totale}s ({vol_repeat}× {vol_time}s)...")

    total_bid_vol = 0.0
    total_ask_vol = 0.0

    for i in range(vol_repeat):
        try:
            ob = exchange.fetch_order_book(symbol)
            bid_vol = sum(level[1] for level in ob["bids"])
            ask_vol = sum(level[1] for level in ob["asks"])
            total_bid_vol += bid_vol
            total_ask_vol += ask_vol
        except Exception as e:
            print(f"  ⚠️ Erreur collecte carnet d'ordres (itération {i + 1}): {e}")
        time.sleep(vol_time)

    # Calcul du ratio bid / ask
    if total_ask_vol > 0:
        control_ratio = round(total_bid_vol / total_ask_vol, 4)
    else:
        control_ratio = 999.0  # Aucun ask = les bulls dominent totalement

    bulls_in_control = total_bid_vol > total_ask_vol
    who = "🐂 Bulls" if bulls_in_control else "🐻 Bears"
    print(f"  {who} en contrôle | Ratio bid/ask : {control_ratio}")
    print(f"  Volume Bid total : {total_bid_vol:,.0f} | Volume Ask total : {total_ask_vol:,.0f}")

    return bulls_in_control, control_ratio


def pnl_close(symbol: str, target_pct: float, max_loss_pct: float) -> None:
    """
    Gestion automatique du P&L (Profit & Loss).

    - Si le P&L ≥ target_pct → analyse du carnet d'ordres puis fermeture si défavorable
    - Si le P&L ≤ -max_loss_pct → fermeture immédiate (stop-loss)

    :param symbol: Le symbole de trading (ex: 'BTC/USDT')
    :param target_pct: Pourcentage de profit cible pour fermer la position (ex: 9.0)
    :param max_loss_pct: Pourcentage de perte maximale toléré (ex: 8.0)
    """
    in_position, size, is_long = get_open_positions(symbol)
    if not in_position:
        print("  ℹ️ Aucune position ouverte — pnl_close ignoré")
        return

    # Récupération du P&L actuel
    try:
        positions = exchange.fetch_positions([symbol])
        pnl = 0.0
        for pos in positions:
            if pos["symbol"] == symbol and float(pos.get("contracts", 0) or 0) > 0:
                pnl = float(pos.get("percentage", 0) or 0)
                break
        print(f"  💰 P&L actuel : {pnl:.2f}%")
    except Exception as e:
        print(f"  ⚠️ Erreur récupération P&L: {e}")
        return

    # Vérification du stop-loss → fermeture immédiate
    if pnl <= -abs(max_loss_pct):
        print(f"  🚨 STOP-LOSS atteint ({pnl:.2f}% ≤ -{max_loss_pct}%) → Fermeture immédiate !")
        kill_switch(symbol)
        return

    # Vérification du take-profit → analyser le carnet d'ordres avant de fermer
    if pnl >= target_pct:
        print(f"  🎯 TAKE-PROFIT atteint ({pnl:.2f}% ≥ {target_pct}%) → Analyse du volume...")

        bulls, ratio = order_book_analysis(symbol, vol_repeat=5, vol_time=1)

        # Si on est long et que les bulls dominent encore → maintenir la position
        # ratio = bid_vol / ask_vol : ratio > VOL_DECIMAL indique que les bulls sont actifs
        if is_long and bulls and ratio > VOL_DECIMAL:
            print(f"  ⏳ Volume favorable pour les bulls (ratio {ratio}) → On maintient la position")
            time.sleep(30)
            return

        # Si on est short et que les bears dominent encore → maintenir la position
        # ratio < (1 - VOL_DECIMAL) confirme que le volume d'offre est significativement supérieur
        if not is_long and not bulls and ratio < (1 - VOL_DECIMAL):
            print(f"  ⏳ Volume favorable pour les bears (ratio {ratio}) → On maintient la position")
            time.sleep(30)
            return

        # Sinon → fermer la position
        print(f"  💸 Conditions défavorables (ratio : {ratio}) → Fermeture de la position")
        kill_switch(symbol)
        return

    print(f"  ⏳ P&L : {pnl:.2f}% — Ni take-profit ni stop-loss atteint, on continue")


# ============================================================
# BOT PRINCIPAL
# ============================================================

def run_sma_bot(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    sma_period: int = DEFAULT_SMA_PERIOD,
    limit: int = 289,
    position_size: int = POSITION_SIZE,
    target_pct: float = TARGET_PROFIT_PCT,
    max_loss_pct: float = MAX_LOSS_PCT,
    pause_minutes: float = PAUSE_TIME_MINUTES,
) -> None:
    """
    Logique principale du bot SMA à exécuter à chaque itération.

    Ordre d'exécution :
    1. Vérifier et gérer le P&L (take-profit / stop-loss)
    2. Vérifier la pause anti-surtrading
    3. Calculer la SMA et générer le signal
    4. Placer les ordres si pas en position

    :param symbol: Symbole à trader
    :param timeframe: Timeframe pour le calcul SMA
    :param sma_period: Période de la SMA
    :param limit: Nombre de bougies OHLCV à récupérer
    :param position_size: Taille totale de la position (divisée en 2 ordres)
    :param target_pct: Pourcentage de profit cible
    :param max_loss_pct: Pourcentage de perte maximale
    :param pause_minutes: Durée de pause entre les trades
    """
    print("\n" + "=" * 60)
    print(f"🤖 BOT SMA EN COURS — {symbol} | {pd.Timestamp.now()}")
    print("=" * 60)

    # Étape 1 : Gestion du P&L
    pnl_close(symbol, target_pct, max_loss_pct)

    # Étape 2 : Anti sur-trading
    if sleep_on_close(symbol, pause_minutes):
        return

    # Étape 3 : Calcul SMA et signal
    df, sma_col = get_sma(symbol, timeframe, limit, sma_period)
    current_ask, current_bid = ask_bid(symbol)
    signal = df["signal"].iloc[-1]
    last_sma = df[sma_col].iloc[-1]

    # Étape 4 : Vérification de la position actuelle
    in_position, current_size, is_long = get_open_positions(symbol)
    open_size = position_size // 2  # Divisé en 2 ordres pour un meilleur prix moyen

    print(f"\n  📋 Résumé de la situation :")
    print(f"     Signal : {signal} | Bid : {current_bid} | SMA : {last_sma:.2f}")
    print(f"     En position : {in_position} | Taille : {current_size}")

    # Étape 5 : Placement des ordres si pas en position
    if not in_position and current_size < position_size:
        exchange.cancel_all_orders(symbol)
        current_ask, current_bid = ask_bid(symbol)  # Rafraîchir les prix

        if signal == "buy" and current_bid > last_sma:
            # === SIGNAL HAUSSIER — Placer des ordres d'ACHAT ===
            # Ordres légèrement en dessous du bid pour un meilleur prix d'entrée
            bp1 = current_bid - BUY_OFFSET        # 1er ordre
            bp2 = current_bid - BUY_OFFSET * 2    # 2ème ordre (encore plus bas)
            print(f"\n  🟢 ACHAT (LONG) | Ordre 1 : {bp1} | Ordre 2 : {bp2}")
            exchange.create_limit_buy_order(symbol, open_size, bp1, PARAMS)
            exchange.create_limit_buy_order(symbol, open_size, bp2, PARAMS)
            print("  ✅ Ordres d'achat soumis → Pause 2 min")
            time.sleep(120)

        elif signal == "sell" and current_bid < last_sma:
            # === SIGNAL BAISSIER — Placer des ordres de VENTE ===
            # Ordres légèrement au-dessus du ask pour un meilleur prix d'entrée
            sp1 = current_ask + SELL_OFFSET        # 1er ordre
            sp2 = current_ask + SELL_OFFSET * 2    # 2ème ordre (encore plus haut)
            print(f"\n  🔴 VENTE (SHORT) | Ordre 1 : {sp1} | Ordre 2 : {sp2}")
            exchange.create_limit_sell_order(symbol, open_size, sp1, PARAMS)
            exchange.create_limit_sell_order(symbol, open_size, sp2, PARAMS)
            print("  ✅ Ordres de vente soumis → Pause 2 min")
            time.sleep(120)

        else:
            print("\n  ⏸️ Pas de signal clair — Pas d'ordre placé → Pause 10 min")
            time.sleep(600)

    else:
        print("\n  📌 Déjà en position — Surveillance du P&L uniquement")


# ============================================================
# LANCEMENT DIRECT (pour les tests)
# ============================================================

if __name__ == "__main__":
    import schedule

    print("🚀 Démarrage du bot SMA")
    print(f"   Symbole     : {DEFAULT_SYMBOL}")
    print(f"   Timeframe   : {DEFAULT_TIMEFRAME}")
    print(f"   SMA         : {DEFAULT_SMA_PERIOD}")
    print(f"   Target      : {TARGET_PROFIT_PCT}%")
    print(f"   Max Loss    : {MAX_LOSS_PCT}%")
    print()

    # Exécuter le bot toutes les 28 secondes
    schedule.every(28).seconds.do(run_sma_bot)

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n⛔ Bot arrêté manuellement")
            break
        except Exception as e:
            print(f"⚠️ Erreur inattendue : {e} — Nouvelle tentative dans 30s")
            time.sleep(30)
