"""
Stratégie 3 : Bot de Breakout avec Support/Résistance
=======================================================
Ce module implémente une stratégie de trading basée sur les cassures (breakouts)
des niveaux de support et de résistance calculés sur une fenêtre glissante.

Fonctionnalités :
- Calcul du support (rolling min) et de la résistance (rolling max)
- Détection des breakouts haussiers et baissiers
- Placement d'ordres légèrement au-delà des niveaux cassés
- Gestion du Take-Profit et Stop-Loss configurables
- Compatible avec le module de backtesting (backtest_breakout.py)
"""

import time
from pathlib import Path

import ccxt
import pandas as pd

# ============================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================
try:
    from config.config import (
        API_KEY,
        API_SECRET,
        DEFAULT_SYMBOL,
        DEFAULT_TIMEFRAME,
        MAX_LOSS_PCT,
        TARGET_PROFIT_PCT,
    )
except ImportError:
    API_KEY = ""
    API_SECRET = ""
    DEFAULT_SYMBOL = "BTC/USDT"
    DEFAULT_TIMEFRAME = "4h"
    TARGET_PROFIT_PCT = 9.0
    MAX_LOSS_PCT = 8.0

# Paramètres de la stratégie breakout
WINDOW = 20              # Fenêtre pour le calcul support/résistance (bougies)
BREAKOUT_OFFSET = 0.001  # 0.1% au-delà du niveau cassé pour confirmer le breakout
POSITION_SIZE = 10       # Taille de la position en unités
TP_PCT = 0.09            # Take-Profit par défaut (9%)
SL_PCT = 0.08            # Stop-Loss par défaut (8%)
PARAMS = {"postOnly": True}  # Maker uniquement

# ============================================================
# CONNEXION À L'EXCHANGE
# ============================================================
exchange = ccxt.phemex({
    "enableRateLimit": True,
    "apiKey": API_KEY,
    "secret": API_SECRET,
})


# ============================================================
# CALCUL SUPPORT / RÉSISTANCE
# ============================================================

def calculate_support_resistance(
    df: pd.DataFrame,
    window: int = WINDOW,
) -> pd.DataFrame:
    """
    Calcule les niveaux de support et de résistance rolling.

    Utilise uniquement les données passées (shift(1)) pour éviter
    tout biais de look-ahead (consultation du futur).

    - Support    : minimum des `window` dernières clôtures (shift(1))
    - Résistance : maximum des `window` dernières clôtures (shift(1))

    :param df: DataFrame OHLCV avec colonne 'close'
    :param window: Taille de la fenêtre glissante en nombre de bougies
    :return: DataFrame enrichi avec les colonnes 'support' et 'resistance'
    """
    df = df.copy()

    # Approche shift(1)-avant-rolling : à chaque instant t, on calcule le min/max
    # des `window` clôtures précédentes [t-window, t-1], excluant la bougie courante.
    # shift(1) décale la série d'un pas en avant, puis rolling() accumule les w valeurs.
    # Cette méthode garantit l'absence totale de look-ahead bias pour le backtesting.
    df["support"] = df["close"].shift(1).rolling(window=window, min_periods=window).min()
    df["resistance"] = df["close"].shift(1).rolling(window=window, min_periods=window).max()

    print(f"  📊 Support/Résistance calculés (fenêtre : {window} bougies)")
    derniere_ligne = df.dropna(subset=["support", "resistance"]).iloc[-1]
    print(f"  🔻 Support actuel    : {derniere_ligne['support']:.2f}")
    print(f"  🔺 Résistance actuelle : {derniere_ligne['resistance']:.2f}")

    return df


# ============================================================
# DÉTECTION DU SIGNAL BREAKOUT
# ============================================================

def get_breakout_signal(
    df: pd.DataFrame,
    current_bid: float,
    offset: float = BREAKOUT_OFFSET,
) -> tuple[str, float, float]:
    """
    Détecte le signal de breakout en comparant le prix actuel aux niveaux S/R.

    Logique :
    - Si bid > résistance précédente × (1 + offset) → BREAKOUT HAUSSIER (long)
    - Si bid < support précédent × (1 - offset)     → BREAKOUT BAISSIER (short)
    - Sinon → Pas de signal

    :param df: DataFrame avec colonnes 'support' et 'resistance'
    :param current_bid: Prix bid actuel
    :param offset: Pourcentage de dépassement requis pour confirmer le breakout
    :return: Tuple (signal, support, resistance)
             - signal : 'long', 'short' ou 'none'
             - support : dernier niveau de support
             - resistance : dernier niveau de résistance
    """
    # Récupérer la dernière ligne avec des valeurs S/R valides
    df_valide = df.dropna(subset=["support", "resistance"])
    if df_valide.empty:
        return "none", 0.0, 0.0

    derniere = df_valide.iloc[-1]
    support = float(derniere["support"])
    resistance = float(derniere["resistance"])

    # Vérification du breakout haussier
    if current_bid > resistance * (1 + offset):
        signal = "long"
        print(f"  🚀 BREAKOUT HAUSSIER détecté ! Bid {current_bid:.2f} > Résistance {resistance:.2f}")
    # Vérification du breakout baissier
    elif current_bid < support * (1 - offset):
        signal = "short"
        print(f"  💥 BREAKOUT BAISSIER détecté ! Bid {current_bid:.2f} < Support {support:.2f}")
    else:
        signal = "none"
        print(f"  ⏸️ Pas de breakout (Support: {support:.2f} | Résistance: {resistance:.2f} | Bid: {current_bid:.2f})")

    return signal, support, resistance


# ============================================================
# CALCUL DES PRIX D'ORDRE, TP ET SL
# ============================================================

def calculate_entry_tp_sl(
    signal: str,
    support: float,
    resistance: float,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    offset: float = BREAKOUT_OFFSET,
) -> tuple[float, float, float]:
    """
    Calcule le prix d'entrée, le take-profit et le stop-loss pour un breakout.

    Placement des ordres :
    - Long  : entrée à 0.1% au-dessus de l'ancienne résistance
    - Short : entrée à 0.1% en dessous de l'ancien support

    :param signal: Type de signal ('long' ou 'short')
    :param support: Niveau de support
    :param resistance: Niveau de résistance
    :param tp_pct: Pourcentage de take-profit (ex: 0.09 = 9%)
    :param sl_pct: Pourcentage de stop-loss (ex: 0.08 = 8%)
    :param offset: Décalage depuis le niveau cassé (ex: 0.001 = 0.1%)
    :return: Tuple (prix_entree, take_profit, stop_loss)
    """
    if signal == "long":
        # Entrer légèrement au-dessus de la résistance cassée
        prix_entree = resistance * (1 + offset)
        take_profit = prix_entree * (1 + tp_pct)
        stop_loss = prix_entree * (1 - sl_pct)
    elif signal == "short":
        # Entrer légèrement en dessous du support cassé
        prix_entree = support * (1 - offset)
        take_profit = prix_entree * (1 - tp_pct)
        stop_loss = prix_entree * (1 + sl_pct)
    else:
        return 0.0, 0.0, 0.0

    print(f"  💰 Entrée : {prix_entree:.2f} | TP : {take_profit:.2f} | SL : {stop_loss:.2f}")
    return prix_entree, take_profit, stop_loss


# ============================================================
# RÉCUPÉRATION DES DONNÉES OHLCV
# ============================================================

def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 200,
) -> pd.DataFrame:
    """
    Récupère les données OHLCV depuis l'exchange via CCXT.

    :param symbol: Symbole de trading (ex: 'BTC/USDT')
    :param timeframe: Timeframe (ex: '4h', '1d')
    :param limit: Nombre de bougies à récupérer
    :return: DataFrame OHLCV avec colonnes : timestamp, open, high, low, close, volume
    """
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        print(f"  📊 {len(df)} bougies récupérées pour {symbol} ({timeframe})")
        return df
    except Exception as e:
        print(f"  ❌ Erreur fetch_ohlcv ({symbol}): {e}")
        raise


# ============================================================
# PLACEMENT D'ORDRES BREAKOUT
# ============================================================

def place_breakout_order(
    symbol: str,
    signal: str,
    prix_entree: float,
    take_profit: float,
    stop_loss: float,
    size: int = POSITION_SIZE,
) -> None:
    """
    Place un ordre de breakout avec les niveaux TP et SL calculés.

    :param symbol: Symbole de trading
    :param signal: 'long' ou 'short'
    :param prix_entree: Prix d'entrée de l'ordre limite
    :param take_profit: Prix du take-profit
    :param stop_loss: Prix du stop-loss
    :param size: Taille de la position
    """
    try:
        exchange.cancel_all_orders(symbol)

        if signal == "long":
            # Ordre d'achat limite légèrement au-dessus de la résistance
            exchange.create_limit_buy_order(symbol, size, prix_entree, PARAMS)
            print(f"  🟢 Ordre LONG placé | Entrée : {prix_entree:.2f} | TP : {take_profit:.2f} | SL : {stop_loss:.2f}")

        elif signal == "short":
            # Ordre de vente limite légèrement en dessous du support
            exchange.create_limit_sell_order(symbol, size, prix_entree, PARAMS)
            print(f"  🔴 Ordre SHORT placé | Entrée : {prix_entree:.2f} | TP : {take_profit:.2f} | SL : {stop_loss:.2f}")

    except Exception as e:
        print(f"  ❌ Erreur place_breakout_order: {e}")
        raise


# ============================================================
# BOT PRINCIPAL BREAKOUT
# ============================================================

def run_breakout_bot(
    symbol: str = DEFAULT_SYMBOL,
    timeframe: str = DEFAULT_TIMEFRAME,
    window: int = WINDOW,
    limit: int = 200,
    tp_pct: float = TP_PCT,
    sl_pct: float = SL_PCT,
    size: int = POSITION_SIZE,
) -> None:
    """
    Logique principale du bot breakout à exécuter à chaque itération.

    Ordre d'exécution :
    1. Récupérer les données OHLCV
    2. Calculer support/résistance
    3. Vérifier le signal de breakout
    4. Placer l'ordre si signal détecté

    :param symbol: Symbole de trading
    :param timeframe: Timeframe OHLCV
    :param window: Fenêtre pour le calcul S/R
    :param limit: Nombre de bougies à récupérer
    :param tp_pct: Pourcentage de take-profit
    :param sl_pct: Pourcentage de stop-loss
    :param size: Taille de la position
    """
    print("\n" + "=" * 60)
    print(f"🤖 BOT BREAKOUT EN COURS — {symbol} | {pd.Timestamp.now()}")
    print("=" * 60)

    try:
        # Étape 1 : Récupérer les données
        df = fetch_ohlcv(symbol, timeframe, limit)

        # Étape 2 : Calculer support / résistance
        df = calculate_support_resistance(df, window)

        # Étape 3 : Récupérer le bid actuel et détecter le breakout
        order_book = exchange.fetch_order_book(symbol)
        current_bid = order_book["bids"][0][0]
        print(f"  💹 Bid actuel : {current_bid}")

        signal, support, resistance = get_breakout_signal(df, current_bid)

        # Étape 4 : Placer l'ordre si signal détecté
        if signal in ("long", "short"):
            prix_entree, take_profit, stop_loss = calculate_entry_tp_sl(
                signal, support, resistance, tp_pct, sl_pct
            )
            place_breakout_order(symbol, signal, prix_entree, take_profit, stop_loss, size)
        else:
            print("  ⏸️ Aucun breakout détecté — En attente...")

    except Exception as e:
        print(f"  ❌ Erreur dans le bot breakout : {e}")


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    import schedule

    print("🚀 Démarrage du bot Breakout")
    print(f"   Symbole    : {DEFAULT_SYMBOL}")
    print(f"   Timeframe  : {DEFAULT_TIMEFRAME}")
    print(f"   Fenêtre S/R : {WINDOW} bougies")
    print(f"   TP : {TP_PCT * 100:.1f}% | SL : {SL_PCT * 100:.1f}%")
    print()

    # Exécuter le bot toutes les 60 secondes
    schedule.every(60).seconds.do(run_breakout_bot)

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
