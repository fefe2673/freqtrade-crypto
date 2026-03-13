# ============================================================
# CONFIGURATION EXEMPLE — Stratégies de Trading
# ============================================================
# ATTENTION : Ne jamais partager vos clés API ni committer ce
# fichier avec de vraies credentials dans votre dépôt !
#
# Instructions :
# 1. Copiez ce fichier : cp config_example.py config.py
# 2. Remplissez vos vraies clés API dans config.py
# 3. config.py est dans .gitignore — il ne sera pas commis
# ============================================================

# --- Clés API de l'exchange (Phemex par défaut) ---
# Remplacez par vos vraies clés API (ne jamais partager !)
API_KEY = "votre_cle_api_ici"
API_SECRET = "votre_secret_api_ici"

# --- Paramètres généraux ---
DEFAULT_SYMBOL = "BTC/USDT"        # Symbole à trader par défaut
DEFAULT_TIMEFRAME = "4h"           # Timeframe par défaut
DEFAULT_SMA_PERIOD = 20            # Période de la SMA
DEFAULT_POSITION_SIZE = 10         # Taille de position par défaut (en contrats ou unités)

# --- Paramètres de risque ---
TARGET_PROFIT_PCT = 9.0            # Pourcentage de profit cible (ex: 9.0 = 9%)
MAX_LOSS_PCT = 8.0                 # Pourcentage de perte maximale toléré (ex: 8.0 = 8%)
PAUSE_TIME_MINUTES = 10            # Durée de pause entre les trades (minutes)

# --- Paramètres de la stratégie SMA ---
SMA_NUM_BARS = 289                 # Nombre de bougies OHLCV à récupérer (~3 jours en 15m)
SMA_BUY_OFFSET = 200               # Décalage en $ pour les ordres d'achat (sous le bid)
SMA_SELL_OFFSET = 200              # Décalage en $ pour les ordres de vente (sur le ask)
SMA_VOL_DECIMAL = 0.4              # Seuil de ratio bid/ask pour maintenir une position

# --- Paramètres de la stratégie Breakout ---
BREAKOUT_WINDOW = 20               # Fenêtre rolling pour le calcul Support/Résistance
BREAKOUT_OFFSET_PCT = 0.1          # % de dépassement requis pour confirmer un breakout
BREAKOUT_TP_PCT = 9.0              # Take-Profit de la stratégie breakout (%)
BREAKOUT_SL_PCT = 8.0              # Stop-Loss de la stratégie breakout (%)

# --- Paramètres du modèle HMM ---
HMM_N_COMPONENTS = 4               # Nombre de régimes de marché (4 ou 7)
HMM_N_ITER = 1000                  # Nombre d'itérations d'entraînement du HMM
HMM_VOLATILITY_WINDOW = 20         # Fenêtre pour le calcul de la volatilité rolling

# --- Exchange ---
EXCHANGE_NAME = "phemex"           # Nom de l'exchange CCXT (ex: 'binance', 'phemex')
EXCHANGE_SANDBOX = False           # True = mode paper trading (testnet)
