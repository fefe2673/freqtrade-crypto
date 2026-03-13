"""
Backtest de la Stratégie Breakout avec la librairie backtesting.py
===================================================================
Ce module permet de tester et d'optimiser la stratégie de breakout
(support/résistance) sur des données historiques OHLCV.

Fonctionnalités :
- Chargement des données OHLCV depuis un CSV
- Implémentation de la stratégie Breakout comme classe Strategy
- Calcul du support et de la résistance rolling (sans biais de look-ahead)
- Exécution du backtest avec Backtest.run()
- Optimisation des paramètres TP et SL avec Backtest.optimize()
- Affichage des résultats et du heat map d'optimisation
"""

import sys
from pathlib import Path

import pandas as pd

# Ajouter le dossier racine au path pour les imports locaux
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError:
    print("❌ Librairie 'backtesting' non installée.")
    print("   Installer avec : pip install backtesting")
    sys.exit(1)

import numpy as np

# ============================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Paramètres par défaut de la stratégie
DEFAULT_WINDOW = 20         # Fenêtre rolling pour support/résistance
DEFAULT_TP_PCT = 9          # Take-profit en % (pour l'optimisation : entier)
DEFAULT_SL_PCT = 8          # Stop-loss en % (pour l'optimisation : entier)
DEFAULT_OFFSET_PCT = 0.1    # Offset de confirmation du breakout (en %)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def load_ohlcv_for_backtest(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV OHLCV et le formate pour la librairie backtesting.py.

    La librairie backtesting.py requiert un DataFrame avec les colonnes :
    Open, High, Low, Close, Volume (majuscules) et un index datetime.

    :param filepath: Chemin vers le fichier CSV
    :return: DataFrame formaté pour backtesting.py
    """
    print(f"  📂 Chargement des données depuis : {filepath}")

    df = pd.read_csv(filepath)

    # Normaliser le nom de la colonne de date
    date_col = None
    for col in ["timestamp", "date", "Datetime", "Date", "Time"]:
        if col in df.columns:
            date_col = col
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
    else:
        df.index = pd.to_datetime(df.iloc[:, 0])
        df = df.iloc[:, 1:]

    # Renommer les colonnes pour backtesting.py (format requis)
    rename_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Vérifier les colonnes requises
    colonnes_requises = ["Open", "High", "Low", "Close", "Volume"]
    colonnes_manquantes = [c for c in colonnes_requises if c not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes manquantes : {colonnes_manquantes}")

    # Convertir en numérique
    for col in colonnes_requises:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=colonnes_requises)
    df = df.sort_index()

    print(f"  ✅ {len(df)} bougies chargées — du {df.index[0]} au {df.index[-1]}")
    return df


def rolling_support(series: pd.Series, window: int) -> pd.Series:
    """
    Calcule le support rolling (minimum) sans biais de look-ahead.

    Utilise shift(1)-avant-rolling : à l'instant t, la valeur est le min des `window`
    clôtures précédentes [t-window, t-1], excluant la bougie courante.
    Cette approche garantit l'absence de look-ahead bias dans le backtest.

    :param series: Série des prix de clôture
    :param window: Taille de la fenêtre rolling
    :return: Série des niveaux de support
    """
    return series.shift(1).rolling(window=window, min_periods=window).min()


def rolling_resistance(series: pd.Series, window: int) -> pd.Series:
    """
    Calcule la résistance rolling (maximum) sans biais de look-ahead.

    Utilise shift(1)-avant-rolling : à l'instant t, la valeur est le max des `window`
    clôtures précédentes [t-window, t-1], excluant la bougie courante.
    Cette approche garantit l'absence de look-ahead bias dans le backtest.

    :param series: Série des prix de clôture
    :param window: Taille de la fenêtre rolling
    :return: Série des niveaux de résistance
    """
    return series.shift(1).rolling(window=window, min_periods=window).max()


# ============================================================
# STRATÉGIE BREAKOUT POUR BACKTESTING.PY
# ============================================================

class BreakoutStrategy(Strategy):
    """
    Stratégie de Breakout avec Support/Résistance pour la librairie backtesting.py.

    Paramètres optimisables :
    - window   : Fenêtre rolling pour le calcul S/R (nombre de bougies)
    - tp_pct   : Take-Profit en % (entier, ex: 9 = 9%)
    - sl_pct   : Stop-Loss en % (entier, ex: 8 = 8%)
    - offset   : Pourcentage de dépassement requis pour confirmer le breakout

    Logique :
    - Long  si Close > Résistance précédente × (1 + offset/100)
    - Short si Close < Support précédent × (1 - offset/100)
    """

    # Paramètres optimisables (modifiables lors de l'optimisation)
    window = DEFAULT_WINDOW
    tp_pct = DEFAULT_TP_PCT
    sl_pct = DEFAULT_SL_PCT
    offset = DEFAULT_OFFSET_PCT

    def init(self):
        """Initialise les indicateurs support et résistance."""
        close = pd.Series(self.data.Close, index=self.data.index)

        # Calcul du support et de la résistance avec la fonction de la stratégie
        self.support = self.I(
            lambda c: rolling_support(pd.Series(c), self.window).values,
            self.data.Close,
            name="Support",
        )
        self.resistance = self.I(
            lambda c: rolling_resistance(pd.Series(c), self.window).values,
            self.data.Close,
            name="Résistance",
        )

    def next(self):
        """Logique de trading exécutée à chaque nouvelle bougie."""
        # Ignorer les premières bougies sans données S/R complètes
        if np.isnan(self.support[-1]) or np.isnan(self.resistance[-1]):
            return

        current_price = self.data.Close[-1]
        support = self.support[-1]
        resistance = self.resistance[-1]

        # Calcul du facteur de confirmation du breakout
        facteur_haussier = resistance * (1 + self.offset / 100)
        facteur_baissier = support * (1 - self.offset / 100)

        # === BREAKOUT HAUSSIER → LONG ===
        if not self.position and current_price > facteur_haussier:
            prix_entree = resistance * (1 + self.offset / 100)
            tp = prix_entree * (1 + self.tp_pct / 100)
            sl = prix_entree * (1 - self.sl_pct / 100)
            self.buy(tp=tp, sl=sl)

        # === BREAKOUT BAISSIER → SHORT ===
        elif not self.position and current_price < facteur_baissier:
            prix_entree = support * (1 - self.offset / 100)
            tp = prix_entree * (1 - self.tp_pct / 100)
            sl = prix_entree * (1 + self.sl_pct / 100)
            self.sell(tp=tp, sl=sl)


# ============================================================
# EXÉCUTION DU BACKTEST
# ============================================================

def run_backtest(
    df: pd.DataFrame,
    cash: float = 10000.0,
    commission: float = 0.001,
    window: int = DEFAULT_WINDOW,
    tp_pct: float = DEFAULT_TP_PCT,
    sl_pct: float = DEFAULT_SL_PCT,
    plot: bool = True,
) -> dict:
    """
    Exécute le backtest de la stratégie Breakout sur les données historiques.

    :param df: DataFrame OHLCV formaté pour backtesting.py
    :param cash: Capital initial en USD
    :param commission: Commission par trade (ex: 0.001 = 0.1%)
    :param window: Fenêtre rolling pour le calcul S/R
    :param tp_pct: Take-Profit en %
    :param sl_pct: Stop-Loss en %
    :param plot: Afficher le graphique du backtest (True/False)
    :return: Dictionnaire des résultats du backtest
    """
    print("\n" + "=" * 60)
    print("📊 BACKTEST STRATÉGIE BREAKOUT")
    print("=" * 60)
    print(f"  Capital initial : {cash:,.0f} USD")
    print(f"  Commission      : {commission * 100:.2f}%")
    print(f"  Fenêtre S/R     : {window} bougies")
    print(f"  Take-Profit     : {tp_pct}%")
    print(f"  Stop-Loss       : {sl_pct}%")

    # Configuration de la stratégie avec les paramètres choisis
    BreakoutStrategy.window = window
    BreakoutStrategy.tp_pct = tp_pct
    BreakoutStrategy.sl_pct = sl_pct

    # Création et exécution du backtest
    bt = Backtest(df, BreakoutStrategy, cash=cash, commission=commission)
    resultats = bt.run()

    # Affichage des résultats clés
    print("\n  📈 RÉSULTATS DU BACKTEST :")
    print(f"  Return total           : {resultats['Return [%]']:.2f}%")
    print(f"  Return Buy & Hold      : {resultats['Buy & Hold Return [%]']:.2f}%")
    print(f"  Sharpe Ratio           : {resultats['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown           : {resultats['Max. Drawdown [%]']:.2f}%")
    print(f"  Nombre de trades       : {resultats['# Trades']}")
    print(f"  Win Rate               : {resultats['Win Rate [%]']:.1f}%")
    print(f"  Profit Factor          : {resultats.get('Profit Factor', 'N/A')}")
    print(f"  Capital final          : {resultats['Equity Final [$]']:,.2f} USD")

    if plot:
        try:
            bt.plot()
        except Exception as e:
            print(f"  ⚠️ Impossible d'afficher le graphique : {e}")

    return dict(resultats)


# ============================================================
# OPTIMISATION DES PARAMÈTRES
# ============================================================

def optimize_backtest(
    df: pd.DataFrame,
    cash: float = 10000.0,
    commission: float = 0.001,
    tp_range: range = range(3, 21, 3),
    sl_range: range = range(3, 21, 3),
    window_range: range = range(10, 51, 10),
    maximize: str = "Sharpe Ratio",
    plot_heatmap: bool = True,
) -> tuple[dict, dict]:
    """
    Optimise les paramètres TP, SL et window de la stratégie Breakout.

    :param df: DataFrame OHLCV formaté pour backtesting.py
    :param cash: Capital initial en USD
    :param commission: Commission par trade
    :param tp_range: Plage de valeurs pour le Take-Profit (%)
    :param sl_range: Plage de valeurs pour le Stop-Loss (%)
    :param window_range: Plage de valeurs pour la fenêtre S/R
    :param maximize: Métrique à maximiser ('Sharpe Ratio', 'Return [%]', etc.)
    :param plot_heatmap: Afficher le heat map d'optimisation
    :return: Tuple (meilleurs_params, meilleurs_résultats)
    """
    print("\n" + "=" * 60)
    print("🔍 OPTIMISATION DES PARAMÈTRES BREAKOUT")
    print("=" * 60)
    print(f"  TP      : {list(tp_range)}")
    print(f"  SL      : {list(sl_range)}")
    print(f"  Fenêtre : {list(window_range)}")
    print(f"  Optimisation de : {maximize}")
    print("  ⏳ Optimisation en cours...")

    bt = Backtest(df, BreakoutStrategy, cash=cash, commission=commission)

    # Lancer l'optimisation sur la grille de paramètres
    stats, heatmap = bt.optimize(
        tp_pct=tp_range,
        sl_pct=sl_range,
        window=window_range,
        maximize=maximize,
        return_heatmap=True,
    )

    meilleurs_params = {
        "tp_pct": stats._strategy.tp_pct,
        "sl_pct": stats._strategy.sl_pct,
        "window": stats._strategy.window,
    }

    print("\n  🏆 MEILLEURS PARAMÈTRES TROUVÉS :")
    print(f"  Take-Profit : {meilleurs_params['tp_pct']}%")
    print(f"  Stop-Loss   : {meilleurs_params['sl_pct']}%")
    print(f"  Fenêtre S/R : {meilleurs_params['window']} bougies")
    print(f"\n  📊 Performance avec les meilleurs paramètres :")
    print(f"  Return total    : {stats['Return [%]']:.2f}%")
    print(f"  Sharpe Ratio    : {stats['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown    : {stats['Max. Drawdown [%]']:.2f}%")
    print(f"  Win Rate        : {stats['Win Rate [%]']:.1f}%")

    # Afficher le heat map TP vs SL
    if plot_heatmap:
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Créer un pivot table pour le heat map (TP vs SL, moyenné sur les windows)
            hm_data = heatmap.reset_index()
            if "window" in hm_data.columns:
                hm_pivot = hm_data.groupby(["tp_pct", "sl_pct"])[maximize].mean().unstack()
            else:
                hm_pivot = hm_data.pivot(index="tp_pct", columns="sl_pct", values=maximize)

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                hm_pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                linewidths=0.5,
            )
            plt.title(f"Heat Map d'optimisation — {maximize}\n(TP% vs SL%)")
            plt.xlabel("Stop-Loss (%)")
            plt.ylabel("Take-Profit (%)")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  ⚠️ seaborn non installé — heat map ignoré (pip install seaborn)")
        except Exception as e:
            print(f"  ⚠️ Impossible d'afficher le heat map : {e}")

    return meilleurs_params, dict(stats)


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    import os

    print("📊 BACKTEST — Stratégie Breakout Support/Résistance")
    print("=" * 60)

    # Rechercher un fichier CSV dans le dossier data/
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        # Créer des données de test synthétiques si aucun CSV n'est trouvé
        print("  ⚠️ Aucun CSV trouvé dans data/ — Génération de données synthétiques pour la démo")

        np.random.seed(42)
        n = 1000
        dates = pd.date_range("2022-01-01", periods=n, freq="4h")
        prix = 30000 + np.cumsum(np.random.randn(n) * 100)
        df_demo = pd.DataFrame({
            "Open": prix + np.random.randn(n) * 50,
            "High": prix + abs(np.random.randn(n)) * 150,
            "Low": prix - abs(np.random.randn(n)) * 150,
            "Close": prix,
            "Volume": abs(np.random.randn(n)) * 1000 + 500,
        }, index=dates)
        df_demo["High"] = df_demo[["Open", "High", "Close"]].max(axis=1)
        df_demo["Low"] = df_demo[["Open", "Low", "Close"]].min(axis=1)

        # Sauvegarder pour référence future
        os.makedirs(DATA_DIR, exist_ok=True)
        demo_path = DATA_DIR / "demo_ohlcv.csv"
        df_demo.to_csv(demo_path)
        print(f"  💾 Données de démo sauvegardées dans : {demo_path}")
        df_test = df_demo

    else:
        csv_path = str(csv_files[0])
        print(f"  📂 Utilisation de : {csv_path}")
        df_test = load_ohlcv_for_backtest(csv_path)

    # --- Backtest simple ---
    resultats = run_backtest(df_test, plot=False)

    # --- Optimisation ---
    print("\n  Lancement de l'optimisation...")
    meilleurs_params, stats_optimises = optimize_backtest(
        df_test,
        tp_range=range(3, 16, 3),
        sl_range=range(3, 16, 3),
        window_range=range(10, 31, 10),
        plot_heatmap=False,
    )

    print("\n✅ Backtest et optimisation terminés")
