"""
Backtest de la Stratégie basée sur les Régimes HMM
====================================================
Ce module combine le modèle Hidden Markov Model (HMM) avec un backtest
pour évaluer la performance d'une stratégie basée sur les régimes de marché.

Logique :
- Entraîner le HMM sur les données historiques
- Identifier les régimes favorables aux achats/ventes
- Backtester la stratégie : acheter en régime risk-on, vendre en régime risk-off
- Comparer avec une stratégie buy & hold de référence

Fonctionnalités :
- Chargement des données OHLCV depuis CSV
- Entraînement HMM et prédiction des régimes
- Simulation des trades basés sur les changements de régime
- Calcul des métriques de performance (return, Sharpe, drawdown)
- Visualisation comparative HMM vs Buy & Hold
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ajouter le dossier racine au path pour les imports locaux
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from strategies.hmm_strategy import (
        REGIME_NAMES_4,
        feature_engineering,
        load_ohlcv_csv,
        train_hmm,
    )
except ImportError as e:
    print(f"❌ Erreur d'import des modules HMM : {e}")
    print("   Assurez-vous que le module strategies/hmm_strategy.py est présent.")
    sys.exit(1)

# ============================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Paramètres HMM
N_COMPONENTS = 4          # Nombre de régimes (4 par défaut)
TRAIN_SPLIT = 0.7         # 70% pour l'entraînement, 30% pour le test

# Paramètres de trading
CASH_INITIAL = 10000.0    # Capital de départ en USD
COMMISSION = 0.001        # Commission par trade (0.1%)
POSITION_SIZE_PCT = 1.0   # Utiliser 100% du capital disponible


# ============================================================
# IDENTIFICATION DES RÉGIMES FAVORABLES
# ============================================================

def identify_bullish_regimes(
    model,
    n_components: int = N_COMPONENTS,
) -> list[int]:
    """
    Identifie les régimes de marché favorables aux positions longues.

    Analyse les moyennes des features par régime pour déterminer
    quels états représentent des conditions haussières (rendements positifs).

    :param model: Modèle HMM entraîné
    :param n_components: Nombre de régimes
    :return: Liste des indices de régimes haussiers
    """
    regimes_haussiers = []

    for i in range(n_components):
        # La première feature est 'returns' (indice 0 dans le vecteur de moyennes)
        rendement_moyen = model.means_[i][0]  # Feature 0 = returns

        if rendement_moyen > 0:
            regimes_haussiers.append(i)
            print(f"  📈 Régime {i} ({REGIME_NAMES_4.get(i, f'R{i}')}) : rendement moyen = {rendement_moyen:.6f} → HAUSSIER")
        else:
            print(f"  📉 Régime {i} ({REGIME_NAMES_4.get(i, f'R{i}')}) : rendement moyen = {rendement_moyen:.6f} → BAISSIER")

    return regimes_haussiers


# ============================================================
# SIMULATION DU BACKTEST HMM
# ============================================================

def simulate_hmm_strategy(
    df: pd.DataFrame,
    states: np.ndarray,
    regimes_haussiers: list[int],
    cash: float = CASH_INITIAL,
    commission: float = COMMISSION,
) -> pd.DataFrame:
    """
    Simule la stratégie basée sur les régimes HMM.

    Logique de trading :
    - Régime HAUSSIER → Acheter (ou maintenir une position longue)
    - Régime BAISSIER → Vendre (ou rester en cash)

    :param df: DataFrame avec les données OHLCV et features
    :param states: Array des régimes prédits par le HMM
    :param regimes_haussiers: Liste des indices de régimes favorables
    :param cash: Capital initial
    :param commission: Taux de commission par trade
    :return: DataFrame avec les trades simulés et la valeur du portefeuille
    """
    print(f"\n  💼 Simulation de la stratégie HMM ({len(df)} points)...")

    portfolio = []
    capital = cash
    position = 0.0       # Nombre d'unités détenues
    en_position = False
    prix_achat = 0.0     # Prix d'achat de la position actuelle (pour le win rate)
    trades_fermes = []   # Liste des trades (prix_achat, prix_vente) pour le win rate

    for i in range(len(df)):
        prix = float(df["close"].iloc[i])
        regime = int(states[i])
        est_haussier = regime in regimes_haussiers

        trade_type = "hold"

        # Signal d'achat : régime haussier sans position ouverte
        if est_haussier and not en_position and capital > 0:
            # Acheter avec tout le capital disponible
            frais = capital * commission
            capital_net = capital - frais
            position = capital_net / prix
            capital = 0.0
            en_position = True
            prix_achat = prix  # Mémoriser le prix d'achat pour calculer le win rate
            trade_type = "buy"

        # Signal de vente : régime baissier avec position ouverte
        elif not est_haussier and en_position and position > 0:
            # Vendre toute la position
            valeur_brute = position * prix
            frais = valeur_brute * commission
            capital = valeur_brute - frais
            position = 0.0
            en_position = False
            # Enregistrer le trade pour le calcul du win rate
            trades_fermes.append({"prix_achat": prix_achat, "prix_vente": prix})
            prix_achat = 0.0
            trade_type = "sell"

        # Valeur totale du portefeuille à ce point
        valeur_portfolio = capital + position * prix

        portfolio.append({
            "date": df.index[i],
            "close": prix,
            "regime": regime,
            "regime_nom": REGIME_NAMES_4.get(regime, f"R{regime}"),
            "est_haussier": est_haussier,
            "en_position": en_position,
            "capital": capital,
            "position": position,
            "valeur_portfolio": valeur_portfolio,
            "trade": trade_type,
        })

    df_portfolio = pd.DataFrame(portfolio)
    df_portfolio = df_portfolio.set_index("date")

    # Forcer la fermeture de la dernière position si nécessaire
    if en_position and position > 0:
        dernier_prix = float(df["close"].iloc[-1])
        valeur_finale = capital + position * dernier_prix
        df_portfolio.iloc[-1, df_portfolio.columns.get_loc("valeur_portfolio")] = valeur_finale
        # Enregistrer ce trade forcé pour le win rate
        trades_fermes.append({"prix_achat": prix_achat, "prix_vente": dernier_prix})
        print(f"  ⚠️ Position fermée à la fin du backtest au prix {dernier_prix:.2f}")

    # Stocker les trades dans un attribut du DataFrame pour les métriques
    df_portfolio.attrs["trades_fermes"] = trades_fermes
    return df_portfolio


# ============================================================
# CALCUL DES MÉTRIQUES DE PERFORMANCE
# ============================================================

def calculate_performance_metrics(
    df_portfolio: pd.DataFrame,
    df_original: pd.DataFrame,
    cash_initial: float = CASH_INITIAL,
) -> dict:
    """
    Calcule et affiche les métriques de performance du backtest HMM.

    Métriques calculées :
    - Return total HMM vs Buy & Hold
    - Sharpe Ratio (annualisé)
    - Maximum Drawdown
    - Nombre de trades
    - Win Rate

    :param df_portfolio: DataFrame résultat de la simulation
    :param df_original: DataFrame OHLCV original
    :param cash_initial: Capital initial
    :return: Dictionnaire des métriques
    """
    print("\n  📊 CALCUL DES MÉTRIQUES DE PERFORMANCE...")

    # --- Stratégie HMM ---
    valeur_finale_hmm = df_portfolio["valeur_portfolio"].iloc[-1]
    return_hmm = (valeur_finale_hmm - cash_initial) / cash_initial * 100

    # Rendements journaliers de la stratégie HMM
    rendements_hmm = df_portfolio["valeur_portfolio"].pct_change().dropna()

    # Sharpe Ratio annualisé (252 jours trading par an)
    if rendements_hmm.std() > 0:
        sharpe_hmm = (rendements_hmm.mean() / rendements_hmm.std()) * np.sqrt(252)
    else:
        sharpe_hmm = 0.0

    # Maximum Drawdown
    valeur_cumul = df_portfolio["valeur_portfolio"]
    pic_cumul = valeur_cumul.cummax()
    drawdown = (valeur_cumul - pic_cumul) / pic_cumul
    max_drawdown_hmm = float(drawdown.min() * 100)

    # Nombre de trades et win rate
    # Utiliser les trades enregistrés pendant la simulation (prix d'achat/vente directs)
    trades = df_portfolio[df_portfolio["trade"].isin(["buy", "sell"])]
    n_trades = len(trades)

    # Win rate : calculé à partir des paires achat/vente enregistrées dans la simulation
    trades_fermes_list = df_portfolio.attrs.get("trades_fermes", [])
    n_trades_fermes = len(trades_fermes_list)
    wins = sum(1 for t in trades_fermes_list if t["prix_vente"] > t["prix_achat"])
    win_rate = (wins / n_trades_fermes * 100) if n_trades_fermes > 0 else 0.0

    # --- Stratégie Buy & Hold (référence) ---
    prix_debut = float(df_original["close"].iloc[0])
    prix_fin = float(df_original["close"].iloc[-1])
    return_bah = (prix_fin - prix_debut) / prix_debut * 100

    rendements_bah = df_original["close"].pct_change().dropna()
    if rendements_bah.std() > 0:
        sharpe_bah = (rendements_bah.mean() / rendements_bah.std()) * np.sqrt(252)
    else:
        sharpe_bah = 0.0

    prix_cumul = df_original["close"]
    pic_bah = prix_cumul.cummax()
    drawdown_bah = (prix_cumul - pic_bah) / pic_bah
    max_drawdown_bah = float(drawdown_bah.min() * 100)

    # Affichage des résultats
    print("\n  " + "=" * 50)
    print("  📈 RÉSULTATS DU BACKTEST HMM")
    print("  " + "=" * 50)
    print(f"  {'Métrique':<30} {'HMM':>12} {'Buy & Hold':>12}")
    print("  " + "-" * 54)
    print(f"  {'Return Total':<30} {return_hmm:>11.2f}% {return_bah:>11.2f}%")
    print(f"  {'Sharpe Ratio':<30} {sharpe_hmm:>12.3f} {sharpe_bah:>12.3f}")
    print(f"  {'Max Drawdown':<30} {max_drawdown_hmm:>11.2f}% {max_drawdown_bah:>11.2f}%")
    print(f"  {'Nombre de trades':<30} {n_trades:>12} {'N/A':>12}")
    print(f"  {'Win Rate':<30} {win_rate:>11.1f}% {'N/A':>12}")
    print(f"  {'Capital final':<30} {valeur_finale_hmm:>11,.2f}$ {cash_initial * (1 + return_bah / 100):>11,.2f}$")
    print("  " + "=" * 50)

    metriques = {
        "return_hmm": return_hmm,
        "return_bah": return_bah,
        "sharpe_hmm": sharpe_hmm,
        "sharpe_bah": sharpe_bah,
        "max_drawdown_hmm": max_drawdown_hmm,
        "max_drawdown_bah": max_drawdown_bah,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "capital_final": valeur_finale_hmm,
    }

    return metriques


# ============================================================
# VISUALISATION
# ============================================================

def plot_hmm_backtest(
    df_portfolio: pd.DataFrame,
    df_original: pd.DataFrame,
    cash_initial: float = CASH_INITIAL,
    save_path: str | None = None,
) -> None:
    """
    Affiche un graphique comparatif HMM vs Buy & Hold.

    :param df_portfolio: Résultats de la simulation HMM
    :param df_original: Données OHLCV originales
    :param cash_initial: Capital initial
    :param save_path: Chemin de sauvegarde (None = afficher seulement)
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle("Backtest Stratégie HMM vs Buy & Hold", fontsize=14, fontweight="bold")

        # --- Graphique 1 : Prix avec régimes ---
        ax1 = axes[0]
        ax1.plot(df_original.index, df_original["close"], color="black", linewidth=0.8, label="Prix")

        couleurs_regimes = {0: "green", 1: "red", 2: "orange", 3: "blue"}
        for i in range(len(df_portfolio)):
            regime = df_portfolio["regime"].iloc[i]
            couleur = couleurs_regimes.get(regime, "gray")
            ax1.axvspan(
                df_portfolio.index[i],
                df_portfolio.index[min(i + 1, len(df_portfolio) - 1)],
                alpha=0.2,
                color=couleur,
            )

        legende = [Patch(facecolor=couleurs_regimes.get(i, "gray"), alpha=0.5,
                         label=REGIME_NAMES_4.get(i, f"R{i}")) for i in range(4)]
        ax1.legend(handles=legende, loc="upper left", fontsize=8)
        ax1.set_ylabel("Prix (USD)")
        ax1.set_title("Prix avec régimes HMM")
        ax1.grid(True, alpha=0.3)

        # --- Graphique 2 : Valeur du portefeuille HMM vs Buy & Hold ---
        ax2 = axes[1]

        # HMM
        ax2.plot(df_portfolio.index, df_portfolio["valeur_portfolio"],
                 color="blue", linewidth=1.5, label="Stratégie HMM")

        # Buy & Hold (normalisé au capital initial)
        prix_norm = df_original["close"] / df_original["close"].iloc[0] * cash_initial
        ax2.plot(df_original.index, prix_norm, color="orange",
                 linewidth=1.5, linestyle="--", label="Buy & Hold")

        # Marquer les achats et ventes
        achats = df_portfolio[df_portfolio["trade"] == "buy"]
        ventes = df_portfolio[df_portfolio["trade"] == "sell"]
        ax2.scatter(achats.index, achats["valeur_portfolio"],
                    marker="^", color="green", s=100, zorder=5, label="Achat")
        ax2.scatter(ventes.index, ventes["valeur_portfolio"],
                    marker="v", color="red", s=100, zorder=5, label="Vente")

        ax2.set_ylabel("Valeur du portefeuille (USD)")
        ax2.set_title("Performance : HMM vs Buy & Hold")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        # --- Graphique 3 : Drawdown ---
        ax3 = axes[2]
        valeur_cumul = df_portfolio["valeur_portfolio"]
        pic_cumul = valeur_cumul.cummax()
        drawdown = (valeur_cumul - pic_cumul) / pic_cumul * 100
        ax3.fill_between(df_portfolio.index, drawdown, 0, color="red", alpha=0.4, label="Drawdown HMM")
        ax3.set_ylabel("Drawdown (%)")
        ax3.set_xlabel("Date")
        ax3.set_title("Drawdown de la stratégie HMM")
        ax3.legend(loc="lower left", fontsize=8)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  📊 Graphique sauvegardé dans : {save_path}")
        else:
            plt.show()

        plt.close()

    except ImportError:
        print("  ⚠️ matplotlib non disponible — graphique ignoré")
    except Exception as e:
        print(f"  ⚠️ Erreur lors du tracé : {e}")


# ============================================================
# PIPELINE COMPLET DU BACKTEST HMM
# ============================================================

def run_hmm_backtest(
    csv_filepath: str,
    n_components: int = N_COMPONENTS,
    train_split: float = TRAIN_SPLIT,
    cash: float = CASH_INITIAL,
    commission: float = COMMISSION,
    plot_save_path: str | None = None,
) -> dict:
    """
    Pipeline complet : entraînement HMM → backtest sur données de test.

    Processus :
    1. Charger et préparer les données
    2. Diviser en ensemble d'entraînement (70%) et de test (30%)
    3. Entraîner le HMM sur les données d'entraînement
    4. Prédire les régimes sur les données de test
    5. Simuler la stratégie et calculer les métriques

    :param csv_filepath: Chemin vers le fichier CSV OHLCV
    :param n_components: Nombre de régimes HMM
    :param train_split: Proportion des données pour l'entraînement
    :param cash: Capital initial
    :param commission: Commission par trade
    :param plot_save_path: Chemin pour sauvegarder le graphique
    :return: Dictionnaire des métriques de performance
    """
    print("\n" + "=" * 60)
    print(f"🧠 BACKTEST STRATÉGIE HMM — {n_components} régimes")
    print("=" * 60)

    # Étape 1 : Chargement et feature engineering
    df = load_ohlcv_csv(csv_filepath)
    df = feature_engineering(df)

    # Étape 2 : Division entraînement / test
    n_train = int(len(df) * train_split)
    df_train = df.iloc[:n_train]
    df_test = df.iloc[n_train:]
    print(f"  📊 Entraînement : {len(df_train)} points | Test : {len(df_test)} points")
    print(f"  📅 Période de test : {df_test.index[0]} → {df_test.index[-1]}")

    # Étape 3 : Entraînement sur les données d'entraînement
    model, scaler, states_train = train_hmm(df_train, n_components=n_components)

    # Étape 4 : Prédiction sur les données de test
    feature_cols = ["returns", "volatility", "volume_change"]
    X_test = df_test[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    states_test = model.predict(X_test_scaled)

    # Étape 5 : Identification des régimes haussiers
    regimes_haussiers = identify_bullish_regimes(model, n_components)

    if not regimes_haussiers:
        print("  ⚠️ Aucun régime haussier identifié — Utilisation du régime 0 par défaut")
        regimes_haussiers = [0]

    # Étape 6 : Simulation de la stratégie
    df_portfolio = simulate_hmm_strategy(
        df_test, states_test, regimes_haussiers, cash, commission
    )

    # Étape 7 : Calcul des métriques
    metriques = calculate_performance_metrics(df_portfolio, df_test, cash)

    # Étape 8 : Visualisation
    plot_hmm_backtest(df_portfolio, df_test, cash, save_path=plot_save_path)

    return metriques


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    print("🧠 BACKTEST — Stratégie basée sur les régimes HMM")
    print("=" * 60)

    # Rechercher un fichier CSV dans le dossier data/
    csv_files = list(DATA_DIR.glob("*.csv"))

    if not csv_files:
        # Créer des données synthétiques de démonstration
        print("  ⚠️ Aucun CSV trouvé dans data/ — Génération de données synthétiques")

        import os
        np.random.seed(42)
        n = 2000
        dates = pd.date_range("2020-01-01", periods=n, freq="4h")
        prix = 10000 + np.cumsum(np.random.randn(n) * 50)
        prix = np.maximum(prix, 1000)  # Éviter les prix négatifs

        df_demo = pd.DataFrame({
            "timestamp": dates,
            "open": prix + np.random.randn(n) * 20,
            "high": prix + abs(np.random.randn(n)) * 80,
            "low": prix - abs(np.random.randn(n)) * 80,
            "close": prix,
            "volume": abs(np.random.randn(n)) * 500 + 200,
        })

        os.makedirs(DATA_DIR, exist_ok=True)
        demo_path = str(DATA_DIR / "demo_hmm_ohlcv.csv")
        df_demo.to_csv(demo_path, index=False)
        print(f"  💾 Données de démo sauvegardées dans : {demo_path}")
        csv_path = demo_path

    else:
        csv_path = str(csv_files[0])
        print(f"  📂 Utilisation de : {csv_path}")

    # Lancer le backtest HMM complet
    metriques = run_hmm_backtest(
        csv_filepath=csv_path,
        n_components=N_COMPONENTS,
        train_split=TRAIN_SPLIT,
        plot_save_path=str(DATA_DIR / "hmm_backtest.png"),
    )

    print("\n✅ Backtest HMM terminé avec succès")
