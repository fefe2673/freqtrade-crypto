"""
Stratégie 2 : Hidden Markov Model (HMM) pour la détection de régimes de marché
================================================================================
Ce module utilise un modèle HMM gaussien pour identifier les différents régimes
de marché (risk-on, risk-off, haute volatilité, basse volatilité) à partir de
données OHLCV historiques.

Fonctionnalités :
- Chargement et préparation des données CSV
- Feature engineering : returns, volatilité, variation de volume
- Entraînement d'un GaussianHMM avec normalisation StandardScaler
- Prédiction des régimes de marché
- Analyse : matrice de transition, moyennes, covariances, importance des features
- Sauvegarde du modèle entraîné avec joblib
- Visualisation des régimes sur le graphique de prix
"""

import os
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

# Supprimer les avertissements de convergence HMM pour une sortie plus propre
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION PAR DÉFAUT
# ============================================================
# Dossier de données par défaut (relatif à la racine du projet)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_DIR = Path(__file__).resolve().parent.parent / "data"

# Paramètres HMM
N_COMPONENTS_DEFAULT = 4    # Nombre de régimes de marché (4 ou 7)
N_ITER = 1000               # Nombre d'itérations d'entraînement
RANDOM_STATE = 42           # Graine aléatoire pour la reproductibilité
VOLATILITY_WINDOW = 20      # Fenêtre pour le calcul de la volatilité rolling

# Noms des régimes pour 4 états
REGIME_NAMES_4 = {
    0: "Risk-On",
    1: "Risk-Off",
    2: "Haute Volatilité",
    3: "Basse Volatilité",
}

# Couleurs associées à chaque régime
REGIME_COLORS_4 = {
    0: "green",
    1: "red",
    2: "orange",
    3: "blue",
}

# Noms des régimes pour 7 états
REGIME_NAMES_7 = {
    0: "Forte Hausse",
    1: "Hausse Modérée",
    2: "Légère Hausse",
    3: "Neutre",
    4: "Légère Baisse",
    5: "Baisse Modérée",
    6: "Forte Baisse",
}


# ============================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================

def load_ohlcv_csv(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV contenant des données OHLCV.

    Le fichier doit contenir au minimum les colonnes :
    'timestamp' (ou 'date'), 'open', 'high', 'low', 'close', 'volume'

    :param filepath: Chemin vers le fichier CSV
    :return: DataFrame pandas avec les données OHLCV indexées par timestamp
    """
    print(f"  📂 Chargement des données depuis : {filepath}")

    try:
        df = pd.read_csv(filepath)

        # Normaliser le nom de la colonne de date
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        else:
            # Essayer la première colonne comme index de date
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0])

        # S'assurer que les colonnes essentielles sont présentes
        colonnes_requises = ["open", "high", "low", "close", "volume"]
        colonnes_manquantes = [c for c in colonnes_requises if c not in df.columns]
        if colonnes_manquantes:
            raise ValueError(f"Colonnes manquantes dans le CSV : {colonnes_manquantes}")

        # Convertir en numérique et supprimer les valeurs manquantes
        for col in colonnes_requises:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=colonnes_requises)

        print(f"  ✅ Données chargées : {len(df)} lignes, du {df.index[0]} au {df.index[-1]}")
        return df

    except FileNotFoundError:
        print(f"  ❌ Fichier non trouvé : {filepath}")
        raise
    except Exception as e:
        print(f"  ❌ Erreur lors du chargement du CSV : {e}")
        raise


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les features nécessaires pour l'entraînement du HMM.

    Features calculées :
    - returns          : variation en % du prix de clôture
    - volatility       : écart-type rolling des returns (fenêtre = VOLATILITY_WINDOW)
    - volume_change    : variation en % du volume

    :param df: DataFrame OHLCV
    :return: DataFrame enrichi avec les nouvelles features (sans NaN)
    """
    print("  🔧 Calcul des features pour le HMM...")
    df = df.copy()

    # Rendement : variation en pourcentage du prix de clôture
    df["returns"] = df["close"].pct_change()

    # Volatilité : écart-type rolling des rendements sur une fenêtre glissante
    df["volatility"] = df["returns"].rolling(window=VOLATILITY_WINDOW).std()

    # Variation du volume : variation en pourcentage du volume échangé
    df["volume_change"] = df["volume"].pct_change()

    # Supprimer les lignes avec des NaN (premières lignes dues au rolling)
    df = df.dropna(subset=["returns", "volatility", "volume_change"])

    print(f"  ✅ Features calculées — {len(df)} lignes utilisables")
    return df


# ============================================================
# ENTRAÎNEMENT DU MODÈLE HMM
# ============================================================

def train_hmm(
    df: pd.DataFrame,
    n_components: int = N_COMPONENTS_DEFAULT,
    n_iter: int = N_ITER,
    random_state: int = RANDOM_STATE,
) -> tuple[GaussianHMM, StandardScaler, np.ndarray]:
    """
    Entraîne un modèle GaussianHMM sur les features préparées.

    :param df: DataFrame contenant les colonnes 'returns', 'volatility', 'volume_change'
    :param n_components: Nombre de régimes (états cachés) à identifier (4 ou 7)
    :param n_iter: Nombre d'itérations maximum pour l'algorithme EM
    :param random_state: Graine aléatoire pour la reproductibilité
    :return: Tuple (model, scaler, states) avec le modèle entraîné, le scaler et les états prédits
    """
    print(f"\n  🧠 Entraînement du HMM avec {n_components} régimes...")

    # Sélection et normalisation des features
    feature_cols = ["returns", "volatility", "volume_change"]
    X = df[feature_cols].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entraînement du modèle GaussianHMM
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )

    try:
        model.fit(X_scaled)
        print(f"  ✅ HMM entraîné — Convergé : {model.monitor_.converged}")
    except Exception as e:
        print(f"  ⚠️ Avertissement lors de l'entraînement : {e}")

    # Prédiction des états pour chaque point de données
    states = model.predict(X_scaled)
    print(f"  📊 Régimes identifiés : {np.unique(states)}")

    return model, scaler, states


# ============================================================
# PRÉDICTION ET ANALYSE
# ============================================================

def predict_regime(
    model: GaussianHMM,
    scaler: StandardScaler,
    features: np.ndarray,
) -> int:
    """
    Prédit le régime de marché actuel à partir de nouvelles features.

    :param model: Modèle HMM entraîné
    :param scaler: StandardScaler ajusté sur les données d'entraînement
    :param features: Array numpy de shape (1, 3) avec [returns, volatility, volume_change]
    :return: Index du régime prédit (entier)
    """
    features_scaled = scaler.transform(features.reshape(1, -1))
    regime = model.predict(features_scaled)[0]
    return int(regime)


def analyze_regimes(
    model: GaussianHMM,
    df: pd.DataFrame,
    states: np.ndarray,
    n_components: int = N_COMPONENTS_DEFAULT,
) -> dict:
    """
    Analyse détaillée des régimes identifiés par le HMM.

    Calcule pour chaque régime :
    - Matrice de transition
    - Moyennes des features par état
    - Covariances par état
    - Importance relative des features
    - Statistiques de prix par régime

    :param model: Modèle HMM entraîné
    :param df: DataFrame avec les données et features
    :param states: Array des états prédits
    :param n_components: Nombre de régimes
    :return: Dictionnaire avec toutes les statistiques d'analyse
    """
    print("\n  📊 Analyse des régimes de marché...")

    feature_cols = ["returns", "volatility", "volume_change"]
    regime_names = REGIME_NAMES_4 if n_components == 4 else REGIME_NAMES_7

    analyse = {
        "matrice_transition": model.transmat_,
        "regimes": {},
        "importance_features": {},
    }

    # Statistiques par régime
    print("\n  === ANALYSE PAR RÉGIME ===")
    for i in range(n_components):
        masque = states == i
        regime_df = df[masque]
        nom = regime_names.get(i, f"Régime {i}")

        stats = {
            "nom": nom,
            "occurrences": int(masque.sum()),
            "pourcentage": float(masque.mean() * 100),
            "rendement_moyen": float(regime_df["returns"].mean() * 100) if len(regime_df) > 0 else 0,
            "volatilite_moyenne": float(regime_df["volatility"].mean() * 100) if len(regime_df) > 0 else 0,
            "prix_moyen": float(regime_df["close"].mean()) if len(regime_df) > 0 else 0,
            "moyennes_features": model.means_[i].tolist(),
            "covariances": model.covars_[i].tolist(),
        }
        analyse["regimes"][i] = stats

        print(f"\n  Régime {i} — {nom} :")
        print(f"    Occurrences    : {stats['occurrences']} ({stats['pourcentage']:.1f}%)")
        print(f"    Rendement moyen : {stats['rendement_moyen']:.4f}%")
        print(f"    Volatilité moy. : {stats['volatilite_moyenne']:.4f}%")
        print(f"    Moyennes feat.  : {[f'{v:.4f}' for v in stats['moyennes_features']]}")

    # Matrice de transition
    print("\n  === MATRICE DE TRANSITION ===")
    print("  (Probabilité de passer du régime ligne au régime colonne)")
    transmat_df = pd.DataFrame(
        model.transmat_,
        index=[regime_names.get(i, f"R{i}") for i in range(n_components)],
        columns=[regime_names.get(i, f"R{i}") for i in range(n_components)],
    )
    print(transmat_df.round(3).to_string())

    # Importance des features (basée sur la variance des moyennes par régime)
    print("\n  === IMPORTANCE DES FEATURES ===")
    for j, feat in enumerate(feature_cols):
        moyennes_feat = [model.means_[i][j] for i in range(n_components)]
        importance = float(np.std(moyennes_feat))
        analyse["importance_features"][feat] = importance
        print(f"    {feat:20s} : {importance:.4f}")

    return analyse


# ============================================================
# SAUVEGARDE ET CHARGEMENT DU MODÈLE
# ============================================================

def save_model(
    model: GaussianHMM,
    scaler: StandardScaler,
    filepath: str | None = None,
) -> str:
    """
    Sauvegarde le modèle HMM et le scaler avec joblib.

    :param model: Modèle HMM entraîné
    :param scaler: StandardScaler ajusté
    :param filepath: Chemin de sauvegarde (optionnel, généré automatiquement si None)
    :return: Chemin du fichier de sauvegarde
    """
    if filepath is None:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filepath = str(MODEL_DIR / "hmm_model.joblib")

    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, filepath)
    print(f"  💾 Modèle sauvegardé dans : {filepath}")
    return filepath


def load_model(filepath: str) -> tuple[GaussianHMM, StandardScaler]:
    """
    Charge un modèle HMM et son scaler depuis un fichier joblib.

    :param filepath: Chemin vers le fichier .joblib
    :return: Tuple (model, scaler)
    """
    print(f"  📂 Chargement du modèle depuis : {filepath}")
    bundle = joblib.load(filepath)
    model = bundle["model"]
    scaler = bundle["scaler"]
    print("  ✅ Modèle chargé avec succès")
    return model, scaler


# ============================================================
# VISUALISATION
# ============================================================

def plot_regimes(
    df: pd.DataFrame,
    states: np.ndarray,
    n_components: int = N_COMPONENTS_DEFAULT,
    title: str = "Régimes de marché HMM",
    save_path: str | None = None,
) -> None:
    """
    Affiche le graphique du prix coloré par régime de marché.

    Génère deux sous-graphiques :
    1. Prix de clôture avec les régimes en arrière-plan coloré
    2. Rendements quotidiens colorés par régime

    :param df: DataFrame contenant les données de prix et les features
    :param states: Array des états/régimes prédits
    :param n_components: Nombre de régimes
    :param title: Titre du graphique
    :param save_path: Chemin pour sauvegarder le graphique (None = afficher seulement)
    """
    regime_names = REGIME_NAMES_4 if n_components == 4 else REGIME_NAMES_7
    regime_colors = REGIME_COLORS_4 if n_components == 4 else {
        i: plt.cm.RdYlGn(i / (n_components - 1)) for i in range(n_components)
    }

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Graphique 1 : Prix avec régimes en arrière-plan ---
    ax1 = axes[0]
    ax1.plot(df.index, df["close"], color="black", linewidth=0.8, label="Prix de clôture")

    # Colorer le fond selon le régime actif
    for i in range(len(states)):
        etat = states[i]
        couleur = regime_colors.get(etat, "gray")
        ax1.axvspan(
            df.index[i],
            df.index[min(i + 1, len(df) - 1)],
            alpha=0.3,
            color=couleur,
        )

    # Légende des régimes
    from matplotlib.patches import Patch
    legende = [
        Patch(facecolor=regime_colors.get(i, "gray"), alpha=0.5, label=regime_names.get(i, f"R{i}"))
        for i in range(n_components)
    ]
    ax1.legend(handles=legende, loc="upper left", fontsize=9)
    ax1.set_ylabel("Prix (USD)")
    ax1.set_title("Prix avec régimes de marché HMM")
    ax1.grid(True, alpha=0.3)

    # --- Graphique 2 : Rendements colorés par régime ---
    ax2 = axes[1]
    for i in range(len(states)):
        etat = states[i]
        couleur = regime_colors.get(etat, "gray")
        ax2.bar(
            df.index[i],
            df["returns"].iloc[i] * 100,
            color=couleur,
            alpha=0.7,
            width=0.8,
        )

    ax2.axhline(y=0, color="black", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Rendements (%)")
    ax2.set_xlabel("Date")
    ax2.set_title("Rendements journaliers par régime")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  📊 Graphique sauvegardé dans : {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================
# PIPELINE COMPLET
# ============================================================

def run_hmm_pipeline(
    csv_filepath: str,
    n_components: int = N_COMPONENTS_DEFAULT,
    model_save_path: str | None = None,
    plot_save_path: str | None = None,
) -> tuple[GaussianHMM, StandardScaler, pd.DataFrame, np.ndarray]:
    """
    Exécute le pipeline complet HMM : chargement → feature engineering → entraînement → analyse.

    :param csv_filepath: Chemin vers le fichier CSV OHLCV
    :param n_components: Nombre de régimes à identifier (4 ou 7)
    :param model_save_path: Chemin pour sauvegarder le modèle (None = pas de sauvegarde)
    :param plot_save_path: Chemin pour sauvegarder le graphique (None = afficher)
    :return: Tuple (model, scaler, df_avec_features, states)
    """
    print("\n" + "=" * 60)
    print(f"🧠 PIPELINE HMM — {n_components} régimes")
    print("=" * 60)

    # Étape 1 : Charger les données
    df = load_ohlcv_csv(csv_filepath)

    # Étape 2 : Feature engineering
    df = feature_engineering(df)

    # Étape 3 : Entraînement
    model, scaler, states = train_hmm(df, n_components=n_components)

    # Ajouter les états au DataFrame pour l'analyse
    df = df.copy()
    df["regime"] = states

    # Étape 4 : Analyse des régimes
    analyse = analyze_regimes(model, df, states, n_components)

    # Étape 5 : Sauvegarde du modèle (si un chemin est fourni)
    if model_save_path:
        save_model(model, scaler, model_save_path)

    # Étape 6 : Visualisation
    plot_regimes(df, states, n_components, save_path=plot_save_path)

    print("\n✅ Pipeline HMM terminé avec succès")
    return model, scaler, df, states


# ============================================================
# LANCEMENT DIRECT
# ============================================================

if __name__ == "__main__":
    import sys

    # Exemple d'utilisation avec un fichier CSV
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Chercher un CSV dans le dossier data
        csv_files = list(DATA_DIR.glob("*.csv"))
        if csv_files:
            csv_path = str(csv_files[0])
            print(f"📂 Utilisation du fichier : {csv_path}")
        else:
            print("❌ Aucun fichier CSV trouvé dans le dossier data/")
            print("   Usage : python strategies/hmm_strategy.py chemin/vers/data.csv")
            exit(1)

    n_regimes = int(sys.argv[2]) if len(sys.argv) > 2 else N_COMPONENTS_DEFAULT

    run_hmm_pipeline(
        csv_filepath=csv_path,
        n_components=n_regimes,
        model_save_path=str(DATA_DIR / "hmm_model.joblib"),
        plot_save_path=str(DATA_DIR / "hmm_regimes.png"),
    )
