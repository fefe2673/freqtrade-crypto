# ![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade_poweredby.svg)

[![Freqtrade CI](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml/badge.svg?branch=develop)](https://github.com/freqtrade/freqtrade/actions/workflows/ci.yml)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.04864/status.svg)](https://doi.org/10.21105/joss.04864)
[![codecov](https://codecov.io/gh/freqtrade/freqtrade/branch/develop/graph/badge.svg?token=AD5BG3ATKI)](https://codecov.io/gh/freqtrade/freqtrade)
[![Documentation](https://readthedocs.org/projects/freqtrade/badge/)](https://www.freqtrade.io)
[![Discord Server](https://img.shields.io/badge/Freqtrade_Discord-4E4E4E?logo=discord)](https://discord.gg/p7nuUNVfP7)

Freqtrade is a free and open source crypto trading bot written in Python. It is designed to support all major exchanges and be controlled via Telegram or webUI. It contains backtesting, plotting and money management tools as well as strategy optimization by machine learning.

![freqtrade](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/assets/freqtrade-screenshot.png)

## Disclaimer

This software is for educational purposes only. Do not risk money which
you are afraid to lose. USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS
AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

Always start by running a trading bot in Dry-Run and do not engage money
before you understand how it works and what profit/loss you should
expect.

We strongly recommend you to have coding and Python knowledge. Do not
hesitate to read the source code and understand the mechanism of this bot.

## Supported Exchange marketplaces

Please read the [exchange-specific notes](https://www.freqtrade.io/en/stable/exchanges/) to learn about special configurations that maybe needed for each exchange.

### Supported Spot Exchanges

- [X] [Binance](https://www.binance.com/)
- [X] [BingX](https://bingx.com/invite/0EM9RX)
- [X] [Bitget](https://www.bitget.com/)
- [X] [Bitmart](https://bitmart.com/)
- [X] [Bybit](https://bybit.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [HTX](https://www.htx.com/)
- [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
- [X] [Kraken](https://kraken.com/)
- [X] [OKX](https://okx.com/)
- [X] [MyOKX](https://okx.com/) (OKX EEA)
- [ ] [potentially many others](https://github.com/ccxt/ccxt/). _(We cannot guarantee they will work)_

### Supported Futures Exchanges

- [X] [Binance](https://www.binance.com/)
- [X] [Bitget](https://www.bitget.com/)
- [X] [Gate.io](https://www.gate.io/ref/6266643)
- [X] [Hyperliquid](https://hyperliquid.xyz/) (A decentralized exchange, or DEX)
- [X] [OKX](https://okx.com/)
- [X] [Bybit](https://bybit.com/)

Please make sure to read the [exchange specific notes](https://www.freqtrade.io/en/stable/exchanges/), as well as the [trading with leverage](https://www.freqtrade.io/en/stable/leverage/) documentation before diving in.

### Community tested

Exchanges confirmed working by the community:

- [X] [Bitvavo](https://bitvavo.com/)
- [X] [Kucoin](https://www.kucoin.com/)

## Documentation

We invite you to read the bot documentation to ensure you understand how the bot is working.

Please find the complete documentation on the [freqtrade website](https://www.freqtrade.io).

## Features

- [x] **Based on Python 3.11+**: For botting on any operating system - Windows, macOS and Linux.
- [x] **Persistence**: Persistence is achieved through sqlite.
- [x] **Dry-run**: Run the bot without paying money.
- [x] **Backtesting**: Run a simulation of your buy/sell strategy.
- [x] **Strategy Optimization by machine learning**: Use machine learning to optimize your buy/sell strategy parameters with real exchange data.
- [X] **Adaptive prediction modeling**: Build a smart strategy with FreqAI that self-trains to the market via adaptive machine learning methods. [Learn more](https://www.freqtrade.io/en/stable/freqai/)
- [x] **Whitelist crypto-currencies**: Select which crypto-currency you want to trade or use dynamic whitelists.
- [x] **Blacklist crypto-currencies**: Select which crypto-currency you want to avoid.
- [x] **Builtin WebUI**: Builtin web UI to manage your bot.
- [x] **Manageable via Telegram**: Manage the bot with Telegram.
- [x] **Display profit/loss in fiat**: Display your profit/loss in fiat currency.
- [x] **Performance status report**: Provide a performance status of your current trades.

## Quick start

Please refer to the [Docker Quickstart documentation](https://www.freqtrade.io/en/stable/docker_quickstart/) on how to get started quickly.

For further (native) installation methods, please refer to the [Installation documentation page](https://www.freqtrade.io/en/stable/installation/).

## Basic Usage

### Bot commands

```
usage: freqtrade [-h] [-V]
                 {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
                 ...

Free, open source crypto trading bot

positional arguments:
  {trade,create-userdir,new-config,show-config,new-strategy,download-data,convert-data,convert-trade-data,trades-to-ohlcv,list-data,backtesting,backtesting-show,backtesting-analysis,edge,hyperopt,hyperopt-list,hyperopt-show,list-exchanges,list-markets,list-pairs,list-strategies,list-hyperoptloss,list-freqaimodels,list-timeframes,show-trades,test-pairlist,convert-db,install-ui,plot-dataframe,plot-profit,webserver,strategy-updater,lookahead-analysis,recursive-analysis}
    trade               Trade module.
    create-userdir      Create user-data directory.
    new-config          Create new config
    show-config         Show resolved config
    new-strategy        Create new strategy
    download-data       Download backtesting data.
    convert-data        Convert candle (OHLCV) data from one format to
                        another.
    convert-trade-data  Convert trade data from one format to another.
    trades-to-ohlcv     Convert trade data to OHLCV data.
    list-data           List downloaded data.
    backtesting         Backtesting module.
    backtesting-show    Show past Backtest results
    backtesting-analysis
                        Backtest Analysis module.
    hyperopt            Hyperopt module.
    hyperopt-list       List Hyperopt results
    hyperopt-show       Show details of Hyperopt results
    list-exchanges      Print available exchanges.
    list-markets        Print markets on exchange.
    list-pairs          Print pairs on exchange.
    list-strategies     Print available strategies.
    list-hyperoptloss   Print available hyperopt loss functions.
    list-freqaimodels   Print available freqAI models.
    list-timeframes     Print available timeframes for the exchange.
    show-trades         Show trades.
    test-pairlist       Test your pairlist configuration.
    convert-db          Migrate database to different system
    install-ui          Install FreqUI
    plot-dataframe      Plot candles with indicators.
    plot-profit         Generate plot showing profits.
    webserver           Webserver module.
    strategy-updater    updates outdated strategy files to the current version
    lookahead-analysis  Check for potential look ahead bias.
    recursive-analysis  Check for potential recursive formula issue.

options:
  -h, --help            show this help message and exit
  -V, --version         show program's version number and exit
```

### Telegram RPC commands

Telegram is not mandatory. However, this is a great way to control your bot. More details and the full command list on the [documentation](https://www.freqtrade.io/en/stable/telegram-usage/)

- `/start`: Starts the trader.
- `/stop`: Stops the trader.
- `/stopentry`: Stop entering new trades.
- `/status <trade_id>|[table]`: Lists all or specific open trades.
- `/profit [<n>]`: Lists cumulative profit from all finished trades, over the last n days.
- `/profit_long [<n>]`: Lists cumulative profit from all finished long trades, over the last n days.
- `/profit_short [<n>]`: Lists cumulative profit from all finished short trades, over the last n days.
- `/forceexit <trade_id>|all`: Instantly exits the given trade (Ignoring `minimum_roi`).
- `/fx <trade_id>|all`: Alias to `/forceexit`
- `/performance`: Show performance of each finished trade grouped by pair
- `/balance`: Show account balance per currency.
- `/daily <n>`: Shows profit or loss per day, over the last n days.
- `/help`: Show help message.
- `/version`: Show version.


## Development branches

The project is currently setup in two main branches:

- `develop` - This branch has often new features, but might also contain breaking changes. We try hard to keep this branch as stable as possible.
- `stable` - This branch contains the latest stable release. This branch is generally well tested.
- `feat/*` - These are feature branches, which are being worked on heavily. Please don't use these unless you want to test a specific feature.

## Support

### Help / Discord

For any questions not covered by the documentation or for further information about the bot, or to simply engage with like-minded individuals, we encourage you to join the Freqtrade [discord server](https://discord.gg/p7nuUNVfP7).

### [Bugs / Issues](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)

If you discover a bug in the bot, please
[search the issue tracker](https://github.com/freqtrade/freqtrade/issues?q=is%3Aissue)
first. If it hasn't been reported, please
[create a new issue](https://github.com/freqtrade/freqtrade/issues/new/choose) and
ensure you follow the template guide so that the team can assist you as
quickly as possible.

For every [issue](https://github.com/freqtrade/freqtrade/issues/new/choose) created, kindly follow up and mark satisfaction or reminder to close issue when equilibrium ground is reached.

--Maintain github's [community policy](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct)--

### [Feature Requests](https://github.com/freqtrade/freqtrade/labels/enhancement)

Have you a great idea to improve the bot you want to share? Please,
first search if this feature was not [already discussed](https://github.com/freqtrade/freqtrade/labels/enhancement).
If it hasn't been requested, please
[create a new request](https://github.com/freqtrade/freqtrade/issues/new/choose)
and ensure you follow the template guide so that it does not get lost
in the bug reports.

### [Pull Requests](https://github.com/freqtrade/freqtrade/pulls)

Feel like the bot is missing a feature? We welcome your pull requests!

Please read the
[Contributing document](https://github.com/freqtrade/freqtrade/blob/develop/CONTRIBUTING.md)
to understand the requirements before sending your pull-requests.

Coding is not a necessity to contribute - maybe start with improving the documentation?
Issues labeled [good first issue](https://github.com/freqtrade/freqtrade/labels/good%20first%20issue) can be good first contributions, and will help get you familiar with the codebase.

**Note** before starting any major new feature work, *please open an issue describing what you are planning to do* or talk to us on [discord](https://discord.gg/p7nuUNVfP7) (please use the #dev channel for this). This will ensure that interested parties can give valuable feedback on the feature, and let others know that you are working on it.

**Important:** Always create your PR against the `develop` branch, not `stable`.

## Requirements

### Up-to-date clock

The clock must be accurate, synchronized to a NTP server very frequently to avoid problems with communication to the exchanges.

### Minimum hardware required

To run this bot we recommend you a cloud instance with a minimum of:

- Minimal (advised) system requirements: 2GB RAM, 1GB disk space, 2vCPU

### Software requirements

- [Python >= 3.11](http://docs.python-guide.org/en/latest/starting/installation/)
- [pip](https://pip.pypa.io/en/stable/installing/)
- [git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
- [TA-Lib](https://ta-lib.github.io/ta-lib-python/)
- [virtualenv](https://virtualenv.pypa.io/en/stable/installation.html) (Recommended)
- [Docker](https://www.docker.com/products/docker) (Recommended)

---

## 🇫🇷 Stratégies de Trading Personnalisées

Ce dépôt inclut un ensemble de **stratégies de trading algorithmique** complémentaires au bot Freqtrade, développées en Python avec CCXT.

---

### 📋 Description du projet

Ce projet implémente **3 stratégies de trading** indépendantes mais complémentaires, avec des outils de backtesting intégrés. Il est conçu pour fonctionner en complément du bot Freqtrade existant.

---

### 🗂️ Structure des fichiers personnalisés

```
strategies/
  __init__.py
  sma_strategy.py          # Stratégie 1 : SMA + carnet d'ordres + kill switch
  hmm_strategy.py          # Stratégie 2 : Hidden Markov Model (régimes de marché)
  breakout_strategy.py     # Stratégie 3 : Breakout support/résistance
backtests/
  __init__.py
  backtest_breakout.py     # Backtest de la stratégie breakout
  backtest_hmm.py          # Backtest basé sur les régimes HMM
config/
  config_example.py        # Exemple de configuration (à copier en config.py)
data/
  .gitkeep                 # Dossier pour vos fichiers CSV de données historiques
```

---

### 📦 Les 3 Stratégies

#### Stratégie 1 — SMA + Carnet d'Ordres + Kill Switch (`strategies/sma_strategy.py`)

Stratégie de **suivi de tendance** basée sur la Simple Moving Average (SMA) :

- **Signal BUY** : Prix actuel **au-dessus** de la SMA → tendance haussière
- **Signal SELL** : Prix actuel **en dessous** de la SMA → tendance baissière
- **Ordres limites décalés** pour obtenir un meilleur prix d'entrée
- **Analyse du carnet d'ordres** pour optimiser les sorties
- **Kill switch** : fermeture élégante via ordres limites
- **Anti-surtrading** : pause configurable après chaque trade fermé
- **Gestion P&L** automatique : take-profit et stop-loss configurables

| Fonction | Description |
|---|---|
| `ask_bid(symbol)` | Récupère le prix ask et bid via CCXT |
| `get_sma(symbol, timeframe, limit, sma_period)` | Calcule la SMA et génère le signal buy/sell |
| `get_open_positions(symbol)` | Récupère les positions ouvertes |
| `kill_switch(symbol)` | Ferme élégamment une position (ordres limites) |
| `sleep_on_close(symbol, pause_time_minutes)` | Anti-surtrading : pause après un trade |
| `order_book_analysis(symbol, vol_repeat, vol_time)` | Analyse du volume du carnet d'ordres |
| `pnl_close(symbol, target_pct, max_loss_pct)` | Gestion take-profit / stop-loss |

#### Stratégie 2 — Hidden Markov Model (`strategies/hmm_strategy.py`)

Stratégie de **détection de régimes de marché** basée sur un modèle HMM gaussien :

- **4 régimes** : Risk-On, Risk-Off, Haute Volatilité, Basse Volatilité
- **Features** : rendements, volatilité rolling, variation du volume
- **Entraînement** avec `GaussianHMM` + normalisation `StandardScaler`
- **Analyse** : matrice de transition, moyennes/covariances par régime
- **Sauvegarde** du modèle avec `joblib`
- **Visualisation** des régimes colorés sur le graphique de prix

#### Stratégie 3 — Breakout Support/Résistance (`strategies/breakout_strategy.py`)

Stratégie de **cassure de niveaux techniques** :

- **Support** : `close.rolling(window).min().shift(1)` (données passées uniquement, sans look-ahead)
- **Résistance** : `close.rolling(window).max().shift(1)` (données passées uniquement, sans look-ahead)
- **Breakout haussier** : bid > résistance × (1 + 0.1%) → ordre LONG
- **Breakout baissier** : bid < support × (1 - 0.1%) → ordre SHORT
- **TP/SL** configurables et optimisables (3% à 20%)

---

### 📊 Backtests

#### Backtest Breakout (`backtests/backtest_breakout.py`)

Utilise la librairie `backtesting.py` pour tester et optimiser la stratégie breakout.
Lance automatiquement un backtest simple puis une optimisation sur grille de paramètres TP/SL.

#### Backtest HMM (`backtests/backtest_hmm.py`)

Entraîne le HMM sur 70% des données et teste sur les 30% restants, avec comparaison vs Buy & Hold.
Génère des métriques : return total, Sharpe Ratio, maximum drawdown, win rate.

---

### 🚀 Installation des dépendances supplémentaires

```bash
pip install -r requirements.txt
```

Packages ajoutés pour les stratégies personnalisées :

| Package | Utilisation |
|---|---|
| `hmmlearn>=0.3.0` | Modèles de Markov cachés (stratégie HMM) |
| `scikit-learn>=1.3.0` | Normalisation des données (StandardScaler) |
| `backtesting>=0.3.3` | Framework de backtesting |
| `matplotlib>=3.7.0` | Visualisation des résultats |

---

### ⚙️ Configuration

```bash
cp config/config_example.py config/config.py
# Éditer config/config.py avec vos clés API et paramètres
```

**Important :** Ne jamais commiter `config/config.py` — ajoutez-le à `.gitignore`.

---

### ▶️ Lancer les stratégies

```bash
# Stratégie SMA (trading en live)
python strategies/sma_strategy.py

# Stratégie HMM (analyse de régimes sur un CSV dans data/)
python strategies/hmm_strategy.py data/btc_4h.csv

# Stratégie Breakout (trading en live)
python strategies/breakout_strategy.py

# Backtest Breakout (nécessite un CSV dans data/)
python backtests/backtest_breakout.py

# Backtest HMM (nécessite un CSV dans data/)
python backtests/backtest_hmm.py
```

---

### ⚠️ Avertissement sur les risques du trading

> **Ce projet est à des fins éducatives uniquement.**

Le trading de cryptomonnaies comporte des **risques financiers importants** :

- Ne jamais investir plus que ce que vous pouvez vous permettre de perdre
- Les performances passées ne garantissent pas les résultats futurs
- Le levier amplifie les gains ET les pertes
- Toujours tester en **paper trading** (testnet) avant d'utiliser de l'argent réel
- Comprendre chaque ligne de code avant de le déployer

**Les auteurs déclinent toute responsabilité pour les pertes financières liées à l'utilisation de ce code.**

---

### 🔗 Complémentarité avec Freqtrade

Ce projet est **complémentaire** au bot Freqtrade existant dans ce dépôt :

| Freqtrade | Stratégies personnalisées |
|---|---|
| Framework complet de trading automatisé | Scripts de trading directs via CCXT |
| Gestion via Telegram / WebUI | Exécution en ligne de commande |
| Backtesting intégré avec FreqAI | Backtesting avec `backtesting.py` |
| Stratégies Freqtrade standard | SMA, HMM, Breakout personnalisés |
