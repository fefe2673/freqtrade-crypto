import logging

import numpy as np  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, merge_informative_pair  # noqa


logger = logging.getLogger(__name__)


class FefeHybridStrategy(IStrategy):
    """
    FefeHybridStrategy — Personal hybrid FreqAI strategy with 5-layer entry filtering,
    dual entry modes (crossover + pullback), ATR-aware risk management, and adaptive
    ML-driven exits.

    Built on top of FreqaiHybridEnhancedStrategy indicators with refined entry/exit logic
    derived from analysis of 12+ published trading strategies.

    Key features:
    - 5-layer entry validation: Context, Timing, Volume, Safety, ML
    - Dual entry modes: classic MACD crossover + MACD pullback (buy-the-dip in trend)
    - Trailing stop with offset for profit protection
    - Multi-layered exits: technical, ML-driven, and DMF sentiment-based
    - Cooldown, MaxDrawdown, and StoplossGuard protections
    - Anti-slippage via confirm_trade_entry

    Timeframe: 1h
    Pairs: BTC/USDT, ETH/USDT (crypto trending assets)

    freqtrade trade --strategy FefeHybridStrategy \
        --strategy-path freqtrade/templates \
        --freqaimodel LightGBMClassifierMultiTarget \
        --config config_examples/config_freqai.example.json
    """

    timeframe = "1h"
    can_short = True

    minimal_roi = {
        "0": 0.15,
        "60": 0.08,
        "120": 0.04,
        "240": 0,
    }

    stoploss = -0.04

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count: int = 100  # Need 90 for DMF Light rolling windows + buffer

    plot_config = {
        "main_plot": {
            "tema": {},
            "ema20": {"color": "blue"},
            "ema50": {"color": "orange"},
            "ema200": {"color": "red"},
            "bb_upperband": {"color": "grey"},
            "bb_lowerband": {"color": "grey"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
                "macdhist": {"color": "grey", "type": "bar"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Stochastic": {
                "fastk": {"color": "blue"},
                "fastd": {"color": "orange"},
            },
            "ADX": {
                "adx": {"color": "purple"},
            },
            "Volume": {
                "volume_sma": {"color": "blue"},
            },
            "OBV": {
                "obv": {"color": "green"},
                "obv_sma": {"color": "orange"},
            },
            "DMF_Light": {
                "dmf_light": {"color": "cyan"},
            },
            "ATR_Pocket": {
                "atr_compression": {"color": "gold"},
                "atr_pocket": {"color": "magenta", "type": "bar"},
            },
            "Entry_Fragger": {
                "entry_fragger": {"color": "lime", "type": "bar"},
            },
            "ML_Prediction": {
                "&-s_close": {"color": "blue"},
            },
            "ML_Confidence": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    # Timing
    buy_rsi = IntParameter(low=20, high=40, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=60, high=80, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=60, high=80, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=20, high=40, default=30, space="buy", optimize=True, load=True)

    # Filters
    adx_threshold = IntParameter(low=15, high=35, default=25, space="buy", optimize=True, load=True)
    volume_spike_multiplier = DecimalParameter(
        low=1.1, high=2.0, default=1.3, decimals=1, space="buy", optimize=True, load=True
    )

    # ML amplitude minimum (new)
    ml_amplitude_threshold = DecimalParameter(
        low=0.005, high=0.02, default=0.01, decimals=3, space="buy", optimize=True, load=True
    )

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/stable/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/stable/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param period: period of the indicator - usage example:
        :param metadata: metadata of current pair
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)
        """

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]

        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=period)

        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        dataframe["%-stoch_fastk-period"] = ta.STOCHF(dataframe, fastk_period=period)["fastk"]
        dataframe["%-atr-period"] = ta.ATR(dataframe, timeperiod=period)
        dataframe["%-obv-period"] = ta.OBV(dataframe) / ta.OBV(dataframe).rolling(period).mean()

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.
        In other words, a single feature defined in this function
        will automatically expand to a total of
        `include_timeframes` * `include_shifted_candles` * `include_corr_pairs`
        numbers of features added to the model.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details on how these config defined parameters accelerate feature engineering
        in the documentation at:

        https://www.freqtrade.io/en/stable/freqai-parameter-table/#feature-parameters

        https://www.freqtrade.io/en/stable/freqai-feature-engineering/#defining-the-features

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-ema-200"] = ta.EMA(dataframe, timeperiod=200)
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        dataframe["%-obv_raw"] = ta.OBV(dataframe)
        dataframe["%-roc_3"] = ta.ROC(dataframe, timeperiod=3)
        # ATR compression ratio for ML
        dataframe["%-atr_compression"] = ta.ATR(dataframe, timeperiod=5) / ta.ATR(
            dataframe, timeperiod=20
        ).replace(0, np.nan)
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This optional function will be called once with the dataframe of the base timeframe.
        This is the final function to be called, which means that the dataframe entering this
        function will contain all the features and columns created by all other
        freqai_feature_engineering_* functions.

        This function is a good place to do custom exotic feature extractions (e.g. tsfresh).
        This function is a good place for any feature that should not be auto-expanded upon
        (e.g. day of the week).

        All features must be prepended with `%` to be recognized by FreqAI internals.

        More details about feature engineering available:

        https://www.freqtrade.io/en/stable/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        usage example: dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        """
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        This strategy uses dual targets:
        - &s-up_or_down: classification target ("up" or "down") for predicted direction.
        - &-s_close: regression target for predicted price change amplitude.

        Note: Using both a classifier and a regressor target requires a multi-output
        model such as LightGBMClassifierMultiTarget.

        More details about feature engineering available:

        https://www.freqtrade.io/en/stable/freqai-feature-engineering

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        usage example: dataframe["&-target"] = dataframe["close"].shift(-1) / dataframe["close"]
        """
        # Classification target: predict whether price will be higher or lower in N candles
        # Use label_period_candles to keep the prediction horizon consistent with the
        # regression target below.
        self.freqai.class_names = ["down", "up"]
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        dataframe["&s-up_or_down"] = np.where(
            dataframe["close"].shift(-label_period) > dataframe["close"], "up", "down"
        )

        # Regression target: predict the average relative price change over the label period
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:  # noqa: C901
        # Run FreqAI model to generate ML predictions (&s-up_or_down, &-s_close, do_predict)
        dataframe = self.freqai.start(dataframe, metadata, self)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (dataframe["close"] - dataframe["bb_lowerband"]) / (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        )
        dataframe["bb_width"] = (
            dataframe["bb_upperband"] - dataframe["bb_lowerband"]
        ) / dataframe["bb_middleband"]

        # TEMA - Triple Exponential Moving Average
        dataframe["tema"] = ta.TEMA(dataframe, timeperiod=9)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # --- Additional indicators ---

        # 1. EMA 20 / EMA 50 / EMA 200 (Trend)
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)

        # 2. ADX (Trend strength filter)
        dataframe["adx"] = ta.ADX(dataframe)

        # 3. Stochastic Fast (Reinforces RSI)
        stoch_fast = ta.STOCHF(dataframe)
        dataframe["fastd"] = stoch_fast["fastd"]
        dataframe["fastk"] = stoch_fast["fastk"]

        # 4. ATR - Average True Range (Dynamic stop-loss / take-profit)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # 5. OBV - On-Balance Volume (Accumulation / Distribution)
        dataframe["obv"] = ta.OBV(dataframe)
        dataframe["obv_sma"] = dataframe["obv"].rolling(20).mean()

        # 6. Volume SMA (Volume confirmation)
        dataframe["volume_sma"] = dataframe["volume"].rolling(20).mean()

        # 7. ROC - Rate Of Change (DMF Light component)
        dataframe["roc"] = ta.ROC(dataframe, timeperiod=10)

        # 8. Entry Fragger Logic (volume trap detection)
        dataframe["volume_spike_sell"] = (
            (dataframe["close"] < dataframe["open"])
            & (dataframe["volume"] > (dataframe["volume_sma"] * self.volume_spike_multiplier.value))
        )

        dataframe["sell_spikes_below_ema"] = 0
        for i in range(1, 11):
            dataframe["sell_spikes_below_ema"] += (
                dataframe["volume_spike_sell"].shift(i)
                & (dataframe["close"].shift(i) < dataframe["ema50"].shift(i))
            ).astype(int)

        dataframe["entry_fragger"] = (
            (dataframe["close"] > dataframe["ema50"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"].shift(1) <= dataframe["ema50"].shift(1))
            & (dataframe["sell_spikes_below_ema"] >= 2)
        )

        # 9. DMF Light — Contrarian sentiment score (0-100, no on-chain data)
        roc_range = dataframe["roc"].rolling(90).max() - dataframe["roc"].rolling(90).min()
        roc_norm = (
            (dataframe["roc"] - dataframe["roc"].rolling(90).min())
            / roc_range.replace(0, np.nan)
        ) * 100

        atr_range = dataframe["atr"].rolling(90).max() - dataframe["atr"].rolling(90).min()
        atr_norm = (
            (dataframe["atr"] - dataframe["atr"].rolling(90).min())
            / atr_range.replace(0, np.nan)
        ) * 100

        vol_ratio = dataframe["volume"] / dataframe["volume_sma"]
        vol_range = vol_ratio.rolling(90).max() - vol_ratio.rolling(90).min()
        vol_norm = (
            (vol_ratio - vol_ratio.rolling(90).min())
            / vol_range.replace(0, np.nan)
        ) * 100

        dataframe["dmf_light"] = (
            roc_norm * 0.4 + (100 - atr_norm) * 0.3 + (100 - vol_norm) * 0.3
        )
        dataframe["dmf_light"] = dataframe["dmf_light"].rolling(14).mean()

        dataframe["dmf_panic"] = dataframe["dmf_light"] < 20
        dataframe["dmf_fomo"] = dataframe["dmf_light"] > 80

        # ============================================================
        # ATR Pocket — Volatility Compression Detector (from Octopus Scanner concept)
        # Detects when volatility compresses while volume holds = potential breakout imminent
        # ============================================================
        atr_fast = ta.ATR(dataframe, timeperiod=5)
        atr_slow = ta.ATR(dataframe, timeperiod=20)
        vol_fast = dataframe["volume"].rolling(5).mean()
        vol_slow = dataframe["volume"].rolling(20).mean()

        # ATR Pocket: volatility compressed to 60% of baseline BUT volume stays at 75%+
        dataframe["atr_pocket"] = (
            (atr_fast < atr_slow * 0.6)    # Volatility is compressed
            & (vol_fast > vol_slow * 0.75)  # But volume is NOT dying
        ).astype(int)

        # ATR ratio for monitoring (how compressed is the volatility right now)
        dataframe["atr_compression"] = atr_fast / atr_slow.replace(0, np.nan)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # =====================================================================
        # ENTER LONG — Mode A: Classic Crossover
        # 5 layers: Context + Timing(crossover) + Volume + Safety + ML
        # =====================================================================
        df.loc[
            (
                # LAYER 1 — CONTEXT: trend is bullish
                (df["ema20"] > df["ema50"])
                & (df["close"] > df["ema200"])
                & (df["adx"] > self.adx_threshold.value)
                # LAYER 2 — TIMING: RSI crossover + MACD momentum
                & (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                & (df["macdhist"] > 0)
                # LAYER 3 — VOLUME: conviction behind the move
                & (df["volume"] > (df["volume_sma"] * 0.8))
                & (df["obv"] > df["obv_sma"])
                # LAYER 4 — SAFETY: no FOMO, ML is confident
                & (df["dmf_light"] < 80)
                & (df["do_predict"] == 1)
                # LAYER 5 — ML: AI confirms direction and amplitude
                & (df["&s-up_or_down"] == "up")
                & (df["&-s_close"] > self.ml_amplitude_threshold.value)
            ),
            "enter_long",
        ] = 1

        # =====================================================================
        # ENTER LONG — Mode B: Pullback Entry (buy the dip in uptrend)
        # Concept from MACD+EMA strategy: MACD hist negative but recovering
        # =====================================================================
        df.loc[
            (
                # LAYER 1 — CONTEXT: uptrend confirmed
                (df["ema20"] > df["ema50"])
                & (df["close"] > df["ema200"])
                & (df["adx"] > self.adx_threshold.value)
                # LAYER 2 — TIMING: pullback recovering
                & (df["macdhist"] < 0)                              # In a pullback
                & (df["macdhist"] > df["macdhist"].shift(1))        # But MACD recovering
                & (df["rsi"] < 50)                                   # RSI not overbought
                & (df["rsi"] > df["rsi"].shift(1))                   # RSI turning up
                # LAYER 3 — VOLUME
                & (df["volume"] > (df["volume_sma"] * 0.8))
                & (df["obv"] > df["obv_sma"])
                # LAYER 4 — SAFETY
                & (df["dmf_light"] < 80)
                & (df["do_predict"] == 1)
                # LAYER 5 — ML
                & (df["&s-up_or_down"] == "up")
                & (df["&-s_close"] > self.ml_amplitude_threshold.value)
            ),
            "enter_long",
        ] = 1

        # =====================================================================
        # ENTER SHORT — Mode A: Classic Crossover
        # =====================================================================
        df.loc[
            (
                # LAYER 1 — CONTEXT: trend is bearish
                (df["ema20"] < df["ema50"])
                & (df["close"] < df["ema200"])
                & (df["adx"] > self.adx_threshold.value)
                # LAYER 2 — TIMING: RSI crosses into overbought + MACD bearish
                & (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                & (df["macdhist"] < 0)
                # LAYER 3 — VOLUME
                & (df["volume"] > (df["volume_sma"] * 0.8))
                & (df["obv"] < df["obv_sma"])
                # LAYER 4 — SAFETY: no panic (don't short the bottom)
                & (df["dmf_light"] > 20)
                & (df["do_predict"] == 1)
                # LAYER 5 — ML
                & (df["&s-up_or_down"] == "down")
                & (df["&-s_close"] < -self.ml_amplitude_threshold.value)
            ),
            "enter_short",
        ] = 1

        # =====================================================================
        # ENTER SHORT — Mode B: Pullback Entry (sell the rally in downtrend)
        # =====================================================================
        df.loc[
            (
                # LAYER 1 — CONTEXT: downtrend confirmed
                (df["ema20"] < df["ema50"])
                & (df["close"] < df["ema200"])
                & (df["adx"] > self.adx_threshold.value)
                # LAYER 2 — TIMING: rally fading
                & (df["macdhist"] > 0)                              # In a rally
                & (df["macdhist"] < df["macdhist"].shift(1))        # But MACD fading
                & (df["rsi"] > 50)                                   # RSI not oversold
                & (df["rsi"] < df["rsi"].shift(1))                   # RSI turning down
                # LAYER 3 — VOLUME
                & (df["volume"] > (df["volume_sma"] * 0.8))
                & (df["obv"] < df["obv_sma"])
                # LAYER 4 — SAFETY
                & (df["dmf_light"] > 20)
                & (df["do_predict"] == 1)
                # LAYER 5 — ML
                & (df["&s-up_or_down"] == "down")
                & (df["&-s_close"] < -self.ml_amplitude_threshold.value)
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # =====================================================================
        # EXIT LONG
        # =====================================================================
        df.loc[
            # Technical exit: RSI overbought + MACD momentum reversing
            (
                (df["rsi"] > self.sell_rsi.value)
                & (df["macdhist"] < 0)
                & (df["macdhist"] < df["macdhist"].shift(1))
                & (df["volume"] > 0)
            )
            # OR ML exit: model predicts downward move
            | (
                (df["&-s_close"] < 0)
                & (df["do_predict"] == 1)
            )
            # OR DMF exit: extreme FOMO detected
            | (
                df["dmf_fomo"]
                & (df["rsi"] > 60)
            ),
            "exit_long",
        ] = 1

        # =====================================================================
        # EXIT SHORT
        # =====================================================================
        df.loc[
            # Technical exit: RSI oversold + MACD momentum reversing up
            (
                (df["rsi"] < self.exit_short_rsi.value)
                & (df["macdhist"] > 0)
                & (df["macdhist"] > df["macdhist"].shift(1))
                & (df["volume"] > 0)
            )
            # OR ML exit: model predicts upward move
            | (
                (df["&-s_close"] > 0)
                & (df["do_predict"] == 1)
            )
            # OR DMF exit: extreme panic detected (likely bottom)
            | (
                df["dmf_panic"]
                & (df["rsi"] < 40)
            ),
            "exit_short",
        ] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        """
        Refuse trade entry if the order rate has drifted more than 0.25% away from the
        last analysed close price. This protects against entering on sudden price spikes
        that occurred between signal generation and order execution.
        """
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()

        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
