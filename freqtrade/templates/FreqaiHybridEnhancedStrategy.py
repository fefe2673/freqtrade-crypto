import logging

import numpy as np  # noqa
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IntParameter, IStrategy, merge_informative_pair  # noqa


logger = logging.getLogger(__name__)


class FreqaiHybridEnhancedStrategy(IStrategy):
    """
    Enhanced hybrid FreqAI strategy that combines the best of FreqaiExampleStrategy (pure ML)
    and FreqaiExampleHybridStrategy (TA + ML classification).

    Key improvements over FreqaiExampleHybridStrategy:
    - Dual ML targets: binary classification (&s-up_or_down) for direction AND
      regression (&-s_close) for predicted amplitude.
    - Entry signals require all three filters: TA timing, ML direction, AND ML amplitude > 1%.
    - Exit signals are enhanced: technical conditions OR ML predicts adverse move.
    - Anti-slippage protection via confirm_trade_entry (refuses entry if price drifts > 0.25%).
    - MACD is now actively computed and available for plotting / further signal logic.

    Launching this strategy:

    freqtrade trade --strategy FreqaiHybridEnhancedStrategy
        --strategy-path freqtrade/templates
        --freqaimodel XGBoostClassifier
        --config config_examples/config_freqai.example.json

    Recommended freqai config block:

    "freqai": {
        "enabled": true,
        "purge_old_models": 2,
        "train_period_days": 15,
        "identifier": "unique-id",
        "feature_parameters": {
            "include_timeframes": ["3m", "15m", "1h"],
            "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
            "label_period_candles": 20,
            "include_shifted_candles": 2,
            "DI_threshold": 0.9,
            "weight_factor": 0.9,
            "principal_component_analysis": false,
            "use_SVM_to_remove_outliers": true,
            "indicator_periods_candles": [10, 20]
        },
        "data_split_parameters": {
            "test_size": 0,
            "random_state": 1
        },
        "model_training_parameters": {
            "n_estimators": 800
        }
    },

    Note: Because this strategy uses both a classifier target (&s-up_or_down) and a
    regression target (&-s_close), you must use a multi-output model such as
    LightGBMClassifierMultiTarget, or train two separate models using the appropriate
    FreqAI model class.

    Warning: This is an educational example strategy. It is not intended to be run live
    in production without further tuning and validation.
    """

    minimal_roi = {
        # "120": 0.0,  # exit after 120 minutes at break even
        "60": 0.01,
        "30": 0.02,
        "0": 0.04,
    }

    plot_config = {
        "main_plot": {
            "tema": {},
            "bb_lowerband": {"color": "grey"},
            "bb_middleband": {"color": "grey"},
            "bb_upperband": {"color": "grey"},
        },
        "subplots": {
            "MACD": {
                "macd": {"color": "blue"},
                "macdsignal": {"color": "orange"},
                "macdhist": {"color": "green"},
            },
            "RSI": {
                "rsi": {"color": "red"},
            },
            "Up_or_down": {
                "&s-up_or_down": {"color": "green"},
            },
            "Predicted_close": {
                "&-s_close": {"color": "blue"},
            },
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    stoploss = -0.05
    use_exit_signal = True
    startup_candle_count: int = 30
    can_short = True

    # Hyperoptable parameters
    buy_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)
    sell_rsi = IntParameter(low=50, high=100, default=70, space="sell", optimize=True, load=True)
    short_rsi = IntParameter(low=51, high=100, default=70, space="sell", optimize=True, load=True)
    exit_short_rsi = IntParameter(low=1, high=50, default=30, space="buy", optimize=True, load=True)

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

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # TA timing: RSI crosses above buy threshold (exits oversold zone)
                (qtpylib.crossed_above(df["rsi"], self.buy_rsi.value))
                & (df["tema"] <= df["bb_middleband"])  # Guard: tema below BB middle
                & (df["tema"] > df["tema"].shift(1))  # Guard: tema is raising
                & (df["volume"] > 0)  # Make sure Volume is not 0
                # ML direction filter: model is confident and predicts upward move
                & (df["do_predict"] == 1)
                & (df["&s-up_or_down"] == "up")
                # ML amplitude filter: predicted price change exceeds +1%
                & (df["&-s_close"] > 0.01)
            ),
            "enter_long",
        ] = 1

        df.loc[
            (
                # TA timing: RSI crosses above short threshold (enters overbought zone)
                (qtpylib.crossed_above(df["rsi"], self.short_rsi.value))
                & (df["tema"] > df["bb_middleband"])  # Guard: tema above BB middle
                & (df["tema"] < df["tema"].shift(1))  # Guard: tema is falling
                & (df["volume"] > 0)  # Make sure Volume is not 0
                # ML direction filter: model is confident and predicts downward move
                & (df["do_predict"] == 1)
                & (df["&s-up_or_down"] == "down")
                # ML amplitude filter: predicted price change below -1%
                & (df["&-s_close"] < -0.01)
            ),
            "enter_short",
        ] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                # TA exit: RSI enters overbought, tema above middle and falling
                (qtpylib.crossed_above(df["rsi"], self.sell_rsi.value))
                & (df["tema"] > df["bb_middleband"])
                & (df["tema"] < df["tema"].shift(1))
                & (df["volume"] > 0)
            )
            | (
                # Early ML exit: model predicts a downward move while in a long position
                (df["&-s_close"] < 0)
                & (df["do_predict"] == 1)
            ),
            "exit_long",
        ] = 1

        df.loc[
            (
                # TA exit: RSI exits oversold, tema below middle and rising
                (qtpylib.crossed_above(df["rsi"], self.exit_short_rsi.value))
                & (df["tema"] <= df["bb_middleband"])
                & (df["tema"] > df["tema"].shift(1))
                & (df["volume"] > 0)
            )
            | (
                # Early ML exit: model predicts an upward move while in a short position
                (df["&-s_close"] > 0)
                & (df["do_predict"] == 1)
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
