from dataclasses import dataclass


@dataclass(frozen=True)
class IndicatorConfig:
    sma_fast: int = 50
    sma_slow: int = 200
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: float = 2
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    rsi_midpoint: float = 50


CONFIG = IndicatorConfig()
MIN_HEADLINES = 10
TRADING_DAYS = 252
