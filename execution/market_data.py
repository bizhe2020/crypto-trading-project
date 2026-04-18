from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from execution.okx_client import OkxClient


@dataclass
class MarketDataBundle:
    primary_candles: pd.DataFrame
    informative_candles: pd.DataFrame
    primary_timeframe: str
    informative_timeframe: str
    candles_15m: pd.DataFrame | None = None
    candles_4h: pd.DataFrame | None = None


class OhlcvRepository:
    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)

    def load_pair(
        self,
        pair: str = "BTC/USDT:USDT",
        client: OkxClient | None = None,
        timeframe: str = "15m",
        informative_timeframe: str = "4h",
    ) -> MarketDataBundle:
        symbol = pair.replace("/", "_").replace(":", "_")
        primary_candles = self._load_timeframe(
            symbol,
            pair,
            timeframe,
            client=client,
            fetch_limit=self._default_fetch_limit(timeframe),
        )
        informative_candles = self._load_timeframe(
            symbol,
            pair,
            informative_timeframe,
            client=client,
            fetch_limit=self._default_fetch_limit(informative_timeframe),
        )
        return MarketDataBundle(
            primary_candles=primary_candles,
            informative_candles=informative_candles,
            primary_timeframe=timeframe,
            informative_timeframe=informative_timeframe,
            candles_15m=primary_candles if timeframe == "15m" else None,
            candles_4h=informative_candles if informative_timeframe == "4h" else None,
        )

    def _load_timeframe(
        self,
        symbol: str,
        pair: str,
        timeframe: str,
        *,
        client: OkxClient | None,
        fetch_limit: int,
    ) -> pd.DataFrame:
        path = self.data_root / f"{symbol}-{timeframe}-futures.feather"
        local_df = self._read_feather(path)
        remote_df = None
        try:
            remote_df = self._fetch_remote_dataframe(client, pair, timeframe, fetch_limit)
        except Exception:
            remote_df = None

        if local_df is None and remote_df is None:
            raise FileNotFoundError(f"No OHLCV data available for {pair} {timeframe}")
        if local_df is None:
            merged = remote_df
        elif remote_df is None:
            merged = local_df
        else:
            merged = self._normalize_ohlcv_dataframe(pd.concat([local_df, remote_df], ignore_index=True))
            merged = self._prefer_remote_tail(local_df, merged, remote_df)

        if merged is None or merged.empty:
            raise ValueError(f"Merged OHLCV data is empty for {pair} {timeframe}")

        self._write_feather(path, merged)
        return merged

    def _read_feather(self, path: Path) -> pd.DataFrame | None:
        if not path.exists():
            return None
        df = pd.read_feather(path)
        if df.empty:
            return None
        return self._normalize_ohlcv_dataframe(df)

    def _fetch_remote_dataframe(
        self,
        client: OkxClient | None,
        pair: str,
        timeframe: str,
        fetch_limit: int,
    ) -> pd.DataFrame | None:
        if client is None:
            return None
        raw = client.fetch_ohlcv(pair, timeframe, limit=fetch_limit)
        if not raw:
            return None
        dataframe = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        dataframe["date"] = pd.to_datetime(dataframe["timestamp"], unit="ms", utc=True)
        return self._normalize_ohlcv_dataframe(dataframe[["date", "open", "high", "low", "close", "volume"]])

    def _prefer_remote_tail(self, local_df: pd.DataFrame, merged_df: pd.DataFrame, remote_df: pd.DataFrame) -> pd.DataFrame:
        if remote_df.empty:
            return merged_df
        cutoff = remote_df["date"].min()
        historical = local_df[local_df["date"] < cutoff]
        return self._normalize_ohlcv_dataframe(pd.concat([historical, remote_df], ignore_index=True))

    def _normalize_ohlcv_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result = dataframe.copy()
        result.columns = [str(column).strip().lower() for column in result.columns]
        if "date" not in result.columns:
            time_column = next((column for column in ("date", "timestamp", "datetime", "open_time") if column in result.columns), None)
            if time_column is None:
                raise ValueError("OHLCV data must contain a date/timestamp column")
            result = result.rename(columns={time_column: "date"})
        required_columns = {"date", "open", "high", "low", "close", "volume"}
        missing = required_columns - set(result.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"OHLCV data is missing required columns: {missing_text}")
        result["date"] = pd.to_datetime(result["date"], utc=True, errors="coerce")
        numeric_columns = ["open", "high", "low", "close", "volume"]
        result[numeric_columns] = result[numeric_columns].apply(pd.to_numeric, errors="coerce")
        result = result.dropna(subset=["date", "open", "high", "low", "close"])
        result = result.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
        if result.empty:
            raise ValueError("OHLCV data is empty after normalization")
        return result

    def _default_fetch_limit(self, timeframe: str) -> int:
        limits = {
            "15m": 500,
            "1h": 500,
            "4h": 240,
            "1d": 240,
        }
        return limits.get(timeframe, 500)

    def _write_feather(self, path: Path, dataframe: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_feather(path)
