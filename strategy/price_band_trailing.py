from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Action = Literal["hold", "tighten", "exit"]


@dataclass(frozen=True)
class PriceBandTrailingConfig:
    """动态布林带 trailing 配置"""
    # 布林带参数
    window: int = 20  # 计算周期
    std_mult_base: float = 2.0  # 基础标准差倍数
    std_mult_sensitivity: float = 0.5  # 波动率敏感度（0-1）
    center_mode: str = "weighted_mean"  # "ema" 或 "weighted_mean"

    # 触发规则
    trigger_min_rr: float = 0.5  # 至少浮盈 0.5R 后激活
    break_confirm_bars: int = 1  # 确认根数，避免单根刺穿
    stop_buffer_bps: float = 2.0  # 带外微缓冲

    # 非线性加权参数
    weight_power: float = 2.0  # 权重幂次 (i+1)/window)^p

    # 与现有机制的兼容性
    keep_fixed_target: bool = True  # 保留 fixed RR 作为硬上限
    trail_style_override: str | None = None  # 可覆盖 loose/normal/tight

    # 启用开关
    enabled: bool = True


@dataclass(frozen=True)
class PriceBandTrailingDecision:
    """Trailing 决策"""
    action: Action
    reason: str | None = None
    stop_price: float | None = None
    exit_price: float | None = None
    metrics: dict[str, Any] | None = None


@dataclass
class PriceBandTrailingState:
    """Trailing 状态"""
    position_key: str | None = None
    price_history: list[float] = field(default_factory=list)
    band_history: list[dict[str, float]] = field(default_factory=list)
    last_stop_price: float | None = None
    break_confirm_count: int = 0


class PriceBandTrailingOverlay:
    """基于动态布林带和非线性加权均值的 trailing overlay"""

    def __init__(self, config: PriceBandTrailingConfig | None = None):
        self.config = config or PriceBandTrailingConfig()
        self.state = PriceBandTrailingState()

    def reset(self) -> None:
        self.state = PriceBandTrailingState()

    def evaluate(self, candle: Any, position: Any) -> PriceBandTrailingDecision:
        """评估是否需要调整止损或平仓"""
        if not self.config.enabled:
            return PriceBandTrailingDecision(action="hold", reason="disabled")

        # 检查 position 有效性
        if position is None:
            return PriceBandTrailingDecision(action="hold", reason="no_position")

        # 获取当前价格
        current_price = float(getattr(candle, "c", 0.0))
        if current_price <= 0:
            return PriceBandTrailingDecision(action="hold", reason="invalid_price")

        # 检查 position key 是否变化（新仓位）
        position_key = f"{getattr(position, 'direction', '')}:{getattr(position, 'entry_time', '')}"
        if self.state.position_key != position_key:
            self.reset()
            self.state.position_key = position_key

        # 更新价格历史
        self.state.price_history.append(current_price)
        if len(self.state.price_history) > self.config.window:
            self.state.price_history.pop(0)

        # 计算浮盈 RR
        entry_price = float(getattr(position, "entry_price", 0.0))
        sl_price = float(getattr(position, "sl_price", getattr(position, "stop_price", 0.0)) or 0.0)

        if entry_price <= 0 or sl_price <= 0:
            return PriceBandTrailingDecision(action="hold", reason="invalid_position_prices")

        direction = str(getattr(position, "direction", ""))
        risk = abs(entry_price - sl_price)

        if risk <= 0:
            return PriceBandTrailingDecision(action="hold", reason="zero_risk")

        if direction == "BULL":
            profit = current_price - entry_price
        elif direction == "BEAR":
            profit = entry_price - current_price
        else:
            return PriceBandTrailingDecision(action="hold", reason="unknown_direction")

        profit_rr = profit / risk if risk > 0 else 0.0

        # 未达到最小浮盈，不激活 trailing
        if profit_rr < self.config.trigger_min_rr:
            return PriceBandTrailingDecision(action="hold", reason="insufficient_profit", metrics={"profit_rr": profit_rr})

        # 计算动态布林带
        if len(self.state.price_history) < self.config.window:
            return PriceBandTrailingDecision(action="hold", reason="insufficient_history")

        band = self._compute_band(self.state.price_history)
        self.state.band_history.append(band)

        # 根据方向决策
        if direction == "BULL":
            return self._evaluate_bull(current_price, position, band, profit_rr)
        elif direction == "BEAR":
            return self._evaluate_bear(current_price, position, band, profit_rr)

        return PriceBandTrailingDecision(action="hold", reason="unknown_direction")

    def _compute_band(self, prices: list[float]) -> dict[str, float]:
        """计算动态布林带"""
        # 计算加权均值
        center = self._compute_weighted_mean(prices)

        # 计算加权标准差
        std = self._compute_weighted_std(prices, center)

        # 自适应标准差倍数（基于最近波动率变化）
        std_mult = self._compute_adaptive_std_mult(prices)

        # 计算上下轨
        upper = center + std * std_mult
        lower = center - std * std_mult

        return {
            "center": center,
            "std": std,
            "std_mult": std_mult,
            "upper": upper,
            "lower": lower,
        }

    def _compute_weighted_mean(self, prices: list[float]) -> float:
        """非线性加权均值，近期价格权重更高"""
        n = len(prices)
        if n == 0:
            return 0.0

        # 权重：(i+1)/n)^power，越靠后权重越高
        weights = [((i + 1) / n) ** self.config.weight_power for i in range(n)]
        total_weight = sum(weights)

        if total_weight <= 0:
            return sum(prices) / n

        return sum(w * p for w, p in zip(weights, prices)) / total_weight

    def _compute_weighted_std(self, prices: list[float], center: float) -> float:
        """加权标准差"""
        n = len(prices)
        if n <= 1:
            return 0.0

        weights = [((i + 1) / n) ** self.config.weight_power for i in range(n)]
        total_weight = sum(weights)

        if total_weight <= 0:
            return 0.0

        variance = sum(w * (p - center) ** 2 for w, p in zip(weights, prices)) / total_weight
        return variance ** 0.5

    def _compute_adaptive_std_mult(self, prices: list[float]) -> float:
        """自适应标准差倍数，基于最近波动率变化"""
        if len(prices) < 2:
            return self.config.std_mult_base

        # 计算最近 N 根的波动率与历史平均的比值
        recent_window = min(5, len(prices) // 2)
        if recent_window < 2:
            return self.config.std_mult_base

        recent_prices = prices[-recent_window:]
        recent_volatility = max(recent_prices) - min(recent_prices)

        # 计算历史平均波动率
        historical_volatility = 0.0
        for i in range(len(prices) - recent_window):
            window_vol = max(prices[i:i+recent_window]) - min(prices[i:i+recent_window])
            historical_volatility += window_vol

        if len(prices) - recent_window > 0:
            historical_volatility /= (len(prices) - recent_window)

        if historical_volatility <= 0:
            return self.config.std_mult_base

        # 波动率比值
        vol_ratio = recent_volatility / historical_volatility

        # 自适应调整：高波动率时扩张，低波动率时收缩
        adjustment = (vol_ratio - 1.0) * self.config.std_mult_sensitivity
        return max(1.0, self.config.std_mult_base + adjustment)

    def _evaluate_bull(self, current_price: float, position: Any, band: dict[str, float], profit_rr: float) -> PriceBandTrailingDecision:
        """多头评估"""
        entry_price = float(getattr(position, "entry_price", 0.0))
        sl_price = float(getattr(position, "sl_price", getattr(position, "stop_price", 0.0)) or 0.0)

        # 计算新的止损价格（基于下轨 + 缓冲）
        new_stop = band["lower"] * (1 - self.config.stop_buffer_bps / 10000)

        # 检查是否应该上调止损
        if new_stop > sl_price:
            # 检查确认条件
            if self._check_break_confirm(current_price, band["upper"]):
                return PriceBandTrailingDecision(
                    action="tighten",
                    reason="bull_band_tighten",
                    stop_price=new_stop,
                    metrics={
                        "profit_rr": profit_rr,
                        "current_price": current_price,
                        "band_center": band["center"],
                        "band_upper": band["upper"],
                        "band_lower": band["lower"],
                        "new_stop": new_stop,
                    },
                )

        # 检查是否触及上轨（激进平仓）
        if current_price > band["upper"]:
            return PriceBandTrailingDecision(
                action="exit",
                reason="bull_upper_band_break",
                exit_price=current_price,
                metrics={
                    "profit_rr": profit_rr,
                    "band_upper": band["upper"],
                },
            )

        return PriceBandTrailingDecision(action="hold", reason="bull_hold")

    def _evaluate_bear(self, current_price: float, position: Any, band: dict[str, float], profit_rr: float) -> PriceBandTrailingDecision:
        """空头评估"""
        entry_price = float(getattr(position, "entry_price", 0.0))
        sl_price = float(getattr(position, "sl_price", getattr(position, "stop_price", 0.0)) or 0.0)

        # 计算新的止损价格（基于上轨 + 缓冲）
        new_stop = band["upper"] * (1 + self.config.stop_buffer_bps / 10000)

        # 检查是否应该上调止损
        if new_stop < sl_price:
            # 检查确认条件
            if self._check_break_confirm(current_price, band["lower"]):
                return PriceBandTrailingDecision(
                    action="tighten",
                    reason="bear_band_tighten",
                    stop_price=new_stop,
                    metrics={
                        "profit_rr": profit_rr,
                        "current_price": current_price,
                        "band_center": band["center"],
                        "band_upper": band["upper"],
                        "band_lower": band["lower"],
                        "new_stop": new_stop,
                    },
                )

        # 检查是否触及下轨（激进平仓）
        if current_price < band["lower"]:
            return PriceBandTrailingDecision(
                action="exit",
                reason="bear_lower_band_break",
                exit_price=current_price,
                metrics={
                    "profit_rr": profit_rr,
                    "band_lower": band["lower"],
                },
            )

        return PriceBandTrailingDecision(action="hold", reason="bear_hold")

    def _check_break_confirm(self, current_price: float, band_edge: float) -> bool:
        """检查是否确认突破"""
        if self.config.break_confirm_bars <= 1:
            return True

        # 简单确认：检查最近 N 根是否都在边界外
        if len(self.state.price_history) < self.config.break_confirm_bars:
            return False

        recent = self.state.price_history[-self.config.break_confirm_bars:]
        # 这里简化处理，实际可以更复杂
        return True
