from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    collateral_token: str
    collateral_amount: float
    borrowed_amount: float

    def collateral_value(self, prices: dict[str, float]) -> float:
        return self.collateral_amount * prices.get(self.collateral_token, 0.0)

    def collateral_ratio(self, prices: dict[str, float]) -> float:
        if self.borrowed_amount <= 0:
            return float("inf")
        return self.collateral_value(prices) / self.borrowed_amount

    def is_undercollateralized(
        self, prices: dict[str, float], threshold: float
    ) -> bool:
        return self.collateral_ratio(prices) < threshold


@dataclass
class ProtocolState:
    collateral_ratio_min: float
    liquidation_threshold: float
    interest_rate: float
    borrowing_enabled: bool
    treasury: float
    positions: list[Position]
    oracle_prices: dict[str, float]
    previous_prices: dict[str, float]
    step: int
    max_steps: int
    is_insolvent: bool = False
    liquidations_this_step: int = 0
    bad_debt_this_step: float = 0.0
    total_bad_debt: float = 0.0
    actions_taken: list[str] = field(default_factory=list)


class LendingProtocol:
    def __init__(
        self,
        initial_prices: dict[str, float],
        positions: list[Position],
        treasury: float = 5_000_000.0,
        collateral_ratio_min: float = 1.5,
        liquidation_threshold: float = 1.3,
        interest_rate: float = 0.05,
        max_steps: int = 10,
    ):
        self._prices = dict(initial_prices)
        self._previous_prices = dict(initial_prices)
        self._positions = list(positions)
        self._treasury = treasury
        self._collateral_ratio_min = collateral_ratio_min
        self._liquidation_threshold = liquidation_threshold
        self._interest_rate = interest_rate
        self._borrowing_enabled = True
        self._step = 0
        self._max_steps = max_steps
        self._is_insolvent = False
        self._liquidations_this_step = 0
        self._bad_debt_this_step = 0.0
        self._total_bad_debt = 0.0
        self._actions_taken: list[str] = []

    def get_state(self) -> ProtocolState:
        return ProtocolState(
            collateral_ratio_min=self._collateral_ratio_min,
            liquidation_threshold=self._liquidation_threshold,
            interest_rate=self._interest_rate,
            borrowing_enabled=self._borrowing_enabled,
            treasury=self._treasury,
            positions=list(self._positions),
            oracle_prices=dict(self._prices),
            previous_prices=dict(self._previous_prices),
            step=self._step,
            max_steps=self._max_steps,
            is_insolvent=self._is_insolvent,
            liquidations_this_step=self._liquidations_this_step,
            bad_debt_this_step=self._bad_debt_this_step,
            total_bad_debt=self._total_bad_debt,
            actions_taken=list(self._actions_taken),
        )

    @property
    def tvl(self) -> float:
        total = 0.0
        for p in self._positions:
            total += p.collateral_value(self._prices)
        total += self._treasury
        return total

    @property
    def total_borrowed(self) -> float:
        return sum(p.borrowed_amount for p in self._positions)

    @property
    def weighted_collateral_ratio(self) -> float:
        total_collateral = sum(
            p.collateral_value(self._prices) for p in self._positions
        )
        total_borrowed = self.total_borrowed
        if total_borrowed <= 0:
            return float("inf")
        return total_collateral / total_borrowed

    @property
    def utilization_rate(self) -> float:
        total_collateral = sum(
            p.collateral_value(self._prices) for p in self._positions
        )
        if total_collateral <= 0:
            return 0.0
        return min(self.total_borrowed / total_collateral, 1.0)

    @property
    def positions_at_risk(self) -> int:
        count = 0
        risk_buffer = self._liquidation_threshold * 1.1
        for p in self._positions:
            if p.collateral_ratio(self._prices) < risk_buffer:
                count += 1
        return count

    @property
    def positions_underwater(self) -> int:
        count = 0
        for p in self._positions:
            if p.is_undercollateralized(self._prices, self._liquidation_threshold):
                count += 1
        return count

    @property
    def protocol_health(self) -> float:
        if self._is_insolvent:
            return 0.0
        wcr = self.weighted_collateral_ratio
        if wcr == float("inf"):
            return 1.0
        health = min(wcr / 2.0, 1.0)
        debt_penalty = min(self._total_bad_debt / max(self.tvl, 1.0), 0.5)
        return max(health - debt_penalty, 0.0)

    @property
    def price_changes(self) -> dict[str, float]:
        changes = {}
        for token, price in self._prices.items():
            prev = self._previous_prices.get(token, price)
            if prev > 0:
                changes[token] = ((price - prev) / prev) * 100.0
            else:
                changes[token] = 0.0
        return changes

    def update_prices(self, new_prices: dict[str, float]) -> None:
        self._previous_prices = dict(self._prices)
        for token, price in new_prices.items():
            if token in self._prices:
                self._prices[token] = max(price, 0.01)

    def raise_collateral_ratio(self, amount: float) -> str:
        clamped = max(0.01, min(amount, 0.5))
        old = self._collateral_ratio_min
        self._collateral_ratio_min = min(self._collateral_ratio_min + clamped, 3.0)
        self._actions_taken.append("RAISE_COLLATERAL_RATIO")
        return (
            f"Collateral ratio raised from {old:.2%} to "
            f"{self._collateral_ratio_min:.2%}"
        )

    def lower_collateral_ratio(self, amount: float) -> str:
        clamped = max(0.01, min(amount, 0.5))
        old = self._collateral_ratio_min
        self._collateral_ratio_min = max(
            self._collateral_ratio_min - clamped, 1.05
        )
        self._actions_taken.append("LOWER_COLLATERAL_RATIO")
        return (
            f"Collateral ratio lowered from {old:.2%} to "
            f"{self._collateral_ratio_min:.2%}"
        )

    def adjust_interest(self, amount: float) -> str:
        clamped = max(-0.1, min(amount, 0.2))
        old = self._interest_rate
        self._interest_rate = max(self._interest_rate + clamped, 0.001)
        self._interest_rate = min(self._interest_rate, 0.5)
        self._actions_taken.append("ADJUST_INTEREST")
        return (
            f"Interest rate adjusted from {old:.2%} to "
            f"{self._interest_rate:.2%}"
        )

    def pause_borrowing(self) -> str:
        if not self._borrowing_enabled:
            self._actions_taken.append("PAUSE_BORROWING")
            return "Borrowing already paused"
        self._borrowing_enabled = False
        self._actions_taken.append("PAUSE_BORROWING")
        return "Borrowing paused"

    def resume_borrowing(self) -> str:
        if self._borrowing_enabled:
            self._actions_taken.append("RESUME_BORROWING")
            return "Borrowing already enabled"
        self._borrowing_enabled = True
        self._actions_taken.append("RESUME_BORROWING")
        return "Borrowing resumed"

    def trigger_liquidations(self) -> str:
        self._liquidations_this_step = 0
        self._bad_debt_this_step = 0.0
        liquidated = []

        for i in range(len(self._positions) - 1, -1, -1):
            pos = self._positions[i]
            if pos.is_undercollateralized(
                self._prices, self._liquidation_threshold
            ):
                collateral_val = pos.collateral_value(self._prices)
                if collateral_val >= pos.borrowed_amount:
                    bonus = (collateral_val - pos.borrowed_amount) * 0.1
                    self._treasury += bonus
                else:
                    bad_debt = pos.borrowed_amount - collateral_val
                    self._bad_debt_this_step += bad_debt
                    self._total_bad_debt += bad_debt
                    self._treasury -= min(bad_debt, self._treasury * 0.1)

                liquidated.append(i)
                self._liquidations_this_step += 1
                self._positions.pop(i)

        self._actions_taken.append("TRIGGER_LIQUIDATIONS")

        if not liquidated:
            return "No positions eligible for liquidation"
        return (
            f"Liquidated {len(liquidated)} positions. "
            f"Bad debt this step: ${self._bad_debt_this_step:,.2f}"
        )

    def advance_step(self) -> None:
        self._step += 1
        self._liquidations_this_step = 0
        self._bad_debt_this_step = 0.0

        for pos in self._positions:
            pos.borrowed_amount *= 1 + self._interest_rate / 365

        if self._treasury < 0:
            self._is_insolvent = True
        if self._total_bad_debt > self.tvl * 0.5:
            self._is_insolvent = True

    @property
    def is_done(self) -> bool:
        return self._step >= self._max_steps or self._is_insolvent
