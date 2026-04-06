from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class MarketEvent:
    description: str
    price_impacts: dict[str, float]  # token -> multiplier (0.9 = 10% drop)
    volatility_boost: float = 0.0


class MarketSimulator:
    def __init__(
        self,
        base_prices: dict[str, float],
        seed: int | None = None,
        base_volatility: float = 0.02,
    ):
        self._base_prices = dict(base_prices)
        self._current_prices = dict(base_prices)
        self._rng = random.Random(seed)
        self._base_volatility = base_volatility
        self._extra_volatility = 0.0
        self._pending_events: list[MarketEvent] = []
        self._step = 0

    @property
    def current_prices(self) -> dict[str, float]:
        return dict(self._current_prices)

    def schedule_event(self, event: MarketEvent, at_step: int) -> None:
        self._pending_events.append((at_step, event))
        self._pending_events.sort(key=lambda x: x[0])

    def schedule_price_trajectory(
        self,
        token: str,
        target_multiplier: float,
        over_steps: int,
        start_step: int = 0,
    ) -> None:
        if over_steps <= 0:
            return
        per_step = target_multiplier ** (1.0 / over_steps)
        for i in range(over_steps):
            cumulative = per_step ** (i + 1)
            event = MarketEvent(
                description=f"{token} price trajectory step {i+1}",
                price_impacts={token: cumulative},
            )
            self._pending_events.append((start_step + i, event))
        self._pending_events.sort(key=lambda x: x[0])

    def step(self) -> tuple[dict[str, float], list[str]]:
        events_fired: list[str] = []
        event_multipliers: dict[str, float] = {}

        remaining = []
        for at_step, event in self._pending_events:
            if at_step <= self._step:
                events_fired.append(event.description)
                self._extra_volatility = max(
                    self._extra_volatility, event.volatility_boost
                )
                for token, mult in event.price_impacts.items():
                    base = self._base_prices.get(token, 0)
                    if base > 0:
                        event_multipliers[token] = mult
            else:
                remaining.append((at_step, event))
        self._pending_events = remaining

        vol = self._base_volatility + self._extra_volatility
        new_prices = {}
        for token, price in self._current_prices.items():
            noise = self._rng.gauss(0, vol)
            new_price = price * (1 + noise)

            if token in event_multipliers:
                target = self._base_prices[token] * event_multipliers[token]
                new_price = price + (target - price) * 0.7
                new_price += new_price * self._rng.gauss(0, vol * 0.3)

            new_prices[token] = max(new_price, 0.01)

        self._current_prices = new_prices
        self._extra_volatility *= 0.7
        self._step += 1

        return dict(new_prices), events_fired
