from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .market import MarketEvent, MarketSimulator
from .protocol import LendingProtocol, Position


@dataclass
class TaskScenario:
    task_id: str
    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    max_steps: int
    seed: int
    initial_prices: dict[str, float]
    positions: list[Position]
    treasury: float
    collateral_ratio_min: float
    liquidation_threshold: float
    interest_rate: float
    market_events: list[tuple[int, MarketEvent]] = field(default_factory=list)
    price_trajectories: list[dict] = field(default_factory=list)
    base_volatility: float = 0.02


def _create_easy_scenario() -> TaskScenario:
    prices = {"ETH": 2000.0, "BTC": 40000.0, "LINK": 15.0}
    positions = [
        Position("ETH", 100.0, 120000.0),
        Position("ETH", 50.0, 55000.0),
        Position("BTC", 5.0, 130000.0),
        Position("ETH", 200.0, 200000.0),
        Position("LINK", 10000.0, 80000.0),
        Position("BTC", 3.0, 70000.0),
        Position("ETH", 80.0, 90000.0),
        Position("BTC", 2.0, 50000.0),
    ]
    events = [
        (2, MarketEvent(
            description="Mild market correction: ETH drops ~8%",
            price_impacts={"ETH": 0.92},
            volatility_boost=0.01,
        )),
        (4, MarketEvent(
            description="LINK oracle update shows weakness: -6%",
            price_impacts={"LINK": 0.94},
            volatility_boost=0.005,
        )),
    ]
    return TaskScenario(
        task_id="mild_dip",
        name="Mild Market Dip",
        description=(
            "Oracle prices dip 5-10% on select assets. "
            "A few positions approach the liquidation threshold. "
            "Adjust protocol parameters to maintain safety margins."
        ),
        difficulty="easy",
        max_steps=10,
        seed=42,
        initial_prices=prices,
        positions=positions,
        treasury=5_000_000.0,
        collateral_ratio_min=1.5,
        liquidation_threshold=1.3,
        interest_rate=0.05,
        market_events=events,
        base_volatility=0.015,
    )


def _create_medium_scenario() -> TaskScenario:
    prices = {"ETH": 2000.0, "BTC": 40000.0, "LINK": 15.0, "AVAX": 35.0}
    positions = [
        Position("ETH", 150.0, 210000.0),   # CR 1.43
        Position("ETH", 80.0, 115000.0),    # CR 1.39
        Position("BTC", 8.0, 230000.0),     # CR 1.39
        Position("LINK", 20000.0, 210000.0), # CR 1.43
        Position("ETH", 300.0, 420000.0),   # CR 1.43
        Position("AVAX", 5000.0, 125000.0), # CR 1.40
        Position("BTC", 4.0, 115000.0),     # CR 1.39
        Position("ETH", 60.0, 86000.0),     # CR 1.40
        Position("LINK", 15000.0, 160000.0), # CR 1.41
        Position("AVAX", 8000.0, 200000.0), # CR 1.40
        Position("ETH", 120.0, 175000.0),   # CR 1.37
        Position("BTC", 6.0, 175000.0),     # CR 1.37
    ]
    events = [
        (1, MarketEvent(
            description="Flash crash begins: ETH -15%, BTC -12%",
            price_impacts={"ETH": 0.85, "BTC": 0.88},
            volatility_boost=0.04,
        )),
        (3, MarketEvent(
            description="Contagion spreads: LINK -20%, AVAX -18%",
            price_impacts={"LINK": 0.80, "AVAX": 0.82},
            volatility_boost=0.03,
        )),
        (5, MarketEvent(
            description="Partial recovery signal: ETH +5%",
            price_impacts={"ETH": 0.90},
            volatility_boost=0.02,
        )),
        (7, MarketEvent(
            description="Secondary dip: BTC -8% from current",
            price_impacts={"BTC": 0.80},
            volatility_boost=0.02,
        )),
    ]
    return TaskScenario(
        task_id="flash_crash",
        name="Flash Crash Recovery",
        description=(
            "A sudden market crash sends ETH and BTC down 15-20%. "
            "Contagion spreads to altcoins. Multiple positions go underwater. "
            "Execute liquidations, adjust parameters, and prevent insolvency."
        ),
        difficulty="medium",
        max_steps=12,
        seed=123,
        initial_prices=prices,
        positions=positions,
        treasury=4_000_000.0,
        collateral_ratio_min=1.5,
        liquidation_threshold=1.3,
        interest_rate=0.05,
        market_events=events,
        base_volatility=0.025,
    )


def _create_hard_scenario() -> TaskScenario:
    prices = {
        "ETH": 2000.0,
        "BTC": 40000.0,
        "LINK": 15.0,
        "AVAX": 35.0,
        "SOL": 100.0,
    }
    positions = [
        Position("ETH", 500.0, 720000.0),    # CR 1.39
        Position("ETH", 200.0, 295000.0),    # CR 1.36
        Position("BTC", 15.0, 440000.0),     # CR 1.36
        Position("LINK", 50000.0, 550000.0), # CR 1.36
        Position("AVAX", 20000.0, 520000.0), # CR 1.35
        Position("SOL", 3000.0, 225000.0),   # CR 1.33
        Position("ETH", 350.0, 520000.0),    # CR 1.35
        Position("BTC", 10.0, 300000.0),     # CR 1.33
        Position("LINK", 30000.0, 340000.0), # CR 1.32
        Position("AVAX", 15000.0, 395000.0), # CR 1.33
        Position("SOL", 5000.0, 380000.0),   # CR 1.32
        Position("ETH", 180.0, 270000.0),    # CR 1.33
        Position("BTC", 7.0, 210000.0),      # CR 1.33
        Position("LINK", 40000.0, 460000.0), # CR 1.30
        Position("SOL", 4000.0, 305000.0),   # CR 1.31
        Position("AVAX", 10000.0, 265000.0), # CR 1.32
        Position("ETH", 400.0, 600000.0),    # CR 1.33
        Position("BTC", 12.0, 365000.0),     # CR 1.32
    ]
    events = [
        (0, MarketEvent(
            description="Oracle manipulation detected: ETH price spike then crash -35%",
            price_impacts={"ETH": 0.65},
            volatility_boost=0.08,
        )),
        (1, MarketEvent(
            description="Cascading liquidations trigger: multi-asset crash. "
                        "BTC -30%, LINK -40%, SOL -35%",
            price_impacts={"BTC": 0.70, "LINK": 0.60, "SOL": 0.65},
            volatility_boost=0.10,
        )),
        (2, MarketEvent(
            description="AVAX contagion: -30%",
            price_impacts={"AVAX": 0.70},
            volatility_boost=0.06,
        )),
        (3, MarketEvent(
            description="Dead cat bounce: brief recovery ETH +5%, then resumed decline",
            price_impacts={"ETH": 0.68, "AVAX": 0.60},
            volatility_boost=0.05,
        )),
        (5, MarketEvent(
            description="Stablecoin depeg fear: all assets -15% from current",
            price_impacts={
                "ETH": 0.55,
                "BTC": 0.60,
                "LINK": 0.50,
                "AVAX": 0.45,
                "SOL": 0.52,
            },
            volatility_boost=0.09,
        )),
        (7, MarketEvent(
            description="Brief stabilization",
            price_impacts={"ETH": 0.58, "BTC": 0.62},
            volatility_boost=0.04,
        )),
        (9, MarketEvent(
            description="Second wave: protocol exploit rumors across DeFi",
            price_impacts={
                "ETH": 0.50,
                "LINK": 0.42,
                "SOL": 0.40,
                "AVAX": 0.38,
            },
            volatility_boost=0.10,
        )),
        (11, MarketEvent(
            description="Final stress: SOL exploit confirmed, AVAX bridge hack",
            price_impacts={"SOL": 0.30, "AVAX": 0.28},
            volatility_boost=0.08,
        )),
    ]
    return TaskScenario(
        task_id="cascading_crisis",
        name="Cascading Liquidation Crisis",
        description=(
            "A multi-stage crisis: oracle manipulation triggers cascading liquidations "
            "across 5 assets and 18 positions. Protocol faces insolvency risk with "
            "total bad debt threatening to exceed treasury. "
            "Requires precise sequencing of liquidations, parameter adjustments, "
            "and borrowing controls over 15 steps."
        ),
        difficulty="hard",
        max_steps=15,
        seed=777,
        initial_prices=prices,
        positions=positions,
        treasury=3_500_000.0,
        collateral_ratio_min=1.5,
        liquidation_threshold=1.3,
        interest_rate=0.05,
        market_events=events,
        base_volatility=0.035,
    )


TASK_SCENARIOS: dict[str, Callable[[], TaskScenario]] = {
    "mild_dip": _create_easy_scenario,
    "flash_crash": _create_medium_scenario,
    "cascading_crisis": _create_hard_scenario,
}
