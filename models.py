from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    RAISE_COLLATERAL_RATIO = "RAISE_COLLATERAL_RATIO"
    LOWER_COLLATERAL_RATIO = "LOWER_COLLATERAL_RATIO"
    ADJUST_INTEREST = "ADJUST_INTEREST"
    PAUSE_BORROWING = "PAUSE_BORROWING"
    RESUME_BORROWING = "RESUME_BORROWING"
    TRIGGER_LIQUIDATIONS = "TRIGGER_LIQUIDATIONS"
    NO_OP = "NO_OP"


class DeFiRiskAction(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    action_type: ActionType = Field(
        description="The type of risk management action to take"
    )
    amount: Optional[float] = Field(
        default=None,
        description=(
            "Numeric parameter for the action. "
            "Required for RAISE_COLLATERAL_RATIO, LOWER_COLLATERAL_RATIO, "
            "ADJUST_INTEREST. Ignored for others."
        ),
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning for taking this action",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PositionInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    collateral_token: str
    collateral_amount: float
    borrowed_amount: float
    collateral_value_usd: float
    collateral_ratio: float
    is_at_risk: bool


class DeFiRiskObservation(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    done: bool = Field(default=False)
    reward: Optional[float] = Field(default=None)

    tvl: float = Field(description="Total Value Locked in USD")
    total_borrowed: float = Field(description="Total borrow outstanding in USD")
    collateral_ratio_min: float = Field(
        description="Current minimum collateral ratio setting"
    )
    weighted_collateral_ratio: float = Field(
        description="Weighted average collateral ratio across all positions"
    )
    liquidation_threshold: float = Field(
        description="Collateral ratio below which positions get liquidated"
    )
    interest_rate: float = Field(description="Current borrow interest rate")
    borrowing_enabled: bool = Field(description="Whether new borrows are allowed")
    oracle_prices: Dict[str, float] = Field(description="Current oracle prices")
    oracle_price_changes: Dict[str, float] = Field(
        description="Price changes since last step in percent"
    )
    protocol_health: float = Field(
        description="Protocol health score 0.0-1.0"
    )
    positions_count: int = Field(description="Total number of active positions")
    positions_at_risk: int = Field(
        description="Positions near liquidation threshold"
    )
    positions_underwater: int = Field(
        description="Positions below liquidation threshold"
    )
    treasury_balance: float = Field(description="Protocol treasury in USD")
    total_bad_debt: float = Field(description="Cumulative bad debt in USD")
    is_solvent: bool = Field(description="Whether the protocol is solvent")
    utilization_rate: float = Field(description="Borrow utilization rate")
    step_number: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum steps in this episode")
    last_action_result: Optional[str] = Field(
        default=None,
        description="Result message from last action taken",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Error message if last action failed",
    )
    market_events: List[str] = Field(
        default_factory=list,
        description="Market events that occurred this step",
    )
    positions_summary: List[PositionInfo] = Field(
        default_factory=list,
        description="Summary of all positions",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeFiRiskState(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )

    episode_id: Optional[str] = Field(default=None)
    step_count: int = Field(default=0, ge=0)
    task_id: str = Field(default="mild_dip")
    task_name: str = Field(default="")
    difficulty: str = Field(default="easy")
    is_terminal: bool = Field(default=False)
    cumulative_reward: float = Field(default=0.0)
    actions_taken: List[str] = Field(default_factory=list)
