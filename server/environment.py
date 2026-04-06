from __future__ import annotations

import sys
import os
from typing import Any, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action,
    Observation,
    State,
)

from models import (
    ActionType,
    DeFiRiskAction,
    DeFiRiskObservation,
    DeFiRiskState,
    PositionInfo,
)
from graders.grader import TaskGrader, StepContext, compute_step_reward
from simulation.market import MarketSimulator
from simulation.protocol import LendingProtocol
from simulation.scenarios import TASK_SCENARIOS, TaskScenario


class DeFiRiskEnvironment(Environment):

    def __init__(self):
        super().__init__()
        self._task_id: str = os.environ.get("DEFI_TASK", "mild_dip")
        self._protocol: Optional[LendingProtocol] = None
        self._market: Optional[MarketSimulator] = None
        self._grader: Optional[TaskGrader] = None
        self._scenario: Optional[TaskScenario] = None
        self._state = DeFiRiskState(episode_id=str(uuid4()))
        self._last_action_result: Optional[str] = None
        self._last_action_error: Optional[str] = None
        self._last_market_events: list[str] = []

    def _init_scenario(self, task_id: str, seed: Optional[int] = None) -> None:
        factory = TASK_SCENARIOS.get(task_id)
        if factory is None:
            raise ValueError(
                f"Unknown task: {task_id}. "
                f"Available: {list(TASK_SCENARIOS.keys())}"
            )
        self._scenario = factory()
        effective_seed = seed if seed is not None else self._scenario.seed

        self._protocol = LendingProtocol(
            initial_prices=self._scenario.initial_prices,
            positions=list(self._scenario.positions),
            treasury=self._scenario.treasury,
            collateral_ratio_min=self._scenario.collateral_ratio_min,
            liquidation_threshold=self._scenario.liquidation_threshold,
            interest_rate=self._scenario.interest_rate,
            max_steps=self._scenario.max_steps,
        )
        self._market = MarketSimulator(
            base_prices=self._scenario.initial_prices,
            seed=effective_seed,
            base_volatility=self._scenario.base_volatility,
        )
        for at_step, event in self._scenario.market_events:
            self._market.schedule_event(event, at_step)

        self._grader = TaskGrader(
            task_id=task_id,
            difficulty=self._scenario.difficulty,
            initial_positions=len(self._scenario.positions),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> DeFiRiskObservation:
        task_id = kwargs.get("task_id", self._task_id)
        self._task_id = task_id
        self._init_scenario(task_id, seed)
        self._last_action_result = None
        self._last_action_error = None
        self._last_market_events = []

        self._state = DeFiRiskState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=task_id,
            task_name=self._scenario.name,
            difficulty=self._scenario.difficulty,
            is_terminal=False,
            cumulative_reward=0.0,
            actions_taken=[],
        )

        return self._build_observation(reward=0.0, done=False)

    def step(
        self,
        action: DeFiRiskAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> DeFiRiskObservation:
        if self._protocol is None or self._market is None:
            self._last_action_error = "Environment not initialized. Call reset() first."
            return self._build_observation(reward=0.0, done=True)

        if self._protocol.is_done:
            self._last_action_error = "Episode already finished."
            return self._build_observation(reward=0.0, done=True)

        state_before = self._protocol.get_state()

        new_prices, events = self._market.step()
        self._protocol.update_prices(new_prices)
        self._last_market_events = events

        action_result = self._execute_action(action)
        self._last_action_result = action_result
        self._last_action_error = None

        self._protocol.advance_step()

        state_after = self._protocol.get_state()

        ctx = StepContext(
            state_before=state_before,
            state_after=state_after,
            action_type=action.action_type.value,
            action_amount=action.amount,
            market_events=events,
        )
        reward = compute_step_reward(ctx)
        self._grader.record_step(reward)

        done = self._protocol.is_done
        self._state.step_count += 1
        self._state.cumulative_reward += reward
        self._state.actions_taken.append(action.action_type.value)
        self._state.is_terminal = done

        if done:
            self._grader.finalize(
                is_solvent=not self._protocol._is_insolvent,
                protocol_health=self._protocol.protocol_health,
                positions_remaining=len(self._protocol._positions),
                total_bad_debt=self._protocol._total_bad_debt,
                treasury_remaining=self._protocol._treasury,
            )

        return self._build_observation(reward=reward, done=done)

    def _execute_action(self, action: DeFiRiskAction) -> str:
        at = action.action_type
        amount = action.amount or 0.0

        if at == ActionType.RAISE_COLLATERAL_RATIO:
            return self._protocol.raise_collateral_ratio(amount)
        elif at == ActionType.LOWER_COLLATERAL_RATIO:
            return self._protocol.lower_collateral_ratio(amount)
        elif at == ActionType.ADJUST_INTEREST:
            return self._protocol.adjust_interest(amount)
        elif at == ActionType.PAUSE_BORROWING:
            return self._protocol.pause_borrowing()
        elif at == ActionType.RESUME_BORROWING:
            return self._protocol.resume_borrowing()
        elif at == ActionType.TRIGGER_LIQUIDATIONS:
            return self._protocol.trigger_liquidations()
        elif at == ActionType.NO_OP:
            return "No action taken"
        else:
            return f"Unknown action type: {at}"

    def _build_observation(
        self, reward: float, done: bool
    ) -> DeFiRiskObservation:
        if self._protocol is None:
            return DeFiRiskObservation(
                done=True,
                reward=0.0,
                tvl=0.0,
                total_borrowed=0.0,
                collateral_ratio_min=0.0,
                weighted_collateral_ratio=0.0,
                liquidation_threshold=0.0,
                interest_rate=0.0,
                borrowing_enabled=False,
                oracle_prices={},
                oracle_price_changes={},
                protocol_health=0.0,
                positions_count=0,
                positions_at_risk=0,
                positions_underwater=0,
                treasury_balance=0.0,
                total_bad_debt=0.0,
                is_solvent=False,
                utilization_rate=0.0,
                step_number=0,
                max_steps=0,
                last_action_result=self._last_action_result,
                last_action_error=self._last_action_error,
                market_events=[],
                positions_summary=[],
            )

        wcr = self._protocol.weighted_collateral_ratio
        if wcr == float("inf"):
            wcr = 999.99

        positions_summary = []
        for p in self._protocol._positions:
            cv = p.collateral_value(self._protocol._prices)
            cr = p.collateral_ratio(self._protocol._prices)
            if cr == float("inf"):
                cr = 999.99
            risk_buffer = self._protocol._liquidation_threshold * 1.1
            positions_summary.append(
                PositionInfo(
                    collateral_token=p.collateral_token,
                    collateral_amount=p.collateral_amount,
                    borrowed_amount=p.borrowed_amount,
                    collateral_value_usd=cv,
                    collateral_ratio=round(cr, 4),
                    is_at_risk=cr < risk_buffer,
                )
            )

        return DeFiRiskObservation(
            done=done,
            reward=round(reward, 4),
            tvl=round(self._protocol.tvl, 2),
            total_borrowed=round(self._protocol.total_borrowed, 2),
            collateral_ratio_min=round(self._protocol._collateral_ratio_min, 4),
            weighted_collateral_ratio=round(wcr, 4),
            liquidation_threshold=round(self._protocol._liquidation_threshold, 4),
            interest_rate=round(self._protocol._interest_rate, 4),
            borrowing_enabled=self._protocol._borrowing_enabled,
            oracle_prices={
                k: round(v, 2) for k, v in self._protocol._prices.items()
            },
            oracle_price_changes={
                k: round(v, 2) for k, v in self._protocol.price_changes.items()
            },
            protocol_health=round(self._protocol.protocol_health, 4),
            positions_count=len(self._protocol._positions),
            positions_at_risk=self._protocol.positions_at_risk,
            positions_underwater=self._protocol.positions_underwater,
            treasury_balance=round(self._protocol._treasury, 2),
            total_bad_debt=round(self._protocol._total_bad_debt, 2),
            is_solvent=not self._protocol._is_insolvent,
            utilization_rate=round(self._protocol.utilization_rate, 4),
            step_number=self._protocol._step,
            max_steps=self._protocol._max_steps,
            last_action_result=self._last_action_result,
            last_action_error=self._last_action_error,
            market_events=list(self._last_market_events),
            positions_summary=positions_summary,
        )

    @property
    def state(self) -> DeFiRiskState:
        return self._state

    @property
    def grader(self) -> Optional[TaskGrader]:
        return self._grader

    def get_score(self) -> float:
        if self._grader is None:
            return 0.0
        return self._grader.score()
