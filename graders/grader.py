from __future__ import annotations

from dataclasses import dataclass, field

from simulation.protocol import ProtocolState


@dataclass
class StepContext:
    state_before: ProtocolState
    state_after: ProtocolState
    action_type: str
    action_amount: float | None
    market_events: list[str]


def compute_step_reward(ctx: StepContext) -> float:
    reward = 0.0

    if ctx.state_after.is_insolvent:
        return -0.7

    if not ctx.state_before.is_insolvent:
        reward += 0.1

    underwater_before = sum(
        1
        for p in ctx.state_before.positions
        if p.is_undercollateralized(
            ctx.state_before.oracle_prices,
            ctx.state_before.liquidation_threshold,
        )
    )
    underwater_after = sum(
        1
        for p in ctx.state_after.positions
        if p.is_undercollateralized(
            ctx.state_after.oracle_prices,
            ctx.state_after.liquidation_threshold,
        )
    )

    has_crisis = underwater_before > 0 or len(ctx.market_events) > 0
    prices_dropping = any(
        ctx.state_after.oracle_prices.get(t, 0) < ctx.state_before.oracle_prices.get(t, 0)
        for t in ctx.state_before.oracle_prices
    )

    if ctx.action_type == "TRIGGER_LIQUIDATIONS":
        if underwater_before > 0:
            cleared = underwater_before - underwater_after
            reward += 0.2 * min(cleared / max(underwater_before, 1), 1.0)
            if ctx.state_after.bad_debt_this_step < ctx.state_before.treasury * 0.05:
                reward += 0.05
            positions_lost = len(ctx.state_before.positions) - len(ctx.state_after.positions)
            healthy_lost = max(positions_lost - underwater_before, 0)
            if healthy_lost > 0:
                reward -= 0.05 * healthy_lost
        else:
            reward -= 0.1

    elif ctx.action_type == "RAISE_COLLATERAL_RATIO":
        if has_crisis or prices_dropping:
            reward += 0.15
            amount = ctx.action_amount or 0
            if 0.05 <= amount <= 0.3:
                reward += 0.05
        else:
            reward -= 0.05

    elif ctx.action_type == "LOWER_COLLATERAL_RATIO":
        if not has_crisis and not prices_dropping:
            reward += 0.05
        else:
            reward -= 0.15

    elif ctx.action_type == "ADJUST_INTEREST":
        amount = ctx.action_amount or 0
        if has_crisis and amount > 0:
            reward += 0.1
        elif not has_crisis and amount < 0:
            reward += 0.05
        elif has_crisis and amount < 0:
            reward -= 0.1
        else:
            reward += 0.02

    elif ctx.action_type == "PAUSE_BORROWING":
        if has_crisis and underwater_before >= 2:
            reward += 0.15
        elif not has_crisis:
            reward -= 0.1

    elif ctx.action_type == "RESUME_BORROWING":
        if not has_crisis and underwater_after == 0:
            reward += 0.1
        elif has_crisis:
            reward -= 0.15

    elif ctx.action_type == "NO_OP":
        if has_crisis and underwater_before > 0:
            reward -= 0.2
        elif not has_crisis:
            reward += 0.05

    if underwater_before > 0 and ctx.action_type != "TRIGGER_LIQUIDATIONS":
        if ctx.action_type != "NO_OP":
            reward -= 0.05 * min(underwater_before, 3)

    health_before = _compute_health(ctx.state_before)
    health_after = _compute_health(ctx.state_after)
    health_delta = health_after - health_before
    reward += health_delta * 0.3

    return max(min(reward, 1.0), -1.0)


def _compute_health(state: ProtocolState) -> float:
    if state.is_insolvent:
        return 0.0
    total_coll = sum(
        p.collateral_amount * state.oracle_prices.get(p.collateral_token, 0)
        for p in state.positions
    )
    total_borr = sum(p.borrowed_amount for p in state.positions)
    if total_borr <= 0:
        return 1.0
    wcr = total_coll / total_borr
    health = min(wcr / 2.0, 1.0)
    tvl = total_coll + state.treasury
    if tvl > 0:
        debt_penalty = min(state.total_bad_debt / tvl, 0.5)
        health -= debt_penalty
    return max(health, 0.0)


@dataclass
class TaskGrader:
    task_id: str
    difficulty: str
    initial_positions: int = 0
    step_rewards: list[float] = field(default_factory=list)
    final_solvent: bool = True
    final_health: float = 0.0
    positions_remaining: int = 0
    total_bad_debt: float = 0.0
    treasury_remaining: float = 0.0

    def record_step(self, reward: float) -> None:
        self.step_rewards.append(reward)

    def finalize(
        self,
        is_solvent: bool,
        protocol_health: float,
        positions_remaining: int,
        total_bad_debt: float,
        treasury_remaining: float,
    ) -> None:
        self.final_solvent = is_solvent
        self.final_health = protocol_health
        self.positions_remaining = positions_remaining
        self.total_bad_debt = total_bad_debt
        self.treasury_remaining = treasury_remaining

    def score(self) -> float:
        if not self.final_solvent:
            raw = max(sum(self.step_rewards) * 0.1, 0.0)
            return min(raw, 0.15)

        solvency_score = 0.15 if self.final_solvent else 0.0

        if self.initial_positions > 0:
            retention = self.positions_remaining / self.initial_positions
        else:
            retention = 0.0

        health_score = self.final_health * max(retention, 0.3) * 0.25

        step_avg = (
            sum(self.step_rewards) / len(self.step_rewards)
            if self.step_rewards
            else 0.0
        )
        action_score = max(min((step_avg + 0.2) / 0.5, 1.0), 0.0) * 0.30

        debt_ratio = (
            self.total_bad_debt / max(self.treasury_remaining, 1.0)
        )
        debt_score = max(1.0 - debt_ratio, 0.0) * 0.15

        retention_score = retention * 0.15

        total = (
            solvency_score + health_score + action_score
            + debt_score + retention_score
        )
        return max(min(total, 1.0), 0.0)
