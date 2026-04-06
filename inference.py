"""
Inference Script for DeFi Protocol Risk Management Environment
===============================================================
Runs an LLM agent against the DeFi risk management environment.

Required environment variables:
    API_BASE_URL   - The API endpoint for the LLM (default: HF router)
    MODEL_NAME     - The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       - Your Hugging Face / API key
    IMAGE_NAME     - Docker image name for the environment

STDOUT format follows the mandatory [START]/[STEP]/[END] protocol.
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from models import ActionType, DeFiRiskAction, DeFiRiskObservation
from server.environment import DeFiRiskEnvironment
from graders.grader import TaskGrader

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = "defi_risk"
MAX_STEPS_OVERRIDE = int(os.getenv("MAX_STEPS", "0"))
TEMPERATURE = 0.3
MAX_TOKENS = 512

TASKS = ["mild_dip", "flash_crash", "cascading_crisis"]

SYSTEM_PROMPT = textwrap.dedent("""\
You are a DeFi protocol risk analyst managing a lending protocol similar to Aave or Compound.

Your job is to monitor protocol metrics, detect liquidation risk, adjust parameters, and prevent insolvency.

Available actions (respond with EXACTLY one JSON object):
{
  "action_type": "<ACTION>",
  "amount": <float or null>,
  "reasoning": "<brief explanation>"
}

Action types:
- RAISE_COLLATERAL_RATIO: Increase minimum collateral ratio (amount: 0.01-0.5)
- LOWER_COLLATERAL_RATIO: Decrease minimum collateral ratio (amount: 0.01-0.5)
- ADJUST_INTEREST: Change interest rate (amount: -0.1 to 0.2)
- PAUSE_BORROWING: Halt new borrows (no amount needed)
- RESUME_BORROWING: Resume borrows (no amount needed)
- TRIGGER_LIQUIDATIONS: Force liquidate undercollateralized positions (no amount needed)
- NO_OP: Do nothing this step (no amount needed)

Key principles:
1. When positions are underwater (collateral ratio below liquidation threshold), TRIGGER_LIQUIDATIONS immediately.
2. When prices are dropping, RAISE_COLLATERAL_RATIO and consider PAUSE_BORROWING.
3. When many positions are at risk, increase interest rates to discourage borrowing.
4. Only RESUME_BORROWING when the crisis has passed and no positions are at risk.
5. Avoid NO_OP during market stress - inaction during crisis is penalized.
6. Balance safety (high collateral ratios) with capital efficiency.

Respond with ONLY the JSON object, no other text.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool, steps: int, score: float, rewards: List[float]
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_observation_prompt(obs_dict: dict) -> str:
    lines = [
        "Current Protocol State:",
        f"  TVL: ${obs_dict['tvl']:,.2f}",
        f"  Total Borrowed: ${obs_dict['total_borrowed']:,.2f}",
        f"  Weighted Collateral Ratio: {obs_dict['weighted_collateral_ratio']:.2%}",
        f"  Min Collateral Ratio Setting: {obs_dict['collateral_ratio_min']:.2%}",
        f"  Liquidation Threshold: {obs_dict['liquidation_threshold']:.2%}",
        f"  Interest Rate: {obs_dict['interest_rate']:.2%}",
        f"  Borrowing Enabled: {obs_dict['borrowing_enabled']}",
        f"  Protocol Health: {obs_dict['protocol_health']:.2%}",
        f"  Utilization Rate: {obs_dict['utilization_rate']:.2%}",
        "",
        f"  Total Positions: {obs_dict['positions_count']}",
        f"  Positions At Risk: {obs_dict['positions_at_risk']}",
        f"  Positions Underwater: {obs_dict['positions_underwater']}",
        "",
        f"  Treasury: ${obs_dict['treasury_balance']:,.2f}",
        f"  Total Bad Debt: ${obs_dict['total_bad_debt']:,.2f}",
        f"  Is Solvent: {obs_dict['is_solvent']}",
        "",
        f"  Step: {obs_dict['step_number']}/{obs_dict['max_steps']}",
        "",
        "Oracle Prices:",
    ]
    for token, price in obs_dict.get("oracle_prices", {}).items():
        change = obs_dict.get("oracle_price_changes", {}).get(token, 0)
        lines.append(f"  {token}: ${price:,.2f} ({change:+.2f}%)")

    if obs_dict.get("market_events"):
        lines.append("")
        lines.append("Market Events This Step:")
        for ev in obs_dict["market_events"]:
            lines.append(f"  - {ev}")

    if obs_dict.get("last_action_result"):
        lines.append("")
        lines.append(f"Last Action Result: {obs_dict['last_action_result']}")

    lines.append("")
    lines.append("Decide your next action. Respond with a single JSON object.")
    return "\n".join(lines)


def parse_action(text: str) -> DeFiRiskAction:
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        text = text[start:end]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return DeFiRiskAction(
            action_type=ActionType.NO_OP,
            reasoning="Failed to parse LLM response",
        )

    action_type_str = data.get("action_type", "NO_OP")
    try:
        action_type = ActionType(action_type_str)
    except ValueError:
        action_type = ActionType.NO_OP

    return DeFiRiskAction(
        action_type=action_type,
        amount=data.get("amount"),
        reasoning=data.get("reasoning"),
    )


def get_model_action(
    client: OpenAI, obs_dict: dict, history: List[dict]
) -> DeFiRiskAction:
    user_prompt = build_observation_prompt(obs_dict)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-6:]:
        messages.append(h)
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        action = parse_action(text)
        return action
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return DeFiRiskAction(
            action_type=ActionType.NO_OP,
            reasoning=f"Model error: {exc}",
        )


def obs_to_dict(obs: DeFiRiskObservation) -> dict:
    return obs.model_dump()


async def run_task(client: OpenAI, task_id: str) -> float:
    env = DeFiRiskEnvironment()
    os.environ["DEFI_TASK"] = task_id
    env._task_id = task_id

    history: List[dict] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset(task_id=task_id)
        obs_dict = obs_to_dict(result)
        max_steps = obs_dict["max_steps"]
        if MAX_STEPS_OVERRIDE > 0:
            max_steps = min(MAX_STEPS_OVERRIDE, max_steps)

        for step in range(1, max_steps + 1):
            if result.done:
                break

            action = get_model_action(client, obs_dict, history)
            action_str = (
                f"{action.action_type.value}"
                f"({action.amount if action.amount else ''})"
            )

            result = env.step(action)
            obs_dict = obs_to_dict(result)

            reward = result.reward or 0.0
            done = result.done
            error = result.last_action_error

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action_type": action.action_type.value,
                            "amount": action.amount,
                            "reasoning": action.reasoning,
                        }
                    ),
                }
            )
            history.append(
                {
                    "role": "user",
                    "content": build_observation_prompt(obs_dict),
                }
            )

            if done:
                break

        score = env.get_score()
        score = min(max(score, 0.0), 1.0)
        success = score >= 0.3

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id}", flush=True)
        print(f"{'='*60}", flush=True)
        score = await run_task(client, task_id)
        scores[task_id] = score
        print(f"Task {task_id} score: {score:.3f}", flush=True)

    print(f"\n{'='*60}", flush=True)
    print("Final Scores:", flush=True)
    for task_id, score in scores.items():
        print(f"  {task_id}: {score:.3f}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  Average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
