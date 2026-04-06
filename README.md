---
title: DeFi Protocol Risk Management Environment
emoji: "\U0001F4C9"
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# DeFi Protocol Risk Management Environment

A realistic DeFi lending protocol risk management environment for training and evaluating AI agents. The agent acts as a risk analyst for an Aave/Compound-style lending protocol, monitoring market conditions, managing collateral parameters, executing liquidations, and preventing protocol insolvency.

## Motivation

DeFi risk management is a genuine, high-stakes task performed by protocol risk teams at Aave, Compound, MakerDAO, and similar protocols. These teams must:

- Monitor oracle price feeds across multiple assets in real-time
- Detect positions approaching liquidation thresholds
- Adjust protocol parameters (collateral ratios, interest rates, borrowing caps)
- Execute liquidations to prevent bad debt accumulation
- Make split-second decisions during market crashes

This environment faithfully simulates these dynamics, including cascading liquidations, oracle price shocks, and protocol insolvency risk.

## Action Space

| Action | Amount | Description |
|--------|--------|-------------|
| `RAISE_COLLATERAL_RATIO` | 0.01 - 0.5 | Increase minimum collateral ratio requirement |
| `LOWER_COLLATERAL_RATIO` | 0.01 - 0.5 | Decrease minimum collateral ratio requirement |
| `ADJUST_INTEREST` | -0.1 to 0.2 | Change borrow interest rate (positive = increase) |
| `PAUSE_BORROWING` | n/a | Halt all new borrows |
| `RESUME_BORROWING` | n/a | Resume new borrows |
| `TRIGGER_LIQUIDATIONS` | n/a | Force liquidate all undercollateralized positions |
| `NO_OP` | n/a | Take no action this step |

Actions are submitted as JSON:
```json
{
  "action_type": "TRIGGER_LIQUIDATIONS",
  "amount": null,
  "reasoning": "3 positions underwater after ETH crash"
}
```

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `tvl` | float | Total Value Locked (USD) |
| `total_borrowed` | float | Outstanding borrows (USD) |
| `collateral_ratio_min` | float | Current min collateral ratio setting |
| `weighted_collateral_ratio` | float | Weighted avg CR across positions |
| `liquidation_threshold` | float | CR below which liquidation occurs |
| `interest_rate` | float | Current borrow interest rate |
| `borrowing_enabled` | bool | Whether new borrows are allowed |
| `oracle_prices` | dict | Current token prices |
| `oracle_price_changes` | dict | Per-token price changes (%) |
| `protocol_health` | float | Health score 0.0-1.0 |
| `positions_count` | int | Active position count |
| `positions_at_risk` | int | Positions near liquidation |
| `positions_underwater` | int | Positions below liquidation threshold |
| `treasury_balance` | float | Protocol treasury (USD) |
| `total_bad_debt` | float | Cumulative bad debt (USD) |
| `is_solvent` | bool | Protocol solvency status |
| `utilization_rate` | float | Borrow utilization rate |
| `positions_summary` | list | Detailed per-position info |
| `market_events` | list | Market events this step |

## Tasks

### Task 1: Mild Market Dip (Easy)
- **ID:** `mild_dip`
- **Assets:** ETH, BTC, LINK
- **Positions:** 8
- **Steps:** 10
- **Scenario:** Select assets dip 5-10%. A few positions approach liquidation thresholds.
- **Objective:** Adjust parameters to maintain safety margins. No positions should be liquidated if managed correctly.
- **Expected score:** 0.6-0.9 for competent agents

### Task 2: Flash Crash Recovery (Medium)
- **ID:** `flash_crash`
- **Assets:** ETH, BTC, LINK, AVAX
- **Positions:** 12
- **Steps:** 12
- **Scenario:** Sudden 15-20% crash across major assets with contagion to altcoins. Multiple positions go underwater.
- **Objective:** Execute rapid liquidations, adjust parameters, and prevent protocol insolvency through a multi-phase crisis.
- **Expected score:** 0.4-0.7 for competent agents

### Task 3: Cascading Liquidation Crisis (Hard)
- **ID:** `cascading_crisis`
- **Assets:** ETH, BTC, LINK, AVAX, SOL
- **Positions:** 18
- **Steps:** 15
- **Scenario:** Oracle manipulation triggers cascading liquidations across 5 assets. Multi-stage crisis with 40%+ drawdowns, dead cat bounces, and repeated stress events. Protocol faces insolvency risk as bad debt threatens to exceed treasury.
- **Objective:** Survive through precise sequencing of liquidations, parameter adjustments, and borrowing controls.
- **Expected score:** 0.2-0.5 for frontier models

## Reward Function

The reward function provides dense, per-step signal:

**Positive signals:**
- +0.10 for maintaining solvency each step
- +0.20 for successfully liquidating underwater positions (proportional to positions cleared)
- +0.15 for raising collateral ratio during crisis
- +0.15 for pausing borrowing during severe crisis
- +0.10 for appropriate interest rate adjustments
- +0.05 for appropriate inaction during calm periods
- Health improvement bonus (proportional to health delta)

**Negative signals:**
- -0.70 for protocol insolvency (episode ends)
- -0.20 for inaction while positions are underwater
- -0.15 for resuming borrowing during active crisis
- -0.15 for lowering collateral ratio during crisis
- -0.10 for unnecessary liquidation triggers
- -0.10 for pausing borrowing when no crisis exists

**Final score** (0.0-1.0) is computed by the grader considering:
- 40% solvency (binary: survived or not)
- 30% protocol health at episode end
- 20% quality of per-step actions
- 10% debt management efficiency

## Setup

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run server locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t defi-risk-env .
docker run -p 8000:8000 defi-risk-env
```

### Run inference
```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## API Endpoints

- `POST /reset` - Reset environment (accepts `{"task_id": "mild_dip"}` in kwargs)
- `POST /step` - Execute action
- `GET /state` - Get current state
- `GET /health` - Health check
- `GET /schema` - Action/observation schemas
- `WS /ws` - WebSocket for persistent sessions

## Project Structure

```
defi-space/
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package config
├── Dockerfile             # Container image
├── requirements.txt       # Dependencies
├── inference.py           # Baseline inference script
├── models.py              # Action, Observation, State Pydantic models
├── client.py              # EnvClient implementation
├── __init__.py            # Package exports
├── simulation/
│   ├── protocol.py        # Lending protocol simulation engine
│   ├── market.py          # Market/oracle price simulator
│   └── scenarios.py       # Task scenario definitions
├── graders/
│   └── grader.py          # Task graders and reward computation
└── server/
    ├── environment.py     # DeFiRiskEnvironment(Environment)
    └── app.py             # FastAPI application
```

## Baseline Scores

Scores from Qwen2.5-72B-Instruct baseline (reproducible with seed):

| Task | Difficulty | Score |
|------|-----------|-------|
| mild_dip | Easy | ~0.70 |
| flash_crash | Medium | ~0.50 |
| cascading_crisis | Hard | ~0.30 |

Scores vary based on model capability. Frontier models (GPT-4, Claude) are expected to score higher on easy/medium tasks but the hard task genuinely challenges even the best models due to the multi-stage crisis requiring precise action sequencing.
