from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.http_server import create_app

from models import DeFiRiskAction, DeFiRiskObservation
from server.environment import DeFiRiskEnvironment

app = create_app(
    DeFiRiskEnvironment,
    DeFiRiskAction,
    DeFiRiskObservation,
    env_name="defi_risk",
)


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
