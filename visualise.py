from kaggle_environments import make
import json, sys

from utils import DEFAULT_CONFIG

agents = sys.argv[1:3] if len(sys.argv) >= 3 else ["main.py", "random"]
seed = int(sys.argv[3]) if len(sys.argv) >= 4 else 42
output = sys.argv[4] if len(sys.argv) >= 5 else "replay.json"

config = {
    "seed": seed,
    "agents": agents,
    **DEFAULT_CONFIG,
}
print("Env config:")
print(json.dumps(config, indent=2))

env = make("orbit_wars", configuration={"seed": seed}, debug=True)
env.run(agents)

replay = {"config": config, "steps": env.steps}
with open(output, "w") as f:
    json.dump(replay, f)
print(f"Replay saved to {output}")
