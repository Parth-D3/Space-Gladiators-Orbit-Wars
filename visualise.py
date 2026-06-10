from kaggle_environments import make
import json, sys

agents = sys.argv[1:3] if len(sys.argv) >= 3 else ["main.py", "random"]
seed = int(sys.argv[3]) if len(sys.argv) >= 4 else 42
output = sys.argv[4] if len(sys.argv) >= 5 else "replay.json"

env = make("orbit_wars", configuration={"seed": seed}, debug=True)
env.run(agents)
with open(output, "w") as f:
    json.dump(env.steps, f)
print(f"Replay saved to {output}")
