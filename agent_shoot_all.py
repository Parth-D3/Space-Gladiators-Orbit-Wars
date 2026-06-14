"""
agent_shoot_all.py — fires 1 ship at every planet it can reach.

Demonstrates compute_attack_angle by attempting a shot from every
owned planet to every other planet with a 1-ship fleet.
"""
from utils import compute_attack_angle, is_orbiting


def parse(obs):
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        step = obs.get("step", 0)
        planets = []
        for p in obs.get("planets", []):
            planets.append({
                "id": int(p[0]), "owner": int(p[1]),
                "x": float(p[2]), "y": float(p[3]),
                "radius": float(p[4]), "ships": float(p[5]),
                "prod": float(p[6]), "comet": False,
            })
        w = float(obs.get("angular_velocity", 0.0) or 0.0)
    else:
        player = obs.player
        step = obs.step
        planets = []
        for p in obs.planets:
            planets.append({
                "id": int(p[0]), "owner": int(p[1]),
                "x": float(p[2]), "y": float(p[3]),
                "radius": float(p[4]), "ships": float(p[5]),
                "prod": float(p[6]), "comet": False,
            })
        w = float(obs.angular_velocity or 0.0)
    return player, step, planets, w


def agent(obs):
    player, step, planets, w = parse(obs)

    if step % 5 != 0:
        return []

    config = None
    moves = []

    my_planets = [p for p in planets if p["owner"] == player and p["ships"] >= 1]
    targets = [p for p in planets]

    for src in my_planets:
        for tgt in targets:
            if tgt["id"] == src["id"]:
                continue
            angle = compute_attack_angle(
                src, tgt, 1, planets, w,
                config=config, comet_paths=None,
            )
            if angle >= 0:
                moves.append([src["id"], float(angle), 1])

    return moves
