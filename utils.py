"""
Utility module for orbital path calculations in Orbit Wars.

Provides compute_attack_angle — launch angle computation with
configurable parameters, comet support, and safety checks.
"""

import math

CENTER = 50.0
ROTATION_RADIUS_LIMIT = 50.0
_LOG1K = math.log(1000.0)

DEFAULT_CONFIG = {
    "shipSpeed": 6.0,
    "sunRadius": 10.0,
    "boardSize": 100.0,
    "cometSpeed": 4.0,
}


def get_config(config=None):
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    return cfg


def compute_fleet_speed(ships, config=None):
    cfg = get_config(config)
    if ships <= 0:
        return 1.0
    return 1.0 + (cfg["shipSpeed"] - 1.0) * (
        math.log(ships) / _LOG1K
    ) ** 1.5


def is_orbiting(planet):
    """True if the planet rotates around the sun.
    
    Planets orbit when orbital radius + radius < 50.
    Comets never orbit regardless of position.
    """
    if planet.get("comet", False):
        return False
    dx = planet["x"] - CENTER
    dy = planet["y"] - CENTER
    return math.hypot(dx, dy) + planet.get("radius", 0) < ROTATION_RADIUS_LIMIT


def point_segment_dist(px, py, ax, ay, bx, by):
    """Minimum Euclidean distance from point P to line segment AB."""
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return math.hypot(px - proj_x, py - proj_y)


def segment_hits_sun(ax, ay, bx, by, config=None):
    cfg = get_config(config)
    center = cfg["boardSize"] / 2.0
    return point_segment_dist(center, center, ax, ay, bx, by) < cfg["sunRadius"]


def predict_planet_pos(planet, turns_ahead, angular_velocity, config=None, comet_paths=None):
    """Predict (x, y) of a planet `turns_ahead` from now.

    Orbiting planets: circular motion around the sun from their
    current position. Comets: looked up from observed path data.
    Static planets: current position unchanged.
    Returns None if a comet will have left the board by then.
    """
    # Comet path lookup
    if planet.get("comet", False) and comet_paths is not None:
        pp = comet_paths.get(planet["id"])
        if pp is not None:
            path, idx = pp
            j = idx + int(round(turns_ahead))
            if 0 <= j < len(path):
                return float(path[j][0]), float(path[j][1])
            return None

    # Static planet — position never changes
    if not is_orbiting(planet):
        return planet["x"], planet["y"]

    # Orbiting planet — circular motion from current angle
    dx = planet["x"] - CENTER
    dy = planet["y"] - CENTER
    r = math.hypot(dx, dy)
    a = math.atan2(dy, dx) + angular_velocity * turns_ahead
    return CENTER + r * math.cos(a), CENTER + r * math.sin(a)


def _solve_intercept(source, target, ships, angular_velocity, config=None, comet_paths=None):
    """Find aim point such that fleet and target arrive together.

    Scans flight time for the first feasible arrival, then bisects
    for sub-turn precision on orbiting targets. Replaces the old
    fixed-point iteration which failed when target tangential speed
    rivaled fleet speed.

    Returns (tx, ty, arrival_turns) or None if unreachable.
    """
    cfg = get_config(config)
    spd = compute_fleet_speed(ships, cfg)
    sx, sy = source["x"], source["y"]
    off = source.get("radius", 0) + 0.1

    moving = target.get("comet", False) or is_orbiting(target)

    # Static target — direct line, no intercept needed
    if not moving:
        d = math.hypot(target["x"] - sx, target["y"] - sy)
        arrival = max(0.0, (d - off) / spd)
        return target["x"], target["y"], arrival

    # Fleet's straight-line reach after t turns (spawn offset + speed * time)
    def reach(t):
        return off + spd * t

    for ti in range(1, 301):
        p = predict_planet_pos(target, ti, angular_velocity, cfg, comet_paths)
        if p is None:
            return None  # comet left the board
        if reach(ti) >= math.hypot(p[0] - sx, p[1] - sy):
            if target.get("comet", False):
                return p[0], p[1], float(ti)
            # Bisect within the winning turn for sub-turn precision
            lo, hi = float(ti - 1), float(ti)
            for _ in range(20):
                mid = (lo + hi) / 2.0
                pm = predict_planet_pos(target, mid, angular_velocity, cfg, comet_paths)
                if pm is None:
                    return None
                if reach(mid) >= math.hypot(pm[0] - sx, pm[1] - sy):
                    hi = mid
                else:
                    lo = mid
            ph = predict_planet_pos(target, hi, angular_velocity, cfg, comet_paths)
            if ph is None:
                return None
            return ph[0], ph[1], hi
    return None


def fleet_hits_obstacles(source, target_id, tx, ty, angle, speed,
                         planets, angular_velocity, config=None, comet_paths=None):
    """True if the fleet collides with any non-target planet en route.

    Static planets: exact segment-to-point distance check.
    Moving planets/comets: sampled every 0.5 turns against predicted
    positions along the flight path.
    """
    cfg = get_config(config)
    sx = source["x"] + math.cos(angle) * (source.get("radius", 0) + 0.1)
    sy = source["y"] + math.sin(angle) * (source.get("radius", 0) + 0.1)

    dist = math.hypot(tx - sx, ty - sy)
    total_ticks = max(1, math.ceil(dist / speed))
    margin = 0.5

    static = []
    moving = []
    for p in planets:
        pid = p["id"]
        if pid == source["id"] or pid == target_id:
            continue
        if p.get("comet", False) or is_orbiting(p):
            moving.append(p)
        else:
            static.append(p)

    for p in static:
        if point_segment_dist(p["x"], p["y"], sx, sy, tx, ty) < p.get("radius", 0) + margin:
            return True

    if moving:
        step = 0.5
        t = step
        while t <= total_ticks + step:
            fx = sx + math.cos(angle) * speed * t
            fy = sy + math.sin(angle) * speed * t
            for p in moving:
                pos = predict_planet_pos(p, t, angular_velocity, cfg, comet_paths)
                if pos is None:
                    continue
                if math.hypot(fx - pos[0], fy - pos[1]) < p.get("radius", 0) + margin:
                    return True
            t += step

    return False


def compute_attack_angle(source, target, ships, planets,
                         angular_velocity, step=0, config=None, comet_paths=None):
    """Compute the launch angle to send ships from source to target.

    Accounts for:
      * Orbiting target motion (time-scan + bisect intercept).
      * Comet paths (looked up from observed trajectory).
      * Sun collision (fleet destroyed if path passes within sun radius).
      * Obstacle planets (any other planet in the flight path).

    Parameters
    ----------
    source : dict
        Source planet with keys id, x, y, radius.
    target : dict
        Target planet with keys id, x, y, radius, comet.
    ships : int
        Number of ships to send.
    planets : list of dict
        All current planets.
    angular_velocity : float
        Radians per turn from obs.
    step : int
        Current game step (kept for backward compatibility).
    config : dict or None
        Override defaults: shipSpeed, sunRadius, boardSize, cometSpeed.
    comet_paths : dict or None
        {planet_id: (path_list, path_index)} from obs.

    Returns
    -------
    float or None
        Launch angle in radians (0 = right, pi/2 = down), or
        None if the target cannot be reached.
    """
    if ships <= 0:
        return None

    cfg = get_config(config)
    spd = compute_fleet_speed(ships, cfg)

    sol = _solve_intercept(source, target, ships, angular_velocity, cfg, comet_paths)
    if sol is None:
        return None
    tx, ty, arrival_t = sol

    angle = math.atan2(ty - source["y"], tx - source["x"])

    # Fleet spawns just outside the source planet surface
    sx = source["x"] + math.cos(angle) * (source.get("radius", 0) + 0.1)
    sy = source["y"] + math.sin(angle) * (source.get("radius", 0) + 0.1)

    if segment_hits_sun(sx, sy, tx, ty, cfg):
        return None

    if fleet_hits_obstacles(source, target["id"], tx, ty, angle, spd,
                            planets, angular_velocity, cfg, comet_paths):
        return None

    return angle
