"""
Utility module for orbital path calculations in Orbit Wars.

Provides:
  compute_attack_angle — compute launch angle accounting for orbiting
    targets, sun/OOB/obstacle avoidance. Returns -1.0 if unreachable.
"""

import math

# Engine constants (matching kaggle_environments.envs.orbit_wars)
CENTER = 50.0
SUN_RADIUS = 10.0
BOARD_SIZE = 100.0
ROTATION_RADIUS_LIMIT = 50.0
DEFAULT_MAX_SPEED = 6.0
MAX_ITER = 6
CONVERGENCE_THRESHOLD = 1e-6


def compute_fleet_speed(ships, max_speed=DEFAULT_MAX_SPEED):
    """Fleet speed based on ship count, matching engine formula.

    speed = 1 + (maxSpeed - 1) * (log(ships) / log(1000)) ** 1.5

    Parameters
    ----------
    ships : int
        Number of ships in the fleet.
    max_speed : float
        Maximum fleet speed (default 6.0).

    Returns
    -------
    float
        Fleet speed in units/turn.
    """
    if ships <= 0:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def predict_planet_pos(planet_id, initial_planets, angular_velocity, at_step):
    """Predict the (x, y) position of a planet at a future step.

    For orbiting planets (orbital_r + radius < 50), computes circular
    motion from the initial position.  For static planets, returns
    the fixed initial position.

    Parameters
    ----------
    planet_id : int
        Planet ID to predict.
    initial_planets : list of [id, owner, x, y, radius, ships, production]
        Planet snapshots from turn 0 (obs.initial_planets).
    angular_velocity : float
        Radians per turn (obs.angular_velocity).
    at_step : float
        Absolute step number to predict at.

    Returns
    -------
    (float, float) or None
        Predicted (x, y), or None if planet_id is not in initial_planets.
    """
    initial = None
    for p in initial_planets:
        if p[0] == planet_id:
            initial = p
            break
    if initial is None:
        return None

    dx = initial[2] - CENTER
    dy = initial[3] - CENTER
    orbital_r = math.hypot(dx, dy)

    if orbital_r + initial[4] >= ROTATION_RADIUS_LIMIT:
        # Static planet — position never changes
        return (initial[2], initial[3])

    # Orbiting planet — circular motion around the sun
    init_angle = math.atan2(dy, dx)
    cur_angle = init_angle + angular_velocity * at_step

    return (
        CENTER + orbital_r * math.cos(cur_angle),
        CENTER + orbital_r * math.sin(cur_angle),
    )


def point_segment_dist(px, py, ax, ay, bx, by):
    """Minimum Euclidean distance from point P to line segment AB."""
    l2 = (ax - bx) ** 2 + (ay - by) ** 2
    if l2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
    proj_x = ax + t * (bx - ax)
    proj_y = ay + t * (by - ay)
    return math.hypot(px - proj_x, py - proj_y)


def segment_hits_sun(ax, ay, bx, by):
    """True if line segment AB passes within the sun radius of the centre."""
    return point_segment_dist(CENTER, CENTER, ax, ay, bx, by) < SUN_RADIUS


def segment_goes_oob(ax, ay, bx, by):
    """True if any part of line segment AB lies outside the [0,100] board.

    Both endpoints are assumed to be inside the board (planets are always
    in-bounds).  The function checks whether the line between them
    intersects any of the four boundary edges.
    """
    min_x = min(ax, bx)
    max_x = max(ax, bx)
    min_y = min(ay, by)
    max_y = max(ay, by)

    # Fast path — bounding box fully inside
    if (min_x >= 0 and max_x <= BOARD_SIZE
            and min_y >= 0 and max_y <= BOARD_SIZE):
        return False

    # Check intersection with each edge: x=0, x=100, y=0, y=100
    # Parametric form: (x,y) = (ax, ay) + t * (bx-ax, by-ay), t in [0,1]

    if min_x < 0 and abs(bx - ax) > 1e-12:
        t = -ax / (bx - ax)
        if 0.0 < t < 1.0:
            y_at = ay + t * (by - ay)
            if 0.0 <= y_at <= BOARD_SIZE:
                return True

    if max_x > BOARD_SIZE and abs(bx - ax) > 1e-12:
        t = (BOARD_SIZE - ax) / (bx - ax)
        if 0.0 < t < 1.0:
            y_at = ay + t * (by - ay)
            if 0.0 <= y_at <= BOARD_SIZE:
                return True

    if min_y < 0 and abs(by - ay) > 1e-12:
        t = -ay / (by - ay)
        if 0.0 < t < 1.0:
            x_at = ax + t * (bx - ax)
            if 0.0 <= x_at <= BOARD_SIZE:
                return True

    if max_y > BOARD_SIZE and abs(by - ay) > 1e-12:
        t = (BOARD_SIZE - ay) / (by - ay)
        if 0.0 < t < 1.0:
            x_at = ax + t * (bx - ax)
            if 0.0 <= x_at <= BOARD_SIZE:
                return True

    return False


def fleet_hits_obstacles(source, target_id, tx, ty, angle, speed, planets,
                         initial_planets, angular_velocity, step):
    """Check whether a fleet collides with any non-target planet en route.

    Discretises the flight into per-tick steps.  At each tick the fleet
    position is compared against every planet (orbiting positions are
    predicted forward).

    Parameters
    ----------
    source : Planet namedtuple
        Launching planet.
    target_id : int
        Destination planet id (exempt from obstacle check).
    tx, ty : float
        Intercept point (final target position at arrival).
    angle : float
        Launch angle in radians.
    speed : float
        Fleet speed (units/tick).
    planets : list of Planet namedtuple
        All planets at current step.
    initial_planets, angular_velocity, step : as in compute_attack_angle.

    Returns
    -------
    bool
        True if the fleet would hit an obstacle planet.
    """
    start_x = source.x + math.cos(angle) * (source.radius + 0.1)
    start_y = source.y + math.sin(angle) * (source.radius + 0.1)

    dist = math.hypot(tx - start_x, ty - start_y)
    total_ticks = max(1, math.ceil(dist / speed))

    for tick in range(1, total_ticks + 1):
        fx = start_x + math.cos(angle) * speed * tick
        fy = start_y + math.sin(angle) * speed * tick

        for p in planets:
            pid = p.id
            if pid == source.id or pid == target_id:
                continue

            # Predict planet position at this future step
            pos = predict_planet_pos(pid, initial_planets, angular_velocity,
                                     step + tick)
            if pos is None:
                px, py = p.x, p.y
            else:
                px, py = pos

            if math.hypot(fx - px, fy - py) < p.radius:
                return True

    return False


def compute_attack_angle(source, target, ships, planets,
                         initial_planets, angular_velocity, step=0):
    """Compute the launch angle to send ships from *source* to *target*.

    Accounts for:
      * Orbiting target motion (iterative fixed-point intercept).
      * Sun collision (segment passes within 10 units of centre).
      * Out-of-bounds (segment exits the 100×100 board).
      * Obstacle planets (any other planet intersects the path).

    Parameters
    ----------
    source : Planet
        Source planet (namedtuple with .id, .x, .y, .radius, …).
    target : Planet
        Target planet.
    ships : int
        Number of ships to send (determines fleet speed).
    planets : list of Planet
        All planets at the current step.
    initial_planets : list of list
        Turn-0 planet snapshot (obs.initial_planets).
    angular_velocity : float
        Planet rotation speed in rad/turn (obs.angular_velocity).
    step : int
        Current game step (default 0).

    Returns
    -------
    float
        Launch angle in radians (0 = right, π/2 = down), or
        -1.0 if the target cannot be reached.
    """
    if ships <= 0:
        return -1.0

    speed = compute_fleet_speed(ships)
    src_id = source.id
    tgt_id = target.id

    # -- Determine whether the target orbits ---------------------------------
    target_orbiting = False
    target_orbital_r = 0.0
    target_init_angle = 0.0
    for p in initial_planets:
        if p[0] == tgt_id:
            dx = p[2] - CENTER
            dy = p[3] - CENTER
            orbital_r = math.hypot(dx, dy)
            if orbital_r + p[4] < ROTATION_RADIUS_LIMIT:
                target_orbiting = True
                target_orbital_r = orbital_r
                target_init_angle = math.atan2(dy, dx)
            break

    # -- Intercept angle -----------------------------------------------------
    if target_orbiting:
        # Fixed-point iteration: aim, estimate flight time, re-aim at future
        # target position, repeat.
        tx, ty = target.x, target.y
        angle = math.atan2(ty - source.y, tx - source.x)

        for _ in range(MAX_ITER):
            dist = math.hypot(tx - source.x, ty - source.y)
            flight_time = dist / speed

            future_angle = (target_init_angle
                            + angular_velocity * (step + flight_time))
            pred_x = CENTER + target_orbital_r * math.cos(future_angle)
            pred_y = CENTER + target_orbital_r * math.sin(future_angle)

            new_angle = math.atan2(pred_y - source.y, pred_x - source.x)

            if abs(new_angle - angle) < CONVERGENCE_THRESHOLD:
                angle = new_angle
                tx, ty = pred_x, pred_y
                break

            angle = new_angle
            tx, ty = pred_x, pred_y
    else:
        angle = math.atan2(target.y - source.y, target.x - source.x)
        tx, ty = target.x, target.y

    # -- Fleet start position (just outside source planet) ------------------
    start_x = source.x + math.cos(angle) * (source.radius + 0.1)
    start_y = source.y + math.sin(angle) * (source.radius + 0.1)

    # -- Feasibility checks --------------------------------------------------

    # 1. Sun collision
    if segment_hits_sun(start_x, start_y, tx, ty):
        return -1.0

    # 2. Out of bounds
    if segment_goes_oob(start_x, start_y, tx, ty):
        return -1.0

    # 3. Obstacle planets
    if fleet_hits_obstacles(source, tgt_id, tx, ty, angle, speed,
                            planets, initial_planets, angular_velocity, step):
        return -1.0

    return angle
