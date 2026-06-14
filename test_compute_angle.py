"""
Test script for compute_attack_angle.

Creates mock scenarios and verifies the function returns correct
angles or -1.0 for blocked launches.
"""

import math
from utils import compute_attack_angle

W = 0.025  # angular velocity


def planet(id, x, y, radius=5, ships=100, owner=0, production=3, comet=False):
    return dict(
        id=id, x=x, y=y, radius=radius,
        ships=ships, owner=owner, prod=production, comet=comet,
    )


def run(name, source, target, ships, planets, comet_paths=None):
    angle = compute_attack_angle(
        source, target, ships, planets, W,
        config=None, comet_paths=comet_paths,
    )
    if angle < 0:
        print(f"  {name}: BLOCKED (-1.0)")
    else:
        print(f"  {name}: angle={angle:.4f} rad ({math.degrees(angle):.1f} deg), "
              f"fleet speed={ships:.0f} ships -> {compute_fleet_speed_from_angle(ships):.2f} u/turn")


def compute_fleet_speed_from_angle(ships):
    from utils import compute_fleet_speed
    return compute_fleet_speed(ships)


# ---------------------------------------------------------------------------
# Scenario 1: Static source -> static target, clear path
# Source at (90, 20), target at (90, 80). Line x=90, y in [20,80].
# Distance from sun center (50,50) = 40, well outside sun radius.
p0 = planet(0, 90, 20, radius=3)
p1 = planet(1, 90, 80, radius=3)
print("1. Static -> Static (clear shot):")
run("direct", p0, p1, 10, [p0, p1])

# ---------------------------------------------------------------------------
# Scenario 2: Static source -> orbiting target
# Target at (30, 50), radius 3, orbital_r=20 -> orbits.
# Source far right at (90, 20), static.
p2 = planet(2, 30, 50, radius=3)
print("\n2. Static -> Orbiting (intercept lead):")
run("lead", p0, p2, 10, [p0, p2])

# ---------------------------------------------------------------------------
# Scenario 3: Comet target
# Comet moves bottom-left to top-right. Fleet from (90, 20).
p3 = planet(0, 90, 20, radius=3)
comet = planet(3, 20, 10, radius=2, owner=-1, comet=True)
path = [(20, 10), (30, 12), (40, 14), (50, 16), (60, 18), (70, 20),
        (80, 22), (90, 24), (100, 26)]
comet_paths = {3: (path, 0)}
print("\n3. Static -> Comet (path lookup):")
run("comet", p3, comet, 10, [p3, comet], comet_paths)

# ---------------------------------------------------------------------------
# Scenario 4: Sun blocks the shot
# Source left at (10, 50), target right at (90, 50).
# Large radii (15) keep them static (orbital_r + radius >= 50).
# Line from (10,50) to (90,50) passes through (50,50) = sun center.
p4 = planet(4, 10, 50, radius=15)
p5 = planet(5, 90, 50, radius=15)
print("\n4. Sun blocked (crosses center):")
run("sun-block", p4, p5, 10, [p4, p5])

# ---------------------------------------------------------------------------
# Scenario 5: Obstacle planet blocks the shot
# Source (90, 20) -> target (90, 80), with a static planet on the line.
# All have large radii (>=15) so they're static.
p6 = planet(6, 90, 20, radius=15)
p7 = planet(7, 90, 80, radius=15)
blocker = planet(8, 90, 50, radius=15, owner=-1)
print("\n5. Obstacle blocked (planet in flight path):")
run("blocked", p6, p7, 10, [p6, p7, blocker])
