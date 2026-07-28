import numpy as np


def foot_rectangle(center, length=0.22, width=0.12):
    x, y = center
    hx = length / 2.0
    hy = width / 2.0
    return np.array([
        [x - hx, y - hy],
        [x + hx, y - hy],
        [x + hx, y + hy],
        [x - hx, y + hy],
    ], dtype=float)


def convex_hull(points):
    pts = sorted(map(tuple, np.asarray(points, dtype=float)))
    if len(pts) <= 1:
        return np.asarray(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def support_polygon(
    footsteps,
    t,
    foot_length=0.22,
    foot_width=0.12,
    left_offset=(0.0, 0.0),
    right_offset=(0.0, 0.0),
):
    phase = footsteps.get_phase_type(t)
    left = np.asarray(footsteps.get_left_position(t), dtype=float) + left_offset
    right = np.asarray(footsteps.get_right_position(t), dtype=float) + right_offset

    if phase == "left":
        return foot_rectangle(right, foot_length, foot_width)
    if phase == "right":
        return foot_rectangle(left, foot_length, foot_width)

    return convex_hull(np.vstack([
        foot_rectangle(left, foot_length, foot_width),
        foot_rectangle(right, foot_length, foot_width),
    ]))


def point_segment_distance(point, a, b):
    point = np.asarray(point, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ab = b - a
    scale = np.dot(point - a, ab) / (np.dot(ab, ab) + 1e-12)
    closest = a + np.clip(scale, 0.0, 1.0) * ab
    return float(np.linalg.norm(point - closest))


def point_in_polygon(point, polygon, eps=1e-9):
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    if len(polygon) < 3:
        return False

    signs = []
    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]
        edge = b - a
        signs.append(edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0]))
    signs = np.asarray(signs)
    return bool(np.all(signs >= -eps) or np.all(signs <= eps))


def polygon_margin(point, polygon):
    polygon = np.asarray(polygon, dtype=float)
    distances = [
        point_segment_distance(point, polygon[i], polygon[(i + 1) % len(polygon)])
        for i in range(len(polygon))
    ]
    margin = min(distances) if distances else -np.inf
    return float(margin if point_in_polygon(point, polygon) else -margin)


def move_inside_polygon(point, polygon, safety_margin=0.0, max_iter=30):
    point = np.asarray(point, dtype=float)
    polygon = np.asarray(polygon, dtype=float)
    center = polygon.mean(axis=0)
    candidate = point.copy()

    for _ in range(max_iter):
        if polygon_margin(candidate, polygon) >= safety_margin:
            return candidate
        candidate = center + 0.75 * (candidate - center)

    return center
