"""
Enhanced pathfinding with multiple algorithms for robot car navigation.
Now includes a shortest-time Hamiltonian (TSP) solver using Dubins time as edge cost.
Width-aware: accounts for car width by inflating obstacles/walls and using an effective turning radius.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from itertools import permutations, combinations
from dataclasses import dataclass

from lib.car import CarState
from lib.path import Obstacle, DubinsPath, DubinsPathType
from config import config


@dataclass
class PathfindingResult:
    """Result of pathfinding with debug info."""
    paths: List[DubinsPath]
    total_length: float
    algorithm_used: str
    debug_info: Dict


# ------------ Dubins planner (with effective radius) -----------------
class DubinsPlanner:
    """Plans Dubins paths for car-like robot motion."""

    def __init__(self, turning_radius: float = None):
        base_r = turning_radius or config.car.turning_radius
        half_w = 0.5 * getattr(config.car, "width", 0.0)
        # Plan with a larger radius so the outer edge of the car clears during arcs
        self.turning_radius = base_r + half_w
        

    def plan_path(self, start: CarState, goal: CarState) -> Optional[DubinsPath]:
        """Plan shortest Dubins path between two car states. Returns None if no valid path exists."""
        best_path = None
        min_length = float('inf')
        for path_type in DubinsPathType:
            try:
                path = self._compute_path(start, goal, path_type)
                if path and path.length > 0 and path.length < min_length:
                    min_length = path.length
                    best_path = path
            except Exception:
                continue
        return best_path

    def _compute_path(self, start: CarState, goal: CarState, path_type: DubinsPathType) -> Optional[DubinsPath]:
        r = self.turning_radius
        if path_type == DubinsPathType.RSR:
            return self._rsr_path(start, goal, r)
        elif path_type == DubinsPathType.RSL:
            return self._rsl_path(start, goal, r)
        elif path_type == DubinsPathType.LSR:
            return self._lsr_path(start, goal, r)
        elif path_type == DubinsPathType.LSL:
            return self._lsl_path(start, goal, r)
        elif path_type == DubinsPathType.RLR:
            return self._rlr_path(start, goal, r)
        else:
            return self._lrl_path(start, goal, r)

    # ---- individual families (same as your working version) ----
    def _rsr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = -dy / d, dx / d  # external tangent
        t1x, t1y = c1x + r * ux, c1y + r * uy
        t2x, t2y = c2x + r * ux, c2y + r * uy
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        length = abs(alpha * r) + d + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RSR, length, start, goal, waypoints, turn_points)

    def _rsl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 2 * r:
            return None
        try:
            phi = math.acos(2 * r / d)
        except ValueError:
            return None
        theta_t = math.atan2(dy, dx) + phi
        tx, ty = math.cos(theta_t), math.sin(theta_t)
        t1x, t1y = c1x + r * tx, c1y + r * ty
        t2x, t2y = c2x - r * tx, c2y - r * ty
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        straight = math.hypot(t2x - t1x, t2y - t1y)
        length = abs(alpha * r) + straight + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RSL, length, start, goal, waypoints, turn_points)

    def _lsr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 2 * r:
            return None
        try:
            phi = math.acos(2 * r / d)
        except ValueError:
            return None
        theta_t = math.atan2(dy, dx) - phi
        tx, ty = math.cos(theta_t), math.sin(theta_t)
        t1x, t1y = c1x + r * tx, c1y + r * ty
        t2x, t2y = c2x - r * tx, c2y - r * ty
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        straight = math.hypot(t2x - t1x, t2y - t1y)
        length = abs(alpha * r) + straight + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LSR, length, start, goal, waypoints, turn_points)

    def _lsl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d < 1e-3:
            return None
        ux, uy = dy / d, -dx / d  # external tangent (opposite sign of RSR)
        t1x, t1y = c1x + r * ux, c1y + r * uy
        t2x, t2y = c2x + r * ux, c2y + r * uy
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        length = abs(alpha * r) + d + abs(beta * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LSL, length, start, goal, waypoints, turn_points)

    def _rlr_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x + r * math.sin(start.theta)
        c1y = start.y - r * math.cos(start.theta)
        c2x = goal.x + r * math.sin(goal.theta)
        c2y = goal.y - r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d > 4 * r or d < 1e-3:
            return None
        mx, my = (c1x + c2x) / 2, (c1y + c2y) / 2
        h_sq = 4 * r * r - d * d / 4
        if h_sq < 0:
            return None
        h = math.sqrt(h_sq)
        if d <= 0:
            return None
        px, py = -dy / d, dx / d
        c3x, c3y = mx + h * px, my + h * py
        t1x, t1y = (c1x + c3x) / 2, (c1y + c3y) / 2
        t2x, t2y = (c2x + c3x) / 2, (c2y + c3y) / 2
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, True)
        beta = self._arc_angle(t1x, t1y, c3x, c3y, t2x, t2y, False)
        gamma = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, True)
        length = abs(alpha * r) + abs(beta * r) + abs(gamma * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.RLR, length, start, goal, waypoints, turn_points)

    def _lrl_path(self, start: CarState, goal: CarState, r: float) -> Optional[DubinsPath]:
        c1x = start.x - r * math.sin(start.theta)
        c1y = start.y + r * math.cos(start.theta)
        c2x = goal.x - r * math.sin(goal.theta)
        c2y = goal.y + r * math.cos(goal.theta)
        dx, dy = c2x - c1x, c2y - c1y
        d = math.hypot(dx, dy)
        if d > 4 * r or d < 1e-3:
            return None
        mx, my = (c1x + c2x) / 2, (c1y + c2y) / 2
        h_sq = 4 * r * r - d * d / 4
        if h_sq < 0:
            return None
        h = math.sqrt(h_sq)
        if d <= 0:
            return None
        px, py = dy / d, -dx / d
        c3x, c3y = mx + h * px, my + h * py
        t1x, t1y = (c1x + c3x) / 2, (c1y + c3y) / 2
        t2x, t2y = (c2x + c3x) / 2, (c2y + c3y) / 2
        alpha = self._arc_angle(start.x, start.y, c1x, c1y, t1x, t1y, False)
        beta = self._arc_angle(t1x, t1y, c3x, c3y, t2x, t2y, True)
        gamma = self._arc_angle(t2x, t2y, c2x, c2y, goal.x, goal.y, False)
        length = abs(alpha * r) + abs(beta * r) + abs(gamma * r)
        waypoints = [(start.x, start.y), (t1x, t1y), (t2x, t2y), (goal.x, goal.y)]
        turn_points = [(t1x, t1y), (t2x, t2y)]
        return DubinsPath(DubinsPathType.LRL, length, start, goal, waypoints, turn_points)

    def _arc_angle(self, px: float, py: float, cx: float, cy: float,
                qx: float, qy: float, clockwise: bool) -> float:
        """
        Return the signed sweep to go from point P on the circle (center C)
        to point Q on the same circle, following the requested direction.
        - CCW  : +angle in [0, 2π)
        - CW   : -angle in (−2π, 0]
        This prevents “extra revolutions” (e.g., −2π − ε) that make the car spin.
        """
        a1 = math.atan2(py - cy, px - cx)   # angle of P around C
        a2 = math.atan2(qy - cy, qx - cx)   # angle of Q around C
        twopi = 2.0 * math.pi

        if clockwise:
            # going CW means decreasing angle; take shortest CW sweep
            sweep = (a1 - a2) % twopi       # in [0, 2π)
            # return negative (CW) sweep; map 0 to 0, others to (-2π, 0]
            return -sweep if sweep != 0.0 else 0.0
        else:
            # CCW: increasing angle; take shortest CCW sweep
            sweep = (a2 - a1) % twopi       # in [0, 2π)
            return sweep if sweep != 0.0 else 0.0



# ------------------- High-level planner with TSP -------------------
class CarPathPlanner:
    def __init__(self):
        self.dubins_planner = DubinsPlanner()
        self.arena_size = config.arena.size
        self.grid_size = config.get_grid_size()
        self.obstacles: List[Obstacle] = []
        
        # Collision grid (1 = blocked, 0 = free)
        self.collision_grid = np.zeros((self.grid_size, self.grid_size), dtype=int)

        # FIXED: More reasonable clearance calculation
        safety = getattr(config.car, "safety_margin_cm", 2.0)
        self.clearance_cm = 0.5 * getattr(config.car, "width", 30.0) + safety
        # For collision detection, use a smaller buffer to avoid over-conservative paths
        self.collision_clearance_cells = max(1, int(math.ceil((self.clearance_cm * 0.7) / config.arena.grid_cell_size)))
        
        print(f"Car clearance: {self.clearance_cm}cm")
        print(f"Collision clearance: {self.collision_clearance_cells} cells")

    # ---- world / obstacle management ----
    def add_obstacle(self, x: int, y: int, image_side: str):
        obstacle = Obstacle(x, y, image_side)
        self.obstacles.append(obstacle)
        self._update_collision_grid()
        print(f"Added obstacle {len(self.obstacles)-1} at ({x}, {y}) with image on {image_side} side")


    def _update_collision_grid(self):
        """FIXED: More reasonable collision grid with proper buffers."""
        self.collision_grid.fill(0)
        cell = config.arena.grid_cell_size

        # Use smaller, more realistic buffers
        obstacle_buffer_cm = 5.0  # Reduced from 8cm
        obstacle_buffer_cells = max(1, int(math.ceil(obstacle_buffer_cm / cell)))
        
        # Border buffer - just enough to keep car body clear
        border_buffer_cm = self.clearance_cm * 0.8  # Reduced multiplier
        border_buffer_cells = max(1, int(math.ceil(border_buffer_cm / cell)))

        print(f"Using obstacle buffer: {obstacle_buffer_cm}cm ({obstacle_buffer_cells} cells)")
        print(f"Using border buffer: {border_buffer_cm}cm ({border_buffer_cells} cells)")

        # Inflate obstacles
        for i, obs in enumerate(self.obstacles):
            obs_min_gx = int(obs.x // cell)
            obs_max_gx = int((obs.x + config.arena.obstacle_size) // cell)
            obs_min_gy = int(obs.y // cell)  
            obs_max_gy = int((obs.y + config.arena.obstacle_size) // cell)
            
            min_gx = max(0, obs_min_gx - obstacle_buffer_cells)
            max_gx = min(self.grid_size, obs_max_gx + obstacle_buffer_cells + 1)
            min_gy = max(0, obs_min_gy - obstacle_buffer_cells)
            max_gy = min(self.grid_size, obs_max_gy + obstacle_buffer_cells + 1)
            
            for gy in range(min_gy, max_gy):
                for gx in range(min_gx, max_gx):
                    if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                        self.collision_grid[gy, gx] = 1

        # Inflate arena borders (more conservative at edges)
        if border_buffer_cells < self.grid_size // 2:
            self.collision_grid[0:border_buffer_cells, :] = 1
            self.collision_grid[self.grid_size - border_buffer_cells:, :] = 1
            self.collision_grid[:, 0:border_buffer_cells] = 1
            self.collision_grid[:, self.grid_size - border_buffer_cells:] = 1

        blocked_count = np.sum(self.collision_grid)
        total_cells = self.grid_size * self.grid_size
        blocked_percentage = 100 * blocked_count / total_cells
        print(f"Collision grid: {blocked_count}/{total_cells} blocked ({blocked_percentage:.1f}%)")

    def get_image_target_position(self, obstacle: Obstacle) -> CarState:
        """Pose where the robot should be to scan the image."""
        d = config.car.camera_distance * 0.8
        size = config.arena.obstacle_size
        if obstacle.image_side == 'S':
            x = obstacle.x + size / 2
            y = obstacle.y - d
            theta = math.pi / 2
        elif obstacle.image_side == 'N':
            x = obstacle.x + size / 2
            y = obstacle.y + size + d
            theta = 3 * math.pi / 2
        elif obstacle.image_side == 'E':
            x = obstacle.x + size + d
            y = obstacle.y + size / 2
            theta = math.pi
        else:  # 'W'
            x = obstacle.x - d
            y = obstacle.y + size / 2
            theta = 0.0
        return CarState(x, y, theta)

    # ------------------ Public planning API ------------------
    def plan_visiting_path(self, start_state: CarState, obstacle_indices: List[int]) -> List[DubinsPath]:
        """Primary method used by the rest of the system."""
        print(f"Planning path from ({start_state.x:.1f}, {start_state.y:.1f}) to visit obstacles: {obstacle_indices}")
        print(f"Available obstacles: {len(self.obstacles)}")

        if not obstacle_indices:
            print("No obstacles to visit")
            return []

        # 1) Try optimal Held–Karp (shortest-time Hamiltonian)
        hk_paths = self.plan_shortest_time_hamiltonian(start_state, obstacle_indices)
        if hk_paths:
            total = sum(p.length for p in hk_paths)
            print(f"Success with shortest_time_hamiltonian: {len(hk_paths)} segments, {total:.1f}cm")
            return hk_paths

        # 2) Fall back to previous strategies if HK fails for any reason
        for algo in (self._greedy_nearest_neighbor, self._exhaustive_search, self._fallback_simple_path):
            try:
                result = algo(start_state, obstacle_indices)
                if result.paths:
                    print(f"Success with {result.algorithm_used}: {len(result.paths)} segments, {result.total_length:.1f}cm")
                    return result.paths
                else:
                    print(f"Failed with {result.algorithm_used}: {result.debug_info}")
            except Exception as e:
                print(f"Error with algorithm: {e}")

        print("All pathfinding algorithms failed")
        return []

    # ------------------ B.3: Held–Karp TSP on time ------------------
    def plan_shortest_time_hamiltonian(self, start_state: CarState, obstacle_indices: List[int]) -> List[DubinsPath]:
        """Compute the true shortest-time visiting order using Held–Karp DP."""
        # Build target states
        targets: List[Tuple[int, CarState]] = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))

        n = len(targets)
        if n == 0:
            return []

        # Precompute costs and keep the actual best Dubins path for each directed edge
        INF = 1e12
        edge_time_start = [INF] * n
        edge_path_start: List[Optional[DubinsPath]] = [None] * n

        edge_time = [[INF] * n for _ in range(n)]
        edge_path: List[List[Optional[DubinsPath]]] = [[None] * n for _ in range(n)]

        # From start -> each target
        for j in range(n):
            dest = targets[j][1]
            path = self.dubins_planner.plan_path(start_state, dest)
            if path and not self._path_intersects_obstacles_strict(path):
                edge_time_start[j] = self._estimate_time_for_path(path)
                edge_path_start[j] = path

        # Between targets
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a = targets[i][1]
                b = targets[j][1]
                path = self.dubins_planner.plan_path(a, b)
                if path and not self._path_intersects_obstacles_strict(path):
                    edge_time[i][j] = self._estimate_time_for_path(path)
                    edge_path[i][j] = path

        # If any target unreachable from start, abort
        if any(t >= INF for t in edge_time_start):
            return []

        # Held–Karp DP: dp[(mask, j)] = (time, prev_j)
        # mask over n targets (0..n-1). Ending at j.
        dp: Dict[Tuple[int, int], Tuple[float, Optional[int]]] = {}

        # Base cases: start -> j
        for j in range(n):
            dp[(1 << j, j)] = (edge_time_start[j], None)

        # Iterate over subset sizes
        for size in range(2, n + 1):
            for subset in combinations(range(n), size):
                mask = 0
                for k in subset:
                    mask |= (1 << k)
                for j in subset:
                    best = (INF, None)
                    prev_mask = mask ^ (1 << j)
                    for i in subset:
                        if i == j:
                            continue
                        if (prev_mask, i) in dp and edge_time[i][j] < INF:
                            cand = dp[(prev_mask, i)][0] + edge_time[i][j]
                            if cand < best[0]:
                                best = (cand, i)
                    if best[0] < INF:
                        dp[(mask, j)] = best

        # Pick best final end node (no return to start needed)
        full_mask = (1 << n) - 1
        best_final = (INF, None)
        best_end = None
        for j in range(n):
            if (full_mask, j) in dp and dp[(full_mask, j)][0] < best_final[0]:
                best_final = dp[(full_mask, j)]
                best_end = j

        if best_end is None:
            return []

        # Reconstruct order
        order: List[int] = []
        mask = full_mask
        j = best_end
        while j is not None:
            order.append(j)
            time_val, prev = dp[(mask, j)]
            if prev is None:
                break
            mask ^= (1 << j)
            j = prev
        order.reverse()

        # Build actual path sequence: start -> first -> ... -> last
        segments: List[DubinsPath] = []
        # Start to first
        first = order[0]
        segments.append(edge_path_start[first])
        # Between targets
        for a, b in zip(order[:-1], order[1:]):
            segments.append(edge_path[a][b])

        return segments

    # ---- helpers: time estimate & collision checks ----
    def _estimate_time_for_path(self, path: DubinsPath) -> float:
        """
        Convert a Dubins path into time using straight vs arc speeds:
          t = L_straight / v_lin + L_arc / (r * omega)
        """
        v_lin = max(1e-6, config.car.linear_speed_cm_s)
        omega = max(1e-6, config.car.angular_speed_rad_s)
        # Use effective radius (includes half-width)
        r = max(1e-6, self.dubins_planner.turning_radius)

        # Straight length: only if the middle segment is 'S'
        straight_len = 0.0
        if 's' in path.path_type.value:
            # waypoints[1] and [2] are tangent points (t1, t2) across the straight
            if len(path.waypoints) >= 3:
                x1, y1 = path.waypoints[1]
                x2, y2 = path.waypoints[2]
                straight_len = math.hypot(x2 - x1, y2 - y1)

        arc_len = max(0.0, path.length - straight_len)
        t_straight = straight_len / v_lin
        t_arc = arc_len / (r * omega)

        # Optional constant per-stop recognition time — same for every target, so it
        # doesn't affect ordering; keep 0 to avoid bias.
        return t_straight + t_arc

    def _path_intersects_obstacles(self, path: DubinsPath) -> bool:
        return self._path_intersects_obstacles_strict(path, buffer_reduction=0)

    def _path_intersects_obstacles_strict(self, path: DubinsPath, buffer_reduction: float = 0) -> bool:
        """FIXED: Proper circular collision detection instead of square."""
        cell = config.arena.grid_cell_size
        check_radius_cells = max(1, int(self.collision_clearance_cells * (1 - buffer_reduction)))
        
        def is_collision_at_point(x: float, y: float) -> bool:
            gx = int(x / cell)
            gy = int(y / cell)
            
            # Check arena bounds
            if (gx < check_radius_cells or gx >= self.grid_size - check_radius_cells or
                gy < check_radius_cells or gy >= self.grid_size - check_radius_cells):
                return True
            
            # FIXED: Check circular region instead of square
            for dx in range(-check_radius_cells, check_radius_cells + 1):
                for dy in range(-check_radius_cells, check_radius_cells + 1):
                    # Only check points within circular distance
                    if dx*dx + dy*dy <= check_radius_cells * check_radius_cells:
                        check_gx = gx + dx
                        check_gy = gy + dy
                        if (0 <= check_gx < self.grid_size and 0 <= check_gy < self.grid_size):
                            if self.collision_grid[check_gy, check_gx] == 1:
                                return True
            return False

        def sample_line(p1, p2, samples=8):
            """Sample points along a line segment."""
            points = []
            for i in range(samples + 1):
                t = i / samples
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                points.append((x, y))
            return points

        def sample_arc(center, start_point, end_point, turn_direction, samples=12):
            """Sample points along an arc."""
            cx, cy = center
            r = self.dubins_planner.turning_radius
            
            a0 = math.atan2(start_point[1] - cy, start_point[0] - cx)
            a1 = math.atan2(end_point[1] - cy, end_point[0] - cx)
            
            if turn_direction == 'l':
                if a1 <= a0: a1 += 2 * math.pi
                angles = [a0 + (a1 - a0) * i / samples for i in range(samples + 1)]
            else:
                if a0 <= a1: a0 += 2 * math.pi
                angles = [a0 - (a0 - a1) * i / samples for i in range(samples + 1)]
                
            points = []
            for angle in angles:
                x = cx + r * math.cos(angle)
                y = cy + r * math.sin(angle)
                points.append((x, y))
            
            return points

        # Check path segments
        wps = path.waypoints
        if len(wps) < 4:
            return False
            
        ptype = (path.path_type.value if hasattr(path.path_type, "value") else str(path.path_type)).lower()
        
        all_sample_points = []
        
        # Segment 0
        if ptype[0] in ('l', 'r'):
            center = self._circle_center(wps[0], path.start_state.theta, ptype[0])
            points = sample_arc(center, wps[0], wps[1], ptype[0])
            all_sample_points.extend(points)
        else:
            points = sample_line(wps[0], wps[1])
            all_sample_points.extend(points)

        # Segment 1 (straight)
        points = sample_line(wps[1], wps[2])
        all_sample_points.extend(points)

        # Segment 2
        if ptype[2] in ('l', 'r'):
            center = self._circle_center(wps[3], path.end_state.theta, ptype[2])
            points = sample_arc(center, wps[2], wps[3], ptype[2])
            all_sample_points.extend(points)
        else:
            points = sample_line(wps[2], wps[3])
            all_sample_points.extend(points)

        # Check all sample points
        for x, y in all_sample_points:
            if is_collision_at_point(x, y):
                return True
                
        return False

    def _circle_center(self, point, heading, turn_char):
        """Helper to calculate circle center for turns."""
        r = self.dubins_planner.turning_radius
        x, y = point
        if turn_char.lower() == 'l':
            return (x - r * math.sin(heading), y + r * math.cos(heading))
        else:
            return (x + r * math.sin(heading), y - r * math.cos(heading))


    # -------- legacy strategies kept as fallbacks --------
    def _greedy_nearest_neighbor(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "greedy_nearest_neighbor", {"error": "no_valid_targets"})

        path_segments = []
        total_length = 0.0
        current_state = start_state
        remaining = targets.copy()

        while remaining:
            best = None
            best_seg = None
            best_len = float('inf')
            for cand_idx, cand_state in remaining:
                seg = self.dubins_planner.plan_path(current_state, cand_state)
                if seg and not self._path_intersects_obstacles(seg) and seg.length < best_len:
                    best_len = seg.length
                    best = (cand_idx, cand_state)
                    best_seg = seg
            if not best_seg:
                return PathfindingResult([], 0, "greedy_nearest_neighbor", {"error": "no_valid_path", "remaining": len(remaining)})
            path_segments.append(best_seg)
            total_length += best_seg.length
            current_state = best_seg.end_state
            remaining.remove(best)

        return PathfindingResult(path_segments, total_length, "greedy_nearest_neighbor", {"segments": len(path_segments)})

    def _exhaustive_search(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        if len(obstacle_indices) > 8:
            return PathfindingResult([], 0, "exhaustive_search", {"error": "too_many_targets"})
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "exhaustive_search", {"error": "no_valid_targets"})

        best_path = None
        min_len = float('inf')
        attempts = 0
        for perm in permutations(targets):
            attempts += 1
            path_segments = []
            total = 0.0
            cur = start_state
            valid = True
            for _, tgt in perm:
                seg = self.dubins_planner.plan_path(cur, tgt)
                if not seg or self._path_intersects_obstacles_strict(seg):
                    valid = False
                    break
                path_segments.append(seg)
                total += seg.length
                cur = tgt
            if valid and total < min_len:
                min_len = total
                best_path = path_segments

        if best_path:
            return PathfindingResult(best_path, min_len, "exhaustive_search", {"attempts": attempts, "segments": len(best_path)})
        return PathfindingResult([], 0, "exhaustive_search", {"error": "no_valid_permutation", "attempts": attempts})

    def _fallback_simple_path(self, start_state: CarState, obstacle_indices: List[int]) -> PathfindingResult:
        targets = []
        for idx in obstacle_indices:
            if 0 <= idx < len(self.obstacles):
                targets.append((idx, self.get_image_target_position(self.obstacles[idx])))
        if not targets:
            return PathfindingResult([], 0, "fallback_simple", {"error": "no_valid_targets"})

        path_segments = []
        total = 0.0
        cur = start_state
        for obs_idx, tgt in targets:
            seg = self.dubins_planner.plan_path(cur, tgt)
            if seg:
                path_segments.append(seg)
                total += seg.length
                cur = tgt
            else:
                print(f"Warning: Could not plan path to obstacle {obs_idx}")
        return PathfindingResult(path_segments, total, "fallback_simple", {"segments": len(path_segments), "warnings": True})
    
    # Add this debugging method to CarPathPlanner class in pathfinding.py

    def debug_collision_grid(self):
        """Print collision grid state for debugging."""
        print(f"Collision grid shape: {self.collision_grid.shape}")
        print(f"Grid cell size: {config.arena.grid_cell_size}cm")
        print(f"Clearance: {self.clearance_cm}cm ({self.clearance_cm} cells)")
        print(f"Arena size: {config.arena.size}cm")
        
        # Print grid (flip y for proper visualization)
        print("\nCollision Grid (1=blocked, 0=free):")
        for y in range(self.grid_size-1, -1, -1):  # top to bottom
            row = ""
            for x in range(self.grid_size):
                row += "█" if self.collision_grid[y, x] == 1 else "·"
            print(f"{y:2d}: {row}")
        
        print("    " + "".join([str(i%10) for i in range(self.grid_size)]))

    def debug_path_collision(self, path: DubinsPath):
        """Debug why a specific path might be colliding."""
        print(f"\nDebugging path collision for {path.path_type.value}")
        print(f"Path length: {path.length:.1f}cm")
        print(f"Waypoints: {path.waypoints}")
        
        cell = config.arena.grid_cell_size
        boundary_buf = max(1, int(self.clearance_cm))
        
        collision_points = []
        
        # Sample the path and check each point
        def check_point(x, y, label):
            gx = int(x / cell)
            gy = int(y / cell)
            print(f"  {label}: world({x:.1f}, {y:.1f}) -> grid({gx}, {gy})")
            
            if (gx < boundary_buf or gx >= self.grid_size - boundary_buf or
                gy < boundary_buf or gy >= self.grid_size - boundary_buf):
                print(f"    -> BOUNDARY COLLISION")
                collision_points.append((x, y, "boundary"))
                return True
                
            for dx in range(-boundary_buf, boundary_buf + 1):
                for dy in range(-boundary_buf, boundary_buf + 1):
                    cx = gx + dx
                    cy = gy + dy
                    if 0 <= cx < self.grid_size and 0 <= cy < self.grid_size:
                        if self.collision_grid[cy, cx] == 1:
                            print(f"    -> OBSTACLE COLLISION at grid({cx}, {cy})")
                            collision_points.append((x, y, "obstacle"))
                            return True
            print(f"    -> OK")
            return False
        
        # Check waypoints
        for i, (x, y) in enumerate(path.waypoints):
            check_point(x, y, f"waypoint_{i}")
        
        return collision_points


if __name__ == "__main__":
    # Simple smoke test
    planner = CarPathPlanner()
    planner.add_obstacle(50, 50, 'S')
    planner.add_obstacle(100, 100, 'E')
    planner.add_obstacle(150, 50, 'N')
    start = CarState(20, 20, 0)
    paths = planner.plan_visiting_path(start, [0, 1, 2])
    if paths:
        total = sum(p.length for p in paths)
        print(f"Final result: {len(paths)} segments, total length: {total:.1f}cm")
        for i, seg in enumerate(paths):
            print(f"  Segment {i}: {seg.path_type.value.upper()} ({seg.length:.1f}cm)")
    else:
        print("No valid path found")
