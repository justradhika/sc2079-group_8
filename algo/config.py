from dataclasses import dataclass


@dataclass
class RobotConfig:
    width: float = 5  # cm
    length: float = 5  # cm
    turning_radius: float = 10.0  # minimum turning radius in cm
    camera_distance: float = 10.0  # optimal distance from obstacle for image recognition

    # Motion model (for time-cost)
    linear_speed_cm_s: float = 10.0      # straight-line speed (cm/s)
    angular_speed_rad_s: float = 1.2     # turning rate (rad/s); arc linear speed = r * omega
    reverse_linear_speed_cm_s: float = 8.0
    image_recognition_time_s: float = 0.0  # constant per target; doesn't change ordering, keep 0 unless needed

    # motion uncertainty (kept as-is)
    forward_motion_error: float = 0.0
    turn_angle_error: float = 0.0
    position_drift: float = 0.0


@dataclass
class Arena:
    size: int = 200  # arena size in cm (200x200)
    grid_cell_size: int = 5  # size of each grid cell in cm
    obstacle_size: int = 5  # physical obstacle size in cm
    collision_buffer: int = 30  # extra space around obstacles for collision avoidance


@dataclass
class PathfindingParams:
    waypoint_tolerance: float = 15.0  # cm - how close to waypoint before advancing
    angle_tolerance: float = 0.3      # radians - when car is considered aligned
    max_forward_step: float = 8.0     # cm - maximum forward movement action
    max_turn_step: float = 0.2        # radians how much is a maximum turn action


class Config:
    def __init__(self, config_file: str = 'car_config.json'):
        self.config_file = config_file
        self.car = RobotConfig()
        self.arena = Arena()
        self.pathfinding = PathfindingParams()

    def get_grid_size(self) -> int:
        return self.arena.size // self.arena.grid_cell_size


# Global configuration (import and use this)
config = Config()