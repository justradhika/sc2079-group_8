"""
Flask server for robot car pathfinding API.
Provides HTTP endpoints for car control, path planning, and status monitoring.
"""

from flask import Flask, request, jsonify
import logging
from typing import Dict, Any

from algo.lib.controller import CarMissionManager
from algo.lib.car import CarState, CarCommand, CarAction
from algo.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global mission manager instance
mission_manager: CarMissionManager = None


@app.before_first_request
def initialize_server():
    """Initialize the mission manager on first request."""
    global mission_manager
    mission_manager = CarMissionManager()
    logger.info("Car pathfinding server initialized")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "robot_car_pathfinding",
        "config": {
            "turning_radius": config.car.turning_radius,
            "arena_size": f"{config.arena.size}x{config.arena.size}cm"
        }
    })


@app.route('/car/initialize', methods=['POST'])
def initialize_car():
    """
    Initialize car at starting position.

    Expected JSON:
    {
        "x": 20.0,
        "y": 20.0,
        "theta": 0.0  // optional, defaults to 0
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        theta = float(data.get('theta', 0))

        # Validate position is within arena
        if not (0 <= x <= config.arena.size and 0 <= y <= config.arena.size):
            return jsonify({"error": "Position outside arena bounds"}), 400

        mission_manager.initialize_car(x, y, theta)

        logger.info(f"Car initialized at ({x}, {y}) facing {theta:.3f} rad")

        return jsonify({
            "status": "success",
            "message": "Car initialized",
            "position": {"x": x, "y": y, "theta": theta}
        })

    except Exception as e:
        logger.error(f"Car initialization failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/obstacles/add', methods=['POST'])
def add_obstacle():
    """
    Add an obstacle to the arena.

    Expected JSON:
    {
        "x": 50,
        "y": 50,
        "image_side": "S"  // 'E', 'N', 'W', 'S'
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = int(data.get('x'))
        y = int(data.get('y'))
        image_side = data.get('image_side', '').upper()

        if image_side not in ['E', 'N', 'W', 'S']:
            return jsonify({"error": "image_side must be E, N, W, or S"}), 400

        # Validate obstacle position
        if not (0 <= x <= config.arena.size - config.arena.obstacle_size and
                0 <= y <= config.arena.size - config.arena.obstacle_size):
            return jsonify({"error": "Obstacle position outside valid area"}), 400

        obstacle_id = mission_manager.add_obstacle(x, y, image_side)

        logger.info(f"Obstacle {obstacle_id} added at ({x}, {y}) with image on {image_side} side")

        return jsonify({
            "status": "success",
            "obstacle_id": obstacle_id,
            "position": {"x": x, "y": y},
            "image_side": image_side
        })

    except Exception as e:
        logger.error(f"Add obstacle failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/mission/plan', methods=['POST'])
def plan_mission():
    """
    Plan a mission to visit specified obstacles.

    Expected JSON:
    {
        "targets": [0, 1, 2]  // obstacle indices to visit
    }
    """
    global mission_manager

    try:
        data = request.get_json() or {}
        targets = data.get('targets', [])

        if not isinstance(targets, list):
            return jsonify({"error": "targets must be a list of obstacle indices"}), 400

        success = mission_manager.plan_mission(targets)

        if not success:
            return jsonify({
                "status": "failed",
                "message": "Could not plan path to visit specified obstacles"
            }), 400

        logger.info(f"Mission planned to visit obstacles: {targets}")

        return jsonify({
            "status": "success",
            "message": f"Mission planned to visit {len(targets)} obstacles",
            "targets": targets
        })

    except Exception as e:
        logger.error(f"Mission planning failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/next_action', methods=['GET'])
def get_next_action():
    """Get the next action for the car to perform."""
    global mission_manager

    try:
        command = mission_manager.get_next_action()

        if not command:
            return jsonify({
                "status": "no_action",
                "message": "No action available (mission not planned or complete)"
            })

        response = {
            "status": "success",
            "action": command.action.value,
            "parameters": command.parameters,
            "expected_end_state": {
                "x": command.expected_end_state.x,
                "y": command.expected_end_state.y,
                "theta": command.expected_end_state.theta
            }
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Get next action failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/execute', methods=['POST'])
def execute_command():
    """
    Execute a car command and update position.

    Expected JSON:
    {
        "action": "forward",
        "parameters": {"distance": 10.0},
        "actual_result": {  // optional - actual sensor readings
            "measured_position": {"x": 25.2, "y": 20.1, "theta": 0.05}
        }
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        action_str = data.get('action', '')
        parameters = data.get('parameters', {})
        actual_result = data.get('actual_result', {})

        # Parse action
        try:
            action = CarAction(action_str)
        except ValueError:
            return jsonify({"error": f"Invalid action: {action_str}"}), 400

        # Create command (we need expected end state, but server doesn't know it)
        # This is a limitation - ideally the command comes from get_next_action
        current_status = mission_manager.car_status
        if not current_status:
            return jsonify({"error": "Car not initialized"}), 400

        # Create a dummy command for execution tracking
        command = CarCommand(action, parameters, current_status.estimated_state)

        result = mission_manager.execute_command(command, actual_result)

        logger.info(f"Executed {action_str} with result: {result['status']}")

        return jsonify(result)

    except Exception as e:
        logger.error(f"Command execution failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/car/position', methods=['POST'])
def update_car_position():
    """
    Manually update car position (for position corrections).

    Expected JSON:
    {
        "x": 25.5,
        "y": 30.2,
        "theta": 0.52
    }
    """
    global mission_manager

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON data required"}), 400

        x = float(data.get('x'))
        y = float(data.get('y'))
        theta = float(data.get('theta'))

        if not mission_manager.car_status:
            return jsonify({"error": "Car not initialized"}), 400

        # Update position directly
        mission_manager.car_status.estimated_state = CarState(x, y, theta)
        mission_manager.car_status.confidence_radius = 1.0  # Reset confidence

        logger.info(f"Car position updated to ({x:.2f}, {y:.2f}, {theta:.3f})")

        return jsonify({
            "status": "success",
            "message": "Position updated",
            "position": {"x": x, "y": y, "theta": theta}
        })

    except Exception as e:
        logger.error(f"Position update failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/status', methods=['GET'])
def get_status():
    """Get comprehensive system status."""
    global mission_manager

    try:
        status = mission_manager.get_status()
        return jsonify({
            "status": "success",
            "data": status
        })

    except Exception as e:
        logger.error(f"Get status failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset_system():
    """Reset the entire system."""
    global mission_manager

    try:
        mission_manager = CarMissionManager()
        logger.info("System reset complete")

        return jsonify({
            "status": "success",
            "message": "System reset complete"
        })

    except Exception as e:
        logger.error(f"System reset failed: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Robot Car Pathfinding Server")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /health              - Health check")
    print("  POST /car/initialize      - Initialize car position")
    print("  POST /obstacles/add       - Add obstacle to arena")
    print("  POST /mission/plan        - Plan mission to visit obstacles")
    print("  GET  /car/next_action     - Get next action for car")
    print("  POST /car/execute         - Execute car command")
    print("  POST /car/position        - Update car position")
    print("  GET  /status              - Get system status")
    print("  POST /reset               - Reset system")
    print("\nKey features:")
    print("  • Dubins path planning for car-like motion")
    print("  • Position uncertainty handling")
    print("  • Hamiltonian path optimization")
    print("  • Collision avoidance")
    print(f"\nStarting server on {config.server.host}:{config.server.port}")
    print("=" * 60)

    app.run(
        host=config.server.host,
        port=config.server.port,
        debug=config.server.debug
    )