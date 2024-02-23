import scipy.special
import numpy as np


class MeasureFunc:
    """
    A wrapper class for storing the metadata required to pass functions an instance of Plotter.
    """
    def __init__(self, function, plot_type):
        """
        Wraps a function in metadata. Sets the default number of bins for distribution functions to 50.

        Arguments:
            - function: A function to measure a gas, must take a gas and a time as arguments.
            - plot_type: The type of plot to make of that function, should be "time", "distribution", or "drawing".
        """
        self.func_name = function.func_name[1:]
        self.plot_type = plot_type
        self._function = function

        if plot_type == "distribution":
            self.bin_number = 50

    def __call__(self, *args):
        return self._function(*args)


def _speed(gas, time):
    """
    Measures the speed of every finite mass particle in a gas.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - A list of the speeds of every finite mass particle.
    """

    speeds = []

    for item in gas.balls:
        if not np.isinf(item.mass):
            velocity = item.vel()
            speeds.append(np.sqrt(np.dot(velocity, velocity)))

    return speeds

speed = MeasureFunc(_speed, "distribution")


def _total_kinetic_energy(gas, time):
    """
    Calculates the total kinetic energy of all balls with finite mass in a gas.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - The total kinetic energy of the balls.
    """

    total_ke = 0

    for item in gas.balls:
        velocity = item.vel()

        if not np.isinf(item.mass):
            total_ke += 0.5 * item.mass * np.dot(velocity, velocity)

    return total_ke

total_kinetic_energy = MeasureFunc(_total_kinetic_energy, "time")


def _total_momentum(gas, time):
    """
    Calculates the total momentum of all balls with finite mas in a gas.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - The magnitude of the total momentum of the balls.
    """

    total_p = np.zeros(gas.dimensions)

    for item in gas.balls:
        velocity = item.vel()

        if not np.isinf(item.mass):
            total_p += item.mass * velocity

    return np.sqrt(np.dot(total_p, total_p))

total_momentum = MeasureFunc(_total_momentum, "time")


def _average_pressure(gas, time):
    """
    Calculates the average pressure on a container in a gas.

    Arguments:
        - gas: Object of type ball.Gas with an instance of ball.Container in Gas.balls
        - time: The current time of the simulation.

    Returns:
        - The average pressure of the gas.
    """
    if not np.isinf(gas.balls[0].mass):
        raise Exception("Pressure is only defined when a container exists.")

    impulse = 0

    for item in gas.balls[1:]:
        impulse += item.container_delta_p

    radius = gas.balls[0].radius

    area = abs((2 * np.pi ** (gas.dimensions / 2.)) * radius ** (gas.dimensions - 1)) / scipy.special.gamma(gas.dimensions / 2.)

    if time == 0:
        return np.inf

    return impulse / (time * area)

average_pressure = MeasureFunc(_average_pressure, "time")


def _angular_momentum(gas, time):
    """
    Calculates the total angular momentum of all balls with finite mas in a gas.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - The magnitude of the total angular momentum of the balls.
    """

    if gas.dimensions > 3:
        raise Exception("Angular momentum is only defined in dimensions less than or equal to 3.")

    total_L = np.zeros(3)

    for item in gas.balls:

        if not np.isinf(item.mass):
            vel = np.zeros(3)
            vel[:item.vel().size] = item.vel()

            pos = np.zeros(3)
            pos[:item.pos().size] = item.pos()

            total_L += item.mass * np.cross(pos, vel)

    return np.sqrt(np.dot(total_L, total_L))

angular_momentum = MeasureFunc(_angular_momentum, "time")


def _brownian_position(gas, time):
    """
    Returns the position of the second ball in a Gas.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - The position of gas.balls[1].
    """
    return list(gas.balls[1].pos())

brownian_position = MeasureFunc(_brownian_position, "drawing")


def _free_path_length(gas, time):
    """
    Measures the distance moved since last time the second ball in a gas changed direction.

    Arguments:
        - gas: Object of type ball.Gas.
        - time: The current time of the simulation.

    Returns:
        - The distance moved since changing direction, if direction has just changed.
        - None if direction has not change.
    """

    if "brownian_free_path_pos" not in gas.stored_values:
        gas.stored_values["brownian_free_path_pos"] = np.array(gas.balls[1].pos())
        gas.stored_values["brownian_free_path_vel"] = np.array(gas.balls[1].vel())
        return None
    else:
        if np.all(gas.balls[1].vel() == gas.stored_values["brownian_free_path_vel"]):
            return None
        else:
            delta_r = gas.balls[1].pos() - gas.stored_values["brownian_free_path_pos"]
            gas.stored_values["brownian_free_path_pos"] = np.array(gas.balls[1].pos())
            gas.stored_values["brownian_free_path_vel"] = np.array(gas.balls[1].vel())
            return np.sqrt(np.dot(delta_r, delta_r))

free_path_length = MeasureFunc(_free_path_length, "distribution")
