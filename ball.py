import itertools
import copy
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import timer


class Ball:
    """
    A ball, modelling a hard particle in a gas that collides elastically.
    """

    _default_position = [0, 0]
    _default_velocity = [0, 0]

    def __init__(self, mass=1, radius=1, position=None, velocity=None, color='r'):
        """
        Initialises a ball.

        Arguments:
            - mass: Mass of the ball. Float or int. Defaults to 1.
            - radius: Radius of the ball. Float or int. Defaults to 1.
            - position: Position of the ball. Must be a list. Defaults to Ball._default_position.
            - velocity: Velocity of the ball. Must be a list of same length as position.
                Defaults to Ball._default.velocity.
            - color: Color of the ball. Defaults to red.
        """

        self.mass = float(mass)
        self.radius = float(radius)

        self._pos = np.array(position or Ball._default_position, dtype=float)
        self._vel = np.array(velocity or Ball._default_velocity, dtype=float)

        if self._pos.size != self._vel.size:
            raise ValueError("Position and velocity lists must be same length.")

        self._color = color

        self._patch = plt.Circle(self._pos, abs(self.radius), color=self._color)

        self.container_delta_p = 0

    def __repr__(self):
        return "Ball(m=%g, rad=%g, pos=%s, vel=%s, col=%s)" % (self.mass, self.radius,
                                                               self._pos, self._vel, self._color)

    def __str__(self):
        return "Ball(mass=%g, radius=%g, position=%s, velocity=%s, color=%s)" % (self.mass, self.radius,
                                                                                 self._pos, self._vel, self._color)

    def pos(self):
        return self._pos

    def vel(self):
        return self._vel

    def patch(self):
        return self._patch

    def move(self, dt):
        """
        Moves the ball to its position a time dt in the future, assuming current velocity is unchanged.
        Also moves the ball's patch.

        Arguments:
            - dt: The time interval through which to move the ball.
        """
        self._pos += self._vel * dt
        self._patch.center = self._pos

    def time_to_collision(self, other):
        """
        Calculates the time until collision with another ball.

        Arguments:
            - other: Another instance of Ball.

        Returns:
            The time dt until collision with other, or inf if no collision will occur given current trajectories.
        """

        delta_v = self._vel - other.vel()
        delta_r = self._pos - other.pos()

        if np.all(delta_v == 0):
            return np.inf

        a = np.dot(delta_v, delta_v)
        b = 2 * np.dot(delta_v, delta_r)
        c = np.dot(delta_r, delta_r) - (self.radius + other.radius)**2

        roots = np.roots([a, b, c])

        roots[abs(roots) < 1e-11] = 0

        # print self._color, other._color, roots

        if (not np.any(np.isreal(roots))) or (not np.any(roots > 0)):
            return np.inf

        else:
            return np.amin(roots[roots > 0])

    def collide(self, other):
        """
        Changes the velocity of self and another ball on collision with each other.
        Also stores the total momentum transferred to infinite mass objects.

        Arguments:
            - other: Another instance of Ball.
        """

        if np.isinf(other.mass) and np.isinf(self.mass):
            raise Exception("Two infinite mass objects collided.")

        vs = copy.deepcopy(self._vel)
        vo = copy.deepcopy(other.vel())

        delta_v = self._vel - other.vel()
        delta_r = self._pos - other.pos()
        delta_r_unit = delta_r / np.sqrt(np.dot(delta_r, delta_r))

        us_parallel = np.dot(delta_v, delta_r_unit) * delta_r_unit

        vs_perpendicular = delta_v - us_parallel

        if np.isinf(self.mass):
            vs_parallel = us_parallel
            vo_parallel = 2 * us_parallel
        elif np.isinf(other.mass):
            vs_parallel = us_parallel * -1
            vo_parallel = 0
        else:
            mass_tot = self.mass + other.mass
            vs_parallel = us_parallel * (self.mass - other.mass) / mass_tot
            vo_parallel = us_parallel * 2 * self.mass / mass_tot

        self._vel = vs_parallel + vs_perpendicular + other.vel()
        other._vel += vo_parallel

        if np.isinf(other.mass) and not np.isinf(self.mass):
            momentum_change = (vs - self.vel()) * self.mass
            self.container_delta_p += np.sqrt(np.dot(momentum_change, momentum_change))
        elif np.isinf(self.mass) and not np.isinf(other.mass):
            momentum_change = (vo - other.vel()) * other.mass
            other.container_delta_p += np.sqrt(np.dot(momentum_change, momentum_change))


class Container(Ball):
    """
    A container functions as a ball with negative radius, such that other balls collide with the inside
    rather than outside of the container.
    """

    _default_position = [0, 0]

    def __init__(self, mass=np.inf, radius=10, position=None, velocity=None, color='k', dimensions=2):

        """
        Initialises a ball.

        Arguments:
            - mass: Mass of the container. Float or int. Defaults to inf.
            - radius: Radius of the container. Float or int. Defaults to 10.
            - position: Position of the container. Must be a list. Defaults to the origin.
            - velocity: Velocity of the container. Must be a list of same length as position. Defaults to rest.
            - color: Color of the container. Defaults to black.
            - dimensions: The number of dimensions of the container.
        """

        Ball.__init__(self, mass, -1*radius, position or np.zeros(dimensions).tolist(),
                      velocity or np.zeros(dimensions).tolist())
        self._patch.set(ec=color, fill=False, ls="solid")
        self._color = color


class Gas:
    """
    A gas consisting of hard, elastically colliding balls and optionally a container.
    Methods allow simulation but only but not a frame rate - only times where collisions take place are simulated.
    """
    def __init__(self, masses, radii, positions, velocities, colors=None, container=None):
        """
        Initialises the balls and, optionally, a container, onto a set of axes.
        Also randomly places balls to prevent overlap and calculates time until first collision.
        Initialises the collision times matrix and determines time of first collision.
        Initial values are stored to self.initial_values for reference.
        Initialises a dictionary for storing extraneous values.

        Arguments:
            - masses: List of ball masses.
            - radii: List of ball radii.
            - positions: List of ball positions.
            - velocities: List of ball velocities.
            - colors: List of ball colors.
            - container: Dictionary of container parameters. If True, initialises default container.
                The container is placed at the start of the list of balls.
        """

        if not all(len(args) == len(masses) for args in [masses, radii, positions, velocities]):
            raise ValueError("Arguments must all be same length.")

        colors = colors or ["r"] * len(masses)

        self.balls = []

        for i in range(len(masses)):
            self.balls.append(Ball(masses[i], radii[i], positions[i], velocities[i], colors[i]))

        self.dimensions = self.balls[0].pos().size

        for item in self.balls:
            self._check_overlap(item, container["radius"] or 10, 100)

        if not container:
            pass
        elif type(container) == bool:
            self.balls.insert(0, Container(dimensions=self.dimensions))
        else:
            container["dimensions"] = self.dimensions
            self.balls.insert(0, Container(**container))

        self.number_of_balls = len(self.balls)

        self.time = 0

        self.collision_times = np.ones(shape=(self.number_of_balls, self.number_of_balls)) * np.inf

        for i, j in itertools.combinations(xrange(self.number_of_balls), 2):
            if i < j:
                self.collision_times[i, j] = (self.balls[i].time_to_collision(self.balls[j]))

        self.next_collision_time = np.amin(self.collision_times)
        self.next_collision_times = self.collision_times[self.collision_times < self.next_collision_time + 1e-10]

        self.next_collision_balls = []
        indices = np.where(self.collision_times == self.next_collision_time)
        self.next_collision_balls_index = [indices[0][0], indices[1][0]]

        self.next_collision_balls.append(self.balls[self.next_collision_balls_index[0]])
        self.next_collision_balls.append(self.balls[self.next_collision_balls_index[1]])

        self.initial_values = {"masses": masses,
                               "radii": radii,
                               "positions": [list(i.pos()) for i in self.balls[1:]],
                               "velocities": velocities,
                               "colors": colors,
                               "container": container}

        self.stored_values = {}


    def _check_overlap(self, ball1, radius, recursion_limit):
        """
        Check if a ball overlaps with any other.
        If it does, randomly move it to another non-overlapping position within the sphere or circle defined by radius.

        Arguments:
            - ball1: A ball from self.balls.
            - radius: Radius within which to place balls.
            - recursion_limit: The number of times to run the function before deciding there is not enough space.
        """
        if recursion_limit == -len(self.balls):
            raise Exception("Recursion limit reached. Could not place balls.")

        for item in self.balls:

            if item == ball1:
                continue

            delta_r = ball1.pos() - item.pos()

            if np.dot(delta_r, delta_r) <= (item.radius + ball1.radius) ** 2:
                vector = (rand.rand(1, self.dimensions)[0] - 0.5) * 2 * radius
                while abs(np.dot(vector, vector)) > (radius - ball1.radius) ** 2:
                    vector = (rand.rand(1, self.dimensions)[0] - 0.5) * 2 * radius
                ball1._pos = vector
                recursion_limit -= 1
                self._check_overlap(ball1, radius, recursion_limit)

    def reset(self):
        """
        Resets the gas to its initial values.
        """
        self.__init__(**self.initial_values)

    def update(self):
        """
        Updates the simulation until a time 1e-14 after the next collision.
        Collides all balls that collide in this interval.
        Updates the next collision time and the balls to collide next, as well as the collision times matrix.
        """

        self.update_index = []

        for item in self.balls:
            item.move(self.next_collision_time - self.time)

        self.next_collision_time = np.amin(self.next_collision_times)
        self.collide()

        for i in xrange(1, len(self.next_collision_times)):
            self.next_collision_time = self.next_collision_times[i]
            indices = np.where(self.collision_times == self.next_collision_time)
            duplicates = 0
            if self.next_collision_times[i] == self.next_collision_times[i-1]:
                duplicates += 1
                self.next_collision_balls_index = [indices[0][duplicates], indices[1][duplicates]]
            else:
                self.next_collision_balls_index = [indices[0][0], indices[1][0]]
            self.next_collision_balls[0] = self.balls[self.next_collision_balls_index[0]]
            self.next_collision_balls[1] = self.balls[self.next_collision_balls_index[1]]
            self.collide()

        for i in np.unique(self.update_index):
            for j in xrange(self.number_of_balls):
                if i < j:
                    self.collision_times[i, j] = (self.balls[i].time_to_collision(self.balls[j])) + self.time
                elif j < i:
                    self.collision_times[j, i] = (self.balls[i].time_to_collision(self.balls[j])) + self.time

        self.next_collision_time = np.amin(self.collision_times)
        self.next_collision_times = self.collision_times[self.collision_times <= (self.next_collision_time + 1e-14)]

        indices = np.where(self.collision_times == self.next_collision_time)
        self.next_collision_balls_index = [indices[0][0], indices[1][0]]
        self.next_collision_balls[0] = self.balls[self.next_collision_balls_index[0]]
        self.next_collision_balls[1] = self.balls[self.next_collision_balls_index[1]]

    def collide(self):
        """
        Update the simulation to an infinitesimal time after the next collision.
        """
        self.next_collision_balls[0].collide(self.next_collision_balls[1])
        self.update_index.extend(self.next_collision_balls_index)
        self.time = self.next_collision_time

    def simulate(self, end_time, start_time=0):
        """
        Runs the simulation until just after the last collision before a set time.

        Arguments:
            - end_time: The value of self.time before which to stop the simulation.
            - start_time: The value of self.time at which to stop the simulation. Defaults to zero.

        Returns:
            - The time of the simulation just after the last collision.
        """

        self.time = start_time

        while self.next_collision_time <= end_time:
            self.update()

        return self.time
