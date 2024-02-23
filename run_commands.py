import warnings
import simulation as sim
import measure_functions
import scipy.optimize as spo
import scipy.integrate as spi
import scipy.special
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt


def ideal_gas_check(simulation):
    """
    Calculates KbT and PV - NkT for a simulated gas. Prints these values and the volume.

    Arguments:
        - simulation: An instance of simulation.Plotter.
    """

    KbT = (2 * simulation.variables["total_kinetic_energy"][-1]) / (simulation.gas.dimensions
                                                                    * (simulation.gas.number_of_balls - 1))

    print "KbT =", KbT

    simulation.variables["KbT"] = KbT

    print "Volume =", simulation.variables["volume"]

    PV = simulation.variables["average_pressure"][-1] * simulation.variables["volume"]
    NkT = (2 * simulation.variables["total_kinetic_energy"][-1]) / simulation.gas.dimensions

    print "PV - NkT =", PV - NkT


def van_der_waals(simulation, data_length):
    """
    Runs multiple simulations at speeds linearly higher than the first simulation
     by adding the original value of each velocity component on to the velocities every time.
    Plots P against T for these simulations and fits values for the Van der Waals constants a and b.


    Arguments:
        - simulation: An instance of simulation.Plotter.
        - data_length: The number of points to plot.

    Returns:
        - The matplotlib figure that has been plotted.
    """
    fig = plt.figure()

    if not np.all(np.array(simulation.gas.initial_values["radii"]) == simulation.gas.initial_values["radii"][0]):
        warnings.warn("The Van der Waals law is not defined for gases with particles of different radii.")

    b_estimated = abs((((np.pi ** (simulation.gas.dimensions / 2.)) /
                        (scipy.special.gamma((simulation.gas.dimensions / 2. + 1)))) *
                       simulation.gas.balls[1].radius ** simulation.gas.dimensions))

    simulations = [simulation]

    velocity_constants = np.array(simulation.gas.initial_values["velocities"])

    for i in xrange(1, data_length):
        simulation.gas.initial_values["velocities"] = (np.array(simulation.gas.initial_values["velocities"])
                                                       + np.sqrt(i) * velocity_constants).tolist()
        simulation.gas.reset()
        simulations.append(sim.Plotter(simulation.gas, funcs=[measure_functions.total_kinetic_energy,
                                                              measure_functions.average_pressure],
                                       frame_time=simulation.frame_time, end_frame=simulation.end_frame))

    x = []
    y = []

    for item in simulations:
        print item
        x.append((2 * item.variables["total_kinetic_energy"][-1]) /
                 (item.gas.dimensions * (item.gas.number_of_balls - 1)))
        y.append(item.variables["average_pressure"][-1])

    N = simulation.gas.number_of_balls - 1
    V = simulation.variables["volume"]

    def vdw_func(x, a, b):
        return (N * x / (V - b*N)) + a * (N / V) ** 2

    po, po_cov = spo.curve_fit(vdw_func, x, y)
    xx = np.linspace(0, max(x) * 1.1, 1000)

    plt.plot(x, y, "bx")
    plt.plot(xx, vdw_func(xx, po[0], po[1]), "r-")

    plt.grid()
    plt.xlabel(r"$\rm k_B T$", fontsize=25)
    plt.ylabel(r"$\rm Pressure$", fontsize=25)

    print "Fitted Van der Waals a =", po[0], "+/-", np.sqrt(po_cov[0][0])
    print "Calculated Van der Waals b =", b_estimated
    print "Fitted Van der Waals b =", po[1], "+/-", np.sqrt(po_cov[1][1])

    return fig


def fit_maxwell(simulation, figure):
    """
    Fits a Maxwell-Boltzmann curve to a simulation's speed distribution using least squares.

    Arguments:
        - simulation: An instance of simulation.Plotter where a speed distribution has been measured.
        - figure: The matplotlib figure onto which the curve should be placed

    Returns:
        A tuple of the fitted values and the covariance matrix for these values.
    """

    if "speed" not in simulation.dist_data:
        raise Exception("A distribution of speeds must exist in order to fit a Maxwell-Boltzmann distribution.")

    if not np.all(np.array(simulation.gas.initial_values["masses"]) == simulation.gas.initial_values["masses"][0]):
        warnings.warn("The Maxwell-Boltzmann distribution is not defined for gases with particles of different masses.")

    if simulation.gas.dimensions > 3:
        warnings.warn("The Maxwell-Boltzmann distribution is not defined for greater than 3 dimensions.")

    plt.figure(figure.number)

    def maxwell(x, a, b):
        return a * x ** (simulation.gas.dimensions - 1) * np.exp( - b * x ** 2)

    x = []
    for i in xrange(1,simulation.dist_data["speed"][1].size):
        x.append((simulation.dist_data["speed"][1][i-1] + simulation.dist_data["speed"][1][i])/2)

    y = simulation.dist_data["speed"][0]

    KbT_estimate = (2 * simulation.variables["total_kinetic_energy"][-1]) / (simulation.gas.dimensions
                                                                             * (simulation.gas.number_of_balls - 1))
    b_initial = 1 / (2 * KbT_estimate)
    po, po_cov = spo.curve_fit(maxwell, x, y, p0=[np.sqrt(2/np.pi), b_initial])

    xx = np.linspace(0, simulation.dist_data["speed"][1].max() * 1.1, 1000)

    area = spi.quad(maxwell, 0, 10 * xx.max(), args=(po[0], po[1]))

    print "Area under Maxwell-Boltzmann distribution =", area[0], "+/-", area[1]

    KbT = po[1] * 2 / simulation.gas.initial_values["masses"][0]
    KbT_err = (np.sqrt(po_cov[1][1]) / po[1]) * KbT

    print "Maxwell-Boltzmann fit KbT =", KbT, "+/-", KbT_err

    plt.plot(xx, maxwell(xx, po[0], po[1]), "r-")
    plt.plot(xx, maxwell(xx, po[0], b_initial), "y-")
    plt.legend(["Fitted function", "Calculated function", "Data"])

    return po, po_cov


def fit_exponential_brownian(simulation, figure):
    """
    Fits an exponential distribution to a simulation's path length distribution using least squares.

    Arguments:
        - simulation: An instance of simulation.Plotter where a path length distribution has been measured.
        - figure: The matplotlib figure onto which the curve should be placed

    Returns:
        A tuple of the fitted values and the covariance matrix for these values.
    """

    if "free_path_length" not in simulation.dist_data:
        raise Exception("A distribution of free path must exist in order to fit the exponential distribution.")

    plt.figure(figure.number)

    def exponential(x, l):
        return l * np.exp(-l * x)

    x = []
    for i in xrange(1,simulation.dist_data["free_path_length"][1].size):
        x.append((simulation.dist_data["free_path_length"][1][i-1] + simulation.dist_data["free_path_length"][1][i])/2)

    y = simulation.dist_data["free_path_length"][0]

    raw_data = np.array(simulation.variables["free_path_length"][int(-simulation.end_frame / 4):]).flatten()
    raw_data = [i for i in raw_data if i is not None]
    l_estimate = np.mean(raw_data)

    po, po_cov = spo.curve_fit(exponential, x, y, 1/l_estimate)

    xx = np.linspace(0, simulation.dist_data["free_path_length"][1].max() * 1.1, 1000)

    area = spi.quad(exponential, 0, 10 * xx.max(), args=(po[0]))

    print "Area under exponential distribution =", area[0], "+/-", area[1]
    print "Calculated lambda =", l_estimate, "+/-", l_estimate**2
    print "lambda =", 1/po[0], "+/-", np.sqrt(po_cov[0][0])/(po[0] ** 2)

    plt.plot(xx, exponential(xx, po[0]), "r-")

    return po, po_cov


def initial_conditions_uniform(dimensions=3, number_of_balls=40, mass=1, radius=0.5, scale=2):
    """
    Generates the initial conditions for a uniform ball.Gas object.
    Positions are left at the origin to be placed later.

    Arguments:
        - dimensions: The number of dimensions of the gas. Defaults to 3.
        - number_of_balls: The number of ball.Ball objects to generate. Defaults to 40.
        - mass: The mass of the balls. Defaults to 1.
        - radius: The radius of the balls. Defaults to 0.5.
        - scale: The standard distribution of the velocity components of the balls.
            Normally distributed about 0. Defaults to 2.

    Returns:
        - A tuple of lists of masses, radii, positions and velocities that can be passed
         to initialise a ball.Gas instance.
    """

    masses = (np.ones(number_of_balls) * mass).tolist()
    radii = (np.ones(number_of_balls) * radius).tolist()
    positions = np.zeros((number_of_balls, dimensions)).tolist()
    velocities = (rand.normal(scale=scale, size=(number_of_balls, dimensions))).tolist()

    return masses, radii, positions, velocities


def initial_conditions_mixture(dimensions=3, number_of_balls=(20, 20), mass=(1, 2), radius=(0.5, 1), scale=(2, 2)):
    """
    Generates the initial conditions for a mixed ball.Gas object.
    Positions are left at the origin to be placed later.

    Arguments:
        - dimensions: The number of dimensions of the gas. Defaults to 3.
        - number_of_balls: A list of the number of ball.Ball objects to generate of each type. Defaults to [20, 20].
        - mass: A list of the masses of each ball type. Defaults to [1, 2].
        - radius: A list of the radii of each ball type. Defaults to [0.5, 1]
        - scale: A list of the standard distribution of the velocity components of the balls of each type.
            Normally distributed about 0. Defaults to [2, 2]

    Returns:
        - A tuple of lists of masses, radii, positions and velocities that can be passed
         to initialise a ball.Gas instance.
    """

    if not all(len(args) == len(number_of_balls) for args in [number_of_balls, mass, radius, scale]):
        raise ValueError("Arguments other than dimensions must all be same length.")

    total_balls = 0
    for n in number_of_balls:
        total_balls += n

    print total_balls

    masses = np.zeros(len(number_of_balls), dtype=object)
    radii = np.zeros(len(number_of_balls), dtype=object)
    positions = np.zeros(len(number_of_balls), dtype=object)
    velocities = np.zeros(len(number_of_balls), dtype=object)

    for i in xrange(len(number_of_balls)):
        masses[i], radii[i], positions[i], velocities[i] = initial_conditions_uniform(dimensions, number_of_balls[i], mass[i], radius[i], scale[i])

    print masses
    print masses.flatten()

    return np.concatenate(masses).tolist(), np.concatenate(radii).tolist(), np.concatenate(positions).tolist(), np.concatenate(velocities).tolist()


def initial_conditions_brownian(dimensions=2, number_of_balls=40, mass=(10, 1), radius=(2.5, 0.5), scale=2):
    """
    Generates the initial conditions for a Brownian motion simulation for a ball.Gas object.
    Positions are left at the origin to be placed later.

    Arguments:
        - dimensions: The number of dimensions of the gas. Defaults to 2.
        - number_of_balls: The number of ball.Ball objects to generate. Defaults to 40.
        - mass: A list of length 2, with the mass of the particle of interest first
            and the mass of the other particles second. Defaults to [10, 1].
        - radius: A list of length 2, with the radius of the particle of interest first
            and the radius of the other particles second. Defaults to [2.5, 0.5].
        - scale: The standard distribution of the velocity components of the balls.
            Normally distributed about 0. Defaults to 2.

    Returns:
        - A tuple of lists of masses, radii, positions and velocities that can be passed
         to initialise a ball.Gas instance.
    """
    return initial_conditions_mixture(dimensions, (1, number_of_balls - 1), mass, radius, (scale, scale))