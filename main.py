import ball
import simulation as sim
import run_commands as run
import measure_functions
import matplotlib.pyplot as plt

end_time = 100  # Sets the end time for the Plotter

container_values = {"radius": 10}  # Sets the Container properties. See help(ball.Container).

# Generate initial conditions
masses, radii, positions, velocities = run.initial_conditions_mixture()
colors = [["c", "r", "g", "b", "m", "y"][i % 6] for i in xrange(len(masses))]  # Sets the ball colours.

# Initialise the gas
gas = ball.Gas(masses, radii, positions, velocities, colors=colors, container=container_values)

# Initialise the data gathering simulation. See help(sim.Plotter).
plot_sim = sim.Plotter(gas, funcs=[measure_functions.total_kinetic_energy,
                                   measure_functions.total_momentum,
                                   measure_functions.speed,
                                   measure_functions.average_pressure,
                                   measure_functions.angular_momentum,
                                   ],
                       frame_time=0.1, end_frame=int(end_time * 10))

# Return the plots
figures = plot_sim.plot()

# Run any extraneous functions, e.g. fitting.
run.fit_maxwell(plot_sim, figures["speed"])

# Reset the gas
gas.reset()

# Initialise the animation
movie = sim.Movie(gas, interval=1, frame_time=0.01)
# Return the animation
anim = movie.get_animation()

# Show plots
plt.show()
