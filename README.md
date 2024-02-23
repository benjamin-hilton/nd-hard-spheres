# nd-hard-spheres

To start a simulation, first instantiate a Gas from the ball module using some initial conditions. These can just be lists of masses, radii, positions and velocities, or gan be generated using the run_commands module, which has functions to generate initial conditions.

Then instantiate a Simulation. Simulations are either Plotter (generates data using functions from the measure_functions module) or Movie (returns the patches for a matplotlib animation). Generally, a Plotter should be instantiated before a Movie so that the Movie can start from the same point as the Plotter - this ensures that the animation shows the gas during the relevant time interval for the data. The gas.reset() command resets a gas so that a new Simulation can be instantiated from the same starting point.

For a Plotter, the Plotter.plot() method returns a dictionary of the figures. For a Movie, the Movie.get_animation() method returns the matplotlib Animation object.

Other extraneous functions can be found in the run_commands module, for example to fit functions to data.

Finally, use matplotlib.pyplot.show to display the figures generated.

A commented example can be found in the main.py file.
