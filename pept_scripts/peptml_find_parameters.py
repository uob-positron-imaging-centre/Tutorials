#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : peptml_analysis.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.08.2019


'''This is an example script that illustrates how to analyse PEPT data using
the `pept` package. This is meant as an aid in choosing clustering parameters.

This script is a simplified version of "peptml_analysis.py", in which all
user-changeable parameters are at the top of this file. It is meant as a
standard analyser, which can be copied into different datasets and run with
minimal user intervention.

The `pept` package is modular, meaning it can be used in many ways, to suit
(m)any circumstances. This is just an example, which you can modify as you
see fit.

If you're not sure what a class or method does, you can always consult the
documentation online, at "uob-positron-imaging-centre.github.io", or by
writing a questions mark in front of a module in the Python shell. For example:
    >>> import pept
    >>> ?pept.LineData  # Short documentation
    >>> ??pept.LineData # Long documentation

'''


import numpy as np
from   tqdm  import tqdm
import pept

from pept.scanners      import ParallelScreens
from pept.tracking      import peptml
from pept.visualisation import PlotlyGrapher


##############################################################################
#                                                                            #
#                      User-changeable parameters                            #
#                                                                            #
##############################################################################

# If we know the maximum number of particles that we may have in our field
# of view, then we can use it to our advantage when setting the sample size.
# However, it is not necessary; we can set the sample size to be as small as
# required.
max_particles =     2

# Reading in data
filepath =          'sample_2p_42rpm.csv'
separation =        712                     # Separation between the PEPT scanner screens, in mm

skiprows =          16                      # Skip the file header when reading the data
max_rows =          10000                   # The number of LoRs to read in

sample_size =       200 * max_particles     # LoRs sample size
overlap =           50 * max_particles

# Only consider the cutpoints from lines that are closer than `max_distance`
max_distance =      0.1  # mm

# A larger `k_` means harsher clustering.
# Clusterer for one-pass => k1, noisy data
k1 =                [0.05, 0.8]

# Number of iterations between k1[0] and k1[1]
iterations =        5

allow_single_cluster_1 = False

# Store the clustered cutpoints, or just the centres of the clusters (i.e.
# particle positions)? Not storing them makes the clustering ~4x faster.
store_labels =      True
noise =             True


# Plotting
rows =              2
cols =              iterations
xlim =              [0, 500]
ylim =              [0, 500]
zlim =              [0, separation]
subplot_titles =    ["k1 = {}".format(k) for k in np.linspace(k1[0], k1[1], iterations)]

##############################################################################


#
# Read the LoR data from a parallel screens scanner into a `LineData` subclass.
#
print('-----------------------------------------------')
print('Initialising\n')
lors = ParallelScreens(
    filepath,
    separation,
    sample_size = sample_size,
    overlap = overlap,
    skiprows = skiprows,
    max_rows = max_rows
)

# Print our data attributes for convenience
print("LoR data number of samples: {}".format(lors.number_of_samples))
print("PEPT screens separation: {}\n".format(separation))
print("PEPT data sample size = {}, overlap = {}\n".format(sample_size, overlap))

#
# Find the cutpoints
#
print('\n\n-----------------------------------------------')
print('Finding the cutpoints\n')

# Stores the found cutpoints in a `PointData` subclass, so we can access all
# its methods.
cutpoints = peptml.Cutpoints(lors, max_distance)

# Print our data attributes for convenience
print("For max_distance = {}, found a total of {} cutpoints".format(max_distance, cutpoints.number_of_points))
print("Cutpoints sample size = {}".format(cutpoints.sample_size))

#
# Cluster our cutpoints using peptml.HDBSCANClusterer
#

size_1 = cutpoints.sample_size / max_particles
ks = np.linspace(k1[0], k1[1], iterations)

# Instantiate a list of clusterers
clusterers = [
    peptml.HDBSCANClusterer(
        min_cluster_size = int(k * size_1)
    ) for k in ks
]

print('\n\n-----------------------------------------------')
print('First pass of clustering, for cutpoints\n')

centres = []
clustered_cutpoints = []
for clusterer in tqdm(clusterers):
    centres_1, clustered_cutpoints_1 = clusterer.fit_cutpoints(
        cutpoints,
        store_labels = store_labels,
        noise = noise,
        verbose = False
    )

    centres.append(centres_1)
    clustered_cutpoints.append(clustered_cutpoints_1)

#
# Plot the results using PlotlyGrapher. It can define any number
# of subplots and automatically configures them to the alternative PEPT
# 3D axes convention, in which the y-axis points upwards.
#

print('\n\n-----------------------------------------------')
print('Plotting...\n')

# Instantiate PlotlyGrapher
grapher = PlotlyGrapher(
    rows = rows,
    cols = cols,
    xlim = xlim,
    ylim = ylim,
    zlim = zlim,
    subplot_titles = subplot_titles
)

# After instantiating the grapher, don't forget to create a figure too
grapher.create_figure()

# Add traces to the Plotly graph. As both `centres_1` and `centres_2` inherit
# from `PointData`, they have methods which create Plotly traces.
for i in range(len(centres)):
    grapher.add_trace(clustered_cutpoints[i].all_points_trace_colorbar(), row = 1, col = i + 1)
    grapher.add_trace(centres[i].all_points_trace_colorbar(), row = 2, col = i + 1)

grapher.show()




