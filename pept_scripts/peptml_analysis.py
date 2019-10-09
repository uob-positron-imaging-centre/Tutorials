#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : peptml_analysis.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.08.2019


'''This is an example script that illustrates how to analyse PEPT data using
the `pept` package.

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
import pept

from pept.scanners import ParallelScreens
from pept.tracking import peptml
from pept.visualisation import PlotlyGrapher


filepath = 'sample_2p_42rpm.csv'
separation = 712    # Separation between the PEPT scanner screens, in mm

# If we know the maximum number of particles that we may have in our field
# of view, then we can use it to our advantage when setting the sample size.
# However, it is not necessary; we can set the sample size to be as small as
# required.
max_particles = 2
sample_size = 200 * max_particles    # LoRs sample size
overlap = 50 * max_particles
skiprows = 16                        # Skip the file header when reading the data
max_rows = 1000000                   # The number of LoRs to read in

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

# Only consider the cutpoints from lines that are closer than `max_distance`
max_distance = 0.1  # mm
# Stores the found cutpoints in a `PointData` subclass, so we can access all
# its methods.
cutpoints = peptml.Cutpoints(lors, max_distance)

# Print our data attributes for convenience
print("For max_distance = {}, found a total of {} cutpoints".format(max_distance, cutpoints.number_of_points))
print("Cutpoints sample size = {}".format(cutpoints.sample_size))

#
# Cluster our cutpoints using peptml.HDBSCANClusterer
#

# We can use the maximum number of particles to our advantage, if it is known.
# The number of cutpoints in a cluster around a particle scales with the number
# of particles. Hence we can use a scaling parameter `k1`: a larger value means
# more points have to be close together to form a cluster. Therefore, a larger
# `k1` means harsher clustering.
k1 = 0.15
min_cluster_size_1 = int(k1 * cutpoints.sample_size / max_particles)

# If we have more than one particle, set this to False, as it increases the
# accuracy (and harshness) of clustering.
allow_single_cluster_1 = False

# Instantiate a clusterer
clusterer_1 = peptml.HDBSCANClusterer(
    min_cluster_size = min_cluster_size_1,
    allow_single_cluster = allow_single_cluster_1
)

print('\n\n-----------------------------------------------')
print('First pass of clustering, for cutpoints\n')

# Store the clustered cutpoints, or just the centres of the clusters (i.e.
# particle positions)? Not storing them makes the clustering ~4x faster.
store_labels = False
noise = False

centres_1, clustered_cutpoints_1 = clusterer_1.fit_cutpoints(
    cutpoints,
    store_labels = store_labels,
    noise = noise
)


# Save the centres to a file?
save_centres_1 = False
save_centres_filepath = 'tracked_data/bms_a26_1pass.csv'

if save_centres_1:
    print("Saving the centres of cutpoints...")
    centres_1.to_csv(save_centres_filepath)

#
# Cluster our centres using peptml.HDBSCANClusterer (two-pass clustering)
#

# Set the sample_size and overlap of `centres_1` so we can iterate through
# the samples when clustering.
centres_1.sample_size = 30 * max_particles
centres_1.overlap = 29 * max_particles      # Big overlap => small reduction in time resolution

# As before, instantiate a clusterer
k2 = 0.7
min_cluster_size_2 = int(k2 * centres_1.sample_size / max_particles)
allow_single_cluster_2 = False

clusterer_2 = peptml.HDBSCANClusterer(
    min_cluster_size = min_cluster_size_2,
    allow_single_cluster = allow_single_cluster_2
)

print('\n\n-----------------------------------------------')
print('Second pass of clustering, for centres\n')

# Store the clustered cutpoints, or just the centres of the clusters (i.e.
# particle positions)? Not storing them makes the clustering ~4x faster.
store_labels = False
noise = False

centres_2, clustered_cutpoints_2 = clusterer_2.fit_cutpoints(
    centres_1,
    store_labels = store_labels,
    noise = noise
)


# Save the centres to a file?
save_centres_2 = False
save_centres_2_filepath = 'tracked_data/bms_a26_2pass.csv'

if save_centres_2:
    print("Saving the centres of cutpoints...")
    centres_2.to_csv(save_centres_2_filepath)


#
# Plot the results using PlotlyGrapher. It can define any number
# of subplots and automatically configures them to the alternative PEPT
# 3D axes convention, in which the y-axis points upwards.
#

print('\n\n-----------------------------------------------')
print('Plotting...\n')

rows = 1
cols = 2
xlim = [0, 500]
ylim = [0, 500]
zlim = [0, separation]
subplot_titles = ['One-pass clustering', 'Two-pass clustering']

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
grapher.add_trace(centres_1.all_points_trace_colorbar(), col = 1)
grapher.add_trace(centres_2.all_points_trace_colorbar(), col = 2)

grapher.show()




