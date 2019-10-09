#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : peptml_minimal.py
# License: License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.08.2019


'''This script illustrates a minimum working example of analysing PEPT data.

This is meant as a quick go-to example. If this is your first time using the
`pept` package, you should consider working through a tutorial first.

For more sophisticated usage, see the following scripts:
    "peptml_analysis.py" : full working example / tutorial
    "peptml_user.py" : full working example

'''

import pept
peptml = pept.tracking.peptml


# Read LoR data; PEPT screen separation = 712 mm
lors = pept.scanners.ParallelScreens(
    'sample_2p_42rpm.csv',
    712,
    sample_size = 400,
    overlap = 200,
    skiprows = 16,
    max_rows = 50000
)

# One-pass clustering
cutpoints = peptml.Cutpoints(lors, 0.1)     # max_distance = 0.1
clusterer = peptml.HDBSCANClusterer(30)     # min_sample_size = 30

centres, clustered_cutpoints = clusterer.fit_cutpoints(cutpoints)

# Two pass clustering
centres.sample_size = 120
centres.overlap = 118

clusterer2 = peptml.HDBSCANClusterer(20)

centres2, clustered_centres = clusterer2.fit_cutpoints(centres)

# Plot centres
grapher = pept.visualisation.PlotlyGrapher(cols = 2)
grapher.create_figure()

grapher.add_trace(centres.all_points_trace_colorbar(), col = 1)
grapher.add_trace(centres2.all_points_trace_colorbar(), col = 2)
grapher.show()


