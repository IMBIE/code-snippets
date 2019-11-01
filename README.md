# code-snippets
definitions for key functions of the IMBIE processor

This repository contains python source files for some of the key methods used by the IMBIE processor.

## combine.py

This file provides the function used to compute averaged time series from individual data
combinations, or from experiment group averages. The `weighted_combine` function can be used to
calculate various different results, including per-epoch weighted means.

## dm_to_dmdt.py

This file defines the method used to compute dM/dt(t) time series from dM(t) time series. This is
implemented by performing a windowed curve fitting across the input dM data and finding the gradient
of the curve at the centre of each window.

