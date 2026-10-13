"""Detector extractor of the xg-detector study: the package's models on shot frames, positions in metres, features.

Modules
    conventions  pitch metres (SoccerPitch 105 x 68, corner origin), the attacked goal, 180-degree alignment, and the
                 StatsBomb -> metres conversion (used by the audit only, never by a feature)
    features     shot situation of one analysed frame (shooter, shot location, roles) and the H3 geometry in metres
    extract      the four declared configurations C1-C4, the package FrameAnalyzer per configuration, frame records
    records      completion records: model, code and input hashes
    guard        refuses reads of StatsBomb or pairing files on the model and feature input paths
    seal         refuses every label of a held-out (gate or evaluation) shot before the single test (ticket 06)
    sanitation   the label-free sanitation and count rules of approach 1 (ticket 03) and the raster points
    approach1    A1 / A2: G-combo with the geometry branch or the raster CNN branch, inputs, plan check, fit and replay
"""
