# This is the official repo of R2RC
# Innovations:
- No angle limits, which means model is able to predict edges at any angle, and this is very different from Sat2Graph.
- After detecting keypoints for the whole graph, model can not only predict the connectivity between detected points, but is also able to insert new points that should exist but are not successfully detected by the keypoint predicting head.
- Get rid of the complicated graph matching processing of the on-the-fly label generating strategy, which matters when generating lables for current point for tracing algorithm.
