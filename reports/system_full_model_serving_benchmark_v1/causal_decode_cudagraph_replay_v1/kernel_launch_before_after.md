# Kernel Launch Before/After

Post-graph PyTorch profiler was not run because the primary graph formal run is blocked on physical GPU1 contamination in this session. Expected physical GPU kernel count does not decrease; graph replay reduces host submissions to one graph replay per captured decode step.
