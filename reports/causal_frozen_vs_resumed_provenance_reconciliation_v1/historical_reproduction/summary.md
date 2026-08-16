# Historical Reproduction

The 8d native freeze worker reproduces the fast regime on GPU 1 under the frozen env. C2048/B1 is 166.829 ms/token and C4096/B8 passes. These values are close enough to the stored frozen ~153-205 ms/token rows to confirm the frozen regime.
