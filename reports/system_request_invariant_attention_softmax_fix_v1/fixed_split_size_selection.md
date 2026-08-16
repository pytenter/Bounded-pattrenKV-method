# Fixed Split Size Selection

The fixed split size is 128, matching the existing PatternKV group size, residual chunk length, and recent window alignment. This avoids importing unrelated page sizes from other runtimes and keeps the split natural for the current packed/pending/recent layout.
