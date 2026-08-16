# Current Reproduction

The 50a current worker does not reproduce the stored slow CAUSAL rows when run directly on GPU 1. It measures 164.628 ms/token at C2048/B1 and 171.212 ms/token at C4096/B1 with the frozen env. C4096/B8 passes with the frozen env and fails without the allocator setting.
