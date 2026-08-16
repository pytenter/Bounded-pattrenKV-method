# Recent-K Transition Contract

Implementation is append-until-full then shift. For request r at step t: `new_recent = old_recent + current_k` while valid_len < 128; once full, `new_recent = concat(old_recent[:, :, 1:, :], current_k)`. Logical order is oldest to newest.
