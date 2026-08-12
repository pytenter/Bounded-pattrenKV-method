# Approximate Amdahl Estimate

Profile-on component shares are used with profile-off TPOT, so these are approximate bounds.

## T=16384

- mixed_v_share_approx: `0.3202`
- cache_mutation_share_approx: `0.2201`
- if mixed-V were free: `1.471x`
- if cache mutation were free: `1.282x`
- if both were free: `2.175x`

## T=32768

- mixed_v_share_approx: `0.3976`
- cache_mutation_share_approx: `0.2046`
- if mixed-V were free: `1.660x`
- if cache mutation were free: `1.257x`
- if both were free: `2.514x`
