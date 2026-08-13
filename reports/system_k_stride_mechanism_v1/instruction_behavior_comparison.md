# Instruction Behavior Comparison

STATIC_CODE_EVIDENCE from `cuobjdump --dump-sass`:

| Kernel | Instructions | Integer add | Integer multiply/MAD | Constant refs | Global loads |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tight K INT2 | 3488 | 173 | 486 | 217 | 126 |
| Strided K INT2 | 5144 | 421 | 918 | 609 | 142 |
| Tight V2 | 2040 | 179 | 324 | 175 | 54 |
| Strided V2 | 2120 | 180 | 363 | 248 | 54 |

Address arithmetic hypothesis: SUPPORTED.

Strided K adds runtime stride multiplications for packed K, scale, zero, and assignment. SASS reflects this: K IMAD count rises sharply, while V2 strided is close to V2 tight.

Register/occupancy hypothesis: REJECTED by static resource evidence.

| Kernel | Registers/thread | Stack | Local |
| --- | ---: | ---: | ---: |
| Tight K INT2 | 69 | 0 | 0 |
| Strided K INT2 | 63 | 0 | 0 |
| Tight V2 | 103 | 0 | 0 |
| Strided V2 | 102 | 0 | 0 |

No stack or local spill is reported for tight or strided K.
