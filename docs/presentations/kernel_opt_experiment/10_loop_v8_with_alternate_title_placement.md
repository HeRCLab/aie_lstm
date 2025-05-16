-----------------

# Loop V8 with Alt Tile Placement

## Loop V8 with Alt Tile Placement

- Kernel is same as V8
- Some tiles are moved back in south.


## Old Tile Placement

![Old Tile Placement](results/array.svg)

## New Tile Placement

![New Tile Placement](results/loop_v8_alt_tile_placement/array.svg) { width="50%" }


## Cycle count (Page 1)

~~~
Loop cycle count: 3267
Loop cycle count: 3269
Loop cycle count: 3267
Loop cycle count: 3294
Loop cycle count: 3525
Loop cycle count: 3526
Loop cycle count: 3844
Loop cycle count: 3877
~~~

## Cycle count (Page 2)

| Loop | Min | Max | Avg | Std |
|------|-----|-----|-----|-----|
| loop_v1 | 5221 | 8086 | 6327.9 | 1057.8 |
| loop_v2 | 6317 | 6525 | 6392.0 | 86.2 |
| loop_v3 | 5535 | 6473 | 5677.2 | 241.6 |
| loop_v4 | 4124 | 4854 | 4278.9 | 247.2 |
| loop_v5 | 4546 | 5900 | 5056.2 | 554.1 |

## Cycle count (Page 3)

| Loop | Min | Max | Avg | Std |
|------|-----|-----|-----|-----|
| loop_v6 | 4124 | 4854 | 4278.9 | 247.2 |
| loop_v7 | 4546 | 5900 | 5056.2 | 554.1 |
| loop_v8 | 3267 | 4623 | 3671.8 | 482.8 |
| loop_v8_buff_opt | 3267 | 4428 | 3554.0 | 420.7 |
| loop_v8_alt_tile | 3267 | 3877 | 3407.5 | 201.3 |

## Performance Analysis

- Simulation AIE clock speed: 1.25GHz
- Avg. time taken (nsec): $t = 3407.5 / 1.25$
- Avg. performance per gate (GMAC/s): $P = 10240 / t = 3.75$
- Theoretical maximum: 8 GMAC/s
- Percent core utilization: 46.95%
- Slight improvement on plain V8.
- Substantial reduction of inter-gate variability.

## Report

- [Kernel compilation log](results/loop_v8_alt_tile_placement/24_4.log)
- [Simulator log](results/loop_v8_alt_tile_placement/AIESimulator.log)
- [Micro-code listing](results/loop_v8_alt_tile_placement/gate_half_buff.o.lst)
- [Report dir](results/loop_v8_alt_tile_placement/)

## Trace

![Timing graph](./results/loop_v8_alt_tile_placement/timing.png)