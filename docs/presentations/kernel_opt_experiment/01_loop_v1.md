-----------------

# Loop V1


## Loop V1

:::::::::::::: {.columns}
::: {.column width="60%"}

~~~ { #large-code-block .cpp}
...
#if defined(GATE_HALF_LOOP_V1)
    // - pipelining at outer-most loop
    // - branch to skip null computation
    // - branch to seperate result block
    // - single accumulator
    rslt_v = zeros<float, VECTOR_LANES>();
    for (int row = 0; row < LSTM_UNITS / 2; row++)
        chess_prepare_for_pipelining {
            if (partial_computation && (row > (LSTM_UNITS_REAL - LSTM_UNITS / 2))) {
                rslt = 0.0f;
            } else {
                acc1 = zeros<accfloat, VECTOR_LANES>();

                // Multiply and accumulate with state weight matrix.
                for (int i = 0; i < state_blocks; i++) {
                    va = *w_state_iter++;
                    vb = *state_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Mutiply and accumulate with input-weight matrix.
                for (int i = 0; i < input_blocks; i++) {
                    va = *w_inp_iter++;
                    vb = *inp_iter++;
                    acc1 = mac(acc1, va, vb);
                }

                // Sum all of the accumulator lanes.
                rslt = reduce_add(acc1.to_vector<float>(0));

                // Reset input and state iterators.
                state_iter -= state_blocks;
                inp_iter -= input_blocks;
            }

            // Push the result scalar into the vector.
            rslt_v.push(rslt);

            // If result vector is full, publish it to output.
            if ((row % VECTOR_LANES) == (VECTOR_LANES - 1)) {
                rslt_v = reverse(rslt_v);
                // Add bias to generate one block of gate results.
                rslt_v = add(*b_iter++, rslt_v);
                *out_iter++ = rslt_v;
            }
        }
...
~~~

:::

::: {.column width="40%"}

- 2 branches
- Pipelining fails
  - No 'HW do-loop' in log

:::

::::::::::::::


---

## Cycle count

```
Loop cycle count: 5221
Loop cycle count: 5227
Loop cycle count: 5685
Loop cycle count: 5687
Loop cycle count: 7149
Loop cycle count: 7157
Loop cycle count: 8085
Loop cycle count: 8086

```

| Loop | Min | Max | Avg | Std |
|------|-----|-----|-----|-----|
| loop_v1 | 5221 | 8086 | 6327.9 | 1057.8 |

---


## Performance Analysis

- Simulation AIE clock speed: 1.25GHz
- Avg. time taken (nsec): $t = 6327.9 / 1.25$
- Avg. performance per gate (GMAC/s): $P = 10240 / t = 2.02$
- Theoretical maximum: 8 GMAC/s
- Percent core utilization: 25.28%

## Trace

![Timing graph](./results/loop_v1/timing.png)

## Report


- [Kernel compilation log](results/loop_v1/24_4.log)
- [Simulator log](results/loop_v1/AIESimulator.log)
- [Micro-code listing](results/loop_v1/gate_half_buff.o.lst)
- [Report dir](results/loop_v1/)


