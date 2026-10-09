# `3080lab optimize` benchmark

int4 GEMV kernels (lab/experiments/gemv3.py). Times: 20%-trimmed mean of the in-kernel span over 60 interleaved launches (seed 1; seed 2 checks correctness). Speedup = ptxas time / arm time.

**Correctness failures: 0**

Noise floor (max deviation between byte-identical binaries, 20%-trimmed mean): 8-20us 2.6% (n=29), <8us 5.0% (n=42), >20us 1.2% (n=68).

## suite A (v1 held out; v3 revised on A, post hoc): 57 kernels

| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |
|---|---|---|---|---|---|
| srcfix | 38 | 1.108 | 1.701 | 0.856 | 7 |
| blind | 56 | 1.033 | 1.571 | 0.926 | 5 |
| guided_v1 | 56 | 1.034 | 1.577 | 0.926 | 6 |
| guided_v3 | 56 | 1.034 | 1.577 | 0.926 | 5 |
| blindD | 56 | 1.023 | 1.567 | 0.926 | 8 |
| guided_v3D | 56 | 1.023 | 1.571 | 0.926 | 7 |

- model v1: decision accuracy 16/19 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 16 neutral); wrongly applied 3, wrongly skipped 0; speedup prediction error median 12.1%, max 56.9%
- model v2: decision accuracy 16/19 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 16 neutral); wrongly applied 3, wrongly skipped 0; speedup prediction error median 12.1%, max 56.9%
- model v3: decision accuracy 15/19 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 16 neutral); wrongly applied 3, wrongly skipped 1; speedup prediction error median 4.0%, max 44.2%

| policy | geomean regret vs oracle | worst outcome |
|---|---|---|
| never rewrite | 3.61% | 1.000 |
| blind (hoist) | 0.36% | 0.926 |
| guided v1 (hoist) | 0.29% | 0.926 |
| guided v3 (hoist) | 0.31% | 0.926 |
| ablation: blind hoist+dedicate | 1.23% | 0.926 |

| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |
|---|---|---|---|---|---|---|---|---|
| o15/R8U2/u_major/default | 5.12 | 1.000 | 1.000 | 0.973 | 1.000 | 0.994 | 1.015 / 1.000 | 8 |
| o15/R4U2/u_major/default | 4.10 | 0.878 | 1.000 | 1.000 | 1.000 | 1.000 | 1.212 / 1.000 | 4 |
| o15/R4U1/u_major/default | 4.10 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| o15/R8U2/lane_contig/default | 5.18 |  | 1.006 | 1.000 | 0.984 | 1.000 |  /  | 0 |
| o15/R4U2/r_major/default | 4.10 | 0.923 | 1.000 | 1.000 | 1.000 | 1.000 | 1.020 / 1.002 | 1 |
| gu15/R8U2/u_major/default | 34.10 | 1.388 | 1.110 | 1.115 | 1.110 | 1.110 | 1.742 / 1.601 | 8 |
| gu15/R4U2/u_major/default | 31.74 | 1.194 | 1.134 | 1.118 | 1.116 | 1.107 | 1.555 / 1.392 | 4 |
| gu15/R4U1/u_major/default | 24.58 |  | 1.001 | 1.005 | 1.000 | 1.000 |  /  | 0 |
| gu15/R8U2/lane_contig/default | 24.58 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| gu15/R4U2/r_major/default | 29.38 | 1.104 | 1.025 | 1.025 | 1.025 | 1.025 | 1.091 / 1.057 | 1 |
| dn15/R8U2/u_major/default | 21.56 | 1.034 | 0.983 | 1.003 | 1.003 | 1.003 | 1.015 / 1.000 | 8 |
| dn15/R4U2/u_major/default | 20.68 | 0.918 | 1.030 | 1.036 | 0.996 | 1.014 | 1.212 / 1.000 | 4 |
| dn15/R4U1/u_major/default | 18.20 |  | 0.989 | 1.003 | 0.995 | 0.992 |  /  | 0 |
| dn15/R8U2/lane_contig/default | 20.79 |  | 1.004 | 0.996 | 0.996 | 0.987 |  /  | 0 |
| dn15/R4U2/r_major/default | 19.46 | 0.864 | 1.000 | 1.000 | 1.000 | 1.000 | 1.020 / 1.002 | 1 |
| qo7/R8U2/u_major/default | 14.08 | 1.146 | 0.982 | 0.956 | 0.946 | 0.938 | 1.472 / 1.184 | 8 |
| qo7/R4U2/u_major/default | 14.34 | 1.016 | 1.000 | 1.000 | 1.000 | 0.973 | 1.343 / 1.134 | 4 |
| qo7/R4U1/u_major/default | 13.26 |  | 1.015 | 1.004 | 1.024 | 1.009 |  /  | 0 |
| qo7/R8U2/lane_contig/default | 12.63 |  | 1.002 | 0.998 | 0.993 | 0.978 |  /  | 0 |
| qo7/R4U2/r_major/default | 13.54 | 0.967 | 0.944 | 0.944 | 0.944 | 0.944 | 1.060 / 1.025 | 1 |
| gu7/R8U2/u_major/default | 94.72 | 1.721 | 1.510 | 1.514 | 1.512 | 1.511 | 1.742 / 1.604 | 8 |
| gu7/R4U2/u_major/default | 83.57 | 1.407 | 1.387 | 1.385 | 1.390 | 1.256 | 1.555 / 1.393 | 4 |
| gu7/R4U1/u_major/default | 55.32 |  | 1.001 | 0.998 | 1.000 | 0.999 |  /  | 0 |
| gu7/R8U2/lane_contig/default | 54.98 |  | 0.998 | 1.000 | 0.996 | 0.998 |  /  | 0 |
| gu7/R4U2/r_major/default | 73.96 | 1.246 | 1.163 | 1.169 | 1.166 | 1.165 | 1.091 / 1.058 | 1 |
| dn7/R8U2/u_major/default | 66.73 | 1.124 | 1.034 | 1.035 | 1.036 | 1.034 | 1.472 / 1.187 | 8 |
| dn7/R4U2/u_major/default | 68.07 | 1.041 | 1.068 | 1.067 | 1.072 | 0.998 | 1.343 / 1.136 | 4 |
| dn7/R4U1/u_major/default | 57.34 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| dn7/R8U2/lane_contig/default | 59.65 |  | 1.003 | 1.004 | 1.004 | 1.005 |  /  | 0 |
| dn7/R4U2/r_major/default | 65.54 | 1.003 | 1.029 | 1.027 | 1.027 | 1.023 | 1.060 / 1.025 | 1 |
| sq4k/R8U2/u_major/default | 18.01 | 1.208 | 1.014 | 1.034 | 1.028 | 1.026 | 1.517 / 1.226 | 8 |
| sq4k/R4U2/u_major/default | 18.35 | 1.124 | 1.045 | 1.045 | 1.029 | 0.995 | 1.395 / 1.163 | 4 |
| sq4k/R4U1/u_major/default | 14.34 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| sq4k/R8U2/lane_contig/default | 14.90 |  | 1.019 | 1.023 | 1.023 | 1.019 |  /  | 0 |
| sq4k/R4U2/r_major/default | 17.44 | 1.066 | 1.002 | 1.002 | 1.002 | 1.002 | 1.069 / 1.031 | 1 |
| up11k/R8U2/u_major/default | 69.69 | 1.701 | 1.571 | 1.577 | 1.577 | 1.567 | 1.742 / 1.602 | 8 |
| up11k/R4U2/u_major/default | 59.48 | 1.434 | 1.298 | 1.300 | 1.321 | 1.209 | 1.555 / 1.392 | 4 |
| up11k/R4U1/u_major/default | 38.57 |  | 1.005 | 1.012 | 1.002 | 1.003 |  /  | 0 |
| up11k/R8U2/lane_contig/default | 40.96 |  | 1.000 | 0.999 | 0.999 | 1.000 |  /  | 0 |
| up11k/R4U2/r_major/default | 51.31 | 1.242 | 1.119 | 1.118 | 1.125 | 1.121 | 1.091 / 1.058 | 1 |
| kv7/R8U2/u_major/default | 8.59 | 1.060 | 0.977 | 0.990 | 1.017 | 1.049 | 1.015 / 1.000 | 8 |
| kv7/R4U2/u_major/default | 6.06 | 0.959 | 1.000 | 1.014 | 1.019 | 0.986 | 1.310 / 1.000 | 4 |
| kv7/R4U1/u_major/default | 5.26 |  | 1.028 | 0.995 | 1.022 | 0.995 |  /  | 0 |
| kv7/R8U2/lane_contig/default | 8.19 |  | 1.000 | 1.011 | 1.014 | 1.007 |  /  | 0 |
| kv7/R4U2/r_major/default | 5.26 | 0.856 | 1.016 | 1.028 | 0.995 | 1.016 | 0.998 / 1.000 | 1 |
| gu15/R8U2/u_major/lb8 | 78.45 | 1.349 | 1.004 | 1.005 | 1.001 | 1.001 |  /  | 8 |
| gu15/R8U2/u_major/O1 | 34.10 | 1.110 | 1.009 | 1.009 | 1.009 | 1.009 | 1.171 / 1.061 | 8 |
| gu15/R4U2/u_major/lb8 | 38.49 | 1.386 | 1.253 | 1.253 | 1.253 | 1.161 | 1.174 / 1.144 | 4 |
| gu15/R4U2/u_major/O1 | 34.25 | 1.123 | 1.011 | 1.013 | 1.013 | 0.999 | 1.269 / 1.190 | 4 |
| qo7/R8U2/u_major/lb8 | 19.46 | 1.093 | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 8 |
| qo7/R8U2/u_major/O1 | 15.87 | 1.033 | 0.991 | 1.013 | 1.002 | 0.984 | 1.071 / 1.016 | 8 |
| qo7/R4U2/u_major/lb8 | 15.96 | 1.039 | 0.974 | 0.974 | 0.974 | 0.974 | 1.001 / 0.959 | 4 |
| qo7/R4U2/u_major/O1 | 14.22 | 1.057 | 0.926 | 0.926 | 0.926 | 0.926 | 1.258 / 1.154 | 4 |
| sq4k/R8U2/u_major/lb8 | 28.47 | 1.314 | 1.005 | 1.000 | 1.002 | 1.006 |  /  | 8 |
| sq4k/R8U2/u_major/O1 | 19.11 | 1.024 | 0.996 | 0.997 | 1.007 | 0.994 | 1.088 / 1.024 | 8 |
| sq4k/R4U2/u_major/lb8 | 19.48 | 1.103 | 0.999 | 0.990 | 0.990 | 0.950 | 1.023 / 0.959 | 4 |
| sq4k/R4U2/u_major/O1 | 18.43 | 1.059 | 0.953 | 0.970 | 0.977 | 0.961 | 1.266 / 1.177 | 4 |

## suite B (v1 and v3 both held out): 43 kernels

| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |
|---|---|---|---|---|---|
| srcfix | 29 | 1.165 | 1.861 | 0.878 | 7 |
| blind | 43 | 1.074 | 1.624 | 0.970 | 1 |
| guided_v1 | 43 | 1.074 | 1.629 | 0.950 | 2 |
| guided_v3 | 43 | 1.067 | 1.634 | 0.961 | 4 |
| blindD | 43 | 1.064 | 1.629 | 0.939 | 3 |
| guided_v3D | 43 | 1.061 | 1.628 | 0.943 | 2 |

- model v1: decision accuracy 13/13 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 14 neutral); wrongly applied 0, wrongly skipped 0; speedup prediction error median 14.9%, max 43.9%
- model v2: decision accuracy 13/13 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 14 neutral); wrongly applied 0, wrongly skipped 0; speedup prediction error median 14.9%, max 43.9%
- model v3: decision accuracy 12/13 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 14 neutral); wrongly applied 0, wrongly skipped 1; speedup prediction error median 4.9%, max 21.6%

| policy | geomean regret vs oracle | worst outcome |
|---|---|---|
| never rewrite | 7.39% | 1.000 |
| blind (hoist) | 0.00% | 1.000 |
| guided v1 (hoist) | -0.02% | 1.000 |
| guided v3 (hoist) | 0.47% | 1.000 |
| ablation: blind hoist+dedicate | 0.68% | 0.939 |

| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |
|---|---|---|---|---|---|---|---|---|
| qo3/R8U2/u_major/default | 5.77 | 1.036 | 1.000 | 1.025 | 1.010 | 0.981 | 1.230 / 1.067 | 8 |
| qo3/R4U2/u_major/default | 5.03 | 0.983 | 1.017 | 1.035 | 1.006 | 1.029 | 1.166 / 1.049 | 4 |
| qo3/R4U1/u_major/default | 4.84 |  | 0.983 | 0.971 | 0.966 | 0.983 |  /  | 0 |
| qo3/R8U2/lane_contig/default | 5.86 |  | 1.020 | 1.010 | 0.986 | 1.025 |  /  | 0 |
| qo3/R4U2/r_major/default | 4.72 | 0.922 | 0.988 | 0.994 | 1.006 | 1.012 | 1.031 / 1.008 | 1 |
| gu3/R8U2/u_major/default | 67.53 | 1.832 | 1.472 | 1.469 | 1.466 | 1.470 | 1.742 / 1.602 | 8 |
| gu3/R4U2/u_major/default | 58.37 | 1.486 | 1.429 | 1.427 | 1.430 | 1.403 | 1.555 / 1.392 | 4 |
| gu3/R4U1/u_major/default | 35.84 |  | 1.000 | 1.000 | 1.000 | 1.001 |  /  | 0 |
| gu3/R8U2/lane_contig/default | 36.86 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| gu3/R4U2/r_major/default | 56.46 | 1.425 | 1.349 | 1.346 | 1.345 | 1.347 | 1.091 / 1.058 | 1 |
| dn3/R8U2/u_major/default | 28.67 | 1.064 | 1.000 | 1.000 | 1.000 | 1.000 | 1.230 / 1.071 | 8 |
| dn3/R4U2/u_major/default | 27.65 | 0.951 | 1.038 | 1.042 | 1.038 | 0.992 | 1.166 / 1.052 | 4 |
| dn3/R4U1/u_major/default | 23.27 |  | 0.998 | 1.001 | 1.011 | 1.004 |  /  | 0 |
| dn3/R8U2/lane_contig/default | 26.91 |  | 1.005 | 1.005 | 1.002 | 1.001 |  /  | 0 |
| dn3/R4U2/r_major/default | 25.63 | 0.878 | 0.993 | 0.999 | 0.998 | 0.996 | 1.031 / 1.008 | 1 |
| gu8/R8U2/u_major/default | 169.16 | 1.861 | 1.624 | 1.629 | 1.634 | 1.629 | 1.742 / 1.605 | 8 |
| gu8/R4U2/u_major/default | 161.19 | 1.675 | 1.579 | 1.591 | 1.583 | 1.410 | 1.555 / 1.394 | 4 |
| gu8/R4U1/u_major/default | 90.11 |  | 0.998 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| gu8/R8U2/lane_contig/default | 90.91 |  | 0.999 | 0.999 | 1.000 | 1.000 |  /  | 0 |
| gu8/R4U2/r_major/default | 143.08 | 1.486 | 1.243 | 1.239 | 1.241 | 1.246 | 1.091 / 1.058 | 1 |
| dn8/R8U2/u_major/default | 56.15 | 1.177 | 1.054 | 1.054 | 1.054 | 1.050 | 1.517 / 1.229 | 8 |
| dn8/R4U2/u_major/default | 59.62 | 1.174 | 1.159 | 1.155 | 1.152 | 1.078 | 1.395 / 1.165 | 4 |
| dn8/R4U1/u_major/default | 46.56 |  | 1.003 | 1.007 | 1.003 | 1.000 |  /  | 0 |
| dn8/R8U2/lane_contig/default | 47.45 |  | 1.000 | 1.002 | 0.998 | 0.994 |  /  | 0 |
| dn8/R4U2/r_major/default | 59.56 | 1.169 | 1.111 | 1.113 | 1.109 | 1.105 | 1.069 / 1.031 | 1 |
| sq1k/R8U2/u_major/default | 5.12 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.015 / 1.000 | 8 |
| sq1k/R4U2/u_major/default | 3.07 | 0.947 | 1.000 | 0.991 | 0.991 | 0.939 | 1.310 / 1.000 | 4 |
| sq1k/R4U1/u_major/default | 2.05 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| sq1k/R8U2/lane_contig/default | 5.12 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| sq1k/R4U2/r_major/default | 3.07 | 0.900 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 / 1.000 | 1 |
| sq2k/R8U2/u_major/default | 5.83 | 1.015 | 1.046 | 1.010 | 1.025 | 1.051 | 1.230 / 1.067 | 8 |
| sq2k/R4U2/u_major/default | 4.81 | 0.939 | 1.000 | 1.024 | 0.977 | 0.988 | 1.166 / 1.049 | 4 |
| sq2k/R4U1/u_major/default | 4.86 |  | 0.983 | 0.950 | 0.961 | 0.983 |  /  | 0 |
| sq2k/R8U2/lane_contig/default | 5.72 |  | 0.990 | 0.995 | 0.976 | 0.966 |  /  | 0 |
| sq2k/R4U2/r_major/default | 4.61 | 0.900 | 0.970 | 0.982 | 0.994 | 0.964 | 1.031 / 1.008 | 1 |
| gu3/R8U2/u_major/lb8 | 112.27 | 1.406 | 1.001 | 1.000 | 0.999 | 1.002 |  /  | 8 |
| gu3/R8U2/u_major/O1 | 49.49 | 1.022 | 1.005 | 1.002 | 1.000 | 0.993 | 1.171 / 1.061 | 8 |
| gu3/R4U2/u_major/lb8 | 65.82 | 1.568 | 1.381 | 1.380 | 1.381 | 1.368 | 1.174 / 1.144 | 4 |
| gu3/R4U2/u_major/O1 | 50.57 | 1.095 | 1.010 | 1.008 | 1.011 | 0.997 | 1.269 / 1.190 | 4 |
| dn8/R8U2/u_major/lb8 | 98.19 | 1.373 | 0.997 | 0.998 | 0.996 | 1.000 |  /  | 8 |
| dn8/R8U2/u_major/O1 | 60.44 | 0.987 | 0.998 | 0.996 | 0.996 | 0.991 | 1.088 / 1.024 | 8 |
| dn8/R4U2/u_major/lb8 | 67.33 | 1.253 | 1.202 | 1.206 | 1.001 | 1.195 | 1.023 / 0.958 | 4 |
| dn8/R4U2/u_major/O1 | 59.45 | 1.119 | 1.013 | 1.016 | 1.012 | 1.000 | 1.266 / 1.179 | 4 |

