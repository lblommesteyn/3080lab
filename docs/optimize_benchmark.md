# `3080lab optimize` benchmark

int4 GEMV kernels (lab/experiments/gemv3.py). Times: 20%-trimmed mean of the in-kernel span over 60 interleaved launches (seed 1; seed 2 checks correctness). Speedup = ptxas time / arm time.

**Correctness failures: 0**

Noise floor (max deviation between byte-identical binaries, 20%-trimmed mean): 8-20us 1.2% (n=29), <8us 7.9% (n=42), >20us 1.5% (n=68).

## suite A (v1 held out; v3 revised on A, post hoc): 57 kernels

| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |
|---|---|---|---|---|---|
| srcfix | 38 | 1.098 | 1.721 | 0.774 | 6 |
| blind | 56 | 1.036 | 1.571 | 0.974 | 1 |
| guided_v1 | 56 | 1.037 | 1.565 | 0.970 | 1 |
| guided_v3 | 56 | 1.036 | 1.576 | 0.948 | 1 |
| blindD | 56 | 1.027 | 1.567 | 0.948 | 3 |
| guided_v3D | 56 | 1.027 | 1.573 | 0.948 | 2 |

- model v1: decision accuracy 14/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 20 neutral); wrongly applied 1, wrongly skipped 0; speedup prediction error median 10.9%, max 56.9%
- model v2: decision accuracy 14/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 20 neutral); wrongly applied 1, wrongly skipped 0; speedup prediction error median 10.9%, max 56.9%
- model v3: decision accuracy 14/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 20 neutral); wrongly applied 1, wrongly skipped 0; speedup prediction error median 4.1%, max 44.2%

| policy | geomean regret vs oracle | worst outcome |
|---|---|---|
| never rewrite | 3.49% | 1.000 |
| blind (hoist) | 0.07% | 0.974 |
| guided v1 (hoist) | -0.07% | 1.000 |
| guided v3 (hoist) | -0.01% | 1.000 |
| ablation: blind hoist+dedicate | 0.74% | 0.954 |

| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |
|---|---|---|---|---|---|---|---|---|
| o15/R8U2/u_major/default | 6.14 | 1.096 | 1.000 | 1.000 | 1.000 | 1.000 | 1.015 / 1.000 | 8 |
| o15/R4U2/u_major/default | 5.12 | 0.896 | 1.029 | 1.011 | 1.000 | 1.000 | 1.212 / 1.000 | 4 |
| o15/R4U1/u_major/default | 4.64 |  | 0.982 | 0.970 | 0.948 | 0.948 |  /  | 0 |
| o15/R8U2/lane_contig/default | 5.72 |  | 1.036 | 0.990 | 0.985 | 1.020 |  /  | 0 |
| o15/R4U2/r_major/default | 5.03 | 0.851 | 1.000 | 1.047 | 1.054 | 1.029 | 1.020 / 1.002 | 1 |
| gu15/R8U2/u_major/default | 33.79 | 1.378 | 1.110 | 1.100 | 1.102 | 1.100 | 1.742 / 1.601 | 8 |
| gu15/R4U2/u_major/default | 30.98 | 1.175 | 1.111 | 1.102 | 1.104 | 1.080 | 1.555 / 1.392 | 4 |
| gu15/R4U1/u_major/default | 24.58 |  | 1.000 | 1.002 | 1.012 | 1.006 |  /  | 0 |
| gu15/R8U2/lane_contig/default | 24.35 |  | 0.994 | 1.002 | 0.993 | 0.995 |  /  | 0 |
| gu15/R4U2/r_major/default | 29.04 | 1.100 | 1.013 | 1.013 | 1.013 | 1.013 | 1.091 / 1.057 | 1 |
| dn15/R8U2/u_major/default | 21.87 | 1.021 | 0.991 | 1.015 | 1.000 | 0.999 | 1.015 / 1.000 | 8 |
| dn15/R4U2/u_major/default | 20.62 | 0.884 | 1.010 | 1.017 | 1.003 | 1.021 | 1.212 / 1.000 | 4 |
| dn15/R4U1/u_major/default | 18.43 |  | 1.000 | 1.003 | 1.005 | 1.000 |  /  | 0 |
| dn15/R8U2/lane_contig/default | 21.42 |  | 0.997 | 0.996 | 1.011 | 0.996 |  /  | 0 |
| dn15/R4U2/r_major/default | 19.46 | 0.831 | 1.000 | 1.000 | 1.000 | 1.000 | 1.020 / 1.002 | 1 |
| qo7/R8U2/u_major/default | 15.36 | 1.154 | 1.000 | 1.000 | 1.011 | 1.004 | 1.472 / 1.184 | 8 |
| qo7/R4U2/u_major/default | 15.36 | 1.063 | 1.019 | 1.049 | 1.034 | 1.019 | 1.343 / 1.134 | 4 |
| qo7/R4U1/u_major/default | 13.31 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| qo7/R8U2/lane_contig/default | 13.31 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| qo7/R4U2/r_major/default | 14.59 | 1.008 | 0.987 | 0.994 | 0.988 | 1.006 | 1.060 / 1.025 | 1 |
| gu7/R8U2/u_major/default | 94.29 | 1.720 | 1.504 | 1.510 | 1.508 | 1.503 | 1.742 / 1.604 | 8 |
| gu7/R4U2/u_major/default | 83.11 | 1.412 | 1.376 | 1.373 | 1.378 | 1.256 | 1.555 / 1.393 | 4 |
| gu7/R4U1/u_major/default | 55.18 |  | 0.998 | 0.998 | 0.998 | 0.998 |  /  | 0 |
| gu7/R8U2/lane_contig/default | 54.93 |  | 0.999 | 0.999 | 0.997 | 1.003 |  /  | 0 |
| gu7/R4U2/r_major/default | 73.73 | 1.246 | 1.164 | 1.164 | 1.164 | 1.170 | 1.091 / 1.058 | 1 |
| dn7/R8U2/u_major/default | 66.84 | 1.124 | 1.040 | 1.042 | 1.044 | 1.034 | 1.472 / 1.187 | 8 |
| dn7/R4U2/u_major/default | 67.87 | 1.036 | 1.059 | 1.057 | 1.056 | 0.991 | 1.343 / 1.136 | 4 |
| dn7/R4U1/u_major/default | 57.63 |  | 1.001 | 1.003 | 1.005 | 1.004 |  /  | 0 |
| dn7/R8U2/lane_contig/default | 59.39 |  | 1.001 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| dn7/R4U2/r_major/default | 65.59 | 1.003 | 1.028 | 1.031 | 1.030 | 1.028 | 1.060 / 1.025 | 1 |
| sq4k/R8U2/u_major/default | 18.43 | 1.200 | 1.009 | 1.020 | 1.011 | 1.014 | 1.517 / 1.226 | 8 |
| sq4k/R4U2/u_major/default | 18.18 | 1.092 | 1.044 | 1.041 | 1.039 | 0.989 | 1.395 / 1.163 | 4 |
| sq4k/R4U1/u_major/default | 14.56 |  | 0.990 | 0.996 | 0.994 | 0.988 |  /  | 0 |
| sq4k/R8U2/lane_contig/default | 15.36 |  | 1.000 | 1.000 | 1.000 | 1.000 |  /  | 0 |
| sq4k/R4U2/r_major/default | 17.41 | 1.034 | 1.000 | 1.000 | 1.000 | 1.000 | 1.069 / 1.031 | 1 |
| up11k/R8U2/u_major/default | 70.09 | 1.721 | 1.571 | 1.565 | 1.576 | 1.567 | 1.742 / 1.602 | 8 |
| up11k/R4U2/u_major/default | 58.77 | 1.432 | 1.280 | 1.275 | 1.275 | 1.177 | 1.555 / 1.392 | 4 |
| up11k/R4U1/u_major/default | 38.66 |  | 1.002 | 1.006 | 0.999 | 0.997 |  /  | 0 |
| up11k/R8U2/lane_contig/default | 40.65 |  | 0.999 | 1.000 | 0.993 | 0.999 |  /  | 0 |
| up11k/R4U2/r_major/default | 52.39 | 1.274 | 1.146 | 1.145 | 1.148 | 1.144 | 1.091 / 1.058 | 1 |
| kv7/R8U2/u_major/default | 9.56 | 1.037 | 0.974 | 1.009 | 1.003 | 1.015 | 1.015 / 1.000 | 8 |
| kv7/R4U2/u_major/default | 7.17 | 0.866 | 1.000 | 1.000 | 1.000 | 0.984 | 1.310 / 1.000 | 4 |
| kv7/R4U1/u_major/default | 6.97 |  | 1.029 | 1.029 | 1.008 | 1.012 |  /  | 0 |
| kv7/R8U2/lane_contig/default | 9.22 |  | 1.000 | 0.997 | 1.000 | 1.000 |  /  | 0 |
| kv7/R4U2/r_major/default | 6.34 | 0.774 | 0.996 | 0.996 | 1.009 | 1.005 | 0.998 / 1.000 | 1 |
| gu15/R8U2/u_major/lb8 | 77.65 | 1.368 | 0.999 | 1.000 | 1.003 | 1.002 |  /  | 8 |
| gu15/R8U2/u_major/O1 | 33.79 | 1.122 | 1.000 | 1.000 | 1.001 | 1.000 | 1.171 / 1.061 | 8 |
| gu15/R4U2/u_major/lb8 | 38.34 | 1.385 | 1.246 | 1.248 | 1.248 | 1.163 | 1.174 / 1.144 | 4 |
| gu15/R4U2/u_major/O1 | 33.91 | 1.121 | 1.003 | 1.003 | 1.003 | 1.003 | 1.269 / 1.190 | 4 |
| qo7/R8U2/u_major/lb8 | 19.06 | 1.095 | 1.012 | 1.001 | 0.997 | 1.001 |  /  | 8 |
| qo7/R8U2/u_major/O1 | 17.46 | 1.003 | 1.003 | 1.003 | 1.002 | 0.972 | 1.071 / 1.016 | 8 |
| qo7/R4U2/u_major/lb8 | 16.38 | 1.036 | 1.000 | 1.000 | 1.000 | 0.988 | 1.001 / 0.959 | 4 |
| qo7/R4U2/u_major/O1 | 16.38 | 1.071 | 1.000 | 1.002 | 1.005 | 1.005 | 1.258 / 1.154 | 4 |
| sq4k/R8U2/u_major/lb8 | 26.62 | 1.243 | 1.000 | 1.001 | 1.000 | 1.001 |  /  | 8 |
| sq4k/R8U2/u_major/O1 | 20.48 | 1.004 | 1.007 | 1.000 | 1.000 | 1.000 | 1.088 / 1.024 | 8 |
| sq4k/R4U2/u_major/lb8 | 19.46 | 1.075 | 1.000 | 1.000 | 1.000 | 0.954 | 1.023 / 0.959 | 4 |
| sq4k/R4U2/u_major/O1 | 19.46 | 1.059 | 1.000 | 1.000 | 1.003 | 1.000 | 1.266 / 1.177 | 4 |

## suite B (v1 and v3 both held out): 43 kernels

| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |
|---|---|---|---|---|---|
| srcfix | 29 | 1.188 | 1.856 | 0.890 | 5 |
| blind | 43 | 1.074 | 1.619 | 0.964 | 1 |
| guided_v1 | 43 | 1.075 | 1.616 | 0.932 | 1 |
| guided_v3 | 43 | 1.072 | 1.619 | 0.979 | 1 |
| blindD | 43 | 1.065 | 1.617 | 0.974 | 4 |
| guided_v3D | 43 | 1.064 | 1.614 | 0.969 | 4 |

- model v1: decision accuracy 14/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 12 neutral); wrongly applied 1, wrongly skipped 0; speedup prediction error median 14.8%, max 39.4%
- model v2: decision accuracy 14/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 12 neutral); wrongly applied 1, wrongly skipped 0; speedup prediction error median 14.8%, max 39.4%
- model v3: decision accuracy 13/15 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 12 neutral); wrongly applied 1, wrongly skipped 1; speedup prediction error median 4.9%, max 20.5%

| policy | geomean regret vs oracle | worst outcome |
|---|---|---|
| never rewrite | 7.38% | 1.000 |
| blind (hoist) | 0.04% | 0.984 |
| guided v1 (hoist) | 0.02% | 1.000 |
| guided v3 (hoist) | 0.44% | 0.982 |
| ablation: blind hoist+dedicate | 0.74% | 0.979 |

| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |
|---|---|---|---|---|---|---|---|---|
| qo3/R8U2/u_major/default | 7.17 | 1.167 | 1.000 | 1.000 | 1.000 | 1.000 | 1.230 / 1.067 | 8 |
| qo3/R4U2/u_major/default | 6.14 | 0.982 | 1.000 | 1.000 | 1.000 | 1.000 | 1.166 / 1.049 | 4 |
| qo3/R4U1/u_major/default | 5.66 |  | 1.015 | 1.021 | 1.021 | 0.990 |  /  | 0 |
| qo3/R8U2/lane_contig/default | 6.14 |  | 1.000 | 1.000 | 1.000 | 0.995 |  /  | 0 |
| qo3/R4U2/r_major/default | 6.14 | 0.995 | 1.000 | 1.000 | 1.000 | 1.000 | 1.031 / 1.008 | 1 |
| gu3/R8U2/u_major/default | 67.58 | 1.833 | 1.473 | 1.467 | 1.469 | 1.467 | 1.742 / 1.602 | 8 |
| gu3/R4U2/u_major/default | 58.37 | 1.496 | 1.425 | 1.425 | 1.426 | 1.390 | 1.555 / 1.392 | 4 |
| gu3/R4U1/u_major/default | 35.67 |  | 1.002 | 1.002 | 1.005 | 1.010 |  /  | 0 |
| gu3/R8U2/lane_contig/default | 36.86 |  | 1.000 | 1.000 | 1.002 | 1.000 |  /  | 0 |
| gu3/R4U2/r_major/default | 55.81 | 1.431 | 1.330 | 1.329 | 1.331 | 1.329 | 1.091 / 1.058 | 1 |
| dn3/R8U2/u_major/default | 28.67 | 1.068 | 1.000 | 1.000 | 1.000 | 1.000 | 1.230 / 1.071 | 8 |
| dn3/R4U2/u_major/default | 27.28 | 0.940 | 1.040 | 1.045 | 1.042 | 0.991 | 1.166 / 1.052 | 4 |
| dn3/R4U1/u_major/default | 23.01 |  | 0.995 | 1.001 | 1.005 | 1.004 |  /  | 0 |
| dn3/R8U2/lane_contig/default | 26.85 |  | 1.005 | 0.998 | 0.999 | 1.000 |  /  | 0 |
| dn3/R4U2/r_major/default | 25.60 | 0.890 | 1.000 | 1.000 | 1.000 | 1.000 | 1.031 / 1.008 | 1 |
| gu8/R8U2/u_major/default | 168.33 | 1.856 | 1.619 | 1.616 | 1.619 | 1.617 | 1.742 / 1.605 | 8 |
| gu8/R4U2/u_major/default | 162.13 | 1.684 | 1.588 | 1.584 | 1.588 | 1.404 | 1.555 / 1.394 | 4 |
| gu8/R4U1/u_major/default | 90.06 |  | 1.001 | 1.000 | 1.001 | 1.000 |  /  | 0 |
| gu8/R8U2/lane_contig/default | 90.62 |  | 0.998 | 0.997 | 0.997 | 0.998 |  /  | 0 |
| gu8/R4U2/r_major/default | 143.56 | 1.492 | 1.243 | 1.244 | 1.242 | 1.242 | 1.091 / 1.058 | 1 |
| dn8/R8U2/u_major/default | 57.77 | 1.206 | 1.088 | 1.087 | 1.090 | 1.085 | 1.517 / 1.229 | 8 |
| dn8/R4U2/u_major/default | 59.62 | 1.173 | 1.135 | 1.142 | 1.143 | 1.068 | 1.395 / 1.165 | 4 |
| dn8/R4U1/u_major/default | 46.68 |  | 0.994 | 0.995 | 0.998 | 0.995 |  /  | 0 |
| dn8/R8U2/lane_contig/default | 47.79 |  | 0.999 | 0.999 | 0.998 | 0.998 |  /  | 0 |
| dn8/R4U2/r_major/default | 59.59 | 1.176 | 1.112 | 1.107 | 1.112 | 1.112 | 1.069 / 1.031 | 1 |
| sq1k/R8U2/u_major/default | 5.40 | 1.011 | 0.964 | 1.033 | 0.979 | 0.974 | 1.015 / 1.000 | 8 |
| sq1k/R4U2/u_major/default | 4.10 | 1.000 | 1.007 | 1.059 | 1.043 | 1.000 | 1.310 / 1.000 | 4 |
| sq1k/R4U1/u_major/default | 2.73 |  | 1.011 | 0.932 | 1.079 | 0.980 |  /  | 0 |
| sq1k/R8U2/lane_contig/default | 5.26 |  | 0.989 | 1.005 | 0.989 | 1.000 |  /  | 0 |
| sq1k/R4U2/r_major/default | 3.73 | 0.910 | 1.023 | 1.016 | 1.016 | 1.074 | 0.998 / 1.000 | 1 |
| sq2k/R8U2/u_major/default | 7.17 | 1.167 | 1.000 | 1.000 | 0.992 | 1.000 | 1.230 / 1.067 | 8 |
| sq2k/R4U2/u_major/default | 6.14 | 0.964 | 1.000 | 1.000 | 1.000 | 1.000 | 1.166 / 1.049 | 4 |
| sq2k/R4U1/u_major/default | 5.86 |  | 0.990 | 1.005 | 0.981 | 0.976 |  /  | 0 |
| sq2k/R8U2/lane_contig/default | 6.14 |  | 1.000 | 0.995 | 0.991 | 1.000 |  /  | 0 |
| sq2k/R4U2/r_major/default | 6.14 | 0.973 | 1.000 | 1.000 | 1.000 | 1.000 | 1.031 / 1.008 | 1 |
| gu3/R8U2/u_major/lb8 | 109.51 | 1.409 | 1.001 | 0.998 | 1.000 | 0.997 |  /  | 8 |
| gu3/R8U2/u_major/O1 | 49.61 | 1.025 | 1.020 | 1.022 | 1.015 | 1.020 | 1.171 / 1.061 | 8 |
| gu3/R4U2/u_major/lb8 | 65.39 | 1.576 | 1.377 | 1.382 | 1.375 | 1.330 | 1.174 / 1.144 | 4 |
| gu3/R4U2/u_major/O1 | 50.74 | 1.101 | 1.011 | 1.011 | 1.010 | 0.999 | 1.269 / 1.190 | 4 |
| dn8/R8U2/u_major/lb8 | 99.10 | 1.376 | 1.002 | 1.003 | 1.003 | 0.998 |  /  | 8 |
| dn8/R8U2/u_major/O1 | 60.30 | 0.983 | 0.984 | 0.993 | 0.982 | 0.979 | 1.088 / 1.024 | 8 |
| dn8/R4U2/u_major/lb8 | 66.73 | 1.237 | 1.172 | 1.177 | 1.005 | 1.187 | 1.023 / 0.958 | 4 |
| dn8/R4U2/u_major/O1 | 59.65 | 1.130 | 1.015 | 1.014 | 1.013 | 1.000 | 1.266 / 1.179 | 4 |

## suite C: high register pressure (v1 and v3 held out): 20 kernels

| arm | kernels | geomean speedup | best | worst | regressions (<0.98) |
|---|---|---|---|---|---|
| srcfix | 10 | 1.369 | 1.744 | 1.005 | 0 |
| blind | 20 | 1.055 | 1.327 | 0.812 | 3 |
| guided_v1 | 20 | 1.053 | 1.327 | 0.811 | 4 |
| guided_v3 | 20 | 1.050 | 1.327 | 0.811 | 3 |

- model v1: decision accuracy 6/9 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 1 neutral); wrongly applied 3, wrongly skipped 0; speedup prediction error median 33.7%, max 57.0%
- model v3: decision accuracy 5/9 on kernels where the rewrite measurably helps or hurts (beyond the identical-binary noise band; 1 neutral); wrongly applied 3, wrongly skipped 1; speedup prediction error median 24.0%, max 47.2%

| policy | geomean regret vs oracle | worst outcome |
|---|---|---|
| never rewrite | 7.13% | 1.000 |
| blind (hoist) | 1.37% | 0.812 |
| guided v1 (hoist) | 1.57% | 0.811 |
| guided v3 (hoist) | 1.95% | 0.811 |

| kernel | ptxas us | srcfix | blind (hoist) | guided v1 | guided v3 | ablation: hoist+dedicate | predicted v1 / v3 | split pairs |
|---|---|---|---|---|---|---|---|---|
| gu3/R16U2/u_major/default | 69.32 | 1.596 | 1.327 | 1.327 | 1.327 |  | 1.767 / 1.770 | 16 |
| gu3/R8U4/u_major/default | 75.63 | 1.442 | 1.288 | 1.288 | 1.286 |  | 1.139 / 1.194 | 24 |
| gu3/R16U1/u_major/default | 38.60 |  | 1.004 | 0.999 | 1.002 |  |  /  | 0 |
| gu3/R16U2/lane_contig/default | 43.01 |  | 0.997 | 0.995 | 0.999 |  |  /  | 0 |
| up11k/R16U2/u_major/default | 74.67 | 1.744 | 1.316 | 1.308 | 1.312 |  | 1.767 / 1.627 | 16 |
| up11k/R8U4/u_major/default | 81.10 | 1.685 | 1.255 | 1.251 | 1.248 |  | 1.139 / 1.194 | 24 |
| up11k/R16U1/u_major/default | 40.90 |  | 1.001 | 0.999 | 0.999 |  |  /  | 0 |
| up11k/R16U2/lane_contig/default | 42.21 |  | 0.993 | 0.993 | 0.999 |  |  /  | 0 |
| qo7/R16U2/u_major/default | 21.50 | 1.091 | 0.958 | 0.955 | 0.955 |  | 1.499 / 1.192 | 16 |
| qo7/R8U4/u_major/default | 17.21 | 1.159 | 0.989 | 0.989 | 0.989 |  | 1.022 / 0.995 | 24 |
| qo7/R16U1/u_major/default | 27.65 |  | 1.000 | 1.000 | 1.000 |  |  /  | 0 |
| qo7/R16U2/lane_contig/default | 19.48 |  | 0.994 | 0.972 | 0.981 |  |  /  | 0 |
| dn8/R16U2/u_major/default | 66.56 | 1.112 | 0.979 | 0.979 | 0.977 |  | 1.537 / 1.239 | 16 |
| dn8/R8U4/u_major/default | 83.40 | 1.537 | 1.080 | 1.083 | 1.002 |  | 1.023 / 0.995 | 24 |
| dn8/R16U1/u_major/default | 94.46 |  | 0.995 | 0.996 | 0.998 |  |  /  | 0 |
| dn8/R16U2/lane_contig/default | 59.65 |  | 1.000 | 0.998 | 0.999 |  |  /  | 0 |
| gu15/R16U2/u_major/default | 53.28 | 1.578 | 1.302 | 1.301 | 1.301 |  | 1.767 / 1.769 | 16 |
| gu15/R8U4/u_major/default | 37.86 | 1.005 | 0.812 | 0.811 | 0.811 |  | 1.139 / 1.194 | 24 |
| gu15/R16U1/u_major/default | 27.31 |  | 1.002 | 0.991 | 0.998 |  |  /  | 0 |
| gu15/R16U2/lane_contig/default | 33.85 |  | 0.992 | 1.003 | 0.998 |  |  /  | 0 |

