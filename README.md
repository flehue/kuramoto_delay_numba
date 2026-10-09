Kuramoto model with delays running natively on numba. Delays are implemented as integer-step lookback into the full phase history array; before a node's delay window has elapsed, missing history is clamped to the initial condition (equivalent to assuming the system sat frozen at t=0 for all t<0).

Extra fast:
Runs extra fast using a trigonometric identity that allows for 2N calculations inside the matrix instead of N^2

Uses triple memory, though.
