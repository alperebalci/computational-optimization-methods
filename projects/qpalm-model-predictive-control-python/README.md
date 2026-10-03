# QPALM Model Predictive Control QP

A Python example using **QPALM** through the `qpsolvers` interface to solve a condensed finite-horizon model-predictive-control quadratic program.

## System

```text
x[k+1] = a x[k] + b u[k]
```

The controller minimizes state deviation and control effort over a finite horizon while enforcing bounds on both predicted states and inputs.

## QP

After eliminating the state recursion, the optimization variable is the input sequence `u`:

```text
minimize    0.5 u' P u + q' u
subject to  G u <= h
            lower_u <= u <= upper_u
```

The state constraints are embedded into `G u <= h` through the prediction matrix.

## Install and run

```bash
python -m pip install -r requirements.txt
python mpc.py
```

The script prints the first control action, complete control trajectory, predicted state trajectory, and maximum state-bound violation.

## Why QPALM

QPALM implements a proximal augmented-Lagrangian method for quadratic programs, including nonconvex QPs. This project uses its convex-QP path in a control workload where sparse/structured QP methods are natural.

## Notes

The dynamics are synthetic and intentionally small enough to inspect directly.
