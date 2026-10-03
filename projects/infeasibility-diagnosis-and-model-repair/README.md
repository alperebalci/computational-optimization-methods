# Infeasibility Diagnosis and Model Repair

This project treats infeasibility as an Operations Research diagnosis problem rather than a terminal solver status.

For linear models in the form `A x <= b`, it implements three complementary tools.

## 1. Inclusion-minimal infeasible subsystem

A deletion filter repeatedly removes constraints that are unnecessary for preserving infeasibility. The result is IIS-like:

- the retained subsystem is infeasible;
- deleting any one retained row makes that subsystem feasible.

It is **inclusion-minimal**, not necessarily minimum-cardinality.

## 2. Minimum-cardinality constraint deletion

For small models, exact subset enumeration finds the smallest number of constraint rows that must be removed to restore feasibility.

This answers a different question from an IIS: "what is the smallest repair set?"

## 3. Weighted L1 feasibility relaxation

Each row receives a nonnegative violation slack:

```text
A x - s <= b
s >= 0
min sum_i w_i s_i
```

Weights encode the relative cost or governance priority of relaxing different requirements. The output identifies both a repaired decision and the magnitude of every relaxed constraint.

## Validation fixture

The regression model contains:

```text
x >= 5
x <= 3
x <= 10
```

The first two rows form the actual conflict; the third is redundant. Tests verify conflict isolation, a one-row minimum repair, and a weighted relaxation that preserves the expensive requirement and relaxes the cheaper one by exactly two units.

## Run

```bash
python -m pip install -e '.[dev]'
pytest
```

## Scope

The implementation is solver-independent and deliberately small. Production conflict refiners may exploit presolve mappings, indicator/SOS constraints, integrality, bound conflicts, SAT-style explanations, and hierarchical repair objectives.
