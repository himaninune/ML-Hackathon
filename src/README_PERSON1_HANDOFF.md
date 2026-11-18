# Person 1 handoff

## Files to share with Person 2
- `experiments/results/train_features.csv`
- `experiments/results/student_test_features.csv`

## Important
`label` is currently derived from whether a candidate arc appears in that trip's
`.route` sequence. This is a route-membership heuristic, not a verified GPS-point
ground-truth alignment. Confirm route file semantics and arc ID indexing with
the dataset specification before reporting supervised model accuracy as definitive.

For a candidate-ranking/classification baseline, Person 2 can group by
`(trip_id, observation_index)` and use the candidate rows for that GPS point.
Do not randomly split candidate rows: split by trip to avoid leakage.
