## Plan: Agentic EV Charging Optimization Workflow

Implement a LangGraph-style planning assistant that analyzes EV charging demand, retrieves infrastructure-planning guidance using RAG, and outputs balanced (congestion + cost) charger placement and scheduling recommendations with explicit confidence and traceability.

**Steps**
1. Phase 0: Delivery Contract and Objective Lock (completed in planning scope).
2. Phase 0.1: Lock fixed response sections in order: Summary, Analysis, Plan, Optimize, References.
3. Phase 0.2: Lock recommendation item schema: zone_id, trigger_condition, action, expected_effect, cost_class, confidence_level, risk_note.
4. Phase 0.3: Lock citation contract: each recommendation cites one local evidence source and one retrieved guideline source.
5. Phase 0.4: Lock failure contract: when evidence is incomplete, return assumptions + uncertainty note, not hard action.
6. Phase 0.5: Lock balanced defaults: congestion_weight=0.5 and cost_weight=0.5, configurable in state.
7. Phase 0.6: Define preflight checks: schema conformance, reference completeness, weight initialization.
8. Phase 1: State Architecture and Graph Contracts.
9. Phase 1.1: Define canonical state object: run_id, objective, scope, horizon, demand_features, infra_features, retrieved_guidelines, optimization_constraints, candidate_actions, ranked_recommendations, confidence_notes, audit_trace.
10. Phase 1.2: Define node-level input/output contracts for all graph nodes; no node may mutate unrelated keys.
11. Phase 1.3: Define state validation guard with fail-fast routing to data-completion path.
12. Phase 1.4: Define state versioning rules for backward compatibility when fields evolve.
13. Phase 1.5: Define audit trail requirements: each node appends timestamped reasoning metadata.
14. Phase 1.6: Add Phase 1 acceptance gate: state completeness >= 100% required keys and successful dry-run transition across all nodes.
15. Phase 2: Data and Retrieval Foundation.
16. Phase 2.1: Build demand adapter from processed and prediction assets; standardize zone-hour keys and timestamps.
17. Phase 2.2: Build infrastructure adapter from zone metadata; compute density and capacity-gap helper features.
18. Phase 2.3: Build guideline corpus with tagged chunks (utilization, reliability, accessibility, equity, cost, rollout policy).
19. Phase 2.4: Build retrieval layer with Chroma/FAISS-style semantics, returning top-k chunks with relevance scores and source ids.
20. Phase 2.5: Build evidence retrieval node for local files (metrics, feature importance, forecasts) keyed by zone and horizon.
21. Phase 2.6: Add retrieval quality controls: minimum relevance threshold, fallback prompts, and no-guidance handling path.
22. Phase 2.7: Add Phase 2 acceptance gate: retrieval precision check and deterministic evidence payload format.
23. Phase 3: Core Optimization Reasoning Nodes.
24. Phase 3.1: Implement Demand Analyzer logic contract: persistence stress, top-decile frequency, peak intensity, volatility band.
25. Phase 3.2: Implement Guideline Retriever contract: map scenario intent to guideline tags and return ranked constraints.
26. Phase 3.3: Implement Infrastructure Gap Analyzer contract: compare stress metrics to charge_count/density and identify deficits.
27. Phase 3.4: Implement Placement Optimizer contract: generate expansion candidates under balanced objective and feasibility constraints.
28. Phase 3.5: Implement Scheduling Optimizer contract: generate slot-shift, maintenance-stagger, and incentive-window actions.
29. Phase 3.6: Implement Justification Composer contract: every action linked to demand evidence + guideline citation.
30. Phase 3.7: Implement Conflict Resolver contract: resolve contradictory guidance and append risk caveats.
31. Phase 3.8: Add Phase 3 acceptance gate: no uncited recommendation, no unresolved conflict state, and complete confidence notes.
32. Phase 4: Ranking, Policy, and Scalability Controls.
33. Phase 4.1: Define final recommendation schema for delivery objects and confidence classes.
34. Phase 4.2: Define balanced scoring formula and normalization policy across zones.
35. Phase 4.3: Add uncertainty penalty so high forecast-error zones are not over-prioritized without caveat.
36. Phase 4.4: Add feasibility filter policy (budget class, deployment lead time, service continuity floor).
37. Phase 4.5: Add scalability strategy: batching, caching of retrieval context, and multi-zone run limits.
38. Phase 4.6: Add policy-level override rules for city-wide versus top-N rollout modes.
39. Phase 4.7: Add Phase 4 acceptance gate: stable rankings across repeated runs and bounded runtime for target workload.
40. Phase 5: Validation and Quality Gates.
41. Phase 5.1: Backtest validation: top-ranked zones must align with repeated high-load periods, not one-off spikes.
42. Phase 5.2: Scenario validation suite: weekday peak, weekend peak, high-price period, uncertainty stress case.
43. Phase 5.3: Output conformance validation: all required sections and mandatory fields present.
44. Phase 5.4: Recommendation consistency validation: no recommendation contradicts cited guidance or uncertainty notes.
45. Phase 5.5: Practicality validation: each action includes execution trigger and expected operational impact.
46. Phase 5.6: Quality gate: pass only if relevance, consistency, practicality, and conformance thresholds are satisfied.
47. Phase 6: Handoff and Operating Model.
48. Phase 6.1: Produce implementation handoff pack: state schema, node contracts, retrieval design, validation checklist.
49. Phase 6.2: Define monitoring KPIs: recommendation acceptance rate, peak-load reduction proxy, unresolved-uncertainty rate.
50. Phase 6.3: Define governance policy: advisory-only versus policy-linked automation mode.
51. Phase 6.4: Define change-control process for objective weights and guideline corpus updates.
52. Phase 6.5: Define runbook for failure scenarios (missing data, low retrieval relevance, conflicting constraints).
53. Phase 6.6: Add Phase 6 acceptance gate: all docs complete and ready for implementation handoff.
54. 
55. **Dependencies and Parallelism**
56. 1. Phases 1 and 2 can begin after Phase 0 completion, with Phase 2.1 and 2.2 running in parallel.
57. 2. Phase 3 depends on completion of Phase 1 and core retrieval from Phase 2.
58. 3. Phase 4 depends on Phase 3 output contracts.
59. 4. Phase 5 depends on Phase 4 ranking and policy outputs.
60. 5. Phase 6 depends on successful Phase 5 quality gate.
61. 
62. **Relevant files**
63. - [README.md](README.md) — problem framing, assumptions, and scope narrative.
64. - [streamlit_app.py](streamlit_app.py) — future integration surface for assistant orchestration.
65. - [data/processed/final_featured_dataset.csv](data/processed/final_featured_dataset.csv) — engineered zone-hour demand features.
66. - [data/processed/merged_hourly_data.csv](data/processed/merged_hourly_data.csv) — merged baseline time-series demand signals.
67. - [data/raw/zone-information.csv](data/raw/zone-information.csv) — infrastructure and geographic attributes.
68. - [data/raw/adj.csv](data/raw/adj.csv) — adjacency constraints for cross-zone actions.
69. - [data/raw/distance.csv](data/raw/distance.csv) — distance constraints for routing feasibility.
70. - [results/predictions/test_predictions.csv](results/predictions/test_predictions.csv) — forecast traces for stress scoring.
71. - [results/metrics/model_metrics.csv](results/metrics/model_metrics.csv) — model reliability for confidence calibration.
72. - [results/feature_importance/lightgbm_occupancy_feature_importance.csv](results/feature_importance/lightgbm_occupancy_feature_importance.csv) — occupancy reasoning support.
73. - [results/feature_importance/lightgbm_volume_feature_importance.csv](results/feature_importance/lightgbm_volume_feature_importance.csv) — volume reasoning support.
74. 
75. **Verification**
76. 1. State Integrity Test: all required keys present at each node transition.
77. 2. Retrieval Relevance Test: top-k average relevance passes threshold with valid sources.
78. 3. Demand Persistence Test: high-priority zones show recurring stress windows.
79. 4. Feasibility Test: each expansion action satisfies at least one feasibility profile.
80. 5. Scheduling Impact Test: recommended scheduling lowers simulated peak concentration versus baseline.
81. 6. Consistency Test: no contradiction between recommendation text, citations, and uncertainty notes.
82. 7. Output Contract Test: response always contains Summary, Analysis, Plan, Optimize, References.
83. 
84. **Decisions**
85. - Objective is balanced: congestion reduction and cost control.
86. - Scope includes planning logic, state management, retrieval grounding, and validation gates.
87. - Scope excludes retraining redesign, real-time ingestion architecture, and external event pipeline buildout.
88. - Assumption: current six-month zone-hour dataset and saved model outputs are sufficient for first implementation cycle.
89. 
90. **Phase Status**
91. - Phase 0: Completed in planning scope.
92. - Phase 1: Completed in planning scope (detailed contracts defined).
93. - Phase 2: Completed in planning scope (retrieval and evidence design defined).
94. - Phase 3: Completed in planning scope (reasoning node contracts defined).
95. - Phase 4: Completed in planning scope (scoring, policy, and scalability controls defined).
96. - Phase 5: Completed in planning scope (validation gates defined).
97. - Phase 6: Completed in planning scope (handoff and operations model defined).
98. 
99. **Further Considerations**
100. 1. Keep fixed 0.5/0.5 weights for MVP or allow scenario-specific weight overrides.
101. 2. Choose delivery mode: city-wide full ranking or top-N critical-zone rollout.
102. 3. Choose governance mode: advisory-only recommendations or policy-linked automation.