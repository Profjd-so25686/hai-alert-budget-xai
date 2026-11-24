P1. Constant tags (18 November 2026). 22 of 79 tags are constant across all training data (train1 to train3; list in Appendix). Decision: \[keep]. Reason: \[keep all 79 with a scaling guard;Features per window-79 × 5 = 395; KernelSHAP- About 40 per cent higher; Attack on a pump command, trip flag or setpoint- PCA and autoencoder can catch it; isolation forest cannot; Isolation forest’s blindness to never-varying tags, as a limitation]. Determined from training data only; test files not opened.



In HAI 21.03, three attack primitives target tags that never move in training:



Primitive	Target tag	Normal value	What it is

AP21	P2\_VTR01	10	Turbine set-point

AP22	P2\_VTR02	10	Turbine set-point

AP23	P2\_RTR	2,880	Turbine trip rate (RPM)



They occur in your validation file test2, for example A201, A207 and A217. Had I dropped the constant tags, every detector would have been blind to those attacks, and because the held-out attacks combine primitives, probably to some held-out attacks too. Option B is no longer just a defensible choice: it is necessary.





p2. P1\_PP01AD sits at 540,833, not 0 or 1. It looks like a coded status value, not a measurement. Under Option B its scale is 1, so if an attack changed it to, say, 0, the scaled feature would jump by about 540,000. Every other feature moves by single digits.



What the values mean for scaling (pilot observation P2)



The 22 constant tags fall into three groups:



Group	Tags	Normal value	If an attack moves it

On/off flags	PP01BD, PP01BR, PP02D, PP02R, STSP, ASD, AutoGO, Emerg, ManualGO, OnOff, TripEx	0 or 1	Changes by 1. Comparable to other scaled features

Set-points and limits	PCV02D (12), VTR01 to VTR04 (10), LH (70), LL (20), MSD (763), RTR (2,880)	Tens to thousands	Changes by units to hundreds

Coded value	PP01AD, PP01AR	540,833	Could change by hundreds of thousands. The documentation calls these Boolean, so this is an encoding quirk



Under the current guard (scale 1), an attack on P2\_RTR or the PP01A tags would produce a scaled value hundreds or thousands of times larger than any other feature. That detects the attack, but it would swamp both the reconstruction error and the SHAP attributions, making the explanations look like “it was all RTR”.



P2. Scaling of constant tags (23 November 2026). Compared raw and relative scaling on validation data (scaler fitted on train1 to train3). Constant-tag features contribute almost nothing to attack windows under either rule (median share 0.0, max 0.2 per cent); the large-valued tags (P1\_PP01AD, P2\_RTR) do not move in validation. Decision: keep the raw rule (scale 1). P1\_PCV02D moves only in attack windows, supporting P1.



P3. SHAP additivity and cost (23 November 2026). TreeSHAP additivity confirmed on real train1 windows (gap 8.9e-15). KernelSHAP projected at 0.6 to 4.9 CPU h, within the 24 h limit. Agreement with path-dependent TreeSHAP was low (median ρ about 0.27 features, 0.50 tags) and did not improve with more samples, indicating a difference in reference distribution, not sampling error. Next: rerun with interventional TreeSHAP on the same background.



P4. Matched-background SHAP comparison (23 November 2026). With TreeSHAP in interventional mode and both explainers using the same unweighted background of training windows (KernelSHAP with l1\_reg disabled), agreement rose to median ρ 0.92 to 0.97 (features) and 0.98 to 0.99 (tags), and increased with sample count, confirming the earlier low agreement came from mismatched references. Projected KernelSHAP cost: 8.8 to 29.7 CPU h. Proposed setting: 100 background windows, 2,838 samples (15.7 CPU h); fallback 50 and 2,838 (8.8 CPU h). Decided before any validation or test data was used for explanation.



P5. Workload (23 November 2026). Normal-window sample fixed at 1,000 per model run. Held-out attack windows estimated at about 638 from attack durations in the HAI documentation (test3 to test5 not opened), giving about 1,638 windows per model run and a projected 15.6 CPU h at the proposed KernelSHAP setting (about 8.7 CPU h at the fallback). Estimation method checked against the validation files: 552 actual attack windows against 540 estimated (2 per cent under).



P6. KernelSHAP cost for all detectors (23 November 2026). Timed on stand-in PCA (95 per cent variance, 65 components) and autoencoder (395-64-16-64-395) on train1, PyTorch limited to one thread, CPU time counted. Projected totals at 1,638 windows per model run (PCA 1 run, isolation forest and autoencoder 5 each): 17.5 CPU h at the proposed setting (100 background, 2,838 samples) and 10.2 CPU h at the fallback (50, 2,838). Both within the 24 CPU h limit. Rule: if the final configurations exceed 24 CPU h at the proposed setting, the fallback is used for all detectors.



P7. Low-variance tags and regime shift (23 November 2026). Features of tags that barely vary in training reach very large standardised values: P1\_PCV02Z up to 16,542 in attack windows only (a genuine attack response), and turbine tags P2\_VT01, P2\_SIT01, P2\_SIT02, P2\_VXT02 up to about 1,650 in normal validation windows, indicating turbine behaviour in test1 and test2 that is absent from training. To be resolved before pre-registration.



P7 (resolved 24 November 2026). 18 of 15,638 normal validation windows (0.12 per cent) have a standardised feature above 50: 15 within 30 minutes after an attack ends (all test2, mainly P2\_VT01), 3 elsewhere. Decision: no change to scaling. Pre-registered sensitivity analysis: false alerts and AC2 recomputed excluding windows that start within 30 minutes after an attack ends.



P8. Tag groups (24 November 2026). Tags linked where absolute Pearson correlation exceeds 0.8 on normal training rows (train1 to train3); groups are connected sets of linked tags. Result: 47 groups from 79 tags, 11 with more than one tag (largest 9), 22 constant tags as single groups. Full list in results/pilot\_tag\_groups\_2026-11-24.json, to be attached to the pre-registration.



P9. Seeds (24 November 2026). Model seeds 101, 202, 303, 404, 505, with 101 primary. Background sample 7001, normal-window sample 7002, bootstrap 7003.



P10. Tuning grids (24 November 2026). PCA variance retained: 90, 95, 99 per cent. Isolation forest trees: 100, 200 (300 removed, because KernelSHAP cost scales with forest size and 300 trees would exceed the 24 CPU-hour limit); sample size: 256, 512. Autoencoder bottleneck: 8, 16, 32; hidden layers: 1, 2; hidden width 64; learning rate 0.001; early stopping on normal validation windows, patience 10, maximum 200 epochs. Selection: highest validation event recall at the alert budget; ties go to the simpler model.



