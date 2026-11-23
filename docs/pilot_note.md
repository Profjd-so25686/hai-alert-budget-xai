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



P3. SHAP additivity and cost (23 November 2026). TreeSHAP additivity confirmed on real train1 windows (gap 8.9e-15). KernelSHAP projected at 0.6 to 4.9 CPU h, within the 24 h limit. Agreement with path-dependent TreeSHAP was low (median ρ about 0.27 features, 0.50 tags) and did not improve with more samples, indicating a difference in reference distribution, not sampling error. Next: rerun with interventional TreeSHAP on the same background.



P4. Matched-background SHAP comparison (23 November 2026). With TreeSHAP in interventional mode and both explainers using the same unweighted background of training windows (KernelSHAP with l1\_reg disabled), agreement rose to median ρ 0.92 to 0.97 (features) and 0.98 to 0.99 (tags), and increased with sample count, confirming the earlier low agreement came from mismatched references. Projected KernelSHAP cost: 8.8 to 29.7 CPU h. Proposed setting: 100 background windows, 2,838 samples (15.7 CPU h); fallback 50 and 2,838 (8.8 CPU h). Decided before any validation or test data was used for explanation.



P5. Workload (23 November 2026). Normal-window sample fixed at 1,000 per model run. Held-out attack windows estimated at about 638 from attack durations in the HAI documentation (test3 to test5 not opened), giving about 1,638 windows per model run and a projected 15.6 CPU h at the proposed KernelSHAP setting (about 8.7 CPU h at the fallback). Estimation method checked against the validation files: 552 actual attack windows against 540 estimated (2 per cent under).

