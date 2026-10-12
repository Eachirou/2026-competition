# Scientific FAQ · Track 04

This page answers recurring scientific questions about the model-facing
contract. The **Track description** defines the phase specifications; the
public [`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
and active phase configuration are the executable warm-up references.
Packaging and upload instructions are in **Get Started → Submission Guide**.

## Warm-up and sealed evaluation

### What changes between the two phases?

| | Warm-up (current) | Sealed final |
|---|---|---|
| **Evaluation data** | Public EMG2Pose `user_stage` test split | Private 2026 Meta Reality Labs cohort |
| **Generalization** | Unseen user-stage combinations | New users, stages, and user-stage combinations |
| **Delivery** | Independent windows in ordinary evaluation batches | One chronological window at a time within each recording |
| **Ranking metric** | Mean absolute angular error reported in degrees | The same prediction and metric contract |

Warm-up scores are indicative only because the datasets and generalization
shifts differ. The public
[warm-up phase config](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/phases/warmup/emg_pose/config.yaml)
selects EMG2Pose; the sealed cohort is described in the
[public dataset directory](https://neural-interfaces26.github.io/tracks.html#dataset-track-4).

### What is disclosed about the sealed cohort?

It is a separate private cohort, not a hidden split of public EMG2Pose. It
includes movement stages outside the 29 public EMG2Pose stages, while keeping
the same joint-angle regression task. Raw recordings, pose trajectories,
participant metadata, labels, exact cohort size, stage composition, and
evaluation order remain private.

## Input and preprocessing

### What does the model receive?

The model receives contiguous, non-overlapping five-second windows of
16-channel wrist sEMG. No filtering, notch filtering, baseline correction,
learned scaling, or clipping is applied. MNE-loaded volts are deterministically
converted to microvolts (`×10⁶`). Incomplete windows and windows containing
invalid inverse-kinematics target frames are excluded. These settings come
from the
[NeuralBench pose-task config](https://github.com/facebookresearch/neuroai/blob/v0.4.0/neuralbench-repo/neuralbench/tasks/emg/pose/config.yaml)
and are loaded by the public
[competition adapter](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/datasets/emg2pose.py).

Construct the model from `meta["n_chans"]`, `meta["n_times"]`,
`meta["sfreq"]`, and `meta["n_joints"]`. The published protocol is fixed at
16 channels and 2 kHz; do not hard-code an evaluation batch size.

### May NinaPro or other public EMG datasets be used?

Yes, subject to their licences and the competition
[Terms](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/terms.md).
Declare every external corpus in the final method description for the
reproducibility audit. Private sealed data and non-permitted closed datasets
may not be used.

## Sequential inference and adaptation

### How are sealed windows delivered, and what does “no future” mean?

In the sealed phase there is no shuffling within a recording: windows are
presented in strict chronological order, one at a time (`B = 1`). A model may
use earlier windows from the current recording, but not later windows. The
complete current five-second window is available when `predict` is called, so
bidirectional processing **within that window** is permitted. The order of
independent recordings, recording identifiers, and cross-recording continuity
are not model-facing signals, so a submission must not depend on state carrying
between recordings.

This differs from the current warm-up runner, whose batch order and composition
must not be assumed and whose public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
evaluates ordinary independent batches from the public EMG2Pose adapter.

### Is unsupervised test-time adaptation allowed?

Yes. A model may update causal statistics or parameters—including gradient
updates inside `predict`—using only the current window and earlier windows in
the current sequential stream. It may not use labels or future windows, and it
must stay within the worker time and memory limits.

## Prediction and scoring

### What must `predict(X)` return?

For `X` with shape `(B, 16, T)`, return 20 joint-angle trajectories with shape
`(B, 20, T_out)` in **radians**. `T_out` may equal `T` or be coarser;
Codabench nearest-neighbor resamples predictions to the target length before
scoring. The public
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py)
contains the executable output and resampling contract.

### Should submissions return radians or degrees?

Radians. Predictions and targets remain in radians during inference and error
computation. Only the final aggregate error is converted to degrees for the
leaderboard.

### How is the ranking score computed?

Codabench computes the direct absolute difference between prediction and
target, averages over all examples, joints, and time points, then multiplies
that aggregate MAE by `180 / π`. It does not wrap errors modulo `2π`. The
aggregation is implemented by the
[shared metric helpers](https://github.com/neural-interfaces26/2026-competition/blob/main/benchmark_utils/metrics.py)
and exposed by the Track 04
[`objective.py`](https://github.com/neural-interfaces26/2026-competition/blob/main/tracks/emg_pose/objective.py).

NeuralBench's `test/mae` on the EMG2Pose `user_stage` split is the same warm-up
error in radians; multiply it by `180 / π` for comparison. Per-subject MAE,
RMSE, and R² are diagnostic and do not replace the ranking metric.

## Resources

**[Track description](https://www.codabench.org/competitions/17984/)** ·
**[NeuralBench guide](https://facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html)** ·
**[Competition implementation](https://github.com/neural-interfaces26/2026-competition/tree/main/tracks/emg_pose)**
