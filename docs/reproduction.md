# Reproduction scope

This version packages the model, anchor-cache preparation, StereoMIS source training, and input-camera inference on prepared recordings. Its scorer consumes saved PNGs and explicit common masks. The scorer requires the canonical evaluation masks as a separate input. DRENDS and iMED expect preprocessed inputs for their respective loaders.

The common inference command uses FP32 settings and selects the archived ray counts for 100 and 1,000 target updates. StereoMIS P3 used 1,024 rays at 100 updates and 512 at 1,000 updates. The unified command uses common target adaptation code and a fixed seed. Historical benchmark runs include dataset-specific seed-reset ordering, TF32 settings and timing instrumentation, so exact score and time reproduction is not established by this command. The original source-training objective, validation probe, architecture and cache tensor schema were preserved.

The right-eye and independent second-endoscope evaluators, ToF alignment and timed comparison campaigns are outside this release. The [baseline list](baselines.md) identifies the external references, without publishing their code or separate fitting jobs.
