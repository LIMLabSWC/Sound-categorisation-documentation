It works but there are a few issues.

* *Noise generator* : sets global seed not just in function

* *Sampler* seeds are equal to model's random state.

* *Cross-Validation* is not randomised. We want to shuffle blocks.
