# Results Summary

The original mpc interacted at every timestep, which worked out great, but had some caveats:
- It needed a meaningful horizon (3 seconds +), leading to very long horizons when the simulation timestep was lower than 0.1 seconds
- Predicting in single larger steps than the simulation was running at went very poorly - the predictions were poor

The reworked mpc does not interact at every timestep, but it PREDICTS at every timestep, so it usese every simulation timestep to predict, giving perfect predictions, but only interacts every set number of simulation steps. This worked great too - but did take a little longer to compute, luckily its a superset of the original mpc so we can always just choose to sample at every timestep and it will work the same.

rework parity testing is the reworked mpc using every timestep interactions to show the results are the same as the original mpc.