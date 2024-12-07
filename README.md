# Policy Plasticity
A tool that measures the plasticity of an RLGym-PPO policy.

## How to use:
Just run `python policy_plasticity.py <your_policy_path>`

If you want more accuracy (and are willing to wait), increase the `--ensemble_size`.

## How it works:
- Loads your policy from the specified path
- Makes [`ensemble_size`] fresh random policies of the same architecture
- Generates a bunch of `randn()` tensors as inputs
- Creates a small "function" model (a small MLP) and passes the random inputs through the function model to make the target outputs (this is the best way I found to make a unique and controllably-complex training task)
- Trains your policy and the ensemble policies to learn to mimic the logic of the function model
- Computes what portion of samples your trained policy guessed better than the average of the trained ensemble
- Multiplies that portion by two to find the plasticity (because beating the ensemble 50% of the time implies 100% plasticity)