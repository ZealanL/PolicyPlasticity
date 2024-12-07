import os.path
import pathlib

import torch
import sys
from collections import OrderedDict, namedtuple
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Measure the plasticity of a trained RLGym-PPO policy')
    parser.add_argument('policy_path', type=str,
                        help='Path of the \"PPO_POLICY.pt\" file')
    parser.add_argument('--ensemble_size', type=int, nargs='?',
                        help='Number of control networks to compare against, higher amounts give a more accurate measurement',
                        default=1)
    return parser.parse_args()

if __name__ == "__main__":
    g_args = parse_args()

from rlgym_ppo.ppo import DiscreteFF

def model_info_from_dict(loaded_dict):
    state_dict = OrderedDict(loaded_dict)

    bias_counts = []
    weight_counts = []
    for key, value in state_dict.items():
        if ".weight" in key:
            weight_counts.append(value.numel())
        if ".bias" in key:
            bias_counts.append(value.size(0))

    inputs = int(weight_counts[0] / bias_counts[0])
    outputs = bias_counts[-1]
    layer_sizes = bias_counts[:-1]

    return inputs, outputs, layer_sizes

# https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

@torch.no_grad
def create_task_data(input_size, num_outputs, batch_size, num_batches, device):
    inputs = torch.randn((num_batches, batch_size, input_size), device=device)

    # Use a single-layer model to generate the outputs
    complexity = 48 # TODO: Make a configurable argument or something
    data_generation_model = torch.nn.Sequential(
        torch.nn.Linear(input_size, complexity, device=device),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(complexity, num_outputs, device=device),
        torch.nn.Softmax(dim=-1)
    )

    target_outputs: torch.Tensor = data_generation_model(inputs.reshape(-1, input_size))
    target_outputs = target_outputs.reshape(num_batches, batch_size, num_outputs)

    # DISABLED: Only be use the highest output from the data generation model
    #target_outputs = torch.nn.functional.one_hot(target_outputs.argmax(dim=-1), num_classes=num_outputs).to(torch.float32)

    return inputs, target_outputs

def train_task(
        trained_policy: DiscreteFF, control_policies,
        inputs: torch.Tensor, target_outputs: torch.Tensor,
        num_epochs,
        device):

    num_batches = inputs.size(0)
    assert target_outputs.size(0) == num_batches

    for policy_num in range(len(control_policies) + 1):
        is_trained = (policy_num == 0)
        policy = trained_policy if is_trained else control_policies[policy_num - 1]

        initial_loss = None
        optim = torch.optim.Adam(policy.parameters(), lr=5e-4)
        loss_fn = torch.nn.L1Loss()
        for j in range(num_batches):
            for i in range(num_epochs):
                output = policy.get_output(inputs[j])
                loss: torch.Tensor = loss_fn(output, target_outputs[j])
                loss.backward()
                optim.step()
                optim.zero_grad()
                loss = loss.detach().cpu().item()
                if initial_loss is None:
                    initial_loss = loss

            if is_trained:
                policy_name = "already-trained policy"
            else:
                policy_name = f"control policy {policy_num}/{len(control_policies)}"

            print_progress_bar(
                j + 1, num_batches,
                "Training " + policy_name, f"| Loss: {loss:.5f}",
                1, 30)
        print("Final loss:", loss)

    with torch.no_grad():
        outputs = []
        for policy_num in range(len(control_policies) + 1):
            is_trained = (policy_num == 0)
            policy = trained_policy if is_trained else control_policies[policy_num - 1]
            outputs.append(policy.get_output(inputs[0]))

        outputs = torch.stack(outputs)
        all_error = (outputs - target_outputs[0].unsqueeze(0)).abs()
        pretrained_error = all_error[0]
        control_mean_error = all_error[1:].mean(dim=0)

        won_frac = (pretrained_error < control_mean_error).to(torch.float32).mean()
        # Multiply by 2 because beating the control ensemble 50% of the time is having equal plasticity
        plasticity = (won_frac * 2).clamp_max(1)

        return won_frac.cpu().item(), plasticity.cpu().item()

def main():
    global g_args

    ensemble_size = g_args.ensemble_size
    if ensemble_size < 1:
        raise ValueError("Ensemble size must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    policy_path = pathlib.Path(g_args.policy_path)
    if os.path.isdir(policy_path):
        policy_path /= "PPO_POLICY.pt"
    if not os.path.isfile(policy_path):
        raise Exception(f"Policy at \"{policy_path}\" doesn't exist!")

    print(f"Loading policy from \"{policy_path}\"...")
    state_dict = torch.load(policy_path, weights_only=True)
    input_amount, action_amount, layer_sizes = model_info_from_dict(state_dict)
    print(f"Inputs: {input_amount}, layer sizes: {layer_sizes}, actions: {action_amount}")

    print("Creating and loading discrete policy module...")
    policy = DiscreteFF(input_amount, action_amount, layer_sizes, device)
    policy.load_state_dict(state_dict)

    print("Creating control policies...")
    control_policies = []
    for i in range(ensemble_size):
        control_policy = DiscreteFF(input_amount, action_amount, layer_sizes, device)
        control_policies.append(control_policy)

    print("Generating task data...")
    inputs, target_outputs = create_task_data(input_amount, action_amount, 10_000, 300, device)

    print(f"Training on task data against {len(control_policies)} control policies...")
    won_frac, plasticity = train_task(policy, control_policies, inputs, target_outputs, 1, device)

    print("===========================")
    print(f" > PLASTICIY: {won_frac * 100:.3f}%")
    print("===========================")

if __name__ == "__main__":
    main()