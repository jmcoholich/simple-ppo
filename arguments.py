import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--total-steps", type=int, default=int(200000), help="Total samples to train on"
    )
    parser.add_argument(
        "--wandb", action="store_true", help="If passed, enables logging with wandb"
    )
    parser.add_argument(
        "--play",
        default="None",
        type=str,
        help="visualize the saved model with name passed in",
    )
    parser.add_argument(
        "--steps-per-update", type=int, default=int(1000), help="Steps per update"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="number of neurons in hidden layers of the policy and value neural networks",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=2,
        help="number of hidden layers in the policy and value neural networks",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="non-linear activation function for hidden layers. (Choose from tanh, sigmoid, or relu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed (default = 1), passed to env and pytorch",
    )
    parser.add_argument(
        "--deterministic",
        default=False,
        action="store_true",
        help="if True, actor will select mean of action distribution, instead of sampling",
    )
    parser.add_argument(
        "--independent-std",
        default=True,
        action="store_true",
        help="if True, standard deviations of the selected actions will not depend on state",
    )
    parser.add_argument(
        "--env-name",
        default="Pendulum-v1",
        type=str,
        help="Open AI gym environment name passed to gym.make()",
    )
    parser.add_argument(
        "--lr",
        default=3e-4,
        type=float,
        help="learning rate of the optimization algorithm",
    )
    parser.add_argument(
        "--opt-alg",
        default="Adam",
        type=str,
        help="name of optimization algorithm (currently only adam or sgd are allowed)",
    )
    parser.add_argument("--gamma", default=0.99, type=float, help="discount factor")
    parser.add_argument(
        "--gae-lambda",
        default=0.95,
        type=float,
        help="lambda parameter for generalized advantage estimation",
    )
    parser.add_argument(
        "--ppo-eps",
        default=0.2,
        type=float,
        help="epsilon clipping parameter for PPO-clip",
    )
    parser.add_argument(
        "--epochs",
        default=8,
        type=int,
        help="number of training epochs per policy/value function update",
    )
    parser.add_argument(
        "--value-loss-coef",
        default=0.5,
        type=float,
        help="weight for value net MSE term of loss function",
    )
    parser.add_argument(
        "--entropy-coef",
        default=0.0,
        type=float,
        help="entropy bonus in policy net loss function",
    )
    args = parser.parse_args()
    return args
