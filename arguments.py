import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-env-steps', type=int, default=int(1e6), 
        help='Total samples to train on')
    parser.add_argument('--num-steps', type=int, default=int(2048), 
        help='Steps per update')
    parser.add_argument('--hidden-size', type=int, default=256, 
        help='hidden size of the policy and value neural networks')
    parser.add_argument('--hidden-layers', type=int, default=2, 
        help='number of hidden layers in the policy and value neural networks')
    parser.add_argument('--activation', type=str, default='relu', 
        help='non-linear activation function for hidden layers. (Choose from tanh, sigmoid, or relu)')
    args = parser.parse_args()
    
    return args