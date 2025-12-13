"""
Configuration for Baseline Cross-Generalization Experiments
Train on one map type, test on all others
Fixed: 3 agents, 15x15 maps
"""

import torch
import os
import multiprocessing as mp
from configs import configure_threads, calculate_ray_resources

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Auto-configure threads
if "OMP_NUM_THREADS" not in os.environ:
    configure_threads(for_ray_actor=False)

############################################################
####################    Environment     ####################
############################################################

num_agents = 3
map_length = 15
obs_radius = 4

# Map type for this training run (change this for each run)
# Options: 'house', 'maze', 'random', 'tunnels', 'warehouse'
MAP_TYPE = 'random'  # <-- Change this for each training run

# Reward function
reward_fn = dict(
    move=-0.075,
    stay_on_goal=0,
    stay_off_goal=-0.075,
    collision=-0.5,
    finish=3
)

obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)
action_dim = 5

############################################################
####################         DQN        ####################
############################################################

# Basic training settings
num_actors = 20
log_interval = 10

# REDUCED training time for faster experiments
training_times = 200000  # Reduced from 600k (should be enough for 15x15)
save_interval = 2000
gamma = 0.99
batch_size = 192
learning_starts = 10000  # Reduced from 50k
target_network_update_freq = 2000

# Save paths (organized by map type)
save_path = f'./models/baseline_{MAP_TYPE}'
os.makedirs(save_path, exist_ok=True)

# Episode settings
max_episode_length = 256  # Reduced from 512 (smaller maps = faster episodes)
seq_len = 16
load_model = False

# SIGMA extensions
Advantage_all = True
Sec_cons = True
lambdas = 0.001

# Actor settings
actor_update_steps = 400

# Gradient clipping
grad_norm_dqn = 40

# N-step learning
forward_steps = 2

# Replay buffer
episode_capacity = 2048

# Prioritized replay
prioritized_replay_alpha = 0.6
prioritized_replay_beta = 0.4

# FIXED CURRICULUM (no progression, just one setting)
init_env_settings = (num_agents, map_length)
max_num_agents = num_agents  # Fixed!
max_map_length = map_length  # Fixed!
pass_rate = 0.9  # Not used (no curriculum)

# Network architecture
cnn_channel = 128
hidden_dim = 256

# Communication
max_comm_agents = 5  # Reduced from 10 (only 3 agents)

# Communication block
num_comm_layers = 2
num_comm_heads = 2

# Ray resources
ray_num_actors, ray_cpus_per_actor, ray_cpus_per_buffer, ray_cpus_per_learner = \
    calculate_ray_resources(desired_num_actors=num_actors)

############################################################
####################    Evaluation      ####################
############################################################

test_seed = 0
num_test_cases = 100  # Episodes per map type during evaluation

# Test on ALL 5 map types
test_map_types = ['house', 'maze', 'random', 'tunnels', 'warehouse']

# Test settings: (map_size, num_agents, density)
# Test on same size and agent count, different map types
test_env_settings = [
    (15, 3, 0.3),  # Same as training
]

print(f"""
╔══════════════════════════════════════════════════════════╗
║        BASELINE EXPERIMENT CONFIGURATION                 ║
╠══════════════════════════════════════════════════════════╣
║  Training Map Type: {MAP_TYPE:36s} ║
║  Map Size:          {map_length}x{map_length:34s} ║
║  Num Agents:        {num_agents:42d} ║
║  Training Steps:    {training_times:42d} ║
║  Save Path:         {save_path:36s} ║
╚══════════════════════════════════════════════════════════╝
""")
