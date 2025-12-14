import torch
import os
import multiprocessing as mp

communication = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def configure_threads(for_ray_actor=False, num_cpus_per_actor=1):
    """
    Automatically detect CPU count and configure PyTorch and OpenMP threads.
    
    Args:
        for_ray_actor: If True, configure for Ray actors.
        num_cpus_per_actor: Number of CPUs allocated to this actor (for thread calculation).
    """
    # Detect CPU count
    cpu_count = mp.cpu_count()
    
    if for_ray_actor:
        # For Ray actors, use threads equal to allocated CPUs
        num_threads = max(1, int(num_cpus_per_actor))
    else:
        # For main process, use all available CPUs
        # Leave 1-2 cores free for system/other processes if we have many cores
        if cpu_count > 8:
            num_threads = max(1, cpu_count - 2)
        else:
            num_threads = cpu_count
    
    # Configure PyTorch
    torch.set_num_threads(num_threads)
    
    # Configure OpenMP
    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    
    if not for_ray_actor:
        print(f"Detected {cpu_count} CPU cores")
        print(f"Configured PyTorch threads: {num_threads}")
        print(f"Configured OMP_NUM_THREADS: {num_threads}")
        print(f"Configured MKL_NUM_THREADS: {num_threads}")
    
    return num_threads

def calculate_ray_resources(desired_num_actors=20):
    """
    Automatically calculate Ray actor resources based on available CPUs.
    Args:
        desired_num_actors: Desired number of actors (may be reduced if not enough CPUs)
    Returns: (num_actors, cpus_per_actor, cpus_per_buffer, cpus_per_learner)
    """
    total_cpus = mp.cpu_count()
    
    # Reserve CPUs for GlobalBuffer and Learner
    cpus_per_buffer = 1
    cpus_per_learner = 1
    reserved_cpus = cpus_per_buffer + cpus_per_learner
    
    # Calculate available CPUs for actors
    available_for_actors = total_cpus - reserved_cpus
    
    if available_for_actors < 1:
        print(f"Warning: Only {total_cpus} CPUs available. Need at least {reserved_cpus + 1} CPUs.")
        print(f"Using minimal configuration: 1 actor with shared CPUs.")
        return 1, 0.5, cpus_per_buffer, cpus_per_learner
    
    # Try to use fractional CPUs to maximize parallelism
    # Target: use all available CPUs efficiently
    if available_for_actors >= desired_num_actors:
        # Enough CPUs for all actors with 1 CPU each
        cpus_per_actor = 1.0
        actual_num_actors = min(desired_num_actors, available_for_actors)
    else:
        # Not enough CPUs, use fractional allocation
        # Prefer more actors with fractional CPUs for better parallelism
        cpus_per_actor = available_for_actors / desired_num_actors
        actual_num_actors = desired_num_actors
        # Ensure at least 0.25 CPU per actor (Ray minimum is 0.1, but 0.25 is more reasonable)
        if cpus_per_actor < 0.25:
            cpus_per_actor = 0.25
            actual_num_actors = int(available_for_actors / cpus_per_actor)

    return actual_num_actors, cpus_per_actor, cpus_per_buffer, cpus_per_learner

# Auto-configure threads for main process (only if not already set)
# This will be called when configs is imported in the main process
if "OMP_NUM_THREADS" not in os.environ:
    configure_threads(for_ray_actor=False)

# Default environment settings (can be overridden)
obs_radius = 4
reward_fn = dict(move=-0.075,
                stay_on_goal=0,
                stay_off_goal=-0.075,
                collision=-0.5,
                finish=3)

obs_shape = (6, 2*obs_radius+1, 2*obs_radius+1)
action_dim = 5



############################################################
####################         DQN        ####################
############################################################

# basic training setting
map_type = 'random'
num_actors = 20
log_interval = 10
training_times = 10000
save_interval=1000
gamma=0.99
batch_size=256
learning_starts=25000
target_network_update_freq=1000
save_path=f'./models/{map_type}'
max_episode_length = 512
seq_len = 16
load_model = False
load_path = './models/save_model/model_house/84000_house.pth'

Advantage_all = True
Sec_cons = True
lambdas = 0.001
actor_update_steps = 400

# gradient norm clipping
grad_norm_dqn=40

# n-step forward
forward_steps = 2

# global buffer
episode_capacity = 2048

# prioritized replay
prioritized_replay_alpha=0.6
prioritized_replay_beta=0.4

# curriculum learning
init_env_settings = (3, 10)
max_num_agents = 3
max_map_length = 20
pass_rate = 0.9
# dqn network setting
cnn_channel = 128
hidden_dim = 256

# Communication (must be >= max_num_agents)
max_comm_agents = 5  # Set to >= max_num_agents

# Communication block
num_comm_layers = 2
num_comm_heads = 2

# Auto-calculate Ray resources based on available CPUs
# This must be called after num_actors is defined
ray_num_actors, ray_cpus_per_actor, ray_cpus_per_buffer, ray_cpus_per_learner = calculate_ray_resources(desired_num_actors=num_actors)

############################################################
####################  Map Type Settings ####################
############################################################

# All available map types
map_types = ['house', 'maze', 'warehouse', 'tunnels', 'random']

# TRAINING: Map type to train on (change this for each training run)
# Options: 'house', 'maze', 'random', 'tunnels', 'warehouse'
train_map_type = 'random'

# TESTING: Map type for evaluation (independent of training)
test_scenario = 'house'  # Used during test.py evaluation

# Save path (can be overridden in training scripts)
save_path = './models'

############################################################
####################    Evaluation      ####################
############################################################

test_seed = 0
num_test_cases = 200

test_env_settings = ((40, 4, 0.3), (40, 8, 0.3), (40, 16, 0.3), (40, 32, 0.3), (40, 64, 0.3), (40, 128, 0.3),
                    (80, 4, 0.3), (80, 8, 0.3), (80, 16, 0.3), (80, 32, 0.3), (80, 64, 0.3), (80, 128, 0.3))

# House-specific test settings
house_test_env_settings = ((40, 4, 0.3), (40, 8, 0.3), (40, 16, 0.3), (40, 32, 0.3), (40, 64, 0.3), (40, 128, 0.3),
                    (60, 4, 0.3), (60, 8, 0.3), (60, 16, 0.3), (60, 32, 0.3), (60, 64, 0.3), (60, 128, 0.3))
