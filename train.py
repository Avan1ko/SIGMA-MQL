import os
import random
import time

import torch
import numpy as np
import ray

from worker import GlobalBuffer, Learner, Actor
import configs

import wandb

# Ensure thread configuration is set for main process
configs.configure_threads(for_ray_actor=False)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def train_for_map_type(map_type, num_actors=configs.ray_num_actors, log_interval=configs.log_interval):
    """Train a model for a specific map type"""
    # Update configs for this map type
    configs.train_map_type = map_type
    configs.save_path = os.path.join('./models', f'model_{map_type}')
    configs.load_model = False  # Start fresh for each map type
    
    # Initialize Ray
    ray.init(local_mode=False, ignore_reinit_error=True)
    
    try:
        buffer = GlobalBuffer.remote()
        learner = Learner.remote(buffer)
        time.sleep(1)
        actors = [Actor.remote(i, 0.4**(1+(i/(num_actors-1))*7), learner, buffer) for i in range(num_actors)]

        for actor in actors:
            actor.run.remote()

        while not ray.get(buffer.ready.remote()):
            time.sleep(5)
            ray.get(learner.stats.remote(5))
            ray.get(buffer.stats.remote(5))

        print(f'Start training for {map_type}')
        buffer.run.remote()
        learner.run.remote()
        
        done = False
        step_count = 0
        while not done:
            time.sleep(log_interval)
            
            # Get stats for console output (existing behavior)
            done = ray.get(learner.stats.remote(log_interval))
            ray.get(buffer.stats.remote(log_interval))
            
            # Get stats for wandb logging
            learner_stats = ray.get(learner.get_wandb_stats.remote(log_interval))
            buffer_stats = ray.get(buffer.get_wandb_stats.remote())
            
            # Combine stats and add map type context
            wandb_metrics = {
                'map_type': map_type,
                # Learner metrics
                'loss': learner_stats.get('loss', 0),
                'updates': learner_stats.get('updates', 0),
                'update_speed': learner_stats.get('update_speed', 0),
                'learning_rate': learner_stats.get('learning_rate', 0),
                # Buffer metrics
                'buffer_size': buffer_stats.get('buffer_size', 0),
                'success_rate': buffer_stats.get('success_rate', 0),
                'avg_episode_length': buffer_stats.get('avg_episode_length', 0),
                'avg_arrival_rate': buffer_stats.get('avg_arrival_rate', 0),
            }
            
            # Log to wandb
            wandb.log(wandb_metrics, step=step_count)
            step_count += 1
            
            print()
        
        print(f'\nTraining completed for {map_type}')
        
    finally:
        # Shutdown Ray to clean up resources
        ray.shutdown()


def main():
    """Train models for all map types"""
    wandb.init(
        entity="sigmamql",
        project="SIGMA-MQL",
    )
    
    for map_type in configs.map_types:
        train_for_map_type(map_type)
        print(f"\nCompleted training for {map_type}. Moving to next map type...\n")
        time.sleep(2)  # Brief pause between training runs
    
    print("\n" + "="*60)
    print("All training completed!")
    print(f"Trained models for: {', '.join(configs.map_types)}")
    print("="*60)
    
    wandb.finish()


if __name__ == '__main__':
    main()
