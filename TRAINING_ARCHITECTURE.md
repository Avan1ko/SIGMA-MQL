# Distributed Training Architecture & Experience Collection

## Overview

This codebase uses **Ray** for distributed reinforcement learning. The key insight is that **only ONE process updates weights** (just like your standard PyTorch loop), while multiple processes collect experience in parallel.

**The training loop is standard PyTorch!** The only "distributed" parts are:
1. Data collection happens in parallel (Actors)
2. Weights are shared via Ray's shared memory
3. Actors periodically sync to latest weights

The actual weight updates (`optimizer.step()`) happen in exactly one place: the `Learner.train()` method, just like a normal PyTorch training loop.

## Architecture Components

### 1. **Learner** (ONE process)
- **Location**: `worker.py`, lines 263-368
- **Role**: Does ALL weight updates (the only process that trains)
- **Training Loop**: Standard PyTorch training loop:
  ```python
  optimizer.zero_grad()      # Clear gradients
  loss.backward()            # Compute gradients
  optimizer.step()           # Update weights ← THIS IS WHERE WEIGHTS CHANGE
  ```
- **Weight Sharing**: Every 5 iterations, stores weights in Ray shared memory using `ray.put()`

### 2. **Actors** (MANY processes)
- **Location**: `worker.py`, lines 371-435
- **Role**: Collect experience using the model for inference
- **Key Point**: They do **NOT** update weights - they only use them!
- **Weight Updates**: Periodically pull latest weights from Learner using `ray.get()`
- **Mode**: Always in `eval()` mode (inference only, no gradients)

### 3. **GlobalBuffer** (ONE process)
- **Location**: `worker.py`, lines 19-261
- **Role**: Shared experience replay buffer
- **Function**: Stores experience from all Actors, samples batches for Learner

## What Are Actor Experiences?

Actor experiences are the **training data** for the reinforcement learning algorithm. They contain everything needed to train the Q-network using Deep Q-Learning (DQN).

### What is an "Experience"?

An experience is a tuple containing:
- **Observation (state)**: What the agent sees at time `t`
- **Action**: What action the agent took
- **Reward**: The reward received for taking that action
- **Next Observation**: What the agent sees at time `t+1`
- **Hidden State**: RNN hidden state (for recurrent networks)
- **Communication Mask**: Which agents can communicate with each other
- **Q-values**: The model's predicted Q-values (used for prioritized replay)

### What Each Component Does

1. **Observation (obs)**
   - The agent's view of the environment
   - Used as input to the Q-network
   - Shape: `(num_agents, *obs_shape)`

2. **Action**
   - The action taken by the agent
   - Used to compute `Q(s, a)` - the Q-value for state `s` and action `a`
   - Shape: `(1,)` - single action index

3. **Reward**
   - The immediate reward received
   - Used in the TD error calculation: `TD_error = Q(s,a) - (r + γ * max Q(s',a'))`
   - Shape: `(1,)` - single reward value

4. **Next Observation (next_obs)**
   - The state after taking the action
   - Used to compute the target Q-value: `Q_target = r + γ * max Q(s', a')`
   - Shape: `(num_agents, *obs_shape)`

5. **Hidden State (hidden)**
   - RNN hidden state for recurrent networks
   - Needed to properly compute Q-values for sequential data
   - Shape: `(num_agents, hidden_dim)`

6. **Communication Mask (comm_mask)**
   - Indicates which agents can communicate with each other
   - Used in the communication block of the network
   - Shape: `(num_agents, num_agents)` - boolean mask

7. **Q-values (q_val)**
   - The model's predicted Q-values at the time of collection
   - Used for prioritized experience replay (higher TD error = higher priority)
   - Shape: `(action_dim,)` - Q-value for each possible action

## Complete Training Flow

```
┌─────────────┐
│   Learner   │  ← Only this process updates weights (standard PyTorch loop)
│  (Training) │
└──────┬──────┘
       │ Stores weights every 5 iterations (ray.put)
       │
       ▼
┌─────────────────┐
│ Ray Shared Mem  │  ← Weights stored here
└────────┬────────┘
         │
         │ Actors pull weights (ray.get)
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
│Actor1│  │Actor2│  │Actor3│  │ActorN│  ← Multiple actors collect experience
└──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘
   │         │         │         │
   │         │         │         │
   │   1. Get observation from environment
   │   2. Model predicts action (inference)
   │   3. Take action, get reward
   │   4. Store experience in LocalBuffer
   │   5. Episode ends → send to GlobalBuffer
   │
   └─────────┴─────────┴─────────┘
              │
              ▼
      ┌───────────────┐
      │ GlobalBuffer  │  ← Experience stored here
      └───────┬───────┘
              │
              │ Samples batches
              ▼
         ┌─────────┐
         │ Learner │  ← Uses batches for training
         └─────────┘
```

### Experience Collection Flow (Detailed)

```
Actor interacts with environment:
┌─────────┐
│  Actor  │
└────┬────┘
     │
     │ 1. Get observation from environment
     ▼
┌─────────────┐
│ Environment │
└────┬────────┘
     │
     │ 2. Model predicts action (inference)
     ▼
┌─────────┐
│  Model  │ → action, q_val, hidden, comm_mask
└────┬────┘
     │
     │ 3. Take action, get reward
     ▼
┌─────────────┐
│ Environment │ → next_obs, reward, done
└────┬────────┘
     │
     │ 4. Store experience
     ▼
┌─────────────┐
│LocalBuffer  │ → (obs, action, reward, next_obs, hidden, comm_mask, q_val)
└────┬────────┘
     │
     │ 5. Episode ends → send to GlobalBuffer
     ▼
┌─────────────┐
│GlobalBuffer │ → Stores experiences from all Actors
└────┬────────┘
     │
     │ 6. Sample batch for training
     ▼
┌─────────┐
│ Learner │ → Uses experiences to train the model
└─────────┘
```

## How Training Works

### Key Differences from Standard PyTorch

#### Standard PyTorch Training:
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()  # Weights updated here
```

#### This Distributed Training:
```python
# Learner (does training - same as above!)
for batch in buffer.get_batch():
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    optimizer.step()  # Weights updated here (ONLY in Learner!)
    if step % 5 == 0:
        store_weights()  # Share with Actors

# Actors (just use weights, never update them!)
while True:
    action = model(obs)  # Inference only
    experience = env.step(action)
    buffer.add(experience)
    if step % N == 0:
        weights = get_weights_from_learner()  # Pull latest weights
        model.load_state_dict(weights)
```

### How Experiences Are Used for Training

In the Learner (`Learner.train()`):

1. **Sample batch of experiences** from GlobalBuffer
   ```python
   data = buffer.get_data()  # Get batch of experiences
   b_obs, b_action, b_reward, b_done, ... = data
   ```

2. **Compute current Q-values**
   ```python
   b_q = model(b_obs, ...).gather(1, b_action)  # Q(s, a)
   ```

3. **Compute target Q-values** (using next observations)
   ```python
   b_q_ = tar_model(b_obs, ...).max(1)[0]  # max Q(s', a')
   target = b_reward + (0.99 ** b_steps) * b_q_  # r + γ * max Q(s', a')
   ```

4. **Compute TD error and loss**
   ```python
   td_error = b_q - target  # TD error = Q(s,a) - (r + γ * max Q(s',a'))
   loss = (weights * huber_loss(td_error)).mean()
   ```

5. **Update weights** (standard PyTorch backprop)
   ```python
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()  # Update Q-network weights
   ```

## Weight Update Process

1. **Learner trains** (lines 327-334 in `worker.py`):
   - Standard PyTorch: `zero_grad()`, `backward()`, `optimizer.step()`
   - This is the **ONLY** place weights are updated

2. **Learner shares weights** (line 340):
   - Every 5 iterations: `self.store_weights()`
   - Uses `ray.put()` to store in shared memory

3. **Actors pull weights** (lines 420-425):
   - Periodically: `self.update_weights()`
   - Uses `ray.get()` to retrieve from shared memory
   - Loads into local model: `self.model.load_state_dict(weights)`

## Experience Storage

### LocalBuffer (per episode)
- Stores experiences for a single episode
- Created when episode starts, sent to GlobalBuffer when episode ends
- Location: `buffer.py`, lines 59-123

### GlobalBuffer (shared across all processes)
- Stores experiences from ALL Actors
- Uses prioritized experience replay (samples important experiences more often)
- Location: `worker.py`, lines 19-261

## Why This Architecture?

- **Parallelism**: Multiple Actors collect experience simultaneously
- **Efficiency**: One GPU does training, many CPUs collect experience
- **Stability**: Only one process updates weights (no gradient conflicts)
- **Scalability**: Easy to add more Actors without changing training code
- **Speed up data collection**: More episodes per second
- **Increase diversity**: Different actors explore different parts of the environment
- **Improve sample efficiency**: More diverse experiences = better training

## Key Code Locations

1. **Experience Collection**: `worker.py`, lines 432-468 (Actor.run())
2. **Experience Storage**: `buffer.py`, lines 87-97 (LocalBuffer.add())
3. **Experience Usage**: `worker.py`, lines 330-346 (Learner.train())
4. **TD Error Calculation**: `worker.py`, lines 343-346
5. **Weight Updates**: `worker.py`, lines 327-334 (Learner.train())
6. **Weight Sharing**: `worker.py`, lines 301-309 (Learner.store_weights())
7. **Weight Pulling**: `worker.py`, lines 470-485 (Actor.update_weights())

## Summary

**The training loop is standard PyTorch!** The distributed parts are just about sharing weights and collecting experience in parallel.

### The Complete Cycle:

1. **Actors collect experiences** by interacting with the environment
   - Experiences contain: (state, action, reward, next_state, hidden, comm_mask)
   - Experiences are stored in GlobalBuffer

2. **Learner samples experiences** and uses them to compute TD errors
   - TD errors are used to update the Q-network weights
   - Standard PyTorch: `zero_grad()`, `backward()`, `optimizer.step()`

3. **Learner shares updated weights** with Actors (every 5 iterations)
   - Uses Ray's shared memory (`ray.put()`)

4. **Actors pull latest weights** and use them for inference
   - Cycle repeats: Actors use new weights → collect new experiences → Learner trains on them

This is the standard **actor-learner** architecture for distributed RL!

