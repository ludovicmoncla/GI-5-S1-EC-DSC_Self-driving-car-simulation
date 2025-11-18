![INSA](https://www.insa-lyon.fr/sites/all/themes/insa/logo.png)


# GI-5-S1-EC-DSC - Self-driving car simulation

This git repository contains the tutorial files for the 2025-26 Data Science course of 5GI INSA Lyon (`GI-5-S1-EC-DSC`).

In this lab, you will explore how artificial agents can learn to drive autonomously without any explicit programming of their behavior.
Using a simplified 2D car simulator, each car is controlled by a small neural network that takes as input a set of distance sensors and outputs steering commands.

Instead of classic reinforcement learning, this lab uses NEAT, an evolutionary algorithm that achieves a similar goal — learning through reward and adaptation — but is simpler to set up and better suited for the time available in this lab session.

Over multiple generations, a population of neural networks evolves: those that perform better (drive farther without crashing) are selected, mutated, and recombined to produce better drivers.

This experiment demonstrates how complex behavior can emerge naturally from simple evolutionary principles, contextual feedback, and reward signals.


## Academic Integrity and Authorized Resources

To ensure fair evaluation and genuine learning, please follow these rules when completing the lab:

✅ You are allowed to:
* Search for information using search engines (Google, Bing, etc.)
* Consult official documentation (e.g., [NEAT-Python](https://neat-python.readthedocs.io/en/latest/), [Pygame](https://www.pygame.org/docs/tut/PygameIntro.html))
* Use educational or technical websites such as Stack Overflow, GeeksforGeeks, or academic blogs
* Discuss ideas and concepts with classmates at a high level (no code sharing)
* Ask the teacher questions during the lab sessions if you need clarification or guidance

🚫 You are not allowed to:
* Use generative AI tools (e.g., ChatGPT, Copilot, Gemini, Claude, etc.) to generate or modify code or written explanations
* Submit code, text, or analyses that were not written or understood by you
* Copy solutions directly from other students or online repositories

🧭 Important note

You are encouraged to understand, test, and experiment with the concepts presented in this lab.
External sources can help you learn — but all submitted work must reflect your own understanding and be authored by you.


## Overview

This tutorial will guide you through setting up and improving a self-driving car simulation using NEAT (NeuroEvolution of Augmenting Topologies) and Pygame. You'll learn how to modify the neural network's behavior, tweak simulation parameters, and introduce new control strategies.


### NEAT

[NEAT](https://neat-python.readthedocs.io/en/latest/) stands for NeuroEvolution of Augmenting Topologies (Stanley & Miikkulainen, 2002).
It is an evolutionary algorithm designed to evolve both the weights and the structure (topology) of neural networks.

Unlike traditional neural networks with a fixed architecture, NEAT starts with very simple networks — often just inputs directly connected to outputs — and gradually increases complexity as needed.

The key mechanisms of NEAT are:
* Weight mutation: randomly adjusts connection weights to fine-tune behaviors.
* Structural mutation: adds new neurons or new connections over time, allowing networks to grow in complexity.
* Speciation: groups similar networks into species, protecting innovation so that new structures can evolve without being immediately discarded.

The goal of evolution is to maximize a fitness function. In this project it is typically the distance traveled before crashing but will you will have to improve it by proposing a better reward function.

After several generations, NEAT often produces neural networks that display emergent, intelligent behaviors — such as turning smoothly, avoiding walls, and following the track — all learned automatically through evolution.

A good introduction to NEAT can be found in this video:
https://www.youtube.com/watch?v=yVtdp1kF0I4

### Pygame

[Pygame](https://www.pygame.org/docs/tut/PygameIntro.html) is a library used to create the simulation environment. It handles rendering, physics, and user interactions.


## Getting started

Follow these steps to set up and run the simulation on your computer.

### Clone the Repository (or Download as ZIP)

```bash
git clone https://github.com/lmoncla/GI-5-S1-EC-DSC_Self-driving-car-simulation.git
```
### (Optional) Option 1. Create a Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### (Optional) Option 2. Create a Conda Environment

```bash
conda create --name self_driving_car python=3.11
conda activate self_driving_car
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Project Structure

### Files Description
* `self-driving-car-simulation.py`: Main simulation script that runs the self-driving car simulation.
* `config-feedforward.txt`: Configuration file for the NEAT algorithm.
* `maps/`: Directory containing different map layouts for the simulation.
* `README.md`: This document, describing the tutorial and experiment tasks.



## Running the Simulation

To start the simulation, simply run the file from your IDE with the following command:

```python
python self-driving-car-simulation.py
```

Make sure the `config-feedforward.txt` file is in the same directory as the simulation script.

* What happens when you run the simulation?


## Improving the Simulation

### 1. Tweak Parameters

* Adjust variables from the `Car` class from `self-driving-car-simulation.py` such as:
    - the speed
    - the rotation angle
    - the number of sensors and sensor angle

* Adjust variables from the NEAT `config-feedforward.txt` file such as:
    - the number of cars per generation 
    - fitness threshold


Note: You should adjust those parameters one by one and observe how these changes affect the car's performance.

* What happens if the rotation angle is set to 90°? 

* What is the best combination of parameters for optimal performance?

---
> Default parameters:
>  * speed = 2 
>  * rotation_angle = 2
>  * cars per generation = 10
>  * only one forward-facing sensor

---


### 2. Designing a Better Reward Function

Redesigning the reward function is one of the most effective ways to improve the agent’s learning performance and encourage more realistic driving behavior.

By default, the reward function is mainly based on the time the car remains alive on the track.
While simple, this can lead to undesirable behaviors — for example, the car might spin in circles to stay alive longer instead of progressing along the road.

To encourage more realistic driving, try modifying the `get_reward()` function to include factors such as:
* Forward progress (distance traveled in the right direction),
* Penalties for collisions or spinning,
* Bonuses for speed or smooth turns.

Experiment by changing the reward function and observe how it affects the car’s behavior and learning performance.

### 3. Activation Function

* Experiment with different activation functions in the output layer and compare their effects.
* Consider alternative strategies for interpreting the network output:
    - Default: `i = output.index(max(output))` 
    - Solution 1: Use a decision threshold
    - Solution 2: Compute a weighted combination of outputs (e.g., transform separate left/right turning values into a single steering output)


### 4. Speed Control

* Add an output variable for controlling speed dynamically. 


### 5. Finish Line Detection

* Implement a finish line detection mechanism. This can be used to:
    - launch a new generation when a car reaches the end of the track,
    - reward cars that reach the end of the track.


### 6. Wrong-way Detection

* Add a mechanism to detect when a car is going in the wrong direction and penalize it accordingly.



### 7. Experiment with a New Map

* Create a new map with different obstacles and road configurations
* Save the model trained on the first map and run it on the new map



## Summary and Further Exploration

Through this lab, you have seen how complex driving behaviors can emerge from simple evolutionary principles.

To go further, you could replace the NEAT algorithm with a **deep reinforcement learning** approach such as a **Deep Q-Network (DQN)**, where a neural network learns to approximate the **Q-function** that maps sensor inputs to driving actions.
Comparing both methods would highlight the differences between **evolutionary learning** and **reward-based optimization**, offering deeper insight into how autonomous agents can learn from interaction and feedback.