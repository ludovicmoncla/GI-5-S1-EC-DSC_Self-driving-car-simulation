
# GI-5-S1-EC-DSC - Self-driving car simulation

This git repository contains the tutorial files for the 2025-26 Data Science course of 5GI INSA Lyon (`GI-5-S1-EC-DSC`).

In this lab, you will explore how artificial agents can learn to drive autonomously without any explicit programming of their behavior.
Using a simplified 2D car simulator, each car is controlled by a small neural network that takes as input a set of distance sensors and outputs steering commands.

Instead of classic reinforcement learning, this lab uses NEAT, an evolutionary algorithm that achieves a similar goal — learning through reward and adaptation — but is simpler to set up and better suited for the time available in this lab session.

Over multiple generations, a population of neural networks evolves: those that perform better (drive farther without crashing) are selected, mutated, and recombined to produce better drivers.

This experiment demonstrates how complex behavior can emerge naturally from simple evolutionary principles, contextual feedback, and reward signals.


## Session Schedule

This lab is organised into three two-hour sessions:

| Session | Date and time | Location | Format |
| --- | --- | --- | --- |
| Session 1 | Tuesday, November 17, 10:00–12:00 | Room Thorgal | Supervised lab session |
| Session 2 | Tuesday, November 24, 10:00–12:00 | Room Thorgal | Supervised lab session |
| Session 3 | Monday, November 30, 10:00–12:00 | Room Thorgal | Independent work |

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


## Session 1 — Getting Started and Tuning the Simulation

### Learning Objectives

By the end of this session, you should be able to:

* Set up the project and run the initial simulation.
* Explain the main roles of Pygame, NEAT, the car sensors, and the fitness function.
* Observe and describe the baseline behaviour of the cars.
* Experiment with the car speed, rotation angle, population size, and sensor configuration.
* Compare several configurations and justify which one performs best.

### Suggested Schedule

* **10:00–10:15:** Introduction, setup, and instructions.
* **10:15–10:35:** Run and analyse the baseline simulation.
* **10:35–11:40:** Modify parameters one at a time, record the results, compare configurations and identify a promising combination.
* **11:40–12:00:** Save the results and prepare a short summary for the next session.

### 1. Run and Observe the Baseline Simulation

Install the required dependencies (see the **Getting started** section below) and run the simulation with the default parameters. Observe how the cars behave.
Start the simulation from your IDE with the following command:

```bash
python self-driving-car-simulation.py
```

Make sure the `config-feedforward.txt` file is in the same directory as the simulation script.

Before making any changes, observe the simulation and consider the following questions:

* How do the cars behave?
* What information do their sensors provide?

### 2. Tweak the Parameters

* Adjust variables from the `Car` class from `self-driving-car-simulation.py` such as:
    * The speed.
    * The rotation angle.
    * The number and angles of the sensors.

* Adjust variables from the NEAT `config-feedforward.txt` file such as:
    * The number of cars per generation.
    * The fitness threshold.

Adjust the parameters one at a time and record how each change affects the cars' performance.

* What happens if the rotation angle is set to 90°?

* What is the best combination of parameters for optimal performance?

---

> **Default parameters:**
>
> * `speed = 2`
> * `rotation_angle = 2`
> * 10 cars per generation
> * One forward-facing sensor

---

### Expected Outcome

At the end of the session, keep a record of:

* The parameters and configurations you tested.
* The behaviour observed during each experiment.
* The configuration you selected and the reasons for your choice.


# Session 2 — Improving the Learning Strategy

**Date:** Tuesday, November 24  
**Time:** 10:00–12:00  
**Location:** Room Thorgal  
**Format:** Supervised lab session

## Learning Objectives

By the end of this session, students should be able to:

* Explain the limitations of a fitness function based only on survival time.
* Design a fitness function that encourages useful driving behaviour.
* Compare different ways of interpreting the neural network outputs.
* Evaluate changes through controlled experiments.
* Optionally add dynamic speed control to the car.

## Suggested Activities

The order and timings below are provided as guidance only. You may complete the activities in a different order and adjust the time spent on each one according to your progress.

### 1. Review the Baseline — 15 minutes

Run the best configuration retained after Session 1 and identify any remaining undesirable behaviours, such as driving in circles, moving too slowly, or surviving without making progress.

### 2. Improve the Fitness Function — 50 minutes

Modify `get_reward()` and test combinations of the following criteria:

* Survival time.
* Distance travelled or forward progress.
* Collision penalties.
* Penalties for spinning or failing to make progress.
* Bonuses for smooth steering or maintaining an appropriate speed.

Change one factor at a time whenever possible. Record the fitness formulation, parameter values, and observed behaviour for each experiment.

The default reward is mainly based on the time for which a car remains alive. This may encourage undesirable behaviour: for example, a car could spin in circles instead of progressing along the road. The revised function should explicitly address this limitation.

### 3. Interpret the Network Outputs — 30 minutes

Compare at least two control strategies:

* Selecting the action associated with the largest output value: `i = output.index(max(output))`.
* Applying a decision threshold.
* Computing a weighted combination of the outputs, for example by transforming separate left and right values into a single steering command.

Students may also experiment with different activation functions and discuss their effect on the controls.

### 4. Extension: Dynamic Speed Control — 15 minutes

If time permits, add a neural network output that controls the car speed. Define a safe speed range and observe whether the car learns to slow down before difficult turns.

### 5. Wrap-up — 10 minutes

Select the most convincing fitness function and control strategy. Save the configuration and write a short explanation of why they were retained.

## Expected Outcome

Students should produce:

* An improved and documented fitness function.
* A comparison of at least two experiments.
* A short explanation linking the chosen reward criteria to the resulting driving behaviour.
* Optionally, a working implementation of dynamic speed control.

# Session 3 — Improving Robustness and Evaluating the Model

**Date:** Monday, November 30  
**Time:** 10:00–12:00  
**Location:** Room Thorgal  
**Format:** Independent work

## Learning Objectives

By the end of this session, students should be able to:

* Implement and test an additional mechanism that makes the simulation more robust.
* Define an appropriate reward or penalty for the new mechanism.
* Assess whether a trained model generalises to a different track.
* Summarise and justify the main design decisions made during the lab.

## Suggested Activities

The order and timings below are provided as guidance only. You may complete the activities in a different order and adjust the time spent on each one according to your progress.

### 1. Implement a Core Improvement — 60 minutes

Choose at least one of the following mechanisms to implement:

* Finish-line detection, used to reward a completed lap or start a new generation when a car reaches the end of the track.
* Wrong-way detection, used to identify and penalise movement in the incorrect direction.

Completing one robust and well-tested mechanism is preferable to attempting both without validating them.

* Define the expected behaviour before writing the implementation. 
* Connect the new mechanism to the fitness function and check that its reward or penalty has the intended influence on learning.

### 3. Extension: Test a New Map — 25 minutes

If the core improvement is complete:

* Select a map with a different road layout.
* Save the best model trained on the original map.
* Run that model on the new map without retraining it first.
* Compare its behaviour before and after additional training.

### 4. Final Summary

Prepare a concise summary containing:

* The parameters and sensor configuration retained.
* The final fitness function and the reasoning behind it.
* The additional mechanism implemented during this session.
* Evidence from the experiments.
* Limitations and possible future improvements.

## Expected Outcome

Students should produce:

* One functional and tested improvement: finish-line detection or wrong-way detection.
* Experimental results showing its effect on the simulation.
* A concise synthesis of the work completed across the three sessions.
* Optionally, an evaluation of the trained model on a new map.


## Summary and Further Exploration

Through this lab, you have seen how complex driving behaviors can emerge from simple evolutionary principles.

To go further, you could replace the NEAT algorithm with a **deep reinforcement learning** approach such as a **Deep Q-Network (DQN)**, where a neural network learns to approximate the **Q-function** that maps sensor inputs to driving actions.
Comparing both methods would highlight the differences between **evolutionary learning** and **reward-based optimization**, offering deeper insight into how autonomous agents can learn from interaction and feedback.



## Getting started

Follow these steps to set up and run the simulation on your computer.
Python 3.13 is recommended for this project.

### Clone the Repository (or Download as ZIP)

```bash
git clone https://github.com/ludovicmoncla/GI-5-S1-EC-DSC_Self-driving-car-simulation.git
```

### (Optional) Option 1. Create a Virtual Environment
```bash
python3.13 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### (Optional) Option 2. Create a Conda Environment

```bash
conda create --name self_driving_car python=3.13
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
