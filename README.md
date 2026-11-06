
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
* **10:35–11:30:** Modify parameters one at a time and record the results.
* **11:30–11:50:** Compare configurations and identify a promising combination.
* **11:50–12:00:** Save the results and prepare a short summary for the next session.

### 1. Run and Observe the Baseline Simulation

Start the simulation from your IDE with the following command:

```bash
python self-driving-car-simulation.py
```

Make sure the `config-feedforward.txt` file is in the same directory as the simulation script.

Before making any changes, observe the simulation and consider the following questions:

* How do the cars behave?
* What information do their sensors provide?
* How does NEAT evaluate and select the cars between generations?

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
