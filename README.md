
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


## Session Instructions

* [Session 1 — Getting Started and Tuning the Simulation](session-1.md)
* [Session 2 — Improving the Learning Strategy](session-2.md)

The detailed instructions for Session 3 are provided below.

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

### 1. Implement a Core Improvement — 50 minutes

Choose at least one of the following mechanisms to implement:

* Finish-line detection, used to reward a completed lap or start a new generation when a car reaches the end of the track.
* Wrong-way detection, used to identify and penalise movement in the incorrect direction.

Completing one robust and well-tested mechanism is preferable to attempting both without validating them.

* Define the expected behaviour before writing the implementation.
* Connect the new mechanism to the fitness function and check that its reward or penalty has the intended influence on learning.

#### Implementation hints

When implementing the new mechanisms, consider the following:
In the `Car` class, `self.pos` is the top-left corner of the sprite and `self.center` is the centre already used for radar and collision calculations. `Car.__init__()` sets the initial centre, and `Car.update()` recalculates it after each move. Inside a new `Car` method (e.g., `def check_finish_line(self):`), use `x, y = self.center` to read the current coordinates.

The position of the finish line or checkpoints can be hard-coded in the method, or passed as parameters. Use the same coordinates for both drawing and detection.
> In the default map, the finish line is a vertical segment from `(650, 650)` to `(650, 750)`, crossed from left to right. 

##### Track the previous position

Initialise the previous coordinates in `Car.__init__()` immediately after assigning `self.center`.

```python
self.prev_x, self.prev_y = self.center
```

In `run_car()`, find the `# Update car physics and fitness` loop. For each living car, run your checks immediately before `car.update(map)`. At this point, the centre reflects the previous loop iteration's move. Once all your checks have read the two positions, save the current centre as the previous position, then let `car.update(map)` move the car again.

Update both saved coordinates every frame, even when the car crosses nothing. If you implement both mechanisms, keep the history update in one place so both checks see the same movement. On the first frame the two positions are equal, so there is no crossing.

##### Option A — Finish-line detection

1. On `maps/map.png`, try a vertical finish line from `(650, 650)` to `(650, 750)`, crossed from left to right. Draw it with `pygame.draw.line()` during rendering, using the same endpoints for drawing and detection.
2. Add a `check_finish_line(self)` method to `Car`. For a first version, test whether the current y-coordinate is in the finish line's span, then use the previous and current x-coordinates to work out whether the car reached or passed `x = 650` from the left. Fill in the missing crossing test and return the result.

   ```python
   def check_finish_line(self):
       x, y = self.center
       in_span = 650 <= y <= 750
       # TODO: Compare self.prev_x and x with 650 to check the direction.
       # TODO: Return True only when the span and crossing tests both pass.
   ```

   A move from `(647, 700)` to `(654, 700)` should count even though neither x-coordinate is exactly `650`. Reaching `650` from the left should count once. Starting the next frame at `650` and moving right should produce no second event. Use this pair of cases to decide which side of the inequality needs a strict comparison.

3. In `run_car()`, use the returned boolean to add a one-time bonus to `genomes[i][1].fitness`. If you want the first finish to end the generation, initialise a flag to `False` at the start of each frame, set it when any car finishes, and use it in the loop's stopping condition after processing all cars.

A correct crossing gives you a finish event. If your design needs to verify a full lap, an extension is to require the car to pass through checkpoints in order before accepting the finish.

##### Option B — Wrong-way detection

The expected direction changes around the track. Place **directed checkpoints** across a few straight stretches so you can compare the actual direction of each crossing with the expected direction at that location. Start with one checkpoint, test both directions, then add more.

In `run_car()`, a checkpoint can be a tuple of `(start, end, expected_direction)`. On `maps/map.png`, you can start with these two examples and use `pygame.draw.line()` to check where they land.

```python
check_points = [
    ((1180, 350), (1350, 350), "up"),
    ((700, 50), (700, 150), "left"),
]
```

Add `check_checkpoints(self, checkpoints)` to `Car` and pass `check_points` when you call it. Start by unpacking each checkpoint with `for start, end, expected_direction in checkpoints:`. Use the endpoints to choose which coordinates to compare.

| Checkpoint | Test the segment's span using | Compare across frames to find direction |
| --- | --- | --- |
| Vertical (same x at both ends) | `y` against the endpoint y-values | `self.prev_x` and `x`. Increasing x means right. |
| Horizontal (same y at both ends) | `x` against the endpoint x-values | `self.prev_y` and `y`. Decreasing y means up in Pygame. |

For the first checkpoint above, use the finish-line idea with the axes swapped. At `x = 1250`, a move from `y = 355` to `y = 345` crosses the line upwards. The reverse move crosses downwards and should trigger a wrong-way event. The same moves at `x = 1100` fall outside the segment. Derive the equivalent tests for left and right crossings of the second checkpoint.

Distinguish three outcomes in your method. A crossing in the expected direction can increment a `self.checkpoints` counter, which starts at zero in `__init__()`. A crossing in the opposite direction should report a wrong-way event. With no crossing, leave the counter and return without reporting an event. This method checks direction at the checkpoint lines, so wrong-way movement between lines may go undetected.

##### Connect the event to fitness and test it

* Keep the Session 2 reward as your baseline. Apply a finish bonus or wrong-way penalty once per event in `run_car()`, alongside the existing `get_reward()` call. A cumulative term such as `laps_completed * bonus` inside `get_reward()` would add the same bonus again on every subsequent frame.
* Test with fixed pairs of centre coordinates before relying on training runs. For either detector, try a valid crossing, a reverse crossing, crossing outside the segment's span, a stationary car on the line, and the frame immediately after a crossing.
* If the span test behaves unexpectedly on fast diagonal moves near an endpoint, try computing where the movement between the two centres intersects the line and test the span at that point.
* Compare runs with and without the bonus or penalty using the same map and Session 2 settings. Record the value you used and an observable outcome, such as how often cars reach the finish or cross a checkpoint backwards.

### 2. Test a New Map — 50 minutes

If the core improvement is complete:

* Save the best model trained on the original map.
* Select a map with a different road layout.
* Run that model on the new map without retraining it first.
* Compare its behaviour before and after additional training.

> **Note:** You'll find the test() function code in the test-function.py file. You can use it to load a saved model and run it on a new map. Copy the function into your simulation script and call it with the appropriate parameters.

### 3. Final Summary

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
