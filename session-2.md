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

