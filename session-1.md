# Session 1 — Getting Started and Tuning the Simulation

**Date:** Tuesday, November 17  
**Time:** 10:00–12:00  
**Location:** Room Thorgal  
**Format:** Supervised lab session

## Learning Objectives

By the end of this session, you should be able to:

* Set up the project and run the initial simulation.
* Explain the main roles of Pygame, NEAT, the car sensors, and the fitness function.
* Observe and describe the baseline behaviour of the cars.
* Experiment with the car speed, rotation angle, population size, and sensor configuration.
* Compare several configurations and justify which one performs best.

## Suggested Schedule

* **10:00–10:15:** Introduction, setup, and instructions.
* **10:15–10:35:** Run and analyse the baseline simulation.
* **10:35–11:40:** Modify parameters one at a time, record the results, compare configurations and identify a promising combination.
* **11:40–12:00:** Save the results and prepare a short summary for the next session.

## 1. Run and Observe the Baseline Simulation

Install the required dependencies (see the **Getting started** section in the [README](README.md#getting-started)) and run the simulation with the default parameters. Observe how the cars behave.
Start the simulation from your IDE with the following command:

```bash
python self-driving-car-simulation.py
```

Make sure the `config-feedforward.txt` file is in the same directory as the simulation script.

Before making any changes, observe the simulation and consider the following questions:

* How do the cars behave?
* What information do their sensors provide?

## 2. Tweak the Parameters

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

## Expected Outcome

At the end of the session, keep a record of:

* The parameters and configurations you tested.
* The behaviour observed during each experiment.
* The configuration you selected and the reasons for your choice.
