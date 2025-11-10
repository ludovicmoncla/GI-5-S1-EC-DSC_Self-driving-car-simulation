import pygame
import os
import math
import sys
import neat
import time
import pickle
import numpy as np

np.random.seed(42)
screen_width = 1500
screen_height = 800
generation = 0


class Car:
    """
    Represents a simulated car with position, movement, and radar sensors
    for use in a NEAT (NeuroEvolution of Augmenting Topologies) environment.
    """
    def __init__(self):
        """
        Initialize a car instance with position, speed, rotation, and sensor attributes.
        Loads the car image, sets up its position, rotation angle, and collision state.
        """
        self.surface = pygame.image.load(os.path.join(path, "assets", "img", "car.png"))
        self.surface = pygame.transform.scale(self.surface, (100, 100))
        self.rotate_surface = self.surface
        self.pos = [700, 650]
        self.angle = 0
        self.speed = 20
        self.center = [self.pos[0] + 50, self.pos[1] + 50]
        self.radars = []
        self.radars_for_draw = []
        self.is_alive = True
        self.goal = False
        self.distance = 0
        self.time_spent = 0
        self.rotation_angle = 2

    def draw(self, screen):
        """
        Draw the car and its radar lines on the screen.

        Args:
            screen (pygame.Surface): The Pygame screen surface to draw on.
        """
        screen.blit(self.rotate_surface, self.pos)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        """
        Draw radar sensor lines and endpoint circles.

        Args:
            screen (pygame.Surface): The Pygame surface to draw radar visuals on.
        """
        for r in self.radars:
            pos, _ = r
            pygame.draw.line(screen, (0, 255, 0), self.center, pos, 1)
            pygame.draw.circle(screen, (0, 255, 0), pos, 5)

    def check_collision(self, map):
        """
        Check whether the car has collided with white pixels on the map.

        Args:
            map (pygame.Surface): The game map image to test collisions against.
        """
        self.is_alive = True
        for p in self.four_points:
            if map.get_at((int(p[0]), int(p[1]))) == (255, 255, 255, 255):
                self.is_alive = False
                break

    def check_radar(self, degree, map):
        """
        Cast a radar ray at a given angle and record the distance to the nearest wall.

        Args:
            degree (float): The angle offset (in degrees) relative to the car's facing direction.
            map (pygame.Surface): The map surface to detect obstacles against.
        """
        len = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        while not map.get_at((x, y)) == (255, 255, 255, 255) and len < 300:
            len = len + 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * len)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * len)

        dist = int(math.sqrt(math.pow(x - self.center[0], 2) + math.pow(y - self.center[1], 2)))
        self.radars.append([(x, y), dist])

    def update(self, map):
        """
        Update car position, rotation, radar readings, and collision detection.

        Args:
            map (pygame.Surface): The track image used for radar and collision detection.
        """
        self.rotate_surface = self.rot_center(self.surface, self.angle)
        self.pos[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        if self.pos[0] < 20:
            self.pos[0] = 20
        elif self.pos[0] > screen_width - 120:
            self.pos[0] = screen_width - 120

        self.distance += self.speed
        self.time_spent += 1
        self.pos[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        if self.pos[1] < 20:
            self.pos[1] = 20
        elif self.pos[1] > screen_height - 120:
            self.pos[1] = screen_height - 120

        # caculate 4 collision points
        self.center = [int(self.pos[0]) + 50, int(self.pos[1]) + 50]
        len = 40
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * len]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * len]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * len]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * len, self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * len]
        self.four_points = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(map)
        self.radars.clear()

        self.check_radar(0, map)

    def get_data(self):
        """
        Return normalized radar distance data as inputs for the neural network.

        Returns:
            list[int]: A list of scaled radar distances.
        """
        radars = self.radars
        ret = [0, 0, 0, 0, 0]
        for i, r in enumerate(radars):
            ret[i] = int(r[1] / 30)
        return ret

    def get_alive(self):
        """
        Check whether the car is still active (not crashed).

        Returns:
            bool: True if the car has not collided, False otherwise.
        """
        return self.is_alive

    def get_reward(self):
        """
        Calculate fitness reward based on survival time.

        Returns:
            float: The time-based reward value.
        """
        return self.time_spent / 100

    def rot_center(self, image, angle):
        """
        Rotate an image while keeping its center position consistent.

        Args:
            image (pygame.Surface): Image surface to rotate.
            angle (float): Angle of rotation in degrees.

        Returns:
            pygame.Surface: Rotated image surface.
        """
        orig_rect = image.get_rect()
        rot_image = pygame.transform.rotate(image, angle)
        rot_rect = orig_rect.copy()
        rot_rect.center = rot_image.get_rect().center
        rot_image = rot_image.subsurface(rot_rect).copy()
        return rot_image


def display_execution_time(screen, start_time):
    """
    Display the total running time of the simulation.

    Args:
        screen (pygame.Surface): The screen surface to draw the text on.
        start_time (float): The start timestamp to calculate elapsed time from.
    """
    font = pygame.font.Font(None, 36)  # Set font and size
    elapsed_time = time.time() - start_time  # Compute elapsed time
    time_text = f"Time: {elapsed_time:.2f}s"
    
    text_surface = font.render(time_text, True, (0, 0, 0))  # Render text in white
    screen.blit(text_surface, (10, 70))  # Draw in top-left corner

def draw_button(screen):
    """
    Draw a "STOP" button on the screen.

    Args:
        screen (pygame.Surface): The screen surface to draw the button on.

    Returns:
        pygame.Rect: The rectangle area of the button for click detection.
    """
    button_color = (200, 0, 0)
    button_rect = pygame.Rect(700, 20, 80, 40)  # Position & size of "STOP" button
    font = pygame.font.Font(None, 30)
    stop_text = font.render("STOP", True, (255, 255, 255))

    pygame.draw.rect(screen, button_color, button_rect)
    screen.blit(stop_text, (button_rect.x + 20, button_rect.y + 10))

    return button_rect

def get_best_genome(genomes):
    """
    Retrieve the genome with the highest fitness score from a population.

    Args:
        genomes (list[tuple[int, neat.DefaultGenome]]): List of genome pairs (id, genome).

    Returns:
        neat.DefaultGenome: The genome object with the best fitness.
    """
    best_genome = None
    best_fitness = 0
    for _, genome in genomes:
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome = genome

    return best_genome

def run_car(genomes, config):
    """
    Execute one generation of the NEAT simulation.

    Initializes neural networks, simulates car movement, updates fitness, and
    stops the simulation when all cars crash or when manually stopped.

    Args:
        genomes (list[tuple[int, neat.DefaultGenome]]): List of genome pairs.
        config (neat.config.Config): The NEAT configuration object.
    """
    start_time = time.time()  
    stop_simulation = False

    # Initialize NEAT networks and cars
    nets = []
    cars = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

     # Initialize Pygame display
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    map = pygame.image.load(os.path.join(path, 'maps', 'map.png'))

    # Main simulation loop
    global generation
    generation += 1
    while True:
        screen.blit(map, (0, 0))
        button_rect = draw_button(screen)  # Draw STOP button

        # Handle user events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                    if button_rect.collidepoint(event.pos):  # If STOP button is clicked
                        stop_simulation = True

        # Neural network decision-making
        for index, car in enumerate(cars):
            output = nets[index].activate(car.get_data())
            i = output.index(max(output)) 
            if i == 0:
                car.angle += car.rotation_angle
            else:
                car.angle -= car.rotation_angle

        # Update car physics and fitness
        remain_cars = 0
        for i, car in enumerate(cars):
            if car.get_alive():
                remain_cars += 1
                car.update(map)
                genomes[i][1].fitness += car.get_reward()

        if remain_cars == 0:
            break

        # Drawing
        for car in cars:
            if car.get_alive():
                car.draw(screen)

        # Display generation info and time
        font = pygame.font.Font(None, 36) 
        text = font.render("Generation : " + str(generation), True, (0, 0, 0))
        screen.blit(text, (10, 10))
        text = font.render("Remain cars : " + str(remain_cars), True, (0, 0, 0))
        screen.blit(text, (10, 40))
        display_execution_time(screen, start_time) 

        if stop_simulation: 
            print("Simulation stopped. Saving best genome...")
            with open(os.path.join(path,"best_genome.pkl"), "wb") as f:
                pickle.dump(get_best_genome(genomes), f)
                print("Best genome saved!")
            sys.exit(0)

        pygame.display.flip()
        clock.tick(0)

def train(config):
    """
    Train the NEAT population using the provided configuration.

    Args:
        config (neat.config.Config): NEAT configuration object.
    """
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    p.run(run_car, 1000)


if __name__ == "__main__":
    # Initialize configuration and start training
    path = os.path.dirname(__file__)
    config_path = os.path.join(path, "config-feedforward.txt")
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    train(config)
    