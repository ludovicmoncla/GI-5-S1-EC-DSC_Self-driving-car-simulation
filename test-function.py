
# This code should be copy-pasted in the main file self-driving-car-simulation.py after the train() function.
# Then, you may change the main() function to call test() instead of train() to test the best genome on a new map.

# This code is for testing the best genome on a new map. 
# It loads the genome from a file, creates a neural network, and runs the car simulation on the specified map.
def test(config, genome_filepath, map_filepath):
    # use the best genom and run it on a new map
    with open(genome_filepath, "rb") as f:
        genome = pickle.load(f)
    
    # Create the neural network
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    # Init my game
    pygame.init()
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()
    map = pygame.image.load(map_filepath)
    # Initialize the car
    car = Car()
    running = True
    while car.is_alive and running:
        #screen.fill((0, 0, 0))  # Clear screen
        screen.blit(map, (0, 0))

        # Neural network controls the car
        turn_left, turn_right, speed_control = net.activate(car.get_data())
        if turn_left > 0.5:
                car.angle += car.rotation_angle
        if turn_right > 0.5:
                car.angle -= car.rotation_angle
        car.speed = max(4, min(15, car.speed + (speed_control - 0.5) * 2))  # Adjust speed
        
        if car.get_alive():
            car.update(map)
            car.draw(screen)

        # Refresh display
        pygame.display.flip()
        clock.tick(0)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

    print("Test completed!")
    pygame.quit()


    while car.is_alive:
        output = net.activate(car.get_data())  # Get action from NN
        car.update(output)  # Move the car

        # Display current progress
        print(f"Speed: {car.speed}, Checkpoints: {len(car.visited_checkpoints)}, Laps: {car.laps_completed}")

        # Stop if the car crashes
        if not car.is_alive:
            break

    print("Test completed!")