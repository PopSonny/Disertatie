import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize Pygame and font
pygame.init()
font = pygame.font.Font("arial.ttf", 25)


# Direction enum and Point
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple("Point", "x, y")

# RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREY = (40, 40, 40)

# Game constants
BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move logic
        self._move(action)
        self.snake.insert(0, self.head)

        # Check collisions
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Food logic
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Boundary collision
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        # Self collision
        if pt in self.snake[1:]:
            return True

        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):  # straight
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self):
        # Fill background
        self.display.fill(BLACK)

        # Draw grid
        for x in range(0, self.w, BLOCK_SIZE):
            pygame.draw.line(self.display, GREY, (x, 0), (x, self.h))
        for y in range(0, self.h, BLOCK_SIZE):
            pygame.draw.line(self.display, GREY, (0, y), (self.w, y))

        # Draw snake body as gradient circles
        for idx, pt in enumerate(self.snake):
            t = idx / len(self.snake)
            color = (
                int(BLUE1[0] * (1 - t) + BLUE2[0] * t),
                int(BLUE1[1] * (1 - t) + BLUE2[1] * t),
                int(BLUE1[2] * (1 - t) + BLUE2[2] * t),
            )
            center = (int(pt.x + BLOCK_SIZE / 2), int(pt.y + BLOCK_SIZE / 2))
            pygame.draw.circle(self.display, color, center, BLOCK_SIZE // 2 - 2)

        # Draw snake head with eyes
        head = self.snake[0]
        cx, cy = head.x + BLOCK_SIZE / 2, head.y + BLOCK_SIZE / 2
        eye_offset = BLOCK_SIZE // 4
        if self.direction == Direction.RIGHT:
            eye_positions = [
                (cx + eye_offset, cy - eye_offset),
                (cx + eye_offset, cy + eye_offset),
            ]
        elif self.direction == Direction.LEFT:
            eye_positions = [
                (cx - eye_offset, cy - eye_offset),
                (cx - eye_offset, cy + eye_offset),
            ]
        elif self.direction == Direction.UP:
            eye_positions = [
                (cx - eye_offset, cy - eye_offset),
                (cx + eye_offset, cy - eye_offset),
            ]
        else:  # DOWN
            eye_positions = [
                (cx - eye_offset, cy + eye_offset),
                (cx + eye_offset, cy + eye_offset),
            ]
        for ex, ey in eye_positions:
            pygame.draw.circle(self.display, WHITE, (ex, ey), 3)

        # Draw pulsing food circle
        phase = (pygame.time.get_ticks() // 200) % 2
        radius = BLOCK_SIZE // 2 - (2 if phase else 0)
        fx, fy = self.food.x + BLOCK_SIZE / 2, self.food.y + BLOCK_SIZE / 2
        pygame.draw.circle(self.display, RED, (fx, fy), radius)

        # Draw score
        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, (10, 10))

        # Update display
        pygame.display.flip()
