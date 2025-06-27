import os
import random
import pygame

# Window always on primary monitor
os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 1100, 600
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
FONT = pygame.font.Font("freesansbold.ttf", 20)
TITLE_FONT = pygame.font.Font("freesansbold.ttf", 30)

# Load assets
ASSET_DIR = os.path.join(os.path.dirname(__file__), "Assets")

def load_image(*path):
    return pygame.image.load(os.path.join(ASSET_DIR, *path))

DINO_RUN = [load_image("Dino", "DinoRun1.png"), load_image("Dino", "DinoRun2.png")]
DINO_JUMP = load_image("Dino", "DinoJump.png")
DINO_DUCK = [load_image("Dino", "DinoDuck1.png"), load_image("Dino", "DinoDuck2.png")]

SMALL_CACTI = [load_image("Cactus", f"SmallCactus{i+1}.png") for i in range(3)]
LARGE_CACTI = [load_image("Cactus", f"LargeCactus{i+1}.png") for i in range(3)]
BIRDS = [load_image("Bird", "Bird1.png"), load_image("Bird", "Bird2.png")]

CLOUD_IMG = load_image("Other", "Cloud.png")
GROUND_IMG = load_image("Other", "Track.png")

# Game classes
class Dino:
    X, Y, Y_DUCK = 80, 310, 340
    JUMP_VELOCITY = 8.5

    def __init__(self):
        self.images = {'run': DINO_RUN, 'duck': DINO_DUCK, 'jump': DINO_JUMP}
        self.state = 'run'
        self.step = 0
        self.velocity = self.JUMP_VELOCITY
        self.image = self.images['run'][0]
        self.rect = self.image.get_rect(x=self.X, y=self.Y)

    def update(self, keys):
        if self.state == 'duck':
            self.duck()
        elif self.state == 'run':
            self.run()
        elif self.state == 'jump':
            self.jump()

        if self.step >= 10:
            self.step = 0

        if keys[pygame.K_SPACE] and self.state != 'jump':
            self.state = 'jump'
        elif keys[pygame.K_DOWN] and self.state != 'jump':
            self.state = 'duck'
        elif not keys[pygame.K_DOWN] and self.state != 'jump':
            self.state = 'run'

    def duck(self):
        self.image = self.images['duck'][self.step // 5]
        self.rect = self.image.get_rect(x=self.X, y=self.Y_DUCK)
        self.step += 1

    def run(self):
        self.image = self.images['run'][self.step // 5]
        self.rect = self.image.get_rect(x=self.X, y=self.Y)
        self.step += 1

    def jump(self):
        self.image = self.images['jump']
        self.rect.y -= self.velocity * 4
        self.velocity -= 0.8
        if self.velocity < -self.JUMP_VELOCITY:
            self.velocity = self.JUMP_VELOCITY
            self.state = 'run'
            self.rect.y = self.Y

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD_IMG
        self.width = self.image.get_width()

    def update(self):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, surface):
        surface.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, images):
        self.type = random.randint(0, len(images) - 1)
        self.image = images[self.type]
        self.rect = self.image.get_rect(x=SCREEN_WIDTH)

    def update(self):
        self.rect.x -= game_speed
        if self.rect.right < 0:
            obstacles.remove(self)

    def draw(self, surface):
        surface.blit(self.image, self.rect)


class Bird(Obstacle):
    def __init__(self):
        super().__init__(BIRDS)
        self.rect.y = 250
        self.index = 0

    def draw(self, surface):
        surface.blit(BIRDS[self.index // 5], self.rect)
        self.index = (self.index + 1) % 10


# Game functions
def draw_background():
    global ground_x
    width = GROUND_IMG.get_width()
    SCREEN.blit(GROUND_IMG, (ground_x, ground_y))
    SCREEN.blit(GROUND_IMG, (ground_x + width, ground_y))
    if ground_x <= -width:
        ground_x = 0
    ground_x -= game_speed


def draw_score():
    global score, game_speed
    score += 1
    if score % 100 == 0:
        game_speed += 1

    score_surface = FONT.render(f"Points: {score}", True, (0, 0, 0))
    SCREEN.blit(score_surface, (850, 40))


def main():
    global game_speed, ground_x, ground_y, score, obstacles
    clock = pygame.time.Clock()
    run_game = True
    player = Dino()
    cloud = Cloud()
    ground_x, ground_y = 0, 380
    game_speed = 20
    score = 0
    obstacles = []

    while run_game:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run_game = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                exit()

        SCREEN.fill((255, 255, 255))
        keys = pygame.key.get_pressed()

        player.update(keys)
        player.draw(SCREEN)

        if not obstacles:
            choice = random.choice([SmallCactus(SMALL_CACTI), LargeCactus(LARGE_CACTI), Bird()])
            obstacles.append(choice)

        for obs in list(obstacles):
            obs.update()
            obs.draw(SCREEN)
            if player.rect.colliderect(obs.rect.inflate(-80, 0)):
                pygame.time.delay(1000)
                return menu(1)

        cloud.update()
        cloud.draw(SCREEN)

        draw_background()
        draw_score()

        clock.tick(30)
        pygame.display.update()


class SmallCactus(Obstacle):
    def __init__(self, images):
        super().__init__(images)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, images):
        super().__init__(images)
        self.rect.y = 300


def menu(deaths):
    global score
    showing_menu = True

    while showing_menu:
        SCREEN.fill((255, 255, 255))
        title = "Press W to Start" if deaths == 0 else "Game Over"
        text_surface = TITLE_FONT.render(title, True, (0, 0, 0))
        text_rect = text_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        SCREEN.blit(text_surface, text_rect)

        if deaths > 0:
            score_surface = TITLE_FONT.render(f"Your Score: {score}", True, (0, 0, 0))
            score_rect = score_surface.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            SCREEN.blit(score_surface, score_rect)

        SCREEN.blit(DINO_RUN[0], (SCREEN_WIDTH//2 - 20, SCREEN_HEIGHT//2 - 140))
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    exit()
                elif event.key == pygame.K_w:
                    return main()


if __name__ == "__main__":
    menu(0)
