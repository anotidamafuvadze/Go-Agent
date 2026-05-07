import pygame
import sys
from go_search_problem import GoProblem, GoState


class GoGUI:
    # Define GUI colors
    BOARD = (210, 180, 140)  # brown
    EMPTY = (0, 0, 0)  # black
    P1 = (0, 0, 0)  # black
    P2 = (255, 255, 255)  # white
    BUTTON = (200, 200, 200)  # grey
    BUTTON_HOVER = (180, 180, 180)  # darker grey
    BUTTON_TEXT = (0, 0, 0)  # black
    COLOR_MAP = [EMPTY, P1, P2]

    def __init__(self, problem: GoProblem):
        # Initialize Pygame
        print("Setting up Board...")
        print("Use the arrow keys to navigate and the enter key to select an action.")
        pygame.init()

        # Constants
        self.WIDTH, self.HEIGHT = 600, 700  # Increased height for pass button
        self.BOARD_SIZE = problem.start_state.size
        self.CELL_SIZE = 600 // self.BOARD_SIZE  # Using original width for board
        
        # Pass button dimensions
        self.BUTTON_WIDTH = 100
        self.BUTTON_HEIGHT = 40
        self.BUTTON_X = (self.WIDTH - self.BUTTON_WIDTH) // 2
        self.BUTTON_Y = 620  # Position below the board
        self.BUTTON_COLOR = self.BUTTON

        # Set up the display
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Go Game")

        # Initialize font
        self.font = pygame.font.Font(None, 36)

        self.problem = problem
        self.state = problem.start_state
        self.cursor_pos = [self.BOARD_SIZE // 2, self.BOARD_SIZE // 2]

    def render(self):
        self.screen.fill(self.BOARD)
        self.draw_board()
        self.draw_pieces()
        self.draw_cursor()
        self.draw_pass_button()
        pygame.display.flip()

    def draw_pass_button(self):
        # Check if mouse is hovering over button
        mouse_pos = pygame.mouse.get_pos()
        button_rect = pygame.Rect(self.BUTTON_X, self.BUTTON_Y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        button_color = self.BUTTON_HOVER if button_rect.collidepoint(mouse_pos) else self.BUTTON_COLOR

        # Draw button
        pygame.draw.rect(self.screen, button_color, button_rect)
        pygame.draw.rect(self.screen, self.BUTTON_TEXT, button_rect, 2)  # Border

        # Draw text
        text = self.font.render("PASS", True, self.BUTTON_TEXT)
        text_rect = text.get_rect(center=button_rect.center)
        self.screen.blit(text, text_rect)

    def process_window_event(self, event):
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    def is_pass_button_clicked(self, pos):
        button_rect = pygame.Rect(self.BUTTON_X, self.BUTTON_Y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT)
        return button_rect.collidepoint(pos)

    def get_user_input_action(self):
        for event in pygame.event.get():
            self.process_window_event(event)
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    if self.is_pass_button_clicked(event.pos):
                        return self.BOARD_SIZE * self.BOARD_SIZE  # Pass move
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
                elif event.key == pygame.K_DOWN:
                    self.cursor_pos[1] = min(
                        self.BOARD_SIZE - 1, self.cursor_pos[1] + 1)
                elif event.key == pygame.K_LEFT:
                    self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
                elif event.key == pygame.K_RIGHT:
                    self.cursor_pos[0] = min(
                        self.BOARD_SIZE - 1, self.cursor_pos[0] + 1)
                elif event.key == pygame.K_RETURN:
                    return self.cursor_pos[1] * self.BOARD_SIZE + self.cursor_pos[0]
                elif event.key == pygame.K_SPACE:  # Added space as alternative for pass
                    return self.BOARD_SIZE * self.BOARD_SIZE
        return None

    def update_state(self, action):
        if action is not None and action in self.problem.get_available_actions(self.state):
            self.state = self.problem.transition(self.state, action)
        elif action is not None:
            self.state = action

    def draw_cursor(self):
        x, y = self.cursor_pos
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (x * self.CELL_SIZE, y * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE), 3)

    def draw_board(self):
        for i in range(self.BOARD_SIZE):
            # Draw horizontal lines
            pygame.draw.line(self.screen, self.EMPTY, (0, i * self.CELL_SIZE),
                             (600, i * self.CELL_SIZE))
            # Draw vertical lines
            pygame.draw.line(self.screen, self.EMPTY, (i * self.CELL_SIZE, 0),
                             (i * self.CELL_SIZE, 600))
        # Draw bottom line
        pygame.draw.line(self.screen, self.EMPTY, (0, self.BOARD_SIZE * self.CELL_SIZE),
                             (600, self.BOARD_SIZE * self.CELL_SIZE))

    def draw_pieces(self):
        board = self.state.get_board()
        for y in range(self.BOARD_SIZE):
            for x in range(self.BOARD_SIZE):
                if board[0][y][x] == 1:
                    self.draw_piece(x, y, self.P1)
                elif board[1][y][x] == 1:
                    self.draw_piece(x, y, self.P2)

    def draw_piece(self, x, y, color):
        center = (x * self.CELL_SIZE + self.CELL_SIZE // 2,
                  y * self.CELL_SIZE + self.CELL_SIZE // 2)
        pygame.draw.circle(self.screen, color, center, self.CELL_SIZE // 2 - 2)


def main():
    problem = GoProblem()
    gui = GoGUI(problem)
    clock = pygame.time.Clock()

    while True:
        action = gui.get_user_input_action()
        gui.update_state(action)
        gui.render()
        clock.tick(60)


if __name__ == "__main__":
    main()