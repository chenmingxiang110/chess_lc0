from pathlib import Path
import subprocess

import chess
import numpy as np
import pygame
import pyperclip

from chess_rl.chess_ai import GreedyWdlAI, MinimaxWdlAI
from chess_rl.model import Lc0Net

BOARD_SIZE = 720
SQUARE_SIZE = BOARD_SIZE//8
PANEL_WIDTH = 420
WDL_BAR_WIDTH = 18
BOARD_GAP = 12
BOARD_X = WDL_BAR_WIDTH+BOARD_GAP
CAPTURE_ROW_HEIGHT = 34
CAPTURE_IMAGE_SIZE = 30
CAPTURE_IMAGE_STEP = 24
BOARD_Y = CAPTURE_ROW_HEIGHT
WINDOW_WIDTH = BOARD_X+BOARD_SIZE+PANEL_WIDTH
WINDOW_HEIGHT = BOARD_Y+BOARD_SIZE+CAPTURE_ROW_HEIGHT
FPS = 60
PIECE_IMAGE_SIZE = 80
PANEL_PAD = 24
HISTORY_HEIGHT = 170
LIST_TOP = 126
LIST_HEIGHT = 102
LIST_GAP = 14
WHITE_WIN_GRAPH_HEIGHT = 64
WDL_ANIMATION_SECONDS = 1.5
WDL_EASING_STRENGTH = 2.0
DEFAULT_LC0_WEIGHTS = Path(__file__).resolve().parent.parent/"ckpts/t1-512x15x8h-distilled-swa-3395000.pb.gz"

LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
SELECTED_SQUARE = (246, 246, 105)
LEGAL_DOT = (180, 192, 180)
LEGAL_DOT_ALPHA = 128
PANEL_BG = (244, 245, 247)
TEXT = (30, 34, 39)
MUTED = (94, 99, 108)
BUTTON = (228, 232, 238)
BUTTON_HOVER = (215, 221, 230)
BUTTON_DISABLED = (232, 232, 232)
INPUT_BG = (255, 255, 255)
INPUT_ACTIVE = (235, 242, 255)
BORDER = (170, 176, 186)
WHITE_BAR = (248, 248, 245)
DRAW_BAR = (155, 158, 166)
BLACK_BAR = (24, 26, 30)
WIN_TEXT = (33, 111, 61)
LOSS_TEXT = (150, 44, 44)

PIECE_IMAGE_FILES = {
    chess.PAWN: "chess_pawn.png",
    chess.KNIGHT: "chess_knight.png",
    chess.BISHOP: "chess_bishop.png",
    chess.ROOK: "chess_rook.png",
    chess.QUEEN: "chess_queen.png",
    chess.KING: "chess_king.png",
}

CAPTURE_ORDER = [chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.PAWN]
STARTING_PIECE_COUNTS = {
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.PAWN: 8,
}
AI_OPTIONS = {
    "greedy": GreedyWdlAI,
    "minimax-2": lambda: MinimaxWdlAI(depth=2),
    "minimax-3": lambda: MinimaxWdlAI(depth=3),
}

class Button:

    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, surface, font, disabled=False):
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos) and not disabled
        color = BUTTON_DISABLED if disabled else BUTTON_HOVER if hovered else BUTTON
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, BORDER, self.rect, width=1, border_radius=6)
        text = font.render(self.label, True, MUTED if disabled else TEXT)
        surface.blit(text, text.get_rect(center=self.rect.center))

    def clicked(self, event, disabled=False):
        return (
            event.type==pygame.MOUSEBUTTONDOWN
            and event.button==1
            and self.rect.collidepoint(event.pos)
            and not disabled
        )

class NumberBox:

    def __init__(self, rect, label, value, min_value, max_value):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = None if value is None else int(value)
        self.min_value = int(min_value)
        self.max_value = int(max_value)
        self.active = False
        self.text = "" if value is None else str(int(value))

    def draw(self, surface, label_font, value_font, disabled=False):
        label = label_font.render(self.label, True, MUTED)
        surface.blit(label, (self.rect.x, self.rect.y-24))
        bg = BUTTON_DISABLED if disabled else INPUT_ACTIVE if self.active else INPUT_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, BORDER, self.rect, width=1, border_radius=6)
        value = value_font.render(self.text, True, MUTED if disabled else TEXT)
        surface.blit(value, value.get_rect(midleft=(self.rect.x+12, self.rect.centery)))

    def handle_event(self, event, disabled=False):
        if disabled:
            self.active = False
            return
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            self.active = self.rect.collidepoint(event.pos)
        if event.type!=pygame.KEYDOWN or not self.active:
            return
        if event.key==pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.active = False
        elif event.unicode.isdigit() and len(self.text)<4:
            self.text += event.unicode
        self._sync_value()

    def _sync_value(self):
        if len(self.text)==0:
            self.value = None
            return
        value = int(self.text)
        value = int(np.clip(value, self.min_value, self.max_value))
        self.value = value

    def normalized_value(self):
        self._sync_value()
        self.text = "" if self.value is None else str(self.value)
        return self.value

class Dropdown:

    def __init__(self, rect, options, selected):
        self.rect = pygame.Rect(rect)
        self.options = list(options)
        self.selected = selected
        self.expanded = False

    def draw(self, surface, font, disabled=False, draw_options=True):
        self._draw_main(surface, font, disabled=disabled)
        if self.expanded and not disabled and draw_options:
            self._draw_options(surface, font)

    def handle_event(self, event, disabled=False):
        if disabled:
            self.expanded = False
            return None
        if event.type!=pygame.MOUSEBUTTONDOWN or event.button!=1:
            return None
        if self.rect.collidepoint(event.pos):
            self.expanded = not self.expanded
            return None
        if self.expanded:
            option = self._option_at(event.pos)
            self.expanded = False
            if option is not None and option!=self.selected:
                self.selected = option
                return option
        return None

    def _draw_main(self, surface, font, disabled=False):
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos) and not disabled
        color = BUTTON_DISABLED if disabled else BUTTON_HOVER if hovered else INPUT_BG
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, BORDER, self.rect, width=1, border_radius=6)
        text = font.render(self.selected, True, MUTED if disabled else TEXT)
        surface.blit(text, text.get_rect(midleft=(self.rect.x+10, self.rect.centery)))
        arrow = "^" if self.expanded else "v"
        icon = font.render(arrow, True, MUTED if disabled else TEXT)
        surface.blit(icon, icon.get_rect(center=(self.rect.right-14, self.rect.centery)))

    def _draw_options(self, surface, font):
        for i, option in enumerate(self.options):
            rect = self._option_rect(i)
            hovered = rect.collidepoint(pygame.mouse.get_pos())
            color = BUTTON_HOVER if hovered else INPUT_BG
            pygame.draw.rect(surface, color, rect)
            pygame.draw.rect(surface, BORDER, rect, width=1)
            text = font.render(option, True, TEXT)
            surface.blit(text, text.get_rect(midleft=(rect.x+10, rect.centery)))

    def _option_at(self, pos):
        for i, option in enumerate(self.options):
            if self._option_rect(i).collidepoint(pos):
                return option
        return None

    def _option_rect(self, index):
        return pygame.Rect(
            self.rect.x,
            self.rect.bottom+index*self.rect.height,
            self.rect.width,
            self.rect.height,
        )

class Lc0WdlEstimator:

    def __init__(self, weights_path=DEFAULT_LC0_WEIGHTS, backend=None):
        self.net = Lc0Net(str(weights_path), backend=backend)
        self.cache = {}

    def evaluate(self, board):
        fen = board.fen()
        if fen not in self.cache:
            wdl = self.net.wdl(board)
            self.cache[fen] = tuple(float(x) for x in wdl)
        return self.cache[fen]

class ChessGameApp:

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Chess RL")
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.small_font = pygame.font.SysFont("Arial", 16)
        self.mono_font = pygame.font.SysFont("Menlo", 15)
        self.large_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.piece_images = self._load_piece_images()
        self.piece_shadows = self._load_piece_shadows()
        self.capture_images = self._load_capture_images()
        self.board = chess.Board()
        self.estimator = Lc0WdlEstimator()
        self.ai_name = "minimax-2"
        self.ai = self._make_ai(self.ai_name)
        self.two_part_wdl_bar = True
        self.controllers = {chess.WHITE: "Human", chess.BLACK: "Human"}
        self.pending_controllers = self.controllers.copy()
        x = BOARD_X+BOARD_SIZE+PANEL_PAD
        list_width = (PANEL_WIDTH-PANEL_PAD*2-LIST_GAP)//2
        self.history_rect = pygame.Rect(x, LIST_TOP, list_width, LIST_HEIGHT)
        self.legal_rect = pygame.Rect(x+list_width+LIST_GAP, LIST_TOP, list_width, LIST_HEIGHT)
        self.total_box = NumberBox((x, 252, list_width, 40), "Total minutes", None, 1, 180)
        self.step_box = NumberBox((self.legal_rect.x, 252, list_width, 40), "Overtime seconds", None, 1, 600)
        self.wdl_button = Button((x, 328, 108, 42), "Show WDL")
        self.bar_style_button = Button((x+120, 328, 104, 42), "Bar Style")
        self.copy_button = Button((x, 88, 80, 30), "Copy")
        self.undo_button = Button((x+92, 88, 80, 30), "Undo")
        self.end_button = Button((x+236, 328, 112, 42), "End Game")
        self.analyze_button = Button((self.legal_rect.x, 88, 92, 30), "Analyze")
        self.white_control_button = Button((x, 388, 108, 30), "White: Human")
        self.black_control_button = Button((x+120, 388, 108, 30), "Black: Human")
        self.apply_control_button = Button((x+236, 388, 112, 30), "Apply")
        self.ai_dropdown = Dropdown((x+82, 426, 146, 30), AI_OPTIONS.keys(), self.ai_name)
        self.reset_game()

    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.game_started = False
        self.game_over = False
        self.result_text = ""
        self.show_wdl = False
        self.wdl_button.label = "Show WDL"
        self.move_history = []
        self.state_history = []
        self.history_scroll = 0
        self.legal_scroll = 0
        self.legal_rows_cache = None
        self.legal_rows_cache_fen = None
        self.legal_moves_analyzed = False
        self.legal_moves_analyzing = False
        self.legal_analysis_items = []
        self.legal_analysis_done = 0
        self.legal_analysis_total = 0
        self.copy_message = ""
        self.copy_message_time = 0.0
        self.controller_message = ""
        self.controller_message_time = 0.0
        self.ai_thinking = False
        self.ai_search_task = None
        self.last_tick = pygame.time.get_ticks()
        self._set_clock_from_boxes()
        self.target_wdl = self.estimator.evaluate(self.board)
        self.display_wdl = self.target_wdl
        self.animation_start_wdl = self.display_wdl
        self.wdl_animation_t = 1.0
        self.white_win_rate_history = [self._current_white_win_rate()]

    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS)/1000.0
            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    running = False
                else:
                    self.handle_event(event)
            self.update(dt)
            self.draw()
            self._maybe_start_ai_search()
        pygame.quit()

    def handle_event(self, event):
        locked = self.game_started
        self.total_box.handle_event(event, disabled=locked)
        self.step_box.handle_event(event, disabled=locked)
        dropdown_was_expanded = self.ai_dropdown.expanded
        ai_name = self.ai_dropdown.handle_event(event)
        if ai_name is not None:
            self._set_ai(ai_name)
        if dropdown_was_expanded and event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            return
        if self.wdl_button.clicked(event):
            self.show_wdl = not self.show_wdl
            self.wdl_button.label = "Hide WDL" if self.show_wdl else "Show WDL"
            if not self.show_wdl:
                self._clear_legal_cache()
        if self.bar_style_button.clicked(event, disabled=not self.show_wdl):
            self.two_part_wdl_bar = not self.two_part_wdl_bar
        if self.copy_button.clicked(event, disabled=len(self.move_history)==0):
            self.copy_history()
        if self.undo_button.clicked(event, disabled=len(self.board.move_stack)==0):
            self.undo_moves()
        if self.analyze_button.clicked(event, disabled=not self.show_wdl or self.legal_moves_analyzing):
            self.analyze_legal_moves()
        if self.end_button.clicked(event):
            self.reset_game()
        if self.white_control_button.clicked(event):
            self._toggle_pending_controller(chess.WHITE)
        if self.black_control_button.clicked(event):
            self._toggle_pending_controller(chess.BLACK)
        if self.apply_control_button.clicked(event, disabled=not self._controllers_changed()):
            self._apply_controllers()
        if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
            if self._inside_board(event.pos) and not self.game_over:
                self.handle_board_click(event.pos)
        if event.type==pygame.MOUSEWHEEL:
            self.handle_panel_scroll(event.y, getattr(event, "pos", pygame.mouse.get_pos()))

    def handle_board_click(self, pos):
        if self._is_ai_turn():
            return
        square = self._square_at(pos)
        piece = self.board.piece_at(square)
        if self.selected_square is None:
            if piece is not None and piece.color==self.board.turn:
                self.selected_square = square
                self._start_clock()
            return

        move = self._move_from_selection(self.selected_square, square)
        if move is not None and move in self.board.legal_moves:
            self._push_move(move)
            self.selected_square = None
            return

        if piece is not None and piece.color==self.board.turn:
            self.selected_square = square
        else:
            self.selected_square = None

    def undo_moves(self):
        self.ai_search_task = None
        steps = 2 if self._has_ai_controller() else 1
        for _ in range(steps):
            if len(self.board.move_stack)==0:
                return
            self._undo_one_move()

    def _undo_one_move(self):
        if len(self.board.move_stack)==0:
            return
        self.board.pop()
        self._clear_legal_cache()
        self.selected_square = None
        if len(self.move_history)>0:
            self.move_history.pop()
        self.history_scroll = min(self.history_scroll, self._max_history_scroll())
        self.legal_scroll = 0
        if len(self.white_win_rate_history)>1:
            self.white_win_rate_history.pop()
        self._set_target_wdl()
        if len(self.state_history)>0:
            self._restore_state(self.state_history.pop())
        else:
            self.game_over = False
            self.result_text = ""

    def update(self, dt):
        self._update_wdl_animation(dt)
        self._update_legal_analysis()
        self._update_ai_search()
        self.copy_message_time = max(0.0, self.copy_message_time-dt)
        self.controller_message_time = max(0.0, self.controller_message_time-dt)
        if not self.game_started or self.game_over:
            return
        self._charge_clock(self.board.turn, dt)

    def draw(self):
        self.screen.fill(PANEL_BG)
        self.draw_board()
        self.draw_captured_pieces()
        if self.show_wdl:
            self.draw_board_wdl_bar()
        self.draw_panel()
        pygame.display.flip()

    def draw_board(self):
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, 7-rank)
                color = LIGHT_SQUARE if (rank+file)%2==0 else DARK_SQUARE
                rect = pygame.Rect(
                    BOARD_X+file*SQUARE_SIZE,
                    BOARD_Y+rank*SQUARE_SIZE,
                    SQUARE_SIZE,
                    SQUARE_SIZE,
                )
                if square==self.selected_square:
                    color = SELECTED_SQUARE
                pygame.draw.rect(self.screen, color, rect)
                self._draw_piece(square, rect)
        self._draw_board_coordinates()
        self._draw_legal_targets()

    def draw_captured_pieces(self):
        self._draw_captured_row(chess.WHITE, BOARD_X, 2)
        self._draw_captured_row(chess.BLACK, BOARD_X, BOARD_Y+BOARD_SIZE+2)

    def draw_panel(self):
        x = BOARD_X+BOARD_SIZE+PANEL_PAD
        title = self.large_font.render("Two Player Chess", True, TEXT)
        self.screen.blit(title, (x, 18))
        self._draw_history(x, 62)
        self._draw_legal_moves(self.legal_rect.x, 62)
        locked = self.game_started
        self.total_box.draw(self.screen, self.small_font, self.font, disabled=locked)
        self.step_box.draw(self.screen, self.small_font, self.font, disabled=locked)
        self.wdl_button.draw(self.screen, self.font)
        self.bar_style_button.draw(self.screen, self.font, disabled=not self.show_wdl)
        self.end_button.draw(self.screen, self.font)
        self._draw_controller_controls()
        self.copy_button.draw(self.screen, self.small_font, disabled=len(self.move_history)==0)
        self.undo_button.draw(self.screen, self.small_font, disabled=len(self.board.move_stack)==0)
        analyze_disabled = not self.show_wdl or self.legal_moves_analyzing
        self.analyze_button.draw(self.screen, self.small_font, disabled=analyze_disabled)
        if self.legal_moves_analyzing:
            self._draw_legal_analysis_progress()
        self._draw_clocks(x, 472)
        self._draw_status(x, 586)
        if self.show_wdl:
            self._draw_wdl(x, 616)
            self._draw_white_win_graph(x, 690)
        self._draw_panel_overlay()

    def draw_board_wdl_bar(self):
        white_win, draw, black_win = self.display_wdl
        if self.two_part_wdl_bar:
            self._draw_two_part_wdl_bar(0, BOARD_Y, white_win, draw, black_win)
        else:
            self._draw_vertical_wdl_bar(0, BOARD_Y, white_win, draw, black_win)

    def _draw_piece(self, square, rect):
        piece = self.board.piece_at(square)
        if piece is None:
            return

        image = self.piece_images[(piece.piece_type, piece.color)]
        target = image.get_rect(center=rect.center)
        shadow = self.piece_shadows[(piece.piece_type, piece.color)]
        self.screen.blit(shadow, target.move(2, 2))
        self.screen.blit(image, target)

    def _draw_history(self, x, y):
        label = self.font.render("Move History", True, TEXT)
        self.screen.blit(label, (x, y))
        rect = self.history_rect
        pygame.draw.rect(self.screen, INPUT_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=6)
        rows = self._history_rows()
        max_rows = self._visible_rows(rect)
        start = max(0, len(rows)-max_rows-self.history_scroll)
        visible_rows = rows[start:start+max_rows]
        if len(visible_rows)==0:
            text = self.small_font.render("No moves yet", True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.y+12))
            return
        for i, row in enumerate(visible_rows):
            text = self.small_font.render(row, True, TEXT)
            self.screen.blit(text, (rect.x+12, rect.y+10+i*20))
        self._draw_list_scrollbar(rect, len(rows), max_rows, start)
        if self.copy_message_time>0.0:
            text = self.small_font.render(self.copy_message, True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.bottom-22))

    def _draw_legal_moves(self, x, y):
        label = self.font.render("Legal Moves", True, TEXT)
        self.screen.blit(label, (x, y))
        rect = self.legal_rect
        pygame.draw.rect(self.screen, INPUT_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=6)
        if not self.show_wdl:
            text = self.small_font.render("Show WDL first", True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.y+12))
            return
        if self.legal_moves_analyzing:
            text = self.small_font.render("Evaluating...", True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.y+12))
            return
        if not self.legal_moves_analyzed:
            text = self.small_font.render("Click Analyze", True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.y+12))
            return
        rows = self._legal_move_rows()
        max_rows = self._visible_rows(rect)
        self.legal_scroll = min(self.legal_scroll, self._max_scroll(len(rows), max_rows))
        start = self.legal_scroll
        visible_rows = rows[start:start+max_rows]
        if len(visible_rows)==0:
            text = self.small_font.render("No moves", True, MUTED)
            self.screen.blit(text, (rect.x+12, rect.y+12))
            return
        for i, row in enumerate(visible_rows):
            text = self.mono_font.render(row, True, TEXT)
            self.screen.blit(text, (rect.x+8, rect.y+10+i*20))
        self._draw_list_scrollbar(rect, len(rows), max_rows, start)

    def _draw_legal_analysis_progress(self):
        rect = pygame.Rect(self.legal_rect.x, self.legal_rect.y-6, self.legal_rect.width, 3)
        total = max(1, self.legal_analysis_total)
        progress = self.legal_analysis_done/total
        pygame.draw.rect(self.screen, BUTTON_DISABLED, rect)
        pygame.draw.rect(self.screen, WIN_TEXT, (rect.x, rect.y, int(rect.width*progress), rect.height))
        label = f"{self.legal_analysis_done}/{self.legal_analysis_total}"
        text = self.small_font.render(label, True, MUTED)
        self.screen.blit(text, (self.legal_rect.right-text.get_width(), rect.y-18))

    def _draw_controller_controls(self):
        self.white_control_button.draw(self.screen, self.small_font)
        self.black_control_button.draw(self.screen, self.small_font)
        self.apply_control_button.draw(
            self.screen,
            self.small_font,
            disabled=not self._controllers_changed(),
        )
        label = self.small_font.render("AI", True, MUTED)
        label_pos = (self.white_control_button.rect.x, self.ai_dropdown.rect.y+6)
        self.screen.blit(label, label_pos)
        self.ai_dropdown.draw(self.screen, self.small_font, draw_options=False)
        if self.controller_message_time>0.0:
            text = self.small_font.render(self.controller_message, True, LOSS_TEXT)
            pos = (self.white_control_button.rect.x, self.ai_dropdown.rect.bottom+6)
            self.screen.blit(text, pos)
        self._draw_ai_progress()

    def _draw_ai_progress(self):
        if self.ai_search_task is None or self.ai_search_task.finished:
            return
        rect = pygame.Rect(
            self.ai_dropdown.rect.x,
            self.ai_dropdown.rect.bottom+8,
            self.ai_dropdown.rect.width,
            3,
        )
        total = max(1, self.ai_search_task.total)
        done = min(total, self.ai_search_task.done)
        progress = done/total
        pygame.draw.rect(self.screen, BUTTON_DISABLED, rect)
        pygame.draw.rect(self.screen, WIN_TEXT, (rect.x, rect.y, int(rect.width*progress), rect.height))
        label = f"AI thinking... {done}/{total}"
        text = self.small_font.render(label, True, MUTED)
        self.screen.blit(text, (rect.x, rect.y+7))

    def _draw_panel_overlay(self):
        if self.ai_dropdown.expanded:
            self.ai_dropdown._draw_options(self.screen, self.small_font)

    def _draw_captured_row(self, color, x, y):
        left = x
        for piece_type in CAPTURE_ORDER:
            missing = STARTING_PIECE_COUNTS[piece_type]-len(self.board.pieces(piece_type, color))
            for _ in range(missing):
                image = self.capture_images[(piece_type, color)]
                self.screen.blit(image, (left, y))
                left += CAPTURE_IMAGE_STEP

    def _draw_legal_targets(self):
        if self.selected_square is None:
            return
        for move in self.board.legal_moves:
            if move.from_square!=self.selected_square:
                continue
            file = chess.square_file(move.to_square)
            rank = 7-chess.square_rank(move.to_square)
            center = (
                BOARD_X+file*SQUARE_SIZE+SQUARE_SIZE//2,
                BOARD_Y+rank*SQUARE_SIZE+SQUARE_SIZE//2,
            )
            self._draw_transparent_circle(center, 17, LEGAL_DOT, LEGAL_DOT_ALPHA)

    def _draw_transparent_circle(self, center, radius, color, alpha):
        surface = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        pygame.draw.circle(surface, (*color, alpha), (radius, radius), radius)
        self.screen.blit(surface, (center[0]-radius, center[1]-radius))

    def _draw_board_coordinates(self):
        for rank in range(8):
            label = str(8-rank)
            text = self.small_font.render(label, True, MUTED)
            y = BOARD_Y+rank*SQUARE_SIZE+6
            self.screen.blit(text, (BOARD_X+6, y))
        for file in range(8):
            label = chr(ord("a")+file)
            text = self.small_font.render(label, True, MUTED)
            x = BOARD_X+file*SQUARE_SIZE+SQUARE_SIZE-text.get_width()-8
            y = BOARD_Y+BOARD_SIZE-text.get_height()-5
            self.screen.blit(text, (x, y))

    def _draw_clocks(self, x, y):
        label = self.font.render("Clocks", True, TEXT)
        self.screen.blit(label, (x, y))
        self._draw_clock_row(x, y+34, "White", chess.WHITE)
        self._draw_clock_row(x, y+74, "Black", chess.BLACK)

    def _draw_clock_row(self, x, y, name, color):
        active = self.game_started and not self.game_over and self.board.turn==color
        main = self._format_time(self.main_time[color])
        step = self._format_time(self.step_time[color])
        prefix = ">" if active else " "
        text = self.font.render(f"{prefix} {name}: main {main}  step {step}", True, TEXT)
        self.screen.blit(text, (x, y))

    def _draw_status(self, x, y):
        if len(self.result_text)==0:
            turn = "White to move" if self.board.turn==chess.WHITE else "Black to move"
            status = turn if not self.game_over else ""
            color = MUTED
        else:
            status = self.result_text
            color = WIN_TEXT if "White" in status else LOSS_TEXT if "Black" in status else MUTED
        text = self.font.render(status, True, color)
        self.screen.blit(text, (x, y))

    def _draw_wdl(self, x, y):
        white_win, draw, black_win = self.estimator.evaluate(self.board)
        lines = [
            self._wdl_line(white_win, "white_win"),
            self._wdl_line(draw, "draw"),
            self._wdl_line(black_win, "black_win"),
        ]
        for i, line in enumerate(lines):
            text = self.mono_font.render(line, True, TEXT)
            self.screen.blit(text, (x, y+i*20))

    def _draw_vertical_wdl_bar(self, x, y, white_win, draw, black_win):
        heights = [
            int(round(BOARD_SIZE*black_win)),
            int(round(BOARD_SIZE*draw)),
            BOARD_SIZE,
        ]
        heights[2] = BOARD_SIZE-heights[0]-heights[1]
        segments = [(BLACK_BAR, heights[0]), (DRAW_BAR, heights[1]), (WHITE_BAR, heights[2])]
        top = y
        for color, height in segments:
            if height>0:
                pygame.draw.rect(self.screen, color, (x, top, WDL_BAR_WIDTH, height))
                top += height
        pygame.draw.rect(self.screen, BORDER, (x, y, WDL_BAR_WIDTH, BOARD_SIZE), width=1)

    def _draw_two_part_wdl_bar(self, x, y, white_win, draw, black_win):
        white_win_rate = white_win+0.5*draw
        white_height = int(round(BOARD_SIZE*white_win_rate))
        black_height = BOARD_SIZE-white_height
        if black_height>0:
            pygame.draw.rect(self.screen, BLACK_BAR, (x, y, WDL_BAR_WIDTH, black_height))
        if white_height>0:
            pygame.draw.rect(
                self.screen,
                WHITE_BAR,
                (x, y+black_height, WDL_BAR_WIDTH, white_height),
            )
        pygame.draw.rect(self.screen, BORDER, (x, y, WDL_BAR_WIDTH, BOARD_SIZE), width=1)

    def _draw_white_win_graph(self, x, y):
        label = self.small_font.render("white_win_rate history", True, MUTED)
        self.screen.blit(label, (x, y))
        rect = pygame.Rect(x, y+22, PANEL_WIDTH-PANEL_PAD*2, WHITE_WIN_GRAPH_HEIGHT)
        pygame.draw.rect(self.screen, INPUT_BG, rect, border_radius=6)
        pygame.draw.rect(self.screen, BORDER, rect, width=1, border_radius=6)
        mid_y = rect.y+rect.height//2
        pygame.draw.line(self.screen, DRAW_BAR, (rect.x+8, mid_y), (rect.right-8, mid_y), width=1)
        values = self.white_win_rate_history[-32:]
        if len(values)==0:
            return
        points = self._graph_points(values, rect)
        if len(points)==1:
            pygame.draw.circle(self.screen, WIN_TEXT, points[0], 3)
        else:
            pygame.draw.lines(self.screen, WIN_TEXT, False, points, width=2)
            for point in points:
                pygame.draw.circle(self.screen, WIN_TEXT, point, 2)
        latest = self.small_font.render(f"{values[-1]:.3f}", True, TEXT)
        self.screen.blit(latest, (rect.right-latest.get_width()-8, rect.y+6))

    def _square_at(self, pos):
        file = int((pos[0]-BOARD_X)//SQUARE_SIZE)
        rank = 7-int((pos[1]-BOARD_Y)//SQUARE_SIZE)
        return chess.square(file, rank)

    def _inside_board(self, pos):
        return BOARD_X<=pos[0]<BOARD_X+BOARD_SIZE and BOARD_Y<=pos[1]<BOARD_Y+BOARD_SIZE

    def _move_from_selection(self, from_square, to_square):
        piece = self.board.piece_at(from_square)
        promotion = None
        if piece is not None and piece.piece_type==chess.PAWN:
            to_rank = chess.square_rank(to_square)
            if to_rank==0 or to_rank==7:
                promotion = chess.QUEEN
        return chess.Move(from_square, to_square, promotion=promotion)

    def _start_clock(self):
        if self.game_started:
            return
        self.game_started = True
        self._set_clock_from_boxes()

    def _reset_step_time(self, color):
        self.step_time[color] = self._step_seconds_from_box()

    def _set_clock_from_boxes(self):
        main_seconds = self._main_seconds_from_box()
        step_seconds = self._step_seconds_from_box()
        self.main_time = {chess.WHITE: main_seconds, chess.BLACK: main_seconds}
        self.step_time = {chess.WHITE: step_seconds, chess.BLACK: step_seconds}

    def _main_seconds_from_box(self):
        value = self.total_box.normalized_value()
        return None if value is None else float(value*60)

    def _step_seconds_from_box(self):
        value = self.step_box.normalized_value()
        return None if value is None else float(value)

    def _sync_game_over(self):
        if not self.board.is_game_over(claim_draw=True):
            return
        self.game_over = True
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            self.result_text = "Draw"
        else:
            self.result_text = "White wins" if outcome.winner==chess.WHITE else "Black wins"

    def _push_move(self, move):
        self._start_clock()
        self.state_history.append(self._state_snapshot())
        self.move_history.append(self.board.san(move))
        self.board.push(move)
        self._clear_legal_cache()
        self._set_target_wdl()
        self.white_win_rate_history.append(self._current_white_win_rate())
        self.history_scroll = 0
        self.legal_scroll = 0
        self._reset_step_time(self.board.turn)
        self.selected_square = None
        self._sync_game_over()

    def _maybe_start_ai_search(self):
        if self.game_over or self.ai_search_task is not None or not self._is_ai_turn():
            return
        self._start_clock()
        if hasattr(self.ai, "start_search"):
            self.ai_search_task = self.ai.start_search(self.board, self.estimator.net)
            return
        move = self.ai.choose_move(self.board, self.estimator.net)
        if move is not None and move in self.board.legal_moves:
            self._push_move(move)

    def _update_ai_search(self):
        if self.ai_search_task is None:
            return
        self.ai_search_task.step()
        if not self.ai_search_task.finished:
            return
        move = self.ai_search_task.best_move
        self.ai_search_task = None
        if move is not None and move in self.board.legal_moves:
            self._push_move(move)

    def _charge_clock(self, color, dt):
        if self.game_over:
            return
        if self.main_time[color] is None:
            return
        if self.main_time[color]>0.0:
            self.main_time[color] = max(0.0, self.main_time[color]-dt)
        else:
            if self.step_time[color] is None:
                return
            self.step_time[color] = max(0.0, self.step_time[color]-dt)
            if self.step_time[color]<=0.0:
                self.game_over = True
                self.result_text = "Black wins on time" if color==chess.WHITE else "White wins on time"

    def _is_ai_turn(self):
        return self.controllers[self.board.turn]=="AI"

    def _has_ai_controller(self):
        return self.controllers[chess.WHITE]=="AI" or self.controllers[chess.BLACK]=="AI"

    def _make_ai(self, ai_name):
        return AI_OPTIONS[ai_name]()

    def _set_ai(self, ai_name):
        self.ai_name = ai_name
        self.ai = self._make_ai(ai_name)
        self.ai_search_task = None

    def _toggle_pending_controller(self, color):
        current = self.pending_controllers[color]
        self.pending_controllers[color] = "AI" if current=="Human" else "Human"
        self.controller_message = ""
        self.controller_message_time = 0.0
        self._sync_controller_labels()

    def _apply_controllers(self):
        if self._pending_controllers_are_both_ai():
            self.controller_message = "Only one AI side is allowed"
            self.controller_message_time = 2.0
            return
        self.controllers = self.pending_controllers.copy()
        self.controller_message = ""
        self.controller_message_time = 0.0
        self._sync_controller_labels()

    def _controllers_changed(self):
        return self.pending_controllers!=self.controllers

    def _pending_controllers_are_both_ai(self):
        return self.pending_controllers[chess.WHITE]=="AI" and self.pending_controllers[chess.BLACK]=="AI"

    def _sync_controller_labels(self):
        self.white_control_button.label = f"White: {self.pending_controllers[chess.WHITE]}"
        self.black_control_button.label = f"Black: {self.pending_controllers[chess.BLACK]}"

    def _history_rows(self):
        rows = []
        for i in range(0, len(self.move_history), 2):
            number = i//2+1
            white_move = self.move_history[i]
            black_move = self.move_history[i+1] if i+1<len(self.move_history) else ""
            rows.append(f"{number}. {white_move}  {black_move}")
        return rows

    def handle_panel_scroll(self, amount, pos=None):
        pos = pygame.mouse.get_pos() if pos is None else pos
        if self.history_rect.collidepoint(pos):
            self.history_scroll = self._scrolled(
                self.history_scroll, amount, self._history_rows(), self.history_rect
            )
        elif self.legal_rect.collidepoint(pos):
            rows = self._legal_move_rows() if self.legal_moves_analyzed else []
            self.legal_scroll = self._scrolled(self.legal_scroll, -amount, rows, self.legal_rect)

    def copy_history(self):
        text = "\n".join(self._history_rows())
        if len(text)==0:
            return
        if self._copy_with_pbcopy(text):
            self.copy_message = "Copied"
            self.copy_message_time = 1.5
            return
        if self._copy_with_pyperclip(text):
            self.copy_message = "Copied"
            self.copy_message_time = 1.5
            return
        try:
            pygame.scrap.init()
            pygame.scrap.put(pygame.SCRAP_TEXT, text.encode("utf-8"))
            self.copy_message = "Copied"
            self.copy_message_time = 1.5
        except Exception:
            self.copy_message = "Copy failed"
            self.copy_message_time = 1.5

    def _copy_with_pbcopy(self, text):
        proc = None
        try:
            proc = subprocess.Popen(["/usr/bin/pbcopy"], stdin=subprocess.PIPE)
            proc.communicate(text.encode("utf-8"), timeout=1.0)
            if proc.returncode!=0:
                return False
            return True
        except subprocess.TimeoutExpired:
            if proc is not None:
                proc.kill()
                proc.communicate()
            return False
        except Exception:
            return False

    def _copy_with_pyperclip(self, text):
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    def _legal_move_rows(self):
        if not self.legal_moves_analyzed:
            return []
        return [] if self.legal_rows_cache is None else self.legal_rows_cache

    def _clear_legal_cache(self):
        self.legal_rows_cache = None
        self.legal_rows_cache_fen = None
        self.legal_moves_analyzed = False
        self.legal_moves_analyzing = False
        self.legal_analysis_items = []
        self.legal_analysis_done = 0
        self.legal_analysis_total = 0

    def analyze_legal_moves(self):
        if self.board.is_game_over(claim_draw=True):
            return
        self.legal_scroll = 0
        self.legal_rows_cache = None
        self.legal_rows_cache_fen = self.board.fen()
        self.legal_moves_analyzed = False
        self.legal_moves_analyzing = True
        self.legal_analysis_items = self._legal_analysis_queue()
        self.legal_analysis_done = 0
        self.legal_analysis_total = len(self.legal_analysis_items)

    def _legal_analysis_queue(self):
        items = []
        for move in self.board.legal_moves:
            san = self.board.san(move)
            items.append({"move": move, "san": san, "rate": None})
        return items

    def _update_legal_analysis(self):
        if not self.legal_moves_analyzing:
            return
        if self.legal_analysis_done>=self.legal_analysis_total:
            self._finish_legal_analysis()
            return
        item = self.legal_analysis_items[self.legal_analysis_done]
        self.board.push(item["move"])
        item["rate"] = self._current_white_win_rate()
        self.board.pop()
        self.legal_analysis_done += 1
        if self.legal_analysis_done>=self.legal_analysis_total:
            self._finish_legal_analysis()

    def _finish_legal_analysis(self):
        reverse = self.board.turn==chess.WHITE
        rows = [(item["rate"], item["san"]) for item in self.legal_analysis_items]
        rows.sort(key=lambda item: item[0], reverse=reverse)
        self.legal_rows_cache = [f"{rate*100:6.2f}%  {san}" for rate, san in rows]
        self.legal_moves_analyzed = True
        self.legal_moves_analyzing = False

    def _visible_rows(self, rect):
        return max(1, (rect.height-18)//20)

    def _max_history_scroll(self):
        return max(0, len(self._history_rows())-self._visible_rows(self.history_rect))

    def _max_scroll(self, row_count, visible_count):
        return max(0, row_count-visible_count)

    def _scrolled(self, current, amount, rows, rect):
        return int(np.clip(current+amount, 0, self._max_scroll(len(rows), self._visible_rows(rect))))

    def _draw_list_scrollbar(self, rect, row_count, visible_count, start):
        if row_count<=visible_count:
            return
        track = pygame.Rect(rect.right-8, rect.y+8, 3, rect.height-16)
        pygame.draw.rect(self.screen, BUTTON_DISABLED, track, border_radius=2)
        thumb_h = max(18, int(track.height*visible_count/row_count))
        max_start = row_count-visible_count
        thumb_y = track.y+int((track.height-thumb_h)*start/max_start)
        pygame.draw.rect(self.screen, BORDER, (track.x, thumb_y, track.width, thumb_h), border_radius=2)

    def _graph_points(self, values, rect):
        left = rect.x+10
        right = rect.right-10
        top = rect.y+10
        bottom = rect.bottom-10
        xs = np.linspace(left, right, len(values))
        ys = [bottom-float(np.clip(value, 0.0, 1.0))*(bottom-top) for value in values]
        return [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]

    def _current_white_win_rate(self):
        white_win, draw, black_win = self.estimator.evaluate(self.board)
        return white_win+0.5*draw

    def _wdl_line(self, value, label):
        return f"{value*100:6.2f}%  {label}"

    def _set_target_wdl(self):
        old_target = self.target_wdl
        self.target_wdl = self.estimator.evaluate(self.board)
        if self.target_wdl==old_target:
            return
        self.animation_start_wdl = self.display_wdl
        self.wdl_animation_t = 0.0

    def _update_wdl_animation(self, dt):
        if self.wdl_animation_t>=1.0:
            return
        self.wdl_animation_t = float(np.clip(self.wdl_animation_t+dt/WDL_ANIMATION_SECONDS, 0.0, 1.0))
        alpha = self._wdl_animation_alpha(self.wdl_animation_t)
        start = np.asarray(self.animation_start_wdl, dtype=np.float64)
        target = np.asarray(self.target_wdl, dtype=np.float64)
        display = start+(target-start)*alpha
        display = np.maximum(display, 0.0)
        display = display/display.sum()
        if self.wdl_animation_t>=1.0:
            display = target
        self.display_wdl = tuple(float(x) for x in display)

    def _wdl_animation_alpha(self, t):
        smooth = 0.5-0.5*np.cos(np.pi*t)
        k = max(0.001, WDL_EASING_STRENGTH)
        left = smooth**k
        right = (1.0-smooth)**k
        return float(left/(left+right))

    def _state_snapshot(self):
        return {
            "main_time": self.main_time.copy(),
            "step_time": self.step_time.copy(),
            "game_started": self.game_started,
            "game_over": self.game_over,
            "result_text": self.result_text,
        }

    def _restore_state(self, snapshot):
        self.main_time = snapshot["main_time"].copy()
        self.step_time = snapshot["step_time"].copy()
        self.game_started = snapshot["game_started"]
        self.game_over = snapshot["game_over"]
        self.result_text = snapshot["result_text"]

    def _format_time(self, seconds):
        if seconds is None:
            return "∞"
        seconds = int(np.ceil(seconds))
        minutes = seconds//60
        seconds = seconds%60
        return f"{minutes:02d}:{seconds:02d}"

    def _load_piece_images(self):
        image_dir = Path(__file__).with_name("src")
        images = {}
        for piece_type, filename in PIECE_IMAGE_FILES.items():
            image = pygame.image.load(str(image_dir/filename)).convert_alpha()
            image = pygame.transform.smoothscale(image, (PIECE_IMAGE_SIZE, PIECE_IMAGE_SIZE))
            white_piece = self._tint_piece(image, (245, 245, 242))
            images[(piece_type, chess.WHITE)] = self._outline_piece(white_piece, (14, 16, 20))
            images[(piece_type, chess.BLACK)] = self._tint_piece(image, (26, 28, 32))
        return images

    def _load_piece_shadows(self):
        image_dir = Path(__file__).with_name("src")
        shadows = {}
        for piece_type, filename in PIECE_IMAGE_FILES.items():
            image = pygame.image.load(str(image_dir/filename)).convert_alpha()
            image = pygame.transform.smoothscale(image, (PIECE_IMAGE_SIZE, PIECE_IMAGE_SIZE))
            shadows[(piece_type, chess.WHITE)] = self._tint_shadow(image, (18, 20, 24), 130)
            shadows[(piece_type, chess.BLACK)] = self._tint_shadow(image, (250, 250, 246), 90)
        return shadows

    def _load_capture_images(self):
        images = {}
        for key, image in self.piece_images.items():
            images[key] = pygame.transform.smoothscale(image, (CAPTURE_IMAGE_SIZE, CAPTURE_IMAGE_SIZE))
        return images

    def _tint_piece(self, image, color):
        tinted = image.copy()
        tinted.fill((*color, 255), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def _tint_shadow(self, image, color, alpha):
        shadow = self._tint_piece(image, color)
        shadow.set_alpha(alpha)
        return shadow

    def _outline_piece(self, image, color):
        outlined = pygame.Surface(image.get_size(), pygame.SRCALPHA)
        outline = self._tint_piece(image, color)
        for dx, dy in [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]:
            outlined.blit(outline, (dx, dy))
        outlined.blit(image, (0, 0))
        return outlined
