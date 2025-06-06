import pygame
from typing import Optional
from game_logic import GameState, BuffType, ActionType

class GameRenderer:
    def __init__(self, cell_size: int = 80):
        # Inizializza Pygame, determina dimensioni finestra in base alla griglia e allo spazio UI
        pygame.init()
        self.cell_size = cell_size
        self.grid_size = 7
        self.width = self.grid_size * cell_size + 400  # 400 px riservati al pannello laterale
        self.height = self.grid_size * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Grid Duel RL: Tactical Arena")

        # Definizione dei colori principali usati per elementi di gioco e UI
        self.BLACK      = (  0,   0,   0)
        self.WHITE      = (255, 255, 255)
        self.GRAY       = (128, 128, 128)
        self.DARK_GRAY  = ( 64,  64,  64)
        self.RED        = (255,   0,   0)
        self.BLUE       = (  0,   0, 255)
        self.GREEN      = (  0, 255,   0)
        self.YELLOW     = (255, 255,   0)
        self.PURPLE     = (128,   0, 128)
        self.ORANGE     = (255, 165,   0)

        # Font standard per testi e font più piccolo per dettagli
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)

    def render(self, game_state: GameState):
        # Pulisce lo schermo con tonalità scura per contrastare la griglia bianca
        self.screen.fill(self.DARK_GRAY)

        # Crea una superficie dedicata all'area di gioco (senza UI)
        grid_surface = pygame.Surface((self.grid_size * self.cell_size,
                                       self.grid_size * self.cell_size))
        grid_surface.fill(self.WHITE)

        # Applica "foschia di guerra" per il giocatore umano:
        # nasconde celle non visibili in base al buff di visione
        visible_cells = game_state.get_visible_cells(game_state.human)
        for x in range(1, self.grid_size + 1):
            for y in range(1, self.grid_size + 1):
                if (x, y) not in visible_cells:
                    rect = pygame.Rect((x - 1) * self.cell_size,
                                       (y - 1) * self.cell_size,
                                       self.cell_size, self.cell_size)
                    pygame.draw.rect(grid_surface, self.DARK_GRAY, rect)

        # Disegna linee di griglia per delimitare chiaramente le celle
        for i in range(self.grid_size + 1):
            start = (i * self.cell_size, 0)
            end_v = (i * self.cell_size, self.height)
            pygame.draw.line(grid_surface, self.BLACK, start, end_v)
            start_h = (0, i * self.cell_size)
            end = (self.height, i * self.cell_size)
            pygame.draw.line(grid_surface, self.BLACK, start_h, end)

        # Rende i buff visibili nelle celle esposte: colore in base al tipo di buff
        for buff in game_state.buffs:
            pos = buff.get_position()
            if pos in visible_cells:
                x, y = pos
                rect = pygame.Rect((x - 1) * self.cell_size + 10,
                                   (y - 1) * self.cell_size + 10,
                                   self.cell_size - 20, self.cell_size - 20)
                # Verde per salute, grigio per armatura, viola per visione, arancio per freeze
                if buff.buff_type == BuffType.HEALTH:
                    color = self.GREEN
                elif buff.buff_type == BuffType.ARMOR:
                    color = self.GRAY
                elif buff.buff_type == BuffType.VISION:
                    color = self.PURPLE
                else:
                    color = self.ORANGE
                pygame.draw.rect(grid_surface, color, rect)
                # Disegna la lettera identificativa del buff al centro del rettangolo
                text = self.small_font.render(buff.buff_type.value[0], True, self.BLACK)
                text_rect = text.get_rect(center=rect.center)
                grid_surface.blit(text, text_rect)

        # Disegna il giocatore umano come ellisse blu
        human_x, human_y = game_state.human.get_position()
        human_rect = pygame.Rect((human_x - 1) * self.cell_size + 5,
                                 (human_y - 1) * self.cell_size + 5,
                                 self.cell_size - 10, self.cell_size - 10)
        pygame.draw.ellipse(grid_surface, self.BLUE, human_rect)

        # Disegna l'IA come ellisse rossa, solo se visibile all'umano
        ai_x, ai_y = game_state.ai.get_position()
        if (ai_x, ai_y) in visible_cells:
            ai_rect = pygame.Rect((ai_x - 1) * self.cell_size + 5,
                                  (ai_y - 1) * self.cell_size + 5,
                                  self.cell_size - 10, self.cell_size - 10)
            pygame.draw.ellipse(grid_surface, self.RED, ai_rect)

        # Incolla la griglia sullo schermo principale
        self.screen.blit(grid_surface, (0, 0))

        # Rende il pannello laterale con informazioni di gioco e controlli
        self.draw_ui(game_state)

        # Se la partita è finita, mostra il vincitore in evidenza
        if game_state.game_over:
            if game_state.winner == 0:
                winner_name = "Human"
            elif game_state.winner == 1:
                winner_name = "AI"
            else:
                winner_name = "DRAW"
            winner_text = self.font.render(f"WINNER: {winner_name}!", True, self.BLACK)
            text_rect = winner_text.get_rect(center=(self.width / 2, 20))
            # Rettangolo giallo per dare risalto al messaggio
            pygame.draw.rect(self.screen, self.YELLOW, text_rect.inflate(20, 10))
            self.screen.blit(winner_text, text_rect)

        # Aggiorna il display con tutto ciò che è stato disegnato
        pygame.display.flip()

    def draw_ui(self, game_state: GameState):
        # Coordinate iniziali per il pannello UI
        ui_x = self.grid_size * self.cell_size + 20
        ui_y = 10

        # Disegna statistiche di un partecipante (human o AI)
        def draw_participant_stats(participant, name, color_label, y_start):
            y = y_start
            # Titolo con nome e colore associato
            header = self.font.render(f"{name} ({color_label})", True, self.WHITE)
            self.screen.blit(header, (ui_x, y))
            y += 25
            # Elenco delle statistiche chiave; hp, armatura, durate buff, stato freeze
            stats = [
                f"HP: {participant.hp}/3",
                f"Armor: {participant.armor}",
                f"Vision Buff: {participant.vision_duration}",
                f"Freeze Attacks: {participant.freeze_attack_count}",
                f"Frozen: {participant.freeze_status}"
            ]
            for stat in stats:
                text = self.small_font.render(stat, True, self.WHITE)
                self.screen.blit(text, (ui_x, y))
                y += 20
            return y

        # Mostra il numero del turno (ogni due mosse incrementa il contatore)
        turn_text = self.font.render(f"Turn: {game_state.turn // 2}", True, self.WHITE)
        self.screen.blit(turn_text, (ui_x, ui_y))
        ui_y += 30

        # Indica chi sta giocando in questo momento
        current_player = "Human" if game_state.current_player == 0 else "AI"
        player_text = self.font.render(f"Current Turn: {current_player}", True, self.WHITE)
        self.screen.blit(player_text, (ui_x, ui_y))
        ui_y += 40

        # Statistiche del giocatore umano
        ui_y = draw_participant_stats(game_state.human, "HUMAN", "Blue", ui_y)
        ui_y += 20
        # Statistiche dell'agente IA
        ui_y = draw_participant_stats(game_state.ai, "AI", "Red", ui_y)
        ui_y += 40

        # Sezione controlli: mostra i tasti utilizzabili
        controls_header = self.font.render("CONTROLS:", True, self.WHITE)
        self.screen.blit(controls_header, (ui_x, ui_y))
        ui_y += 25
        controls = ["WASD: Move", "SPACE: Attack", "F: Freeze Attack", "R: Restart", "ESC: Quit"]
        for control in controls:
            ctrl_text = self.small_font.render(control, True, self.WHITE)
            self.screen.blit(ctrl_text, (ui_x, ui_y))
            ui_y += 18

    def get_human_action(self, event) -> Optional[ActionType]:
        # Mappa gli eventi di pressione tasto alle azioni di gioco
        if event.type != pygame.KEYDOWN:
            return None
        key = event.key

        if key == pygame.K_w:
            return ActionType.MOVE_UP
        if key == pygame.K_s:
            return ActionType.MOVE_DOWN
        if key == pygame.K_a:
            return ActionType.MOVE_LEFT
        if key == pygame.K_d:
            return ActionType.MOVE_RIGHT
        if key == pygame.K_SPACE:
            return ActionType.ATTACK
        if key == pygame.K_f:
            return ActionType.FREEZE

        return None
