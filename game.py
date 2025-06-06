import pygame
import sys
from game_logic import GameState, ActionType
from renderer import GameRenderer
from dqn_agent import DQNAgent
import random

class Game:
    def __init__(self):
        # Inizializza lo stato di gioco e il renderer grafico
        self.game_state = GameState()
        self.renderer = GameRenderer()

        # Configura l'agente DQN per il RL:
        # l'osservazione include lo stato della griglia (4 canali per cella) più 10 feature addizionali
        observation_size = (self.game_state.grid_size ** 2 * 4) + 10
        action_size = len(ActionType)
        # Si utilizza DQN per sfruttare il replay buffer e stabilizzare l'apprendimento
        self.ai_agent = DQNAgent(observation_size, action_size)

    def train(self, num_episodes):
        print(f"--- Avvio addestramento per {num_episodes} episodi ---")
        # Dopo ogni quarto del training, aumentiamo la difficoltà dell'avversario controllato da policy semplice
        threshold = num_episodes / 4
        update = threshold
        difficulty = 1
        wins = 0

        for episode in range(num_episodes):
            # Aumenta la difficoltà in modo graduale per evitare un salto troppo brusco nella capacità dell'IA
            if episode >= threshold:
                difficulty += 1
                threshold += update

            # Imposta un nuovo episodio di gioco
            self.game_state.initialize_game()
            state = self.game_state.get_ai_observation()

            while not self.game_state.game_over:
                # Turno dell'avversario controllato da policy semplice
                if self.game_state.current_player == 0:
                    action = self.get_simple_opponent_action(difficulty * 0.1)
                    self.game_state.execute_action(action)

                # Turno dell'agente RL
                elif self.game_state.current_player == 1:
                    # Seleziona l'azione tramite epsilon-greedy (is_training=True abilita esplorazione)
                    action_idx = self.ai_agent.get_action(state, is_training=True)
                    action = ActionType(action_idx)

                    # Esegue l'azione e riceve la ricompensa
                    _, reward = self.game_state.execute_action(action)
                    next_state = self.game_state.get_ai_observation()
                    done = self.game_state.game_over

                    # Memorizza la transizione nel replay buffer per apprendimento batch
                    self.ai_agent.replay_buffer.push(state, action_idx, reward, next_state, done)
                    # Aggiorna i pesi della rete con un mini-batch estratto dal buffer
                    self.ai_agent.learn()

                    state = next_state

            # Conta le vittorie per calcolare il winrate a intervalli regolari
            wins += self.game_state.winner
            if (episode + 1) % 100 == 0:
                # Stampa il tasso di vittorie ogni 100 episodi per monitorare i progressi
                print(f"Episodio {episode + 1}/{num_episodes} completato. Winrate: {(wins / 100):.2f}")
                wins = 0
                # Salva il modello per conservare lo stato corrente dell'apprendimento
                self.ai_agent.save_model()

        print("--- Addestramento completato ---")
        # Salvataggio finale del modello dopo tutti gli episodi
        self.ai_agent.save_model()

    def play(self):
        # Avvia una nuova partita in modalità interattiva
        self.game_state.initialize_game()
        running = True
        clock = pygame.time.Clock()

        while running:
            # Turno del giocatore umano: acquisizione input da tastiera o mouse
            if self.game_state.current_player == 0 and not self.game_state.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.game_state.initialize_game()

                    action = self.renderer.get_human_action(event)
                    if action:
                        self.game_state.execute_action(action)

            # Turno dell'IA: utilizza la politica appresa senza esplorazione
            elif self.game_state.current_player == 1 and not self.game_state.game_over:
                state = self.game_state.get_ai_observation()
                action_idx = self.ai_agent.get_action(state, is_training=False)
                action = ActionType(action_idx)
                self.game_state.execute_action(action)
                # Piccola pausa per rendere visibile la mossa dell'IA all'utente
                pygame.time.delay(200)

            # Gestione dello stato di fine partita: permette di riavviare o chiudere
            if self.game_state.game_over:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self.game_state.initialize_game()

            # Renderizza lo stato attuale del gioco e controlla il framerate
            self.renderer.render(self.game_state)
            clock.tick(30)

        # Pulizia delle risorse e chiusura del gioco
        pygame.quit()
        sys.exit()

    def get_simple_opponent_action(self, difficulty):
        # Policy di base per l'avversario durante l'addestramento:
        # se adiacente all'IA, decide tra ATTACK e FREEZE in base a una soglia variabile
        ai_pos = self.game_state.ai.get_position()
        human_pos = self.game_state.human.get_position()

        if self.game_state.is_adjacent(human_pos, ai_pos):
            # La difficoltà influisce sulla probabilità di attacco diretto
            if random.uniform(0, 1) + difficulty > 0.75:
                return ActionType.ATTACK
            else:
                return ActionType.FREEZE

        # Altrimenti, muove l'avversario verso l'IA scegliendo la direzione dominante
        dx, dy = ai_pos[0] - human_pos[0], ai_pos[1] - human_pos[1]
        if abs(dx) > abs(dy):
            return ActionType.MOVE_RIGHT if dx > 0 else ActionType.MOVE_LEFT
        else:
            return ActionType.MOVE_DOWN if dy > 0 else ActionType.MOVE_UP
