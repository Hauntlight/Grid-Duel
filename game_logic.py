import random
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np

# --- ENUMERAZIONI --- 

class Direction(Enum):
    UP = "Up"
    DOWN = "Down"
    LEFT = "Left"
    RIGHT = "Right"
    NONE = "None"

class BuffType(Enum):
    HEALTH = "Health"
    ARMOR = "Armor"
    VISION = "Vision Range"
    FREEZE = "Freeze Attack"

class ActionType(Enum):
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    ATTACK = 4
    FREEZE = 5

# --- CLASSE PARTECIPANTE --- 

class Participant:
    def __init__(self, x: int, y: int, is_human: bool = True):
        self.x = x
        self.y = y
        self.hp = 3
        self.armor = 0
        self.vision_duration = 0
        self.freeze_status = 0
        self.last_movement_direction = Direction.NONE
        self.freeze_attack_count = 0
        self.is_human = is_human

    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def set_position(self, x: int, y: int):
        self.x = x
        self.y = y

    def take_damage(self) -> bool:
        # Se c'è armatura, questa viene consumata prima della salute
        if self.armor > 0:
            self.armor = 0
            return False
        else:
            self.hp -= 1
            return True

    def heal(self) -> bool:
        # Cura fino a un massimo di 3 punti salute
        if self.hp < 3:
            self.hp += 1
            return True
        return False

    def add_armor(self) -> bool:
        # Aggiunge armatura solo se attualmente assente
        if self.armor == 0:
            self.armor = 1
            return True
        return False

    def add_vision_buff(self):
        # Incrementa la durata del buff visione per consentire visibilità estesa
        self.vision_duration = 3

    def add_freeze_attack(self) -> bool:
        # Concede un solo attacco freeze alla volta
        if self.freeze_attack_count == 0:
            self.freeze_attack_count = 1
            return True
        return False

    def use_freeze_attack(self) -> bool:
        # Consuma l'attacco freeze se disponibile
        if self.freeze_attack_count > 0:
            self.freeze_attack_count = 0
            return True
        return False

    def get_vision_radius(self) -> int:
        # Raggio di visione base 1, esteso a 2 quando è attivo il buff
        return 2 if self.vision_duration > 0 else 1

    def update_vision_duration(self):
        # Riduce la durata del buff visione a ogni turno
        if self.vision_duration > 0:
            self.vision_duration -= 1

# --- CLASSE BUFF --- 

class BuffToken:
    def __init__(self, x: int, y: int, buff_type: BuffType):
        self.x = x
        self.y = y
        self.buff_type = buff_type
        self.duration = 5  # Durata in turni prima di scomparire

    def get_position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    def expire(self) -> bool:
        # Restituisce True se il buff deve essere rimosso
        self.duration -= 1
        return self.duration <= 0

# --- STATO DI GIOCO --- 

class GameState:
    def __init__(self):
        self.grid_size = 7
        self.turn = 1
        self.current_player = 0  # 0 = umano, 1 = IA
        self.human: Optional[Participant] = None
        self.ai: Optional[Participant] = None
        self.buffs: List[BuffToken] = []
        self.game_over = False
        self.winner: Optional[int] = None
        self.last_action_valid = True
        self.max_turns = 1000  # Numero massimo di mezzi-turni

    def initialize_game(self):
        # Posiziona casualmente umano e IA su celle diverse
        positions = [(x, y) for x in range(1, 8) for y in range(1, 8)]
        human_pos, ai_pos = random.sample(positions, 2)
        self.human = Participant(human_pos[0], human_pos[1], is_human=True)
        self.ai = Participant(ai_pos[0], ai_pos[1], is_human=False)
        self.turn = 1
        self.current_player = 0
        self.buffs.clear()
        self.game_over = False
        self.winner = None
        self.last_action_valid = True

    def get_current_participant(self) -> Participant:
        return self.human if self.current_player == 0 else self.ai

    def get_opponent(self) -> Participant:
        return self.ai if self.current_player == 0 else self.human

    def spawn_buff(self):
        # Genera un nuovo buff ogni 3 turni in una cella libera
        if self.turn % 3 == 0:
            occupied = {self.human.get_position(), self.ai.get_position()}
            occupied.update(buff.get_position() for buff in self.buffs)
            all_cells = {(x, y) for x in range(1, 8) for y in range(1, 8)}
            vacant = list(all_cells - occupied)
            if vacant:
                x, y = random.choice(vacant)
                buff_type = random.choice(list(BuffType))
                self.buffs.append(BuffToken(x, y, buff_type))

    def expire_buffs(self):
        # Rimuove i buff la cui durata è terminata
        self.buffs = [buff for buff in self.buffs if not buff.expire()]

    def get_visible_cells(self, participant: Participant) -> List[Tuple[int, int]]:
        # Calcola le celle visibili in base a raggio e ultima direzione movimento
        visible = set()
        base_radius = participant.get_vision_radius()
        x0, y0 = participant.x, participant.y
        # Visibilità circolare
        for dx in range(-base_radius, base_radius + 1):
            for dy in range(-base_radius, base_radius + 1):
                x, y = x0 + dx, y0 + dy
                if 1 <= x <= self.grid_size and 1 <= y <= self.grid_size:
                    visible.add((x, y))
        # Effetto "scia" in direzione di movimento per migliorare esplorazione
        if participant.last_movement_direction != Direction.NONE:
            for k in (1, 2):
                if participant.last_movement_direction == Direction.UP:
                    x, y = x0, y0 - base_radius - k
                elif participant.last_movement_direction == Direction.DOWN:
                    x, y = x0, y0 + base_radius + k
                elif participant.last_movement_direction == Direction.LEFT:
                    x, y = x0 - base_radius - k, y0
                elif participant.last_movement_direction == Direction.RIGHT:
                    x, y = x0 + base_radius + k, y0
                else:
                    continue
                if 1 <= x <= self.grid_size and 1 <= y <= self.grid_size:
                    visible.add((x, y))
        return list(visible)

    def is_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) == 1

    def execute_action(self, action: ActionType) -> Tuple[bool, float]:
        participant = self.get_current_participant()
        opponent = self.get_opponent()
        reward = 0.0
        is_ai_turn = not participant.is_human

        # Se il partecipante è congelato, salta il turno decrementando lo stato freeze
        if participant.freeze_status > 0:
            participant.freeze_status -= 1
            self.next_turn()
            return True, 0.0

        # Movimenti: aggiorna posizione e assegna ricompensa in caso di raccolta buff
        if action in (ActionType.MOVE_UP, ActionType.MOVE_DOWN,
                      ActionType.MOVE_LEFT, ActionType.MOVE_RIGHT):
            new_x, new_y = participant.x, participant.y
            direction = Direction.NONE
            if action == ActionType.MOVE_UP:
                new_y -= 1; direction = Direction.UP
            elif action == ActionType.MOVE_DOWN:
                new_y += 1; direction = Direction.DOWN
            elif action == ActionType.MOVE_LEFT:
                new_x -= 1; direction = Direction.LEFT
            elif action == ActionType.MOVE_RIGHT:
                new_x += 1; direction = Direction.RIGHT

            if (1 <= new_x <= self.grid_size and 1 <= new_y <= self.grid_size
                    and (new_x, new_y) != opponent.get_position()):
                participant.set_position(new_x, new_y)
                participant.last_movement_direction = direction
                for buff in self.buffs[:]:
                    if buff.get_position() == (new_x, new_y):
                        reward += self.collect_buff(participant, buff)
                        self.buffs.remove(buff)
                        break
                self.last_action_valid = True
            else:
                self.last_action_valid = False
                if is_ai_turn:
                    reward = -0.2  # Penalità per azione non valida

        # Attacco corpo a corpo: ricompensa positiva se infligge danno
        elif action == ActionType.ATTACK:
            if self.is_adjacent(participant.get_position(), opponent.get_position()):
                if opponent.take_damage():
                    if is_ai_turn:
                        reward = 0.5
                else:
                    if is_ai_turn:
                        reward = 0.1  # Armor assorbito
                self.last_action_valid = True
            else:
                self.last_action_valid = False
                if is_ai_turn:
                    reward = -0.2

        # Attacco freeze: condizionato a possesso del buff e posizione relativa
        elif action == ActionType.FREEZE:
            if participant.freeze_attack_count > 0:
                opp_x, opp_y = opponent.get_position()
                par_x, par_y = participant.get_position()
                participant.use_freeze_attack()
                # Freeze riesce se su stessa riga, colonna o diagonale
                if (par_x == opp_x or par_y == opp_y or
                    abs(par_x - opp_x) == abs(par_y - opp_y)):
                    opponent.freeze_status = 2
                    if is_ai_turn:
                        reward = 0.5
                else:
                    if is_ai_turn:
                        reward = -0.1
                self.last_action_valid = True
            else:
                self.last_action_valid = False
                if is_ai_turn:
                    reward = -0.2

        # Ogni altra azione è considerata invalida
        else:
            self.last_action_valid = False
            if is_ai_turn:
                reward = -0.2

        # Verifica condizioni di vittoria o pareggio
        if opponent.hp <= 0:
            self.game_over = True
            self.winner = self.current_player
            reward = 1.0 if is_ai_turn else -1.0
        elif self.turn >= self.max_turns:
            self.game_over = True
            self.winner = -1  # Pareggio
            reward = 0.0

        # Aggiorna durata buff visione e passa al turno successivo
        participant.update_vision_duration()
        if not self.game_over:
            self.next_turn()

        return self.last_action_valid, reward

    def collect_buff(self, participant: Participant, buff: BuffToken) -> float:
        reward = 0.0
        is_ai_turn = not participant.is_human

        # Umano raccoglie senza ricompense RL
        if not is_ai_turn:
            if buff.buff_type == BuffType.HEALTH:
                participant.heal()
            elif buff.buff_type == BuffType.ARMOR:
                participant.add_armor()
            elif buff.buff_type == BuffType.VISION:
                participant.add_vision_buff()
            elif buff.buff_type == BuffType.FREEZE:
                participant.add_freeze_attack()
            return 0.0

        # Logica di ricompensa per l'IA: promuove raccolta di buff utili
        if buff.buff_type == BuffType.HEALTH:
            reward = 0.3 if participant.heal() else -0.1
        elif buff.buff_type == BuffType.ARMOR:
            reward = 0.3 if participant.add_armor() else -0.1
        elif buff.buff_type == BuffType.VISION:
            participant.add_vision_buff()
            reward = 0.3
        elif buff.buff_type == BuffType.FREEZE:
            reward = 0.3 if participant.add_freeze_attack() else -0.1
        return reward

    def next_turn(self):
        # Alterna il giocatore corrente, incrementa contatore turni e gestisce buff
        self.current_player = 1 - self.current_player
        self.turn += 1
        self.spawn_buff()
        self.expire_buffs()

    def get_ai_observation(self) -> np.ndarray:
        # Costruisce vettore di osservazione con maschera griglia e feature addizionali
        obs = np.zeros((self.grid_size, self.grid_size, 4), dtype=np.float32)
        visible_cells = self.get_visible_cells(self.ai)
        for (x, y) in visible_cells:
            obs[y - 1, x - 1, 0] = 1.0  # celle visibili
        ai_x, ai_y = self.ai.get_position()
        obs[ai_y - 1, ai_x - 1, 1] = 1.0  # posizione IA
        human_pos = self.human.get_position()
        if human_pos in visible_cells:
            obs[human_pos[1] - 1, human_pos[0] - 1, 2] = 1.0  # posizione umano
        for buff in self.buffs:
            bx, by = buff.get_position()
            if (bx, by) in visible_cells:
                obs[by - 1, bx - 1, 3] = 1.0  # posizioni buff
        # Evita ambiguità: celle occupate non marcate come visibili libere
        occupied_mask = (obs[:, :, 1] > 0) | (obs[:, :, 2] > 0) | (obs[:, :, 3] > 0)
        obs[occupied_mask, 0] = 0.0

        # Flatten della griglia e features numeriche: stato salute, armatura, buff, ultima direzione
        grid_flat = obs.flatten()
        features = [
            self.ai.hp / 3.0,
            float(self.ai.armor),
            self.ai.vision_duration / 3.0,
            float(self.ai.freeze_status),
            float(self.ai.freeze_attack_count),
            float(self.ai.last_movement_direction == Direction.UP),
            float(self.ai.last_movement_direction == Direction.DOWN),
            float(self.ai.last_movement_direction == Direction.LEFT),
            float(self.ai.last_movement_direction == Direction.RIGHT),
            1.0  # bias costante per rete neurale
        ]
        # Concatenazione in vettore di osservazione continuo per DQN
        return np.concatenate([grid_flat, np.array(features, dtype=np.float32)])
