import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os

# Architettura DQN: rete MLP per mappare lo stato alle azioni
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        # Strato di input a 256 unità: bilanciamento tra capacità di apprendimento e complessità
        self.layer1 = nn.Linear(input_size, 256)
        # Strato nascosto più ampio per rappresentare funzioni Q complesse
        self.layer2 = nn.Linear(256, 512)
        # Riduzione graduale per evitare eccessiva dimensionalità intermedia
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 128)
        # Strato di output: una unità per ciascuna azione possibile
        self.layer5 = nn.Linear(128, output_size)

    def forward(self, x):
        # Funzione di attivazione ReLU per introdurre non linearità e stabilità
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        # Output senza attivazione: valori Q per ogni azione
        return self.layer5(x)


# Replay buffer per memorizzare transizioni ed estrarre campioni non correlati
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        # Memoria con capacità massima per evitare consumo eccessivo di RAM
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Inserisce una nuova esperienza nella memoria
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        # Estrae un batch casuale per rompere la correlazione temporale tra transizioni
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Agente DQN: gestisce esplorazione, apprendimento e inferenza
class DQNAgent:
    def __init__(self, state_size, action_size, model_path="dqn_model.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.model_path = model_path

        # Fattore di sconto per valorizzare ricompense future (vicine a 1 favoriscono orizzonte lungo)
        self.gamma = 0.99
        # Parametri epsilon-greedy: esplorazione massima all'inizio, minima alla fine
        self.epsilon_start = 1.0
        self.epsilon_end = 0.1
        self.epsilon_decay = 10000
        # Tasso di apprendimento per l'ottimizzatore Adam, bilanciato per stabilità
        self.learning_rate = 0.0005
        # Dimensione batch per gli aggiornamenti di rete
        self.batch_size = 128
        # Dimensione massima del replay buffer
        self.replay_buffer_size = 20000
        # Frequenza di aggiornamento della rete target per stabilizzare il training
        self.target_update_frequency = 1000

        # Se disponibile, sfrutta GPU per velocizzare le operazioni tensoriali
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Reti di policy e target
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        # Carica modello preesistente se presente
        self.load_model()
        # Inizializza la rete target con gli stessi pesi della rete policy
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Ottimizzatore per la rete policy
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        # Buffer per memorizzare esperienze
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)
        self.steps_done = 0

    def get_action(self, state, is_training=True):
        # Calcola epsilon corrente con decadimento esponenziale
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

        # Esplora con probabilità epsilon durante il training, altrimenti sfrutta la policy appresa
        if is_training and random.random() < epsilon:
            return random.randrange(self.action_size)
        else:
            with torch.no_grad():
                # Prepara lo stato per la rete
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                # Seleziona l'azione con il valore Q massimo
                return q_values.max(1)[1].item()

    def learn(self):
        # Attende di avere abbastanza esperienze prima di aggiornare la rete
        if len(self.replay_buffer) < self.batch_size:
            return

        # Preleva un batch di esperienze
        experiences = self.replay_buffer.sample(self.batch_size)
        batch = Experience(*zip(*experiences))

        # Converte i dati in tensori per il calcolo
        state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # Calcola Q(s, a) per le azioni effettivamente eseguite
        q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Calcola il valore target: r + gamma * max_a' Q_target(s', a') (senza gradiente)
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))

        # Loss Huber per robustezza a outlier nelle ricompense
        loss = F.smooth_l1_loss(q_values, expected_q_values.unsqueeze(1))

        # Backpropagation: azzera i gradienti, calcola e applica l'ottimizzazione
        self.optimizer.zero_grad()
        loss.backward()
        # Clamping dei gradienti per evitare esplosione dei gradienti
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Ogni tot step, aggiorna la rete target per migliorare la stabilità
        if self.steps_done % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        # Salva i pesi della rete policy su file
        print("--- Saving model ---")
        torch.save(self.policy_net.state_dict(), self.model_path)

    def load_model(self):
        # Se il file esiste, carica i pesi precedentemente salvati
        if os.path.exists(self.model_path):
            print("--- Loading existing model ---")
            self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print("--- No model found, starting fresh ---")
