import argparse
from game import Game

def main():
    # Inizializza il parser per gli argomenti da linea di comando,
    # permettendo di configurare modalità di esecuzione e numero di episodi
    parser = argparse.ArgumentParser(
        description="Grid Duel RL - Un gioco tattico ad arena con un'IA basata su RL."
    )
    # Opzione per selezionare se addestrare l'IA o giocare contro di essa
    parser.add_argument(
        '--mode',
        type=str,
        default='play',
        choices=['train', 'play'],
        help="Scegli 'train' per addestrare l'IA oppure 'play' per sfidarla."
    )
    # Numero di episodi per l'addestramento: un valore elevato favorisce la convergenza
    parser.add_argument(
        '--episodes',
        type=int,
        default=10000,
        help="Numero di episodi per l'addestramento dell'IA (default 10000)."
    )

    args = parser.parse_args()

    # Crea un'istanza del gioco; qui vengono inizializzate le strutture per lo stato e le politiche RL
    game = Game()

    if args.mode == 'train':
        # Avvia la fase di addestramento con il numero di episodi specificato
        game.train(num_episodes=args.episodes)
    elif args.mode == 'play':
        # Avvia la modalità interattiva, utilizzando la politica appresa durante il training
        game.play()

if __name__ == "__main__":
    main()
