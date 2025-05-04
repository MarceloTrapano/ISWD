import numpy as np
from player import Player

class Kondys_Dabrowski(Player):
    def lowestCardDeclaired(self, declared_card):
        def sorter(x):
            if x[0] >= declared_card[0]:
                return x[0]
            return np.inf
        return min([(-1,0)] + self.cards, key= lambda x: sorter(x))
    def putCard(self, declared_card) -> tuple[int,int]|str:
        num_of_cards = len(self.cards)
        if num_of_cards > 10:
            if declared_card is not None and len(np.array(self.cards)[np.array(self.cards)[:,0] >= declared_card[0]]) > 8:
                return sorted(self.cards, key=lambda x: x[0])[::-1][4], sorted(self.cards, key=lambda x: x[0])[::-1][4]
        minimal_card = min(self.cards, key=lambda x: x[0]) # znajdywanie najmniejszej karty
        if declared_card is None or minimal_card[0] >= declared_card[0]:
            return minimal_card, minimal_card
        if num_of_cards < 2: # dla małej liczby kart taktyka mniej ryzykowna
            prob = 0.8
        elif num_of_cards < 4: # dla lekko wygrywania wybieramy trochę bardziej ryzykowne granie
            prob = 0.6
        elif num_of_cards < 7: # dla lekko wygrywania wybieramy trochę bardziej ryzykowne granie
            prob = 0.3
        elif num_of_cards < 10: # dla lekko wygrywania wybieramy trochę bardziej ryzykowne granie
            prob = 0.1
        else: # dla przegrywania duża szansa na ryzyko
            prob = 0
        
        # taktyka działa tak, że jak mamy sytuację, gdzie nasza minimalna karta jest niższa od zadeklarowanej
        # to mamy możliwość z prawodpodobieństwem p, że deklarujemy najlepszą kartę jaką mamy i stawiamy najgorszą lub zagrywamy
        # najlepszą możliwą kartę

        fake_card = self.lowestCardDeclaired(declared_card)
        if fake_card[0] == -1:
            if num_of_cards != 1:
               return np.random.choice(np.array([(minimal_card, (np.random.randint(declared_card[0], 15), np.random.randint(0,4))), "draw"], dtype = object), p = [0.8, 0.2])
            return 'draw'
        idx = np.random.choice([0,1], p = [prob, 1-prob])
        choiche = [(minimal_card, fake_card), (fake_card, fake_card)][idx]
        return choiche[0], choiche[1]
    
    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards: 
            return True # sprawdzenie czy zadeklarowana karta przeciwnika jest w twojej ręce
        match opponent_declaration[0]:
            case 9:
                prob = 0
            case 10:
                prob = 0
            case 11:
                prob = 0.2
            case 12:
                prob = 0.5
            case 13:
                prob = 0.8
            case 14:
                prob = 0.9
        
        return np.random.choice([True, False], p = [prob, 1-prob])