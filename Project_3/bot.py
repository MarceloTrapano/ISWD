from player import Player
import random


class KK(Player):
    def __init__(self, name):
        self.name = name
        self.cards = None
        self.last_action = None
        self.current_declare = None
        # Flaga: czy ostatnie sprawdzenie było błędne (bot stracił)
        self.last_check_failed = False
        self.skip_next_check = False

        # Wypłaty gracza
        self.payoffs = {
            ('B', 'C'): -2,
            ('B', 'N'): 1,
            ('T', 'C'): 0,
            ('T', 'N'): 1
        }

        # Wypłaty przeciwnika
        self.opponent_payoffs = {
            ('B', 'C'): 0,
            ('B', 'N'): 0,
            ('T', 'C'): -3,
            ('T', 'N'): 0
        }

        self.check_count = 1
        self.total_moves = 2
        self.opponent_decl_history = []
        self.known_cards = set()
        self.all_cards = self._init_all_cards()
        # Pool kart, które przeciwnik może mieć
        self.card_pool = set(self.all_cards)

    def _init_all_cards(self):
        return [(rank, suit) for rank in range(9, 15) for suit in range(4)]

    def startGame(self, cards):
        self.cards = cards.copy()
        # Zaznaczamy własne karty jako znane (nie w puli przeciwnika)
        self.known_cards = set(self.cards)
        self.card_pool -= self.known_cards

    def putCard(self, declared_card):
        # Aktualizacja puli przeciwnika po jego zagraniu
        if declared_card is not None:
            self.card_pool.discard(declared_card)

        # Sprawdzenie remisu
        if declared_card is not None and len(self.cards) == 1:
            current_rank = self.cards[0][0]
            if current_rank < declared_card[0]:
                return "draw"

        # Oblicz prawdopodobieństwo prawdomównego zagrania
        p_truth, _ = self._compute_mixed_nash()

        # Wyznacz docelową rangę deklaracji (poprzednia +1 lub 9 na start)
        target_rank = 9 if declared_card is None else min(declared_card[0] + 1, 14)

        # Jeżeli ostatnie sprawdzenie przez bota było błędne,
        # zagrywamy najgorszą kartę i blefujemy, deklarując target_rank
        if self.last_check_failed:
            actual_card = min(self.cards, key=lambda c: c[0])
            self.last_action = 'B'
            self.current_declare = target_rank
            self.last_check_failed = False
            return actual_card, (self.current_declare, actual_card[1])

        # Normalna logika: prawda vs blef
        legal = [c for c in self.cards if c[0] == target_rank]
        if legal and random.random() < p_truth:
            actual_card = legal[0]
            self.last_action = 'T'
        else:
            actual_card = random.choice(self.cards)
            self.last_action = 'B'

            # Jeżeli blefujemy, możemy zadeklarować wyżej niż mamy
        if self.last_action == 'B' and actual_card[0] < target_rank:
            # blef: deklarujemy wyższą rangę
            self.current_declare = target_rank

            # szukamy, czy mamy w ręce tę samą rangę z jakimś suwitem
            own_suits = [s for (r, s) in self.cards if r == self.current_declare]
            if own_suits:
                # blefujemy na własny kolor
                declared_suit = random.choice(own_suits)
            else:
                # jeśli nie mamy tej rangi w ręce, losujemy dowolny kolor
                declared_suit = actual_card[1]
        else:
            # zagranie prawdziwe lub ranga >= target_rank
            self.current_declare = actual_card[0]
            declared_suit = actual_card[1]

        return actual_card, (self.current_declare, declared_suit)

    def checkCard(self, opponent_declaration):
        if opponent_declaration in self.cards:
            return True

        if opponent_declaration is None or self.skip_next_check:
            self.skip_next_check = False  # Tylko raz pomijamy
            return False
        declared_rank, declared_suit = opponent_declaration
        bluff_count = sum(1 for decl, real in self.opponent_decl_history if decl != real[0])
        bluff_ratio = bluff_count / len(self.opponent_decl_history) if self.opponent_decl_history else 0.5

        count_remaining = len([c for c in self.card_pool if c[0] == declared_rank])
        rank_suspicion = 1 - (count_remaining / 4)

        suits_seen = {s for (r, s) in self.known_cards if r == declared_rank}
        suit_suspicion = 1.0 if declared_suit in suits_seen else 0.0

        _, q_check = self._compute_mixed_nash()
        final_q = 0.4 * q_check + 0.3 * bluff_ratio + 0.2 * rank_suspicion + 0.1 * suit_suspicion

        return random.random() < final_q

    def getCheckFeedback(self, checked, iChecked, iDrewCards, revealedCard, noTakenCards, log=True):
        # Jeżeli bot sprawdził (iChecked) i nie było karty do ujawnienia => był w błędzie
        if iChecked and not revealedCard:
            self.last_check_failed = True
            self.skip_next_check = True
        # standardowe aktualizacje statystyk
        if iChecked:
            self.check_count += 1
        self.total_moves += 1

        if revealedCard:
            self.known_cards.add(revealedCard)
            self.card_pool.discard(revealedCard)
            self.opponent_decl_history.append((self.current_declare, revealedCard))

        if isinstance(iDrewCards, (list, tuple)):
            for c in iDrewCards:
                self.known_cards.add(c)
                self.card_pool.discard(c)

    def _estimate_opponent_possible_cards(self):
        return list(self.card_pool)

    def count_possible_rank(self, rank):
        return len([c for c in self.card_pool if c[0] == rank])

    def _compute_mixed_nash(self):
        A = [
            [self.payoffs[('T', 'C')], self.payoffs[('T', 'N')]],
            [self.payoffs[('B', 'C')], self.payoffs[('B', 'N')]]
        ]
        B = [
            [self.opponent_payoffs[('T', 'C')], self.opponent_payoffs[('T', 'N')]],
            [self.opponent_payoffs[('B', 'C')], self.opponent_payoffs[('B', 'N')]]
        ]

        denom_q = (B[0][0] - B[0][1]) - (B[1][0] - B[1][1])
        q = (B[1][1] - B[0][1]) / denom_q if abs(denom_q) > 1e-6 else 0.5
        q = max(0, min(1, q))

        denom_p = (A[0][0] - A[1][0]) - (A[0][1] - A[1][1])
        p = (A[1][1] - A[1][0]) / denom_p if abs(denom_p) > 1e-6 else 0.5
        p = max(0, min(1, p))

        return p, q