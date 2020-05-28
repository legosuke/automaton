from collections import deque
from typing import Dict, List, Set, Tuple

import networkx as nx

State = str
Label = str
Word = str


class Dfa:
    """決定性有限オートマトン (DFA) を扱うクラス"""

    def __init__(self, states: Set[State], input_symbols: Set[Label],
                 transition_function: Dict[Tuple[State, Label], State],
                 start_state: State, final_states: Set[State]) -> None:

        for (s, l), d in transition_function.items():
            assert s in states
            assert d in states
            assert l in input_symbols

        assert start_state in states

        for s in final_states:
            assert s in states

        self.__states = states
        self.__input_symbols = input_symbols
        self.__transition_function = transition_function
        self.__start_state = start_state
        self.__final_states = final_states

    @property
    def states(self):
        return self.__states

    @states.setter
    def states(self, states: State):
        self.__states = states

    @property
    def input_symbols(self):
        return self.__input_symbols

    @input_symbols.setter
    def input_symbols(self, input_symbols: Set[Label]):
        self.__input_symbols = input_symbols

    @property
    def transition_function(self):
        return self.__transition_function

    @transition_function.setter
    def transition_function(self, transition_function: Dict[Tuple[State, Label], State]):
        self.__transition_function = transition_function

    @property
    def start_state(self):
        return self.__start_state

    @start_state.setter
    def start_state(self, start_state: State):
        self.__start_state = start_state

    @property
    def final_states(self):
        return self.__final_states

    @final_states.setter
    def final_states(self, final_states: Set[State]):
        self.__final_states = final_states

    def accept(self, word: Word) -> bool:
        """word の受理判定"""

        s = self.start_state
        for c in word:
            if c not in self.input_symbols:
                return False
            if (s, c) not in self.transition_function:
                return False
            s = self.transition_function[(s, c)]
        return s in self.final_states

    def complement(self) -> 'Dfa':
        """ある言語の補集合を受理するDFA"""

        states = self.states
        input_symbols = self.input_symbols
        transition_function = self.transition_function
        start_state = self.start_state
        final_states = {s for s in states if s not in self.final_states}
        return Dfa(states, input_symbols, transition_function, start_state, final_states)

    def intersection(self, x: 'Dfa') -> 'Dfa':
        """2つの言語の共通部分を受理するDFA"""

        assert self.input_symbols == x.input_symbols
        states = {'0'}
        input_symbols = self.input_symbols
        start_state = '0'
        transition_function = {}
        final_states = set()
        state_index = {(self.start_state, x.start_state): '0'}
        deq = [(self.start_state, x.start_state)]
        while deq:
            s0, s1 = deq.pop()
            for c in input_symbols:
                d0 = self.transition_function[(s0, c)]
                d1 = x.transition_function[(s1, c)]
                D = (d0, d1)
                if D not in state_index:
                    state_index[D] = str(len(states))
                    states.add(state_index[D])
                    deq.append(D)
                transition_function[(state_index[(s0, s1)], c)] = state_index[D]
        for (s0, s1), idx in state_index.items():
            if (s0 in self.final_states) and (s1 in x.final_states):
                final_states.add(idx)
        return Dfa(states, input_symbols, transition_function, start_state, final_states)

    def get_image(self, filename: str = None, format: str = None) -> None:
        """DFAの画像を作成"""

        filename += '.' + format
        nodes = []
        edges = []
        for s in self.states:
            attr = {'shape': 'circle'}
            if s in self.final_states:
                attr['shape'] = 'doublecircle'
            if s == self.start_state:
                nodes.append(('_', {'label': '', 'shape': 'plaintext'}))
                edges.append(('_', s, {'label': '開始'}))
            nodes.append((s, attr))
        vis = {}
        for (s, l), d in self.transition_function.items():
            if (s, d) in vis:
                vis[(s, d)].append(l)
            else:
                vis[(s, d)] = [l]
        for (s, d), L in vis.items():
            L = sorted(L)
            edges.append((s, d, {'label': ', '.join(L)}))
        g = nx.MultiDiGraph(rankdir="LR")
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        nx.nx_agraph.to_agraph(g).draw(filename, prog='dot', format=format)

    def table_filling_algorithm(self) -> List[Tuple[State, State]]:
        """穴埋めアルゴリズム"""

        num = len(self.states)
        states_list = sorted(list(self.states))
        states_order = {states_list[i]: i for i in range(num)}
        table = [[False] * num for i in range(num)]
        dependent_list = {}
        for i in range(num):
            for j in range(i+1, num):
                for c in self.input_symbols:
                    di = states_order[self.transition_function[(states_list[i], c)]]
                    dj = states_order[self.transition_function[(states_list[j], c)]]
                    if di > dj:
                        di, dj = dj, di
                    if (di, dj) not in dependent_list:
                        dependent_list[(di, dj)] = {(i, j)}
                    else:
                        dependent_list[(di, dj)].add((i, j))

        deq = deque()
        for s in self.final_states:
            order = states_order[s]
            for i in range(0, num):
                if states_list[i] not in self.final_states:
                    mn = min(i, order)
                    mx = max(i, order)
                    table[mn][mx] = True
                    deq.appendleft((mn, mx))

        while deq:
            distinguishable_pair = deq.pop()
            if distinguishable_pair in dependent_list:
                for ss in dependent_list[distinguishable_pair]:
                    if not table[ss[0]][ss[1]]:
                        table[ss[0]][ss[1]] = True
                        deq.append((ss[0], ss[1]))

        res = []
        for i in range(num):
            for j in range(i+1, num):
                if not table[i][j]:
                    res.append((states_list[i], states_list[j]))

        return res

    def minimize_dfa(self) -> 'Dfa':
        """DFA の最小化"""

        equivalent_pair = self.table_filling_algorithm()
        parent = list(range(len(self.states)))
        order = {}
        for a, b in equivalent_pair:
            if parent[b] == b:
                parent[b] = a
        for i in range(len(self.states)):
            if parent[i] == i:
                order[i] = len(order)

        states = list(order.values())
        input_symbols = self.input_symbols
        transition_function = {(order[parent[s]], l): order[parent[d]] for (s, l), d in self.transition_function.items()}
        start_state = order[parent[self.start_state]]
        final_states = {s for s in order if s in self.final_states}
        return Dfa(states, input_symbols, transition_function, start_state, final_states)


if __name__ == '__main__':

    # L1 = { w | wは偶数個の'a'と偶数個の'b'を含む }
    states1 = {'0', '1', '2', '3'}
    input_symbols1 = {'a', 'b'}
    transition_function1 = {
        ('0', 'a'): '1',
        ('0', 'b'): '2',
        ('1', 'a'): '0',
        ('1', 'b'): '3',
        ('2', 'a'): '3',
        ('2', 'b'): '0',
        ('3', 'a'): '2',
        ('3', 'b'): '1',
    }
    start_state1 = '0'
    final_states1 = {'0'}

    # L2 = { w | wは2回以上連続で'a'が続かない }
    states2 = {'0', '1', '2'}
    input_symbols2 = {'a', 'b'}
    transition_function2 = {
        ('0', 'a'): '1',
        ('0', 'b'): '0',
        ('1', 'a'): '2',
        ('1', 'b'): '0',
        ('2', 'a'): '2',
        ('2', 'b'): '2'
    }
    start_state2 = '0'
    final_states2 = {'0', '1'}

    # DFAの構築
    D1 = Dfa(states1, input_symbols1, transition_function1, start_state1, final_states1)
    D2 = Dfa(states2, input_symbols2, transition_function2, start_state2, final_states2)

    # 補集合を受理するDFAを構築する
    D1_complement = D1.complement()

    # L1 かつ L2 を満たす言語を受理するオートマトンの構成
    D12 = D1.intersection(D2)

    # print(D12.transition_function)

    # 最小化DFAの構成
    # minD12 = D12.minimize_dfa()

    # D1で受理される文字列を判定
    words1 = ['ab', 'abab', 'aabb', 'bbabba', 'bab', 'bbb', 'bbbb', 'baab', 'babab', 'bababa', 'babababa']
    print('DFA D1')
    for word in words1:
        print(f'  "{word}" is accepted ? : {D1.accept(word)}')
    print()

    # D2で受理される文字列を判定
    print('DFA D2')
    for word in words1:
        print(f'  "{word}" is accepted ? : {D2.accept(word)}')
    print()

    # D12で受理される文字列を判定
    words2 = ['ab', 'abab', 'aabb', 'bbabba', 'bab', 'bbb', 'bbbb', 'baab', 'babab', 'bababa', 'babababa']
    print('DFA D12')
    for word in words2:
        print(f'  "{word}" is accepted ? : {D12.accept(word)}')
    print()

    # PNG形式で出力
    D1.get_image('D1', 'png')
    D1_complement.get_image('D1_complement', 'png')
    D2.get_image('D2', 'png')
    D12.get_image('D12', 'png')
    # minD12.get_image('minD12', 'png')

'''
型定義
- State: str
- Label: str
- Word: str
※Stateは数字でないと多重辺が描画されないバグがある

メンバ変数
- states: Set[State]
- input_symbols: Set[Label]
- transition_function: Dict[Tuple[State, Label], State]
- start_state: State
- final_states: Set[State]

メンバ関数
- accept: Callable[[Word], bool]
- complement: Callable[[], Dfa]
- intersection: Callable[[Dfa], Dfa]
- table_filling_algorithm: Callable[[Dfa], List[Tuple[State, State]]]
- minimize_dfa: Callable[[Dfa], Dfa]
- get_image: Callable[[str, str], None]

'''
