import heapq
import itertools
import math


def a_star(start, h, goal, open) -> list:
    closed_set = set()
    open_set = set()
    g_value = {}
    f_value = []
    parent = {}

    # initialization
    f_start = h(start)
    g_value[start] = 0
    open_set.add(start)
    heapq.heappush(f_value, (f_start, start))

    while open_set:
        cost, node = heapq.heappop(f_value)

        if goal(node):
            return make_path(node, parent)

        closed_set.add(node)
        open_set.remove(node)

        for n in open(node):
            tentative_g_score = g_value[node] + 1

            # only works if your heuristic is consistent (monotonic)
            if n in closed_set:
                continue

            if n not in open_set or tentative_g_score < g_value[n]:
                parent[n] = node
                g_value[n] = tentative_g_score
                actual_f_value = tentative_g_score + h(n)

                if n in open_set:
                    # O(N) rebuild of the heap (Python heap doesn't have decrease-key operation)
                    for i, (p, x) in f_value:
                        if x == n:
                            f_value[i] = (actual_f_value, n)
                            break
                    heapq.heapify(f_value)
                else:
                    open_set.add(n)
                    heapq.heappush(f_value, (actual_f_value, n))


def make_path(node, parent) -> list:
    path = [node]

    while node in parent:
        node = parent[node]
        # path.insert(0, node)
        path.append(node)

    path.reverse()
    return path


if __name__ == '__main__':
    import time

    from game import Game


    def goal(game):
        return game.is_game_finished()


    def open(game):
        if game.is_game_finished():
            return

        for move in "nsew":
            new_game = game.play(move)

            if (new_game != game):
                yield new_game


    def trivial_heuristic(game):
        return 0


    def numberOfTiles_heuristic(game):
        numberOfTiles = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                if game.board[i][j] > 0:
                    numberOfTiles += 1
        return numberOfTiles


    lg_d = {2 ** t: t for t in range(32)}
    pw_d = {t: 2 ** t for t in range(32)}
    rt_d = {t: math.sqrt(2 ** t) for t in range(32)}
    #ideals = {1: [2], 2: [2, 2], 3: [4, 2], 4: [4, 2, 2], 5: [8, 2], 6: [8, 2, 2], 7: [8, 4, 2], 8: [8, 4, 2, 2], 9: [8, 4, 4, 2], 10: [8, 8, 2, 2], 11: [16, 4, 2]}
    #ideals_c = {1: [1], 2: [2, 0], 3: [1, 1], 4: [2, 1], 5: [1, 0, 1], 6: [2, 0, 1], 7: [1, 1, 1], 8: [2, 1, 1], 9: [1, 2, 1], 10: [2, 0, 2], 11: [1, 1, 0, 1]}

    ideals = {0:[2]}
    last_ = ideals[0]

    for i in range(1024 + 9):
        next_ = []

        prev_ = None
        for n in last_:
            if prev_ is None:
                prev_ = n
            elif prev_ == n:
                next_.append(n * 2)
                prev_ = None
            else:
                next_.append(prev_)
                prev_ = n
        if prev_ is not None:
            next_.append(prev_)
        next_.append(2)

        ideals[i+1] = next_

        last_ = next_

    ideals_c = {k: [lg_d[v] for v in ideals[k]] for k in ideals.keys()}
    ideals_s = {k: sum(ideals[k]) for k in ideals.keys()}
    ideals_s2 = {k: sum([w ** 2 for w in ideals[k]]) for k in ideals.keys()}
    ideals_tc = {k: len(ideals[k]) for k in ideals.keys()}
    stats_w = ()


    def i_range(a: int, b: int):
        if a < b:
            return a + 1, b - 1
        else:
            return b + 1, a - 1



    path = None

    game = Game(4, 64)
    lgg = lg_d[game.goal]
    sdc = game.goal + (lgg - 1) # wrong
    target_sum = game.goal + (lgg - 2) * 2
    goal_2 = game.goal ** 2
    target_sum_2 = goal_2 + ideals_s2[lgg - 2]

    start = time.time()
    # path = a_star(game, min_perfect_heuristic_dc, goal, open)  #

    delta_t = time.time() - start

    if path is not None:
        print(path)
        print(len(path))
        print(delta_t)

        # print(game)
