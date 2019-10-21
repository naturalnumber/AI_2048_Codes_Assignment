import heapq
import itertools
import math
import random
import statistics


def a_star(start, h, goal, open, h_args, start_time, max_time) -> list:
    closed_set = set()
    open_set = set()
    g_value = {}
    f_value = []
    parent = {}

    # initialization
    f_start = h(start, h_args)
    g_value[start] = 0
    open_set.add(start)
    heapq.heappush(f_value, (f_start, start))

    while open_set:
        if time.time() - start_time > max_time:
            return None
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
                actual_f_value = tentative_g_score + h(n, h_args)

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


    def trivial_heuristic(game, *ignore):
        return 0


    def numberOfTiles_heuristic(game, *ignore):
        numberOfTiles = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                if game.board[i][j] > 0:
                    numberOfTiles += 1
        return numberOfTiles


    lg_d = {2 ** t: t for t in range(32)}
    pw_d = {t: 2 ** t for t in range(32)}
    rt_d = {t: math.sqrt(2 ** t) for t in range(32)}
    ideals = {1: [2], 2: [2, 2], 3: [4, 2], 4: [4, 2, 2], 5: [8, 2], 6: [8, 2, 2], 7: [8, 4, 2], 8: [8, 4, 2, 2],
              9: [8, 4, 4, 2], 10: [8, 8, 2, 2], 11: [16, 4, 2]}
    ideals_c = {1: [1], 2: [2, 0], 3: [1, 1], 4: [2, 1], 5: [1, 0, 1], 6: [2, 0, 1], 7: [1, 1, 1], 8: [2, 1, 1],
                9: [1, 2, 1], 10: [2, 0, 2], 11: [1, 1, 0, 1]}
    ideals_s = {k: sum(ideals[k]) for k in ideals.keys()}
    ideals_s2 = {k: sum([w ** 2 for w in ideals[k]]) for k in ideals.keys()}
    stats_w = ()


    def tile_stats_basic(game: Game) -> tuple:
        sumOfTiles = 0
        counts = [0] * 32
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    sumOfTiles += tile
                    counts[lg_d[tile]] += 1
        return sumOfTiles, counts


    def min_remaining_perfect_merges(game: Game, counts: list = None, *ignored) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        num_ = 2
        while check > 1 and counts[check] < num_:
            min_ += 1
            num_ -= counts[check]
            num_ *= 2
            check -= 1

        if check == 1:
            min_ += max(num_ - counts[check], 0)

        return min_


    def compress(game: Game) -> tuple:
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    #ltile = lg_d[tile]
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)

        return row_comp, col_comp


    def i_range(a: int, b: int):
        if a < b:
            return a + 1, b - 1
        else:
            return b + 1, a - 1


    def find_adj_distance(game, tile, count):
        found = []

        for i in range(game.dimension):
            for j in range(game.dimension):
                if game.board[i][j] != tile:
                    continue
                found.append((i, j))
                if len(found) == count:
                    break
            if len(found) == count:
                break

        for i in range(count):
            for j in range(i, count):
                hi = found[i]
                hj = found[j]

                di = abs(hi[0] - hj[0])
                dj = abs(hi[1] - hj[1])

                if di + dj == 1:
                    return 1

                adjacent = True
                if di == 0:
                    for s in range(*i_range(hi[1], hj[1])):
                        if game.board[hi[0]][s] != 0:
                            # min_ = min(3, min_)
                            adjacent = False
                            break
                    if adjacent:
                        return 1
                elif dj == 0:
                    for s in range(*i_range(hi[0], hj[0])):
                        if game.board[s][hj[1]] != 0:
                            # min_ = min(3, min_)
                            adjacent = False
                            break
                    if adjacent:
                        return 1
        return 2


    def find_cadj_distance(game, tile, count, row_comp, col_comp):
        found = []

        for i in range(game.dimension):
            for j in range(game.dimension):
                if game.board[i][j] != tile:
                    continue
                found.append((i, j))
                if len(found) == count:
                    break
            if len(found) == count:
                break

        c_adjacent = False
        for i in range(count):
            for j in range(i, count):
                hi = found[i]
                hj = found[j]

                dr = abs(hi[0] - hj[0])
                dc = abs(hi[1] - hj[1])

                if dr + dc == 1:
                    return 1

                adjacent = True
                if dr == 0:
                    for s in range(*i_range(hi[1], hj[1])):
                        if game.board[hi[0]][s] != 0:
                            # min_ = min(3, min_)
                            adjacent = False
                            break
                    if adjacent:
                        return 1
                elif dc == 0:
                    for s in range(*i_range(hi[0], hj[0])):
                        if game.board[s][hj[1]] != 0:
                            # min_ = min(3, min_)
                            adjacent = False
                            break
                    if adjacent:
                        return 1
                elif not c_adjacent:
                    if dr == 1:
                        for k in range(min(row_comp[hi[0]], row_comp[hj[0]])):
                            if row_comp[hi[0]][k] == row_comp[hj[0]][k] or \
                               row_comp[hi[0]][-1-k] == row_comp[hj[0]][-1-k]:
                                c_adjacent = True
                                break
                    elif dc == 1:
                        for k in range(min(col_comp[hi[1]], col_comp[hj[1]])):
                            if col_comp[hi[1]][k] == col_comp[hj[1]][k] or \
                               col_comp[hi[1]][-1-k] == col_comp[hj[1]][-1-k]:
                                c_adjacent = True
                                break
        if c_adjacent:
            return 2
        return 3


    def min_remaining_perfect_merges_d(game: Game, counts: list = None, cut_=-1, *ignored) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        num_ = 2
        while check > 1 and counts[check] < num_:
            min_ += 1
            num_ -= counts[check]
            num_ *= 2
            check -= 1

        if check == 1:
            min_ += max(num_ - counts[check], 0)

        if min_ + 2 < cut_:
            return cut_
        return max(min_ + find_adj_distance(game, pw_d[check], counts[check]), cut_)


    def min_remaining_perfect_merges_dc(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        num_ = 2
        while check > 1 and counts[check] < num_:
            min_ += 1
            num_ -= counts[check]
            num_ *= 2
            check -= 1

        if check == 1:
            min_ += max(num_ - counts[check], 0)

        if min_ + 3 < cut_:
            return cut_
        return max(min_ + find_cadj_distance(game, pw_d[check], counts[check], *compress(game)), cut_)


    def weighted_heuristic(game: Game, stats_w) -> float:
        global target_sum
        global target_sum_2
        small = 1e-3
        sum_ = 0
        sum_2 = 0
        count = 0
        w_comp = max(abs(stats_w[8]), abs(stats_w[9]), abs(stats_w[10]))
        w_cnts = max(abs(stats_w[6]), abs(stats_w[7]), abs(stats_w[13]))
        if w_comp > small:
            counts = [0] * 32
            row_comp = [[] for i in range (game.dimension)]
            col_comp = [[] for i in range (game.dimension)]
            for i in range(len(game.board)):
                for j in range(len(game.board)):
                    tile = game.board[i][j]
                    if tile > 0:
                        sum_ += tile
                        sum_2 += tile ** 2
                        counts[lg_d[tile]] += 1
                        count += 1
                        row_comp[i].append(tile)
                        col_comp[j].append(tile)
        elif w_cnts > small:
            counts = [0] * 32
            for i in range(len(game.board)):
                for j in range(len(game.board)):
                    tile = game.board[i][j]
                    if tile > 0:
                        sum_ += tile
                        sum_2 += tile ** 2
                        counts[lg_d[tile]] += 1
                        count += 1
        else:
            for i in range(len(game.board)):
                for j in range(len(game.board)):
                    tile = game.board[i][j]
                    if tile > 0:
                        sum_ += tile
                        sum_2 += tile ** 2
                        count += 1

        cut_ = (target_sum - sum_)
        cut_2 = (target_sum_2 - sum_2)
        # 0 max(cut_, 0) max(target_sum_2 - sum_2, 0)
        wsum_ = stats_w[0] * (1 + stats_w[1] * cut_) * cut_ + \
                stats_w[2] * (1 + stats_w[3] * cut_2) * cut_2 + \
                + stats_w[5] * count
                #stats_w[4] * (target_sum_2 // sum_2) \


        # sum min adjacency diff

        # sum total expected merges

        # tiles over needed

        # if abs(stats_w[0]) > small:
        #    wsum_ += stats_w[0] * cut_
        # if abs(stats_w[1]) > small:
        #    wsum_ += stats_w[1] * (cut_ ** 2)
        # if abs(stats_w[2]) > small:
        #    wsum_ += stats_w[2] * (target_sum_2 - sum_2)
        # if abs(stats_w[3]) > small:
        #    wsum_ += stats_w[3] * ((target_sum_2 - sum_2) ** 2)
        if abs(stats_w[4]) > small:
            wsum_ += stats_w[4] * (target_sum_2 // sum_2)
        # if abs(stats_w[5]) > small:
        #    wsum_ += stats_w[5] * count

        wsum_ *= 2

        if w_cnts > small:
            if abs(stats_w[6]) > small:
                wsum_ += stats_w[6] * min_remaining_perfect_merges(game, counts)
            if abs(stats_w[7]) > small:
                wsum_ += stats_w[7] * min_remaining_perfect_merges_d(game, counts, cut_2*2)
            if abs(stats_w[13]) > small:
                wsum_ += stats_w[13] * min_remaining_perfect_merges_dc(game, counts, cut_2*2)
        if w_comp > small:
            if abs(stats_w[8]) > small:
                adj = 0

                for i in range(game.dimension):
                    for j in range(len(row_comp[i]) - 1):
                        adj += abs(lg_d[row_comp[i][j]] - lg_d[row_comp[i][j + 1]])
                    for j in range(len(col_comp[i]) - 1):
                        adj += abs(lg_d[col_comp[i][j]] - lg_d[col_comp[i][j + 1]])

                wsum_ += stats_w[8] * adj
            if abs(stats_w[9]) > small:
                sum_e = 0
                for i in range(game.dimension):
                    if len(row_comp[i]) > 2:
                        sum_e += row_comp[i][0] + row_comp[i][-1]
                    elif len(row_comp[i]) > 2:
                        sum_e += row_comp[i][0]

                    if len(col_comp[i]) > 2:
                        sum_e += col_comp[i][0] + col_comp[i][-1]
                    elif len(col_comp[i]) > 2:
                        sum_e += col_comp[i][0]
                wsum_ += stats_w[9] * -sum_e
            if abs(stats_w[10]) > small:
                sum_e2 = 0
                for i in range(game.dimension):
                    if len(row_comp[i]) > 2:
                        sum_e2 += row_comp[i][0]**2 + row_comp[i][-1]**2
                    elif len(row_comp[i]) > 2:
                        sum_e2 += row_comp[i][0]**2

                    if len(col_comp[i]) > 2:
                        sum_e2 += col_comp[i][0]**2 + col_comp[i][-1]**2
                    elif len(col_comp[i]) > 2:
                        sum_e2 += col_comp[i][0]**2
                wsum_ += stats_w[10] * -sum_e2
        if abs(stats_w[11]) > small:
            sum_e = 0
            for i in range(game.dimension):
                sum_ += game.board[i][0] + game.board[i][-1]
                sum_ += game.board[0][i] + game.board[-1][i]
            wsum_ += stats_w[11] * -sum_e
        if abs(stats_w[12]) > small:
            sum_e2 = 0
            for i in range(game.dimension):
                sum_ += game.board[i][0]**2 + game.board[i][-1]**2
                sum_ += game.board[0][i]**2 + game.board[-1][i]**2
            wsum_ += stats_w[12] * -sum_e2
        return wsum_*5


    num_stats = 14

    path = None
    do_evo = True

    game = Game(4, 32)
    lgg = lg_d[game.goal]
    sdc = game.goal + (lgg - 1)  # wrong
    target_sum = game.goal + (lgg - 2) * 2
    goal_2 = game.goal ** 2
    target_sum_2 = goal_2 + ideals_s2[lgg - 2]

    num_algs = 200
    init_range = 2

    initial = []

    for i in range(num_stats):
        base = [0] * num_stats
        base[i] = 1
        initial.append(base)

    while len(initial) < num_algs and do_evo:
        initial.append([random.uniform(-init_range / 4, init_range) for s in range(num_stats)])

    running = True
    n_runs = 1000
    n_parents = num_algs//3
    next_ = []
    last_ = []
    parents = []

    this_ = initial


    def fitness(t, mt, l, ml, c, mc):
        return (mt - t) / (0.001 + mt) + 0.1 * (ml - l) / ml - c / (10 + mc) / 10  # 1-math.atan(t)
        # return math.atan(0.25 + 0.5 * ((mt - t)/mt + 0.1 * (ml - l)/ml)) #1-math.atan(t)

    def mix(x, y):
        p = random.uniform(0.05, 0.95)
        return p * x + (1 - p) * y

    def mutate():
        if random.random() > 0.95:
            return random.uniform(-1, 1) + random.uniform(-.25, .25) \
                   + random.uniform(-.25, .25) + random.uniform(-.25, .25) + 0.1
        return 0


    if not do_evo:
        #     #diff diff_2 2diff 2diff2 ratio count mperf mperfd ladj -edge -edge2
        alg = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        print(f"Starting: {alg}")
        start = time.time()
        path = a_star(game, weighted_heuristic, goal, open, alg, start, 7)
        delta_t = time.time() - start
        if path is None:
            print("Failed")
        else:
            length = len(path)
            print(f"Finished in {delta_t}s in {length} moves")

    ran = 0
    max_t = 10
    print("Starting")
    while running and do_evo:
        temp = next_
        next_ = last_
        last = next_
        next_.clear()

        for alg in this_:
            if ran % 5 == 4: print(f"Starting: {alg}")
            start = time.time()
            path = a_star(game, weighted_heuristic, goal, open, alg, start, max_t*1.5)
            delta_t = time.time() - start
            if path is None:
                if ran % 5 == 4: print("Failed")
                continue
            length = len(path)
            next_.append(([], delta_t, length, alg, sum([x**2 for x in alg])))
            if ran % 5 == 4: print(f"Finished in {delta_t}s in {length} moves")

        max_t = 0
        max_l = 0
        max_c = 0
        tot_f = 0
        for e in next_:
            if max_t + max_l == 0:
                max_t = e[1]
                max_l = e[2]
                max_c = e[4]
            else:
                max_t = max(max_t, e[1])
                max_l = max(max_l, e[2])
                max_c = max(max_c, e[4])
        for e in next_:
            fit = fitness(e[1], max_t, e[2], max_l, e[4], max_c)
            tot_f += fit
            e[0].append(fit)
        next_.sort()
        cum_f = 0
        for e in next_:
            fit = e[0][0] / tot_f
            cum_f += fit
            e[0].append(fit)
            e[0].append(cum_f)

        ran += 1

        if ran % 10 == 0 or ran > n_runs:
            best_coefs = [[] for i in range(num_stats)]
            for i in range(min(25, len(next_))):
                for j in range(num_stats):
                    best_coefs[j].append(next_[i][3][j])
            means = [statistics.mean(best_coefs[i]) for i in range(num_stats)]
            std_devs = [statistics.stdev(best_coefs[i], means[i]) for i in range(num_stats)]

            print(f"{ran}: Convergence values {list(zip(means,std_devs))}")

            if ran > n_runs:
                running = False
                break


        parents.clear()

        for i in range(1, min(11, len(next_))):
            parents.append(next_[-i][3])

        while len(parents) < n_parents:
            s = random.uniform(0, 1) + 0.05
            for e in next_:
                if e[0][2] > s:
                    parents.append(e[3])

        this_.clear()

        for i in range(1, min(11, len(next_))):
            this_.append(next_[-i][3])

        for x in random.sample(parents, 5):
            this_.append(x)

        while len(this_) < num_algs:
            pair = random.sample(parents, 2)
            this_.append([mix(x, y)+mutate() for x, y in zip(*pair)])

        #print([x[0] for x in next_])
        #print(this_)
        #running = False

next_.reverse()

print(next_)

for i in range(25):
    print(next_[i])


for i in range(25):
    print(next_[i][3])


