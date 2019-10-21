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


    def max_union_meta_heuristic(game):
        return max(numberOfTiles_heuristic(game), perfect_moves_heuristic(game))


    def max_union_meta_heuristic_x2(game):
        return max(numberOfTiles_heuristic(game), perfect_moves_heuristic_x2(game))


    def perfect_remaining_merges_arch(game: Game, counts: list = None) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        while check > 0 and counts[check] < 2:
            min_ += 1
            check -= 1
        return min_


    def perfect_moves_heuristic_arch(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        return max(perfect_remaining_merges_arch(game, counts), sdc - sum_)


    def perfect_remaining_merges(game: Game, counts: list = None) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        num_ = 2
        while check > 0 and counts[check] < num_:
            num_ -= counts[check]
            num_ *= 2
            min_ += 1
            check -= 1

        # if 2 ...

        return min_


    def perfect_moves_heuristic(game: Game) -> int:
        sum_, counts = tile_stats_basic(game)
        return max(perfect_remaining_merges(game, counts), (target_sum - sum_) / 2)


    def perfect_moves_heuristic_x2(game: Game) -> int:
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        return max(perfect_remaining_merges(game, counts), target_sum - sum_)


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

    def find_distance_arch(game, tile, count):
        found = []
        min_ = 3

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
                else:
                    min_ = min(2, min_)
        return min_


    def min_remaining_merges_arch(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        while check > 0 and counts[check] < 2:
            min_ += 1
            check -= 1
        if min_ + 2 < cut_:
            return cut_
        return max(min_+find_distance_arch(game, pw_d[check], counts[check]), cut_)


    def min_moves_heuristic_arch(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges_arch(game, counts, cut_)


    def min_remaining_merges_meta_arch(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        while check > 0 and counts[check] < 2:
            min_ += 1
            check -= 1

        penalty = counts[check] - 2

        if min_ + 2 < cut_:
            return cut_+penalty
        return max(min_+find_distance_arch(game, pw_d[check], counts[check]), cut_)+penalty


    def min_moves_heuristic_meta_arch(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges_meta_arch(game, counts, cut_)


    def min_moves_heuristic_meta2_arch(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        #penalty = 0
        #if game.goal < sum_:
        #    penalty = sum_ - game.goal
        return min_remaining_merges_meta_arch(game, counts, cut_)#+penalty


    def find_distance_meta(game, tile, count):
        found = []
        min_ = 3

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
                else:
                    min_ = min(2, min_)
        return min_


    def min_remaining_merges(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        while check > 0 and counts[check] < 2:
            min_ += 1
            check -= 1
        if min_ + 2 < cut_:
            return cut_
        return max(min_ + find_distance_meta(game, pw_d[check], counts[check]), cut_)


    def min_moves_heuristic(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_) / 2
        return min_remaining_merges(game, counts, cut_)


    def min_moves_heuristic_x2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges(game, counts, cut_)


    def min_remaining_merges_meta(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        while check > 0 and counts[check] < 2:
            min_ += 1
            check -= 1

        penalty = counts[check] - 2

        if min_ + 2 < cut_:
            return cut_ + penalty
        return max(min_ + find_distance_meta(game, pw_d[check], counts[check]), cut_) + penalty


    def min_moves_heuristic_meta(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_) / 2
        return min_remaining_merges_meta(game, counts, cut_)


    def min_moves_heuristic_meta_x2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges_meta(game, counts, cut_)


    def min_moves_heuristic_meta2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats_basic(game)
        # if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_) / 2
        penalty = 0
        if game.goal < sum_:
            penalty = sum_ - game.goal
        return min_remaining_merges_meta(game, counts, cut_) + penalty

    ###################################################################################

    def goal_diff_heuristic(game: Game) -> int:
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
        #return max((game.goal - sum_) // 2, 0)
        return (game.goal - sum_) // 2


    def goal_diff_heuristic_2(game: Game) -> int:
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
        return (game.goal - sum_) ** 2


    def sum_diff_heuristic(game: Game) -> int:
        global target_sum
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
        #return max((target_sum - sum_) // 2, 0)
        return (target_sum - sum_) // 2


    def sum_diff_heuristic_2(game: Game) -> int:
        global target_sum
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
        return (target_sum - sum_) ** 2


    def sum_diff_heuristic_x2(game: Game) -> int:
        global target_sum
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
        return max(target_sum - sum_, 0)
        #return target_sum - sum


    def goal2_diff_heuristic(game: Game) -> int:
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_2 += game.board[i][j] ** 2
        return max(goal_2 - sum_2, 0)
        #return goal_2 - sum_2


    def goal2_diff_heuristic_2(game: Game) -> int:
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_2 += game.board[i][j] ** 2
        return (goal_2 - sum_2) ** 2


    def sum2_diff_heuristic(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_2 += game.board[i][j] ** 2
        return max(target_sum_2 - sum_2, 0)
        #return target_sum_2 - sum_2


    def sum2_diff_heuristic_2(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_2 += game.board[i][j] ** 2
        return (target_sum_2 - sum_2) ** 2


    def min_remaining_perfect_merges(game: Game, counts: list = None) -> int:
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


    def min_remaining_perfect_merges_d(game: Game, counts: list = None, cut_=-1) -> int:
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


    def min_perfect_heuristic(game):
        global target_sum
        sum_, counts = tile_stats_basic(game)
        return max(min_remaining_perfect_merges(game, counts), (target_sum - sum_) // 2)


    def min_perfect_heuristic_d(game):
        global target_sum
        sum_, counts = tile_stats_basic(game)
        cut_ = (target_sum - sum_) / 2
        return max(min_remaining_perfect_merges_d(game, counts, cut_), cut_)


    def min_perfect_heuristic_dc(game):
        global target_sum
        sum_, counts = tile_stats_basic(game)
        cut_ = (target_sum - sum_) / 2
        return max(min_remaining_perfect_merges_dc(game, counts, cut_), cut_)


    def near_perfect_heuristic(game):
        global target_sum_2
        sum_2 = 0
        counts = [0] * 32
        #row_comp = [[] * game.dimension] * game.dimension
        #col_comp = [[] * game.dimension] * game.dimension
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    #sum_ += tile
                    sum_2 += tile ** 2
                    counts[lg_d[tile]] += 1
                    #count += 1
                    #row_comp[i].append(tile)
                    #col_comp[j].append(tile)
        return max(min_remaining_perfect_merges(game, counts), target_sum_2 - sum_2)


    def near_perfect_heuristic_g(game):
        global goal_2
        sum_2 = 0
        counts = [0] * 32
        #row_comp = [[] * game.dimension] * game.dimension
        #col_comp = [[] * game.dimension] * game.dimension
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    #sum_ += tile
                    sum_2 += tile ** 2
                    counts[lg_d[tile]] += 1
                    #count += 1
                    #row_comp[i].append(tile)
                    #col_comp[j].append(tile)
        return max(min_remaining_perfect_merges(game, counts), goal_2 - sum_2)


    def near_perfect_heuristic_d(game):
        global target_sum_2
        sum_2 = 0
        counts = [0] * 32
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    sum_2 += tile ** 2
                    counts[lg_d[tile]] += 1
        cut_ = target_sum_2 - sum_2
        return max(min_remaining_perfect_merges_d(game, counts, cut_), cut_)


    def near_perfect_heuristic_dc(game):
        global target_sum_2
        sum_2 = 0
        counts = [0] * 32
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    sum_2 += tile ** 2
                    counts[lg_d[tile]] += 1
        cut_ = target_sum_2 - sum_2
        return max(min_remaining_perfect_merges_dc(game, counts, cut_), cut_)


    def near_perfect_heuristic_d_g(game):
        global goal_2
        sum_2 = 0
        counts = [0] * 32
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    sum_2 += tile ** 2
                    counts[lg_d[tile]] += 1
        cut_ = goal_2 - sum_2
        return max(min_remaining_perfect_merges_d(game, counts, cut_), cut_)


    def ratio_heuristic_g(game):
        global target_sum
        sum_ = 0
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
                sum_2 += game.board[i][j] ** 2
        return max(goal_2//sum_2, (target_sum - sum_) // 2)


    def ratio_heuristic_t(game):
        global target_sum
        sum_ = 0
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_ += game.board[i][j]
                sum_2 += game.board[i][j] ** 2
        return max(target_sum_2//sum_2, (target_sum - sum_) // 2)


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


    def adjacency_diff_heuristic(game):
        row_comp, col_comp = compress(game)

        adj = 0

        for i in range(game.dimension):
            for j in range(len(row_comp[i]) - 1):
                adj += abs(row_comp[i][j] - row_comp[i][j + 1])
            for j in range(len(col_comp[i]) - 1):
                adj += abs(col_comp[i][j] - col_comp[i][j + 1])

        return adj


    def adjacency_ldiff_heuristic(game):
        row_comp, col_comp = compress(game)

        adj = 0

        for i in range(game.dimension):
            for j in range(len(row_comp[i]) - 1):
                adj += abs(lg_d[row_comp[i][j]] - lg_d[row_comp[i][j + 1]])
            for j in range(len(col_comp[i]) - 1):
                adj += abs(lg_d[col_comp[i][j]] - lg_d[col_comp[i][j + 1]])

        return adj


    def adjacency_min_ldiff_heuristic(game):
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        adjs = [[[] for i in range(game.dimension)] for j in range(game.dimension)]


        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    ltile = lg_d[tile]

                    if len(row_comp[i]) > 0:
                        ra = row_comp[i][-1]
                        a = abs(lg_d[ra[0]] - ltile)
                        adjs[i][j].append(a)
                        adjs[i][ra[2]].append(a)

                    if len(col_comp[i]) > 0:
                        ca = col_comp[i][-1]
                        a = abs(lg_d[ca[0]] - ltile)
                        adjs[i][j].append(a)
                        adjs[ca[1]][j].append(a)

                    row_comp[i].append((tile, i, j))
                    col_comp[j].append((tile, i, j))


        adj = 0

        for i in range(game.dimension):
            for j in range(game.dimension):
                if len(adjs[i][j]) > 0: adj += min(adjs[i][j])

        return adj


    def adjacency_min1_ldiff_heuristic(game):
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        adjs = [[[] for i in range(game.dimension)] for j in range(game.dimension)]


        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    ltile = lg_d[tile]

                    if len(row_comp[i]) > 0:
                        ra = row_comp[i][-1]
                        a = abs(lg_d[ra[0]] - ltile)
                        adjs[i][j].append(a)
                        adjs[i][ra[2]].append(a)

                    if len(col_comp[i]) > 0:
                        ca = col_comp[i][-1]
                        a = abs(lg_d[ca[0]] - ltile)
                        adjs[i][j].append(a)
                        adjs[ca[1]][j].append(a)

                    row_comp[i].append((tile, i, j))
                    col_comp[j].append((tile, i, j))


        adj = 0

        for i in range(game.dimension):
            for j in range(game.dimension):
                if len(adjs[i][j]) > 0:
                    min_a = min(adjs[i][j])
                    if min_a > 1:
                        adj += min_a

        return adj


    def edge_sum(game):
        row_comp, col_comp = compress(game)

        sum_ = 0

        for i in range(game.dimension):
            if len(row_comp[i]) > 2:
                sum_ += row_comp[i][0] + row_comp[i][-1]
            elif len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]

            if len(col_comp[i]) > 2:
                sum_ += col_comp[i][0] + col_comp[i][-1]
            elif len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]

        return sum_


    def edge_sum2(game):
        row_comp, col_comp = compress(game)

        sum_ = 0

        for i in range(game.dimension):
            if len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]**2 + row_comp[i][-1]**2
            elif len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]**2

            if len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]**2 + col_comp[i][-1]**2
            elif len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]**2

        return sum_


    def true_edge_sum(game):
        sum_ = 0

        for i in range(game.dimension):
            sum_ += game.board[i][0] + game.board[i][-1]
            sum_ += game.board[0][i] + game.board[-1][i]

        return sum_


    def true_edge_sum2(game):
        row_comp, col_comp = compress(game)

        sum_ = 0

        for i in range(game.dimension):
            sum_ += game.board[i][0]**2 + game.board[i][-1]**2
            sum_ += game.board[0][i]**2 + game.board[-1][i]**2

        return sum_

    def sum2_diff_heuristic_e(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    #ltile = lg_d[tile]
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)
                sum_2 += tile ** 2

        sum_ = 0

        for i in range(game.dimension):
            if len(row_comp[i]) > 2:
                sum_ += row_comp[i][0] + row_comp[i][-1]
            elif len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]

            if len(col_comp[i]) > 2:
                sum_ += col_comp[i][0] + col_comp[i][-1]
            elif len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]

        return max((target_sum_2 - sum_2) - sum_, 0)

    def sum2_diff_heuristic_e2(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    #ltile = lg_d[tile]
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)
                sum_2 += tile ** 2

        sum_ = 0

        for i in range(game.dimension):
            if len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]**2 + row_comp[i][-1]**2
            elif len(row_comp[i]) > 2:
                sum_ += row_comp[i][0]**2

            if len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]**2 + col_comp[i][-1]**2
            elif len(col_comp[i]) > 2:
                sum_ += col_comp[i][0]**2

        return max((target_sum_2 - sum_2) - sum_, 0)

    def sum2_diff_heuristic_te(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    #ltile = lg_d[tile]
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)
                sum_2 += tile ** 2

        sum_ = 0

        for i in range(game.dimension):
            sum_ += game.board[i][0] + game.board[i][-1]
            sum_ += game.board[0][i] + game.board[-1][i]

        return max((target_sum_2 - sum_2) - sum_, 0)

    def sum2_diff_heuristic_te2(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        row_comp = [[] for i in range(game.dimension)]
        col_comp = [[] for i in range(game.dimension)]

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    #ltile = lg_d[tile]
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)
                sum_2 += tile ** 2

        sum_ = 0

        for i in range(game.dimension):
            sum_ += game.board[i][0]**2 + game.board[i][-1]**2
            sum_ += game.board[0][i]**2 + game.board[-1][i]**2

        return max((target_sum_2 - sum_2) - sum_, 0)


    def tile_stats(game: Game) -> tuple:
        sum_ = 0
        sum_2 = 0
        sum_h = 0
        counts = [0] * (lgg + 1)
        rows_cnts = [[0] * lgg for i in range (game.dimension)]
        cols_cnts = [[0] * lgg for i in range (game.dimension)]

        row_comp = [[] for i in range (game.dimension)]
        col_comp = [[] for i in range (game.dimension)]

        adj_cnts = [0] * (lgg + 1)
        adj_cnts_i = [0] * (lgg + 1)
        adj_cnts_j = [0] * (lgg + 1)
        adj_cnts_cr = [0] * (lgg + 1)
        adj_cnts_cc = [0] * (lgg + 1)

        # nadj_cnts = [0] * 32
        # nadj_cnts_c = [0] * 32

        for i in range(game.dimension):
            for j in range(game.dimension):
                tile = game.board[i][j]
                if tile > 0:
                    # tile_2 = tile*2
                    # tile_h = tile//2
                    sum_ += tile
                    sum_2 += tile * tile
                    sum_h += rt_d[tile]
                    ltile = lg_d[tile]
                    counts[ltile] += 1
                    rows_cnts[i][ltile] += 1
                    cols_cnts[j][ltile] += 1
                    row_comp[i].append(tile)
                    col_comp[j].append(tile)

                    if i > 0:
                        if game.board[i - 1][j] == tile:
                            adj_cnts_i[ltile] += 1
                    if i < game.dimension - 1:
                        if game.board[i + 1][j] == tile:
                            adj_cnts_i[ltile] += 1
                    if j > 0:
                        if game.board[i][j - 1] == tile:
                            adj_cnts_j[ltile] += 1
                    if j < game.dimension - 1:
                        if game.board[i][j + 1] == tile:
                            adj_cnts_j[ltile] += 1
        for i in range(game.dimension):
            for j in range(len(row_comp[i]) - 1):
                if row_comp[i][j] == row_comp[i][j + 1]:
                    adj_cnts_cr[row_comp[i][j]] += 1
            for j in range(len(col_comp[i]) - 1):
                if col_comp[i][j] == col_comp[i][j + 1]:
                    adj_cnts_cc[col_comp[i][j]] += 1

        for i in range(lgg):
            adj_cnts[i] = adj_cnts_cr[i] + adj_cnts_cc[i]

        return sum_, sum_2, sum_h, counts, rows_cnts, cols_cnts, row_comp, col_comp, adj_cnts

    def weighted_heuristic(game: Game) -> float:
        global target_sum
        global target_sum_2
        global stats_w
        sum_ = 0
        sum_2 = 0
        count = 0
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
        cut_ = (target_sum - sum_)
        # 0 max(cut_, 0) max(target_sum_2 - sum_2, 0)
        wsum_ = stats_w[0] * cut_ + stats_w[1] * (cut_ ** 2) + \
                stats_w[2] * (target_sum_2 - sum_2) + stats_w[3] * ((target_sum_2 - sum_2) ** 2) + \
                stats_w[4] * (target_sum_2 // sum_2) + stats_w[5] * count

        # sum min adjacency diff

        # sum total expected merges

        # tiles over needed

        #if abs(stats_w[0]) > 1e-5:
        #    wsum_ += stats_w[0] * cut_
        #if abs(stats_w[1]) > 1e-5:
        #    wsum_ += stats_w[1] * (cut_ ** 2)
        #if abs(stats_w[2]) > 1e-5:
        #    wsum_ += stats_w[2] * (target_sum_2 - sum_2)
        #if abs(stats_w[3]) > 1e-5:
        #    wsum_ += stats_w[3] * ((target_sum_2 - sum_2) ** 2)
        #if abs(stats_w[4]) > 1e-5:
        #    wsum_ += stats_w[4] * (target_sum_2 // sum_2)
        #if abs(stats_w[5]) > 1e-5:
        #    wsum_ += stats_w[5] * count
        if abs(stats_w[6]) > 1e-3:
            wsum_ += stats_w[6] * min_remaining_perfect_merges(game, counts)
        if abs(stats_w[7]) > 1e-3:
            wsum_ += stats_w[7] * min_remaining_perfect_merges_d(game, counts, cut_ // 2)
        if abs(stats_w[8]) > 1e-3:
            wsum_ += stats_w[8] * adjacency_ldiff_heuristic(game)
        if abs(stats_w[9]) > 1e-3:
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
            wsum_ += stats_w[9] * sum_e
        return wsum_

    num_stats = 10




    def diff_tile_count_heuristic(game: Game):
        sum_t = 0
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                if game.board[i][j] > 0:
                    sum_t += 1
                    sum_ += game.board[i][j]
        return sum_t - ideals_tc[sum_]


    def merge_count_heuristic(game: Game):
        global target_sum
        sum_t = 0
        sum_ = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                if game.board[i][j] > 0:
                    sum_t += 1
                    sum_ += game.board[i][j]
        return (sum_ - target_sum)/2 + sum_t - 1








    def modified_find_adj_distance(game, tile, count):
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

    # without num good?
    def modified_min_remaining_perfect_merges(game: Game, counts: list = None) -> int:
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

    # num bad?
    def modified_min_remaining_perfect_merges_d(game: Game, counts: list = None, cut_=-1) -> int:
        global lgg

        check = lgg

        if counts is None:
            _, counts = tile_stats_basic(game)

        if counts[check] > 0:
            return 0
        min_ = 1
        check -= 1
        num_ = 2
        while check > 1 and counts[check] < num_:#2
            min_ += 1
            num_ -= counts[check]#
            num_ *= 2#
            check -= 1

        if check == 1:
            min_ += max(num_ - counts[check], 0)

        #penalty = counts[check] - 2

        if min_ + 2 < cut_:
            return cut_#+penalty
        return max(min_ + find_adj_distance(game, pw_d[check], counts[check]), cut_)#+penalty


    def modified_min_remaining_perfect_merges_dc(game: Game, counts: list = None, cut_=-1) -> int:
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


    # -edge was bad
    # sdc diff better?
    def modified_sum2_diff_heuristic(game: Game) -> int:
        global target_sum_2
        sum_2 = 0
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                sum_2 += game.board[i][j] ** 2
        return max(target_sum_2 - sum_2, 0)



    path = None

    # path = a_star(game, numberOfTiles_heuristic, goal, open) # 20 1.9442737102508545

    # game = Game(5, 32)
    # path = a_star(game, numberOfTiles_heuristic, goal, open)  # A: 17
    # .01
    # path = a_star(game, perfect_moves_heuristic_x2, goal, open)  # 0.005237579345703125

    game = Game(4, 64)
    lgg = lg_d[game.goal]
    sdc = game.goal + (lgg - 1) # wrong
    target_sum = game.goal + (lgg - 2) * 2
    goal_2 = game.goal ** 2
    target_sum_2 = goal_2 + ideals_s2[lgg - 2]

    start = time.time()
    #
    # game = Game(4, 256) (newer2)
    # path = a_star(game, min_perfect_heuristic_dc, goal, open)  #
    # path = a_star(game, sum2_diff_heuristic_e, goal, open)  #170 0.041991472244262695
    # path = a_star(game, sum2_diff_heuristic_e2, goal, open)  #175 0.297802209854126
    # path = a_star(game, sum2_diff_heuristic_te, goal, open)  #317 0.09587335586547852
    # path = a_star(game, sum2_diff_heuristic_te2, goal, open)  #198 5.878733158111572
    # path = a_star(game, near_perfect_heuristic_dc, goal, open)  #217 0.05271625518798828

    # game = Game(4, 64) (newer2)
    # path = a_star(game, min_perfect_heuristic_dc, goal, open)  #37 0.15231752395629883 0.12286758422851562
    # game = Game(4, 128) (newer2)
    # path = a_star(game, min_perfect_heuristic_dc, goal, open)  #70 2.6523494720458984 2.316368341445923 //89 2.2958924770355225?


    # game = Game(4, 2048) (newer)
    # path = a_star(game, goal_diff_heuristic, goal, open)  #...
    # path = a_star(game, goal_diff_heuristic_2, goal, open)  #...
    # path = a_star(game, sum_diff_heuristic, goal, open)  #...
    # path = a_star(game, sum_diff_heuristic_2, goal, open)  #...
    # path = a_star(game, sum_diff_heuristic_x2, goal, open)  #...
    # path = a_star(game, goal2_diff_heuristic, goal, open)  #1076 0.24562883377075195
    # path = a_star(game, goal2_diff_heuristic_2, goal, open)  #1076 0.32156825065612793
    # path = a_star(game, sum2_diff_heuristic, goal, open)  #1076 0.24927306175231934
    # path = a_star(game, sum2_diff_heuristic_2, goal, open)  #1076 0.24576592445373535
    # path = a_star(game, min_perfect_heuristic, goal, open)  #
    # path = a_star(game, min_perfect_heuristic_d, goal, open)  #
    # path = a_star(game, ratio_heuristic_g, goal, open)  #...
    # path = a_star(game, ratio_heuristic_t, goal, open)  #...
    # path = a_star(game, adjacency_diff_heuristic, goal, open)  #...
    # path = a_star(game, adjacency_ldiff_heuristic, goal, open)  #...
    # path = a_star(game, edge_sum, goal, open)  #...?
    # path = a_star(game, edge_sum2, goal, open)  #...?
    # path = a_star(game, sum2_diff_heuristic_e, goal, open)  #1329 1.165055751800537
    # path = a_star(game, near_perfect_heuristic, goal, open)  #1076 0.3258171081542969
    # path = a_star(game, near_perfect_heuristic_g, goal, open)  #1076 0.36853480339050293
    # path = a_star(game, near_perfect_heuristic_d, goal, open)  #1076 0.2709364891052246
    # path = a_star(game, near_perfect_heuristic_d_g, goal, open)  #1076 0.25841617584228516

    # path = a_star(game, perfect_moves_heuristic_arch, goal, open)  #1038 2.0726425647735596
    # path = a_star(game, min_moves_heuristic_arch, goal, open)  #1038 1.4208390712738037
    # path = a_star(game, min_moves_heuristic_meta_arch, goal, open)  #1037 6.770117521286011


    # game = Game(4, 2048) // Was with *2 on sum dif Old
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # #//1038 1.990800380706787 1.9348342418670654 1.9320755004882812 1.9480178356170654
    # path = a_star(game, min_moves_heuristic, goal, open)  # 1038 1.3966424465179443 1.3767163753509521 1.3767163753509521 1.3767163753509521
    # path = a_star(game, min_moves_heuristic_meta, goal, open)  # 1037 6.478724002838135 6.49899697303772 //?2.384185791015625e-07

    # game = Game(4, 32) (newer2)
    # path = a_star(game, diff_tile_count_heuristic, goal, open)  #20 8.357316255569458
    # path = a_star(game, merge_count_heuristic, goal, open)  #20 15.879167556762695
    # path = a_star(game, min_perfect_heuristic_dc, goal, open)  #20 0.01876378059387207
    # path = a_star(game, adjacency_min_ldiff_heuristic, goal, open)  #20 7.629758358001709
    # path = a_star(game, adjacency_min1_ldiff_heuristic, goal, open)  #20 27.745695114135742
    # path = a_star(game, sum2_diff_heuristic_e, goal, open)  #20 0.008059263229370117
    # path = a_star(game, sum2_diff_heuristic_e2, goal, open)  #20 0.046025753021240234
    # path = a_star(game, sum2_diff_heuristic_te, goal, open)  #20 0.012686491012573242
    # path = a_star(game, sum2_diff_heuristic_te2, goal, open)  #20 0.033293962478637695
    # path = a_star(game, near_perfect_heuristic_dc, goal, open)  #20 0.0075664520263671875
    
    # game = Game(4, 32) (newer)
    # path = a_star(game, numberOfTiles_heuristic, goal, open)  #20 6.985862731933594 6.31069803237915
    # path = a_star(game, goal_diff_heuristic, goal, open)  #20 32.8656051158905 //61 0.013225317001342773
    # path = a_star(game, goal_diff_heuristic_2, goal, open)  #20 1.0471439361572266 0.9942343235015869
    # path = a_star(game, sum_diff_heuristic, goal, open)  #20 32.99239540100098 //61 0.01377105712890625
    # path = a_star(game, sum_diff_heuristic_2, goal, open)  #20 0.027636289596557617 0.02658867835998535
    # path = a_star(game, sum_diff_heuristic_x2, goal, open)  #21 0.039237260818481445 //61 0.01254415512084961
    # path = a_star(game, goal2_diff_heuristic, goal, open)  #20 0.008071660995483398 //20 0.004005908966064453
    # path = a_star(game, goal2_diff_heuristic_2, goal, open)  #20 0.004817962646484375
    # path = a_star(game, sum2_diff_heuristic, goal, open)  #20 0.003881692886352539 //20 0.00497126579284668 # Best test 2
    # path = a_star(game, sum2_diff_heuristic_2, goal, open)  #20 0.003829479217529297 0.004212141036987305
    # path = a_star(game, min_perfect_heuristic, goal, open)  #20 0.0076389312744140625 //0.0468449592590332
    # path = a_star(game, min_perfect_heuristic_d, goal, open)  #20 0.005569934844970703
    # path = a_star(game, ratio_heuristic_g, goal, open)  #20 1.0053856372833252
    # path = a_star(game, ratio_heuristic_t, goal, open)  #20 0.6583983898162842
    # path = a_star(game, adjacency_diff_heuristic, goal, open)  #20 ...bad
    # path = a_star(game, adjacency_ldiff_heuristic, goal, open)  #20 2.3737778663635254
    # path = a_star(game, edge_sum, goal, open)  #21 4.716487407684326
    # path = a_star(game, edge_sum2, goal, open)  #21 5.264557361602783
    # path = a_star(game, sum2_diff_heuristic_e, goal, open)  #20 0.004191398620605469 //20 0.007627248764038086
    # path = a_star(game, near_perfect_heuristic, goal, open)  #20 0.004109382629394531
    # path = a_star(game, near_perfect_heuristic_g, goal, open)  #20 0.004243373870849609
    # path = a_star(game, near_perfect_heuristic_d, goal, open)  #20 0.0038378238677978516
    # path = a_star(game, near_perfect_heuristic_d_g, goal, open)  #20 0.005160093307495117

    # game = Game(4, 64)
    # path = a_star(game, numberOfTiles_heuristic, goal, open)  # 37 1278.5932025909424 1272.3251564502716

    delta_t = time.time() - start
    if path is not None:
        print(path)
        print(len(path))
        print(delta_t)

        # print(game)

    do_min = False
    do_perfect = True
    do_test = path is None and not do_min
    test_game = (4, 2048)
    test_runs = 10
    test_groups = 10
    test_list = [  # numberOfTiles_heuristic,
        #goal_diff_heuristic, #/goal_diff_heuristic_2,
        #sum_diff_heuristic, sum_diff_heuristic_2, sum_diff_heuristic_x2,
        goal2_diff_heuristic, goal2_diff_heuristic_2,
        sum2_diff_heuristic, sum2_diff_heuristic_2,
        #min_perfect_heuristic, min_perfect_heuristic_d, min_perfect_heuristic_dc,
        #ratio_heuristic_t, adjacency_diff_heuristic,
        #adjacency_ldiff_heuristic,
        #adjacency_min_ldiff_heuristic, adjacency_min1_ldiff_heuristic, ######
        sum2_diff_heuristic_e, sum2_diff_heuristic_e2,
        sum2_diff_heuristic_te, #sum2_diff_heuristic_te2,
        near_perfect_heuristic, near_perfect_heuristic_g,
        near_perfect_heuristic_d, near_perfect_heuristic_d_g,
        near_perfect_heuristic_dc,
        perfect_moves_heuristic_arch,
        min_moves_heuristic_arch, min_moves_heuristic_meta_arch
    ]


    if do_test:
        print(f"Doing test... {test_game}")
        import statistics

        game = Game(*test_game)
        lgg = lg_d[game.goal]
        target_sum = game.goal + (lgg - 2) * 2
        goal_2 = game.goal ** 2
        target_sum_2 = goal_2 + ideals_s2[lgg - 2]
        results_dict = {}
        for f in test_list:
            results_dict[f.__name__] = [[], []]
        for g in range(test_groups):
            print(f"group {g + 1} / {test_groups}")
            for f in test_list:
                print(f"function {f.__name__}")
                results_list = results_dict[f.__name__]
                for _ in itertools.repeat(None, test_runs):
                    start = time.time()
                    path = a_star(game, f, goal, open)
                    delta_t = time.time() - start

                    results_list[0].append(len(path))
                    results_list[1].append(delta_t)

        sorted = []
        for k in results_dict.keys():
            results_list = results_dict[k]
            sorted.append((statistics.median(results_list[1]), sum(results_list[0]) / len(results_list[0]), k))

        sorted.sort()

        print(f"On {test_game[0]}x{test_game[0]} board with goal of {test_game[1]}")
        for t in sorted:
            print(t[2] + f": path length {t[1]} in {t[0]}")

        if do_perfect:
            print("\nmin_perfect_heuristic: ")
            start = time.time()
            path = a_star(game, min_perfect_heuristic, goal, open)
            delta_t = time.time() - start
            print(path)
            print(len(path))
            print(delta_t)

            print("\nmin_perfect_heuristic_d: ")
            start = time.time()
            path = a_star(game, min_perfect_heuristic_d, goal, open)
            delta_t = time.time() - start
            print(path)
            print(len(path))
            print(delta_t)

    if do_min:
        print("Doing min...")
        import numpy as np
        from scipy.optimize import minimize

        game = Game(4, 32)
        lgg = lg_d[game.goal]
        target_sum = game.goal + (lgg - 2) * 2
        goal_2 = game.goal ** 2
        target_sum_2 = goal_2 + ideals_s2[lgg - 2]

        f = weighted_heuristic

        def f_t(x):
            global stats_w
            stats_w = x
            start = time.time()
            path = a_star(game, f, goal, open)
            delta_t = time.time() - start
            return delta_t

        def f_d(x):
            global stats_w
            stats_w = x
            start = time.time()
            path = a_star(game, f, goal, open)
            delta_t = time.time() - start
            return len(path)

        def f_h(x):
            global stats_w
            stats_w = x
            start = time.time()
            path = a_star(game, f, goal, open)
            delta_t = time.time() - start
            return len(path)*delta_t

        min_method = 'nelder-mead'
        # min_method = 'BFGS'

        x0 = np.array([1]*num_stats)
        res = minimize(f_t, x0, method=min_method, options = {'xtol': 1e-4, 'disp': False})
        print(f"time minimization {res}")
        print()

        #x0 = np.array([1]*num_stats)
        #res = minimize(f_d, x0, method=min_method, options = {'xtol': 1e-4, 'disp': False})
        #print(f"path minimization {res}")
        #print()

        x0 = np.array([1]*num_stats)
        res = minimize(f_h, x0, method=min_method, options = {'xtol': 1e-4, 'disp': False})
        print(f"hybrid minimization {res}")
        print()

        min_method = 'BFGS'

        x0 = np.array([1]*num_stats)
        res = minimize(f_t, x0, method=min_method, options = {'disp': False})
        print(f"time minimization {res}")
        print()

        #x0 = np.array([1]*num_stats)
        #res = minimize(f_d, x0, method=min_method, options = {'disp': False})
        #print(f"path minimization {res}")
        #print()

        x0 = np.array([1]*num_stats)
        res = minimize(f_h, x0, method=min_method, options = {'disp': False})
        print(f"hybrid minimization {res}")
        print()




    # Broken
    # path = a_star(game, perfect_moves_heuristic_x2, goal, open)  # 37 0.01250910758972168
    # path = a_star(game, min_moves_heuristic_x2, goal, open)  # 37 0.024863719940185547

    # game = Game(4, 32)
    # path = a_star(game, numberOfTiles_heuristic, goal, open)  # 20 6.00048303604126 6.003154993057251
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # 20 12.098444938659668 11.02060604095459
    # path = a_star(game, perfect_moves_heuristic_x2, goal, open)  # 20 0.00446009635925293 0.004324436187744141 0.004242897033691406 0.0046117305755615234
    # path = a_star(game, max_union_meta_heuristic, goal, open)  # 20 6.331159830093384 6.328663349151611
    # path = a_star(game, max_union_meta_heuristic_x2, goal, open)  # 20 0.0882420539855957 0.08676409721374512
    # path = a_star(game, min_moves_heuristic, goal, open)  # 20 11.976464986801147 12.508132219314575
    # path = a_star(game, min_moves_heuristic_x2, goal, open)  # 20 0.0061643123626708984 0.005979299545288086 0.006215572357177734  0.011497020721435547 0.006448030471801758
    # path = a_star(game, min_moves_heuristic_meta, goal, open)  # 20 8.367883682250977 8.16808557510376
    # path = a_star(game, min_moves_heuristic_meta_x2, goal, open)  # 21 0.03540658950805664 0.036478281021118164
    # path = a_star(game, min_moves_heuristic_meta2, goal, open)  # 20 21.07951021194458

    # path to file to diff

    # game = Game(4, 64)
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # 37 0.014313936233520508 0.013176202774047852 0.01551198959350586
    # path = a_star(game, max_union_meta_heuristic, goal, open)  # 37 0.17189908027648926
    # path = a_star(game, min_moves_heuristic, goal, open)  # 37 0.023751497268676758 0.014847517013549805 0.014628887176513672
    # path = a_star(game, min_moves_heuristic_meta, goal, open)  # 38 0.4536604881286621

    # path = a_star(game, numberOfTiles_heuristic, goal, open) # 37 1278.5932025909424

    # game = Game(3, 32)
    # path = a_star(game, trivial_heuristic, goal, open) # 20 4.363207101821899
    # path = a_star(game, numberOfTiles_heuristic, goal, open) # 20 1.9442737102508545
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # 20 0.004635810852050781 0.005115985870361328
    # path = a_star(game, max_union_meta_heuristic, goal, open)  # 20 0.09067487716674805
    # path = a_star(game, min_moves_heuristic, goal, open)  # 20 0.004836559295654297

'''

    def find_distance(game, tile, count):
        found = 1
        min = game.dimension
        for i in range(game.dimension):
            for j in range(game.dimension):
                if game.board[i][j] != tile:
                    continue
                if found == count:
                    return min
                found += 1
                if i > 0:
                    if game.board[i-1][j] == tile or game.board[i-1][j] == 0 and i > 1 and game.board[i-2][j] == tile:
                        return 1
                if i < game.dimension-1:
                    if game.board[i+1][j] == tile:
                        return 1
                if j > 0:
                    if game.board[i][j-1] == tile:
                        return 1
                if j < game.dimension-1:
                    if game.board[i][j+1] == tile:
                        return 1
                startj = j+1
                for i2 in range(i, game.dimension):
                    for j2 in range(startj, game.dimension):
                        if game.board[i][j] != tile:
                            continue
                        di = abs(i - i2)
                        dj = abs(j - j2)
                        
                    startj = 1
                        
                
                    
        pass


    def min_remaining_approx_merges_g_r(game: Game, sum_2) -> float:
        r = (goal_2 % sum_2)
        r_ = r/(goal_2-r)
        return lg_d[goal_2//sum_2] + r_*(1-r_/2)


    def min_remaining_approx_merges_t_r(game: Game, sum_2) -> float:
        r = (target_sum_2 % sum_2)
        r_ = r/(goal_2-r)
        return lg_d[target_sum_2//sum_2] + r_*(1-r_/2)
        
        
        

        return stats_w[0] * cut_ + stats_w[1] * (cut_ ** 2) + \
               stats_w[2] * (target_sum_2 - sum_2) + stats_w[3] * ((target_sum_2 - sum_2) ** 2) + \
               stats_w[4] * min_remaining_perfect_merges(game, counts) + \
               stats_w[5] * min_remaining_perfect_merges_d(game, counts, cut_ // 2) + \
               stats_w[6] * (target_sum_2 // sum_2) + stats_w[7] * count
'''
