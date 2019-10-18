import heapq


def a_star(start, h, goal, open):
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


def make_path(node, parent):
    path = [node]

    while node in parent:
        node = parent[node]
        path.insert(0, node)

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


    def lg(num):
        return lg_d[num]


    def tile_stats(game: Game) -> tuple:
        sumOfTiles = 0
        counts = [0] * 32
        for i in range(len(game.board)):
            for j in range(len(game.board)):
                tile = game.board[i][j]
                if tile > 0:
                    sumOfTiles += tile
                    counts[lg(tile)] += 1
        return sumOfTiles, counts


    def max_union_meta_heuristic(game):
        return max(numberOfTiles_heuristic(game), perfect_moves_heuristic(game))


    def max_union_meta_heuristic_x2(game):
        return max(numberOfTiles_heuristic(game), perfect_moves_heuristic_x2(game))


    def perfect_remaining_merges(game: Game, counts: list = None) -> int:
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

        # if 2 ...

        return min_


    def perfect_moves_heuristic(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        return max(perfect_remaining_merges(game, counts), (sdc - sum_)/2)


    def perfect_moves_heuristic_x2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        return max(perfect_remaining_merges(game, counts), sdc - sum_)


    def i_range(a: int, b: int):
        if a < b:
            return a + 1, b - 1
        else:
            return b + 1, a - 1


    def find_distance(game, tile, count):
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
        return max(min_+find_distance(game, pw_d[check], counts[check]), cut_)


    def min_moves_heuristic(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_)/2
        return min_remaining_merges(game, counts, cut_)


    def min_moves_heuristic_x2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges(game, counts, cut_)


    def min_remaining_merges_meta(game: Game, counts: list = None, cut_=-1) -> int:
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
        return max(min_+find_distance(game, pw_d[check], counts[check]), cut_)+penalty


    def min_moves_heuristic_meta(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_)/2
        return min_remaining_merges_meta(game, counts, cut_)


    def min_moves_heuristic_meta_x2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = sdc - sum_
        return min_remaining_merges_meta(game, counts, cut_)


    def min_moves_heuristic_meta2(game: Game) -> int:
        global sdc
        sum_, counts = tile_stats(game)
        #if sdc is None:
        #    sdc = game.goal + (lg(game.goal) - 1)
        cut_ = (sdc - sum_)/2
        penalty = 0
        if game.goal < sum_:
            penalty = sum_ - game.goal
        return min_remaining_merges_meta(game, counts, cut_)+penalty


    # path = a_star(game, numberOfTiles_heuristic, goal, open) # 20 1.9442737102508545

    game = Game(4, 32)
    lgg = lg(game.goal)
    sdc = game.goal + (lgg - 1)

    start = time.time()
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

    delta_t = time.time() - start


    # game = Game(4, 2048) // Was with *2 on sum dif
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # #//1038 1.990800380706787 1.9348342418670654 1.9320755004882812 1.9480178356170654
    # path = a_star(game, min_moves_heuristic, goal, open)  # 1038 1.3966424465179443 1.3767163753509521 1.3767163753509521 1.3767163753509521
    # path = a_star(game, min_moves_heuristic_meta, goal, open)  # 1037 6.478724002838135 6.49899697303772 //?2.384185791015625e-07

    # path to file to diff


    # game = Game(4, 64)
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # 37 0.014313936233520508 0.013176202774047852 0.01551198959350586
    # path = a_star(game, max_union_meta_heuristic, goal, open)  # 37 0.17189908027648926
    # path = a_star(game, min_moves_heuristic, goal, open)  # 37 0.023751497268676758 0.014847517013549805 0.014628887176513672
    # path = a_star(game, min_moves_heuristic_meta, goal, open)  # 38 0.4536604881286621



    #path = a_star(game, numberOfTiles_heuristic, goal, open) # 37 1278.5932025909424



    # game = Game(3, 32)
    # path = a_star(game, trivial_heuristic, goal, open) # 20 4.363207101821899
    # path = a_star(game, numberOfTiles_heuristic, goal, open) # 20 1.9442737102508545
    # path = a_star(game, perfect_moves_heuristic, goal, open)  # 20 0.004635810852050781 0.005115985870361328
    # path = a_star(game, max_union_meta_heuristic, goal, open)  # 20 0.09067487716674805
    # path = a_star(game, min_moves_heuristic, goal, open)  # 20 0.004836559295654297



    print(path)
    print(len(path))
    print(delta_t)


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
'''
