class Game:
    dimension:int
    goal:int
    def __init__(self, dimension, goal):
        self.board = []
        for i in range(dimension):
            self.board.append([0] * dimension)
        self.goal = goal
        self.dimension = dimension
        self.board[0][self.dimension-1] = 2

    def __str__(self):
        s = ''
        for i in range(len(self.board)):
            s += str(self.board[i]) + "\n"
        return s

    def __eq__(self, other):
        return self.board == other.board

    def __lt__(self, other):
        return 1

    def __repr__(self):
        return "Game <%s>" % repr(self.board)

    def __hash__(self):
        return hash(str(self.board))

    #permited values: n, e, s, w
    def play(self, move):
        new_game = Game(self.dimension, self.goal)
        new_game.board = [[v for v in l] for l in self.board]
        new_game.move(move)
        new_game.locate_new_box(move)

        return new_game

    def move(self, move):
        if move == 'n' or move == 's':
            for c in range(len(self.board)):
                column = self.get_column(c)
                column = self.remove_zeros(column)
                column = self.sum_equals(column, move == 's')
                self.fill_column(c, column, move == 's')
        elif move == 'w' or move == 'e':
            for r in range(len(self.board)):
                row = self.get_row(r)
                row = self.remove_zeros(row)
                row = self.sum_equals(row, move == 'e')
                self.fill_row(r, row, move == 'e')
        else:
            raise ValueError

    def locate_new_box(self, move):
        if move == 'n':
            for r in range(len(self.board) - 1, -1, -1):
                for c in range(len(self.board) - 1, -1, -1):
                    if self.board[r][c] == 0:
                        self.board[r][c] = 2
                        return True
        elif move == 'e':
            for c in range(len(self.board)):
                for r in range(len(self.board) - 1, -1, -1):
                    if self.board[r][c] == 0:
                        self.board[r][c] = 2
                        return True
        elif move == 's':
            for r in range(len(self.board)):
                for c in range(len(self.board)):
                    if self.board[r][c] == 0:
                        self.board[r][c] = 2
                        return True
        elif move == 'w':
            for c in range(len(self.board) - 1, -1, -1):
                for r in range(len(self.board)):
                    if self.board[r][c] == 0:
                        self.board[r][c] = 2
                        return True
        else:
            raise ValueError
        return False

    def is_game_over(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] == 0:
                    return False
                if i > 0 and self.board[i-1][j] == self.board[i][j]:
                    return False
                if j > 0 and self.board[i][j-1] == self.board[i][j]:
                    return False
        return True

    def is_game_finished(self):
        for i in range(len(self.board)):
            for j in range(len(self.board)):
                if self.board[i][j] >= self.goal:
                    return True
        return False

    def get_column(self, c):
        column = []
        for i in range(len(self.board)):
            column.append(self.board[i][c])
        return column

    def get_row(self, r):
        row = []
        for i in range(len(self.board)):
            row.append(self.board[r][i])
        return row

    def fill_column(self, column_index, column, reverse=False):
        for row in range(len(self.board)):
            self.board[row if not reverse else -1 - row][column_index] = \
                column[row if not reverse else -1 - row] if row < len(column) else 0

    def fill_row(self, row_index, row, reverse=False):
        for column in range(len(self.board)):
            self.board[row_index][column if not reverse else -1 - column] = \
                row[column if not reverse else -1 - column] if column < len(row) else 0

    def remove_zeros(self, l):
        _l = []
        for i in range(len(l)):
            if l[i] > 0:
                _l.append(l[i])
        return _l

    def sum_equals(self, l, reverse):
        if not l:
            return []

        if reverse:
            _l = [l[-1]]
            i = -2
            while i >= -len(l):
                if l[i] == l[i+1]:
                    _l[-1] *= 2
                    i -= 1
                    if i >= -len(l):
                        _l.append(l[i])
                else:
                    _l.append(l[i])
                i -= 1
            _l.reverse()
            return _l
        else:
            _l = [l[0]]
            i = 1
            while i < len(l):
                if l[i] == l[i - 1]:
                    _l[-1] *= 2
                    i += 1
                    if i < len(l):
                        _l.append(l[i])
                else:
                    _l.append(l[i])
                i += 1
            return _l


if __name__ == '__main__':
    game = Game(4, 2048)

    while not game.is_game_over() and not game.is_game_finished():
        print(game)
        move = input()
        while move != 'e' and move != 's' and move != 'w'  and move != 'n':
            move = input()
        game = game.play(move)

    print(game)
    print(game.is_game_finished())