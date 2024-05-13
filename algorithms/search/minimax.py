def print_board(board):
    for row in board:
        print(f'{row[0]} | {row[1]} | {row[2]}')
        print('-' * 10)

def is_winner(board, player):
    win_conditions = [
        [board[0][0], board[0][1], board[0][2]],
        [board[1][0], board[1][1], board[1][2]],
        [board[2][0], board[2][1], board[2][2]],
        [board[0][0], board[1][0], board[2][0]],
        [board[0][1], board[1][1], board[2][1]],
        [board[0][2], board[1][2], board[2][2]],
        [board[0][0], board[1][1], board[2][2]],
        [board[0][2], board[1][1], board[2][0]]
    ]

    return [player, player, player] in win_conditions

def get_empty_positions(board):
    positions = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                positions.append((i, j))
    return positions

def minimax(board, depth, is_max, alpha, beta):
    player = "X" if is_max else "O"
    opponent = "O" if is_max else "X"

    if is_winner(board, opponent):
        return -1 if is_max else 1

    if not get_empty_positions(board):
        return 0

    if is_max:
        best = -float('inf')
        for (i, j) in get_empty_positions(board):
            board[i][j] = player
            value = minimax(board, depth + 1, False, alpha, beta)
            board[i][j] = ' '
            best = max(best, value)
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best
    else:
        best = float('inf')
        for (i, j) in get_empty_positions(board):
            board[i][j] = opponent
            value = minimax(board, depth + 1, True, alpha, beta)
            board[i][j] = ' '
            best = min(best, value)
            beta = min(beta, value)
            if beta <= alpha:
                break
        return best

def best_move(board, player):
    best_val = -float('inf')
    move = (-1, -1)
    for (i, j) in get_empty_positions(board):
        board[i][j] = player
        move_val = minimax(board, 0, False, -float('inf'), float('inf'))
        board[i][j] = ' '
        if move_val > best_val:
            best_val = move_val
            move = (i, j)
    return move

def play_tic_tac_toe():
    import random
    board = [[' ' for _ in range(3)] for _ in range(3)]
    current_player = "X" if random.randint(0, 1) == 0 else "O"

    while True:
        print("Board:")
        print_board(board)
        if not get_empty_positions(board) or is_winner(board, "X") or is_winner(board, "O"):
            break
        row, col = best_move(board, current_player) if current_player == "X" else map(int, input(f"{current_player}'s turn. Enter row and column: ").split())
        print(f"{current_player} moves to ({row}, {col})")
        print('\n')
        board[row][col] = current_player
        current_player = "O" if current_player == "X" else "X"

    if is_winner(board, "X"):
        print("X wins!")
    elif is_winner(board, "O"):
        print("O wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    play_tic_tac_toe()