def minimax(state, player, depth=0, print_moves=False):
    max_player = "O"
    other_player = "X" if player == "O" else "O"

    if state.current_winner == other_player:
        return {
            "position": None,
            "score": (
                1 * (state.num_empty_squares() + 1)
                if other_player == max_player
                else -1 * (state.num_empty_squares() + 1)
            ),
        }

    elif not state.empty_squares():
        return {"position": None, "score": 0}

    if player == max_player:
        best = {"position": None, "score": -float("inf")}
    else:
        best = {"position": None, "score": float("inf")}

    all_scores = []

    for possible_move in state.available_moves():
        state.make_move(possible_move, player)
        sim_score = minimax(state, other_player, depth + 1)

        state.board[possible_move] = " "
        state.current_winner = None
        sim_score["position"] = possible_move

        if player == max_player:
            if sim_score["score"] > best["score"]:
                best = sim_score
        else:
            if sim_score["score"] < best["score"]:
                best = sim_score

        all_scores.append(sim_score)

    if depth == 0 and print_moves == True:
        print("=====================================")
        print("All possible moves and their scores:")
        for move_score in all_scores:
            print(f"Move: {move_score['position']}, Score: {move_score['score']}")
        print("=====================================\n")

    return best
