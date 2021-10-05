def valid(row, column, max_row, max_column, map_matrix):
    if 0 <= row and row < max_row and 0 <= column and column < max_column and map_matrix[row][column]:
        return True
    return False

def route_exists(from_row, from_column, to_row, to_column, map_matrix):

    max_row = len(map_matrix)
    max_column = len(map_matrix[0])
    
    queue = [(from_row, from_column)]
    visited = {(from_row, from_column)}
    
    ds = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while queue:
        row, column = queue.pop(0)
        if row == to_row and column == to_column:
            return True
        for d in ds:
            if valid(row+d[0], column+d[1], max_row, max_column, map_matrix) and (row+d[0], column+d[1]) not in visited:
                queue.append((row+d[0], column+d[1]))
                visited.add((row+d[0], column+d[1]))
    return False

if __name__ == '__main__':
    map_matrix = [
        [True, False, False],
        [True, True, False],
        [False, True, True]
    ]

    print(route_exists(0, 0, 2, 2, map_matrix))