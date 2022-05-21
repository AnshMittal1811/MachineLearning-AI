import puz
import re
import unicodedata
import sys

def puz_to_json(fname):
    """ Converts a puzzle in .puz format to .json format
    """
    p = puz.read(fname)
    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index:cell + col_index + 1] == '.':
                grid[row_idx][col_index] = 'BLACK'
            else:
                grid[row_idx][col_index] = ["", row_solution[col_index: col_index + 1]]

    across_clues = {}
    for clue in numbering.across:
        answer = ''.join(p.solution[clue['cell'] + i] for i in range(clue['len']))
        across_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])

    down_clues = {}
    for clue in numbering.down:
        answer = ''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len']))
        down_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])


    mydict = {'metadata': {'date': None, 'rows': p.height, 'cols': p.width}, 'clues': {'across': across_clues, 'down': down_clues}, 'grid': grid}
    return mydict

def puz_to_pairs(filepath):
    """ Takes in a filepath pointing to a .puz file and returns a list of (clue, fill) pairs in a list
    """
    p = puz.read(filepath)

    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index:cell + col_index + 1] == '.':
                grid[row_idx][col_index] = 'BLACK'
            else:
                grid[row_idx][col_index] = ["", row_solution[col_index: col_index + 1]]

    pairs = {}
    for clue in numbering.across:
        answer = ''.join(p.solution[clue['cell'] + i] for i in range(clue['len']))
        pairs[clue['clue']] = answer

    for clue in numbering.down:
        answer = ''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len']))
        pairs[clue['clue']] = answer

    return [(k, v) for k, v in pairs.items()]

