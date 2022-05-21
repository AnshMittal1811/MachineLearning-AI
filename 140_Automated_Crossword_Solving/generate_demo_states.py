import pandas as pd
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
import tokenizers
import json
import puz
import os
import numpy as np
import streamlit as st
import scipy

import sys
import subprocess
import copy
import json

from itertools import zip_longest
from copy import deepcopy
import regex

from solver.Crossword import Crossword
from solver.BPSolver import BPSolver
from models import setup_closedbook, setup_t5_reranker, DPRForCrossword

def call_solver(f):
    with open(f, "r") as infile:
        print("Running solver on file", infile)
        puzzle = json.load(infile)
        crossword = Crossword(puzzle)
    solver = BPSolver(crossword, max_candidates=500000)
    grid, states = solver.solve(num_iters=10, iterative_improvement_steps=5, return_greedy_states=True, return_ii_states=True)
    return grid, states

def fill_black_squares(solver_grid, ref_grid):
    """
    Arguments:
    - drfill_out: [({"clue": "fill"...}, {"clue2: "fill2"...})...]
    - grid_nums: dict mapping numbering to indices of grid: {1: (0, 0)...}
    - grid: 2d array grid
    """
    solver_grid = deepcopy(solver_grid)
    for r in range(len(solver_grid)):
        for c in range(len(solver_grid[r])):
            if ref_grid[r][c] == ".":
                solver_grid[r][c] = "."
            if solver_grid[r][c] == "":
                solver_grid[r][c] = " "
            solver_grid[r][c] = solver_grid[r][c].upper()
    return solver_grid

def diff(state1, state2):
    """
    Infers the anchors of fill made. 
    Arguments:
    - two adjacent (t-1, t) states of solver.
    Returns list of anchor points which changed
    """
    def index_(grid):
        def at(anchor): # (r, c)
            return grid[anchor[0]][anchor[1]]
        return at
    # Find differing squares
    diff_squares = []
    for r in range(len(state1)):
        for c in range(len(state1[r])):
            if state1[r][c] != state2[r][c]:
                diff_squares.append((r, c))
    if all([x != " " for x in sum(state1, [])]):
        return diff_squares
    diff_squares = sorted(diff_squares) # first element is earliest square
    if len(diff_squares) == 0:
        return []
    
    across = {"incr": lambda r, c: (r, c + 1), "decr": lambda r, c: (r, c - 1), "index": 0}
    down = {"incr": lambda r, c: (r + 1, c), "decr": lambda r, c: (r - 1, c), "index": 1}
    for direction in (across, down):
        incr, decr = direction["incr"], direction["decr"]
        # decrement to find the earliest anchor
        r, c = diff_squares[0] # start randomly
        while min(decr(r, c)) >= 0 and index_(state2)(decr(r, c)) != ".":
            r, c = decr(r, c)
        # increment until we hit the final anchor, adding to anchors
        anchor_pts = [(r, c)]
        while max(incr(r, c)) < len(state2) and index_(state2)(incr(r, c)) != ".":
            r, c = incr(r, c)
            anchor_pts.append((r, c))
        if any([index_(state2)(loc) == " " for loc in anchor_pts]): # If not all the cells are filled it can't be this row
            continue
        r_or_c = direction["index"]
        dir_only = [pair[r_or_c] for pair in diff_squares]
        if dir_only.count(dir_only[0]) == len(dir_only): # if this is the right direction then stop
            break
    return anchor_pts
    
   
def find_incorrect_squares(state_2d, sol_2d):
    """
    Arguments: state_2d, sol_2d: 2d array of characters
    Returns: List of coordinates representing incorrect squares
    """
    incorrect = []
    for x in range(len(state_2d)):
        for y in range(len(state_2d[0])):
            if state_2d[x][y] != sol_2d[x][y] and state_2d[x][y] != " ":
                incorrect.append((x, y))
    return incorrect 

def solve(fpath):
    flatten = lambda lst: [item for sublist in lst for item in sublist]
    if fpath.endswith(".json"):
        with open(fpath, "r") as f:
            p = json.load(f)
        width, height = p["metadata"]["cols"], p["metadata"]["rows"]
        title, author, copyright = "NYT" + p["metadata"]["date"], "Unknown", "New York Times"
        solution = "".join(sum([[c[1] if c != "BLACK" else "." for c in x] for x in p["grid"]], []))
        def make_numbering(d, grid):
            numbers_only = flatten([[int(cell[0]) if type(cell) == list and cell[0] != "" else None for cell in row] for row in grid])
            lst = []
            for i, stuff in d.items():
                lst.append({"num": int(i), "clue": stuff[0], "cell": numbers_only.index(int(i))})
            return lst
        numbering_across, numbering_down = make_numbering(p["clues"]["across"], p["grid"]), make_numbering(p["clues"]["down"], p["grid"])
    else:
        p = puz.read(fpath)
        width, height = p.width, p.height
        title, author, copyright = p.title, p.author, p.copyright
        solution = p.solution
        numbering = p.clue_numbering()
        numbering_across, numbering_down = numbering.across, numbering.down

    gridnums = [0]*(height*width)
    across_clues = []
    for clue in numbering_across:
        across_clues.append(str(clue['num']) + '. ' + clue['clue']) 
        gridnums[int(clue['cell'])]  = clue['num']

    down_clues = []
    for clue in numbering_down:
        down_clues.append(str(clue['num']) + '. ' + clue['clue']) 
        gridnums[int(clue['cell'])]  = clue['num']

    grid_mapping = {}
    for i in range(len(gridnums)):
        if gridnums[i] != 0:
            grid_mapping[gridnums[i]] = (i // width, i % width)

    grid = [" " if c.isalpha() else c for c in solution]
    grid_2d = [[c for c in grid[x:x + width]] for x in range(0, len(grid), width)]
    sol_2d = [[c for c in solution[x:x + width]] for x in range(0, len(grid), width)]

    across_mapping_2d = copy.deepcopy(grid_2d)
    j = 0
    for clue in numbering_across:
        i = clue['num']
        anchor_x, anchor_y = grid_mapping[i]
        while anchor_y < width and across_mapping_2d[anchor_x][anchor_y] != ".":
            across_mapping_2d[anchor_x][anchor_y] = j
            anchor_y += 1
        j += 1
    across_mapping = flatten(across_mapping_2d)
    across_mapping = [c if c != "." else 0 for c in across_mapping]

    down_mapping_2d = copy.deepcopy(grid_2d)
    j = 0
    for clue in numbering_down:
        i = clue['num']
        anchor_x, anchor_y = grid_mapping[i]
        while anchor_x < height and down_mapping_2d[anchor_x][anchor_y] != ".":
            down_mapping_2d[anchor_x][anchor_y] = j
            anchor_x += 1
        j += 1
    down_mapping = flatten(down_mapping_2d)
    down_mapping = [c if c != "." else 0 for c in down_mapping]

    puzz_jsons = []
    final_puzz, states = call_solver(fpath)
    states = [fill_black_squares(s, grid_2d) for s in states]
    last_squares = [[]] + [diff(states[i], states[i + 1]) for i in range(len(states) - 1)]
    for i in range(len(states)):
        state_2d = [["" if letter == "." or letter == " " else letter for letter in row] for row in states[i]]
        puzz_jsons.append(state_2d)
    return puzz_jsons, height, width, p

if __name__ == "__main__":
    puzzles = ("5-2-2021", "5-9-2021")
    mapping = {p: f"data/2021/{p}.json" for p in puzzles}
    for puzzle_name in puzzles:
        puzzle_dir = mapping[puzzle_name]
        puzz_jsons, height, width, p = solve(puzzle_dir)
        with open("demo/" + puzzle_name + "_state.json", "w") as f:
            f.write(json.dumps(puzz_jsons))
        with open("demo/" + puzzle_name + "_puzzle.json", "w") as f:
            f.write(json.dumps(p))
