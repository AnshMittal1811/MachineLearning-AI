import json
import puz
import wordsegment
import math
from wordsegment import load, segment, clean
load()

dictionary = set([a.strip() for a in open('solver/words_alpha.txt','r').readlines()])

def num_words(fill):
    '''segment the text into multiple words and count how many words the text has in total'''
    segmented = segment(fill)
    prob = 0.0
    for word in segmented:
        if word not in dictionary:
            return 999, -9999999999999
        prob += math.log(wordsegment.UNIGRAMS[word])
    return (len(segmented), prob)

def get_word_flips(fill, num_candidates=10):
    '''
    We take as input a word/phrase that is probably mispelled, something like iluveyou. We then try flipping each one of the letters
    to all other letters. We then segment those texts into multiple words using num_words, e.g., iloveyou -> i love you. We return the candidates
    that segment into the fewest number of words.
    '''
    results = {}
    min_length = 999
    fill = clean(fill)
    for index, char in enumerate(fill):
        for new_letter in 'abcdefghijklmnopqrstuvwxyz':
            new_fill = list(fill)
            new_fill[index] = new_letter
            new_fill = ''.join(new_fill)
            curr_num_words, prob = num_words(new_fill)
            if curr_num_words not in results:
                results[curr_num_words] = []
            results[curr_num_words].append((new_fill, prob))
            if curr_num_words < min_length:
                min_length = curr_num_words
    if min_length == 999:
        return [fill.upper()]
    all_results = sum([sorted(results[length], key=lambda x:-x[1]) for length in sorted(list(results.keys()))], [])
    return [a[0].upper() for a in all_results[0:num_candidates]]

def convert_puz(fname):
    # requires pypuz library to run
    # converts a puzzle in .puz format to .json format
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

def clean(text):
    '''
    :param text: question or answer text
    :return: text with line breaks and trailing spaces removed
    '''
    return " ".join(text.strip().split())

def print_grid(letter_grid):
    for row in letter_grid:
        row = [" " if val == "" else val for val in row]
        print("".join(row), flush=True)
