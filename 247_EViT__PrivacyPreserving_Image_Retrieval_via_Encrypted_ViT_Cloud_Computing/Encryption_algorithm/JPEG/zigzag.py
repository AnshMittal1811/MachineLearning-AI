import numpy as np
'''
Zig-zag scanning
This function is used to rearrange a matrix of any size into a 1-D array
by implementing the ZIG-ZAG SCANNING procedure.
IN specifies the input matrix of any size
OUT is the resulting zig-zag scanned (1-D) vector
having length equal to the total number of elements in the 2-D input matrix

As an example,
LN = [1	 2	6	7
      3	 5	8	11
      4	 9	10	12];
OUT = ZIGZAG(IN)
OUT=
	1     2     3     4     5     6     7     8     9    10    11    12



 Oluwadamilola (Damie) Martins Ogunbiyi
 University of Maryland, College Park
 Department of Electrical and Computer Engineering
 Communications and Signal Processing
 22-March-2010
 Copyright 2009-2010 Black Ace of Diamonds.
'''

def zigzag(ln):
    [num_rows, num_cols] = ln.shape
    out = np.zeros(num_rows*num_cols)
    cur_row = 0; cur_col = 0; cur_index = 0
    while cur_row < num_rows and cur_col < num_cols:
        if cur_row==0 and (cur_row+cur_col)%2==0 and cur_col!=(num_cols-1):
            out[cur_index] = ln[cur_row, cur_col]
            cur_col = cur_col + 1                     # move right at the top
            cur_index = cur_index + 1
        elif cur_row == (num_rows-1) and (cur_row+cur_col)%2!=0 and cur_col!=(num_cols-1):
            out[cur_index] = ln[cur_row, cur_col]
            cur_col = cur_col + 1                    # move right at the bottom                                       
            cur_index = cur_index + 1
        elif cur_col==0 and (cur_row+cur_col)%2!=0 and cur_row!=(num_rows-1):
            out[cur_index] = ln[cur_row, cur_col]
            cur_row = cur_row + 1                      # move down at the left
            cur_index = cur_index + 1
        elif cur_col==(num_cols-1) and (cur_row+cur_col)%2==0 and cur_row!=(num_rows-1):
            out[cur_index] = ln[cur_row,cur_col]
            cur_row = cur_row + 1                      # move down at the right
            cur_index = cur_index + 1
        elif cur_col!=0 and cur_row!=(num_rows-1) and (cur_row+cur_col)%2!=0:
            out[cur_index] = ln[cur_row, cur_col]
            cur_row = cur_row + 1; cur_col = cur_col - 1 # move diagonally left down
            cur_index = cur_index + 1
        elif cur_row!=0 and cur_col!=(num_cols-1) and (cur_row+cur_col)%2==0:
            out[cur_index] = ln[cur_row, cur_col]
            cur_row = cur_row - 1; cur_col = cur_col + 1 # move diagonally right up
            cur_index = cur_index + 1
        elif cur_row == (num_rows-1) and cur_col == (num_cols-1):
            out[num_rows*num_cols-1] = ln[num_rows-1, num_cols-1]  #obtain the bottom right elements                                                        
            break                                                  #end of the operation
    return out                                                     #terminate the operation