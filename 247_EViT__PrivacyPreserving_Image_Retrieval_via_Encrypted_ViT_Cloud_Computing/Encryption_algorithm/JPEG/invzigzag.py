import numpy as np

def invzigzag(ln, num_rows, num_cols):
    tot_elem = len(ln)
    
    # check if matrix dimensions correspond
    if tot_elem!=num_rows*num_cols:
        print('Matrix dimensions do not coincide')
        
    # Initialise the output matrix
    out = np.zeros([num_rows, num_cols])
    
    cur_row = 0; cur_col = 0; cur_index = 0
    
    while cur_index < tot_elem:
        if cur_row == 0 and (cur_row+cur_col)%2==0 and cur_col != (num_cols-1):
            out[cur_row, cur_col] = ln[cur_index]
            cur_col = cur_col + 1                   # move right at the top
            cur_index = cur_index + 1
        
        elif cur_row==(num_rows-1) and (cur_row+cur_col)%2!=0 and cur_col!=(num_cols-1):
            out[cur_row, cur_col] = ln[cur_index]
            cur_col = cur_col + 1                    # move right at the bottom
            cur_index = cur_index + 1
        
        elif cur_col==0 and (cur_row+cur_col)%2!=0 and cur_row!=(num_rows-1):
            out[cur_row,cur_col] = ln[cur_index]
            cur_row = cur_row + 1                     # move down at the left
            cur_index = cur_index + 1
            
        elif cur_col==(num_cols-1) and (cur_row+cur_col)%2==0 and cur_row!=(num_rows-1):
            out[cur_row,cur_col] = ln[cur_index]
            cur_row = cur_row + 1                     # move down at the right
            cur_index = cur_index + 1
            
        elif cur_col!=0 and cur_row!=(num_rows-1) and (cur_row+cur_col)%2!=0:
            out[cur_row,cur_col] = ln[cur_index]
            cur_row = cur_row + 1                      # move diagonally left down
            cur_col = cur_col - 1
            cur_index = cur_index + 1
        
        elif cur_row!=0 and cur_col != (num_cols-1) and (cur_row+cur_col)%2==0:
            out[cur_row, cur_col] = ln[cur_index]
            cur_row = cur_row - 1
            cur_col = cur_col + 1                      # move diagonally right down
            cur_index = cur_index + 1
        
        elif cur_index == tot_elem-1:
            out[num_rows-1,num_cols-1] = ln[tot_elem-1]
            break
    return out