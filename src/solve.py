import copy
import sys
from PIL import Image, ImageFont, ImageDraw
import os
import cv2

# %%
N = 9
class Solver():
    def __init__(self) -> None:
        self.board=[[3, 0, 6, 5, 0, 8, 4, 0, 0], 
                    [5, 2, 0, 0, 0, 0, 0, 0, 0], 
                    [0, 8, 7, 0, 0, 0, 0, 3, 1], 
                    [0, 0, 3, 0, 1, 0, 0, 8, 0], 
                    [9, 0, 0, 8, 6, 3, 0, 0, 5], 
                    [0, 5, 0, 0, 9, 0, 6, 0, 0], 
                    [1, 3, 0, 0, 0, 0, 2, 5, 0], 
                    [0, 0, 0, 0, 0, 0, 0, 7, 4], 
                    [0, 0, 5, 2, 0, 6, 3, 0, 0] ]

        


    def output(self, a):
        print(str(a), end='')

    def print_board(self, board):
        if not board:
            print('No Solution')
            return

        for i in range(N):
            for j in range(N):
                cell = board[i][j]
                
                if cell == 0:
                    self.output('.')
                else:
                    self.output(cell) if not isinstance(cell, set) else self.output(0)

                if (j+1) % 3 == 0:
                    self.output(' |')

                if j != 8:
                    self.output(' ')
            self.output('\n')
            if (i+1) % 3 == 0 and i < 8:
                self.output("- - - - - - - - - - -\n")

    def convert(self, board):
        """for each square, list all possible nums"""
        state = copy.deepcopy(board)
        for i in range(N):
            for j in range(N):
                cell = state[i][j]
                if cell == 0:
                    state[i][j] = set(range(1,10)) # make all zeros a set of possible values from 1-9
        return state


    def check_done(self, state):
        for i in state:
            for j in i:
                if isinstance(j, set): # check if any values are still undecided
                    return False
        return True

    def calculate(self, state):

        new_val = False

        for i in range(N): # compute rows
            row = state[i]
            values = set([x for x in row if not isinstance(x, set)]) #find the values in the row
            # #print(values)
            for j in range(N): # row
                if isinstance(state[i][j], set): # if square doesn't have a value
                    state[i][j] -= values # remove values in the same row from each box in that row
                    if len(state[i][j]) == 1:
                        temp = state[i][j].pop() # temp is the only value in the set of possible nums
                        state[i][j] = temp
                        values.add(temp)
                        new_val = True
                        #print('new')
                        #print(str(i) + " | " + str(j))
                    elif len(state[i][j]) == 0: # no possible values for this square
                        return False, None


        for i in range(N): # compute columns [row][column]
            column = [state[x][i] for x in range(N)] # make variable for the column
            values = set([x for x in column if not isinstance(x, set)]) # find the values in the column
            for j in range(N): # column
                if isinstance(state[j][i], set):
                    state[j][i] -= values # remove values in the same column from each box in that column
                    if len(state[j][i]) == 1:
                        temp = state[j][i].pop()
                        state[j][i] = temp
                        values.add(temp)
                        new_val = True
                        #print('new1')
                        #print(str(i) + " | " + str(j))
                    elif len(state[j][i]) == 0:
                        return False, None

        for i in range(3): # compute 3x3 boxes
            for j in range(3):
                values = set()
                for x in range(3): # find the values in the box
                    for y in range(3):
                        cell = state[3*i+x][3*j+y]
                        if not isinstance(cell, set):
                            values.add(cell)

                for x in range(3*i, 3*i+3):
                    for y in range(3*j, 3*j+3):
                        if isinstance(state[x][y], set): # narrow down the potential values based on what's in the 3x3 box
                            state[x][y] -= values
                            if len(state[x][y]) == 1:
                                temp = state[x][y].pop()
                                state[x][y] = temp
                                values.add(temp)
                                new_val = True
                                #print('new2')
                                #print(str(x) + " | " + str(y))
                            elif len(state[x][y]) == 0:
                                return False, None

        return True, new_val

    def solve_loop(self, state):
        while True:
            possible, new_val = self.calculate(state)

            if not possible:
                return False

            if not new_val:
                return True


    def solve_board(self, state):
        isPossible = self.solve_loop(state)

        if not isPossible:
            return None

        if self.check_done(state):
            return state

        # if there are values that we can't calculate due to not enough info, do trial and error on a value to see if it makes the puzzle solvable
        for i in range(N):
            for j in range(N):
                cell = state[i][j]
                if isinstance(cell, set):
                    for val in cell:
                        temp = copy.deepcopy(state)
                        temp[i][j] = val
                        done = self.solve_board(temp)
                        if done is not None:
                            return done
                    return None



    def display_board(self, state):
        size = 100
        im = Image.new('RGB', (size*N,size*N), (255,255,255))
        dr = ImageDraw.Draw(im)
        font = ImageFont.truetype("basic_sans_serif_7.ttf", size=30)
                # dr.rectangle(((0+(j)*size,0+(j)*size),(size+(j)*size, size+(j)*size)), fill="blue", outline = "black")

        # draw the 9 3x3 boxes with a thicker outline
        for i in range(3):
            for j in range(3):
                dr.rectangle([(j*size*3, i*size*3), ((j+1)*size*3, ((i+1)*size*3))], fill="white", outline="black", width=5)

        for i in range(N):
            for j in range(N):
                # draw rectangles for each square and fill the square with the corresponding number on the grid
                dr.rectangle([(0+j*size,0+i*size),(size+j*size,size+i*size)], outline = "black", width=2)
                dr.text((j*size+(size/2)-5, i*size+(size/2)-5), str(state[i][j]) if state[i][j] != 0 and not isinstance(state[i][j], set) else '', fill=(0, 0, 0, 255), font=font)
        im.show()

    def solve(self):
        state = self.convert(self.board)

        self.display_board(self.board) # display board before solve

        state = self.solve_board(state) # solve board

        self.print_board(state)

        self.display_board(state) # display board after solve
