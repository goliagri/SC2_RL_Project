'''
Idea is 

MLP with: 

Input: 
    per probe health/shield/iscarrying/position
    per zling health/position (up to 10)

Output:
    per probe softmax vector prob of taking action:
        gather from mins 1-8, attack zling 1-10, run, return min
        (maybe move and then seperate output where to move?)

So we feed in all positions + 0/1 probe and zergling still alive? + 0/1 is decision maker? 

'''