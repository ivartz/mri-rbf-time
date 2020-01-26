import sys
import numpy as np

# Run:
# python scripts/timeints-divide-and-partition.py \
#   <time step days> \
#   <max num intervals with lengths larger than the interval specified>

# f.ex. for creating time intervals for each fourth days while not allowing a line contain 
# sum of intervals larger than 2 * the maximum interval in the .txt file
# python scripts/timeints-divide-and-partition.py ../Elies-longitudinal-data-test/timeints_mod.txt 4 2

#with open("../Elies-longitudinal-data-test/timeints_mod.txt") as file:
with open(sys.argv[1]) as file:
    lines = file.readlines()

# Remove newlines and create list of ints
# Note: the last line in the file should be the last
# integer and should not contain a newline
timeints = [int(time[:-1]) if i < len(lines)-1 else int(time) for i, time in enumerate(lines)]

# 
timeints_mod = np.array(timeints)//int(sys.argv[2])

#print(timeints_mod)

specif_interval = np.max(timeints_mod) # can be median , max or min. Set to max if little RAM available

#print("median interval was %i , using this as splitting measure" %specif_interval)

# 
max_num_specif_intervals = int(sys.argv[3])

timeints_mod_copy = timeints_mod.copy()

buf = 0

j = 0

for i, interval in enumerate(timeints_mod):
    buf += interval
    j += 1
    #print("buf is %i" % buf)
    if buf > max_num_specif_intervals*specif_interval or i == len(timeints_mod)-1:
        for intervalint in timeints_mod_copy[:j]:
            print(intervalint, end=" ")
        print("")
        timeints_mod_copy = timeints_mod_copy[j:]
        buf = 0
        j = 0
