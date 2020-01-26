import sys
import numpy as np

# Run:
# python scripts/mod-timeints-partitions.py \
#   <pat/to/mod-timeints-and-partitions.py> \
#   <time step days> \
#   <num run partitions>

# f.ex. for creating time intervals for each fourth days printed with three lines
# python scripts/mod-timeints-and-partitions.py ../Elies-longitudinal-data-test/timeints_mod.txt 4 3

#with open("../Elies-longitudinal-data-test/timeints_mod.txt") as file:
with open(sys.argv[1]) as file:
    lines = file.readlines()

# Remove newlines and create list of ints
# Note: the last line in the file should be the last
# integer and should not contain a newline
timeints = [int(time[:-1]) if i < len(lines)-1 else int(time) for i, time in enumerate(lines)]

# 
timeints_mod = np.array(timeints)//int(sys.argv[2])

print(timeints_mod)

# 
partition_length = len(timeints_mod)//int(sys.argv[3])

for p in range(len(timeints_mod)//partition_length):
    if p == len(timeints_mod)//partition_length - 1:
        for i in timeints_mod[p*partition_length:]:
            print(i, end=" ")
    else:
        for t in range(partition_length):
            print(timeints_mod[p*partition_length+t], end=" ")
    print("")

