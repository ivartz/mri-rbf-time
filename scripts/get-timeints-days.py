import sys
import datetime

# Run
# python get-timeints-days.py </path/to/ack-times.txt>

#with open("ack-times.txt") as file:
with open(sys.argv[1]) as file:
    lines = file.readlines()

#print("number of time points: %i" % len(lines))

for i, line in enumerate(lines[:-1]):
    
    time1 = \
    datetime.datetime.strptime(line, '%d/%m/%Y, %H:%M\n')
    
    if i == len(lines[:-1]) - 1:
        time2 = \
        datetime.datetime.strptime(lines[i+1], '%d/%m/%Y, %H:%M')
    else:
        time2 = \
        datetime.datetime.strptime(lines[i+1], '%d/%m/%Y, %H:%M\n')

    print((time2-time1).days)
        
    

