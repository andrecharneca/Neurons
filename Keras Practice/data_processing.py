from numpy import loadtxt
import numpy as np

#company name
name = 'apple'
#number of days per row
ndays = 10

#import data
raw_data = loadtxt('stocks_data_' + name +'.txt', delimiter=',')

N = len(raw_data) #no. of available days, 0->N

#high and low from each day
hi_lo = raw_data[:, 2:4]


#array,starts at 2nd day, has value 1 if next day it goes up, 0 otherwise
growth = []
for i in range(N-1):
    if (hi_lo[i+1][0] - hi_lo[i][0]) >= 0:
        growth.append(1)
    else:
        growth.append(0)



#create data for stocks.py, rows of ndays days with growth of the next day at the end of row
data =[]
temp = []



for i in range(N):
    if i+ndays-1 >= N-1: break
    for k in range(ndays):
        temp.append(hi_lo[i+k][0]) #high
        temp.append(hi_lo[i+k][1]) #low
    temp.append(growth[i+(ndays-1)]) #growth of high from ndaysth to ndays+1th day
    data.append(temp)
    temp=[]


file1 = open(r"stocks_data_processed_" + name +".txt", "w")

for i in range(len(data)):
    file1.write(",".join( repr(e) for e in data [i]) + "\n") #removes the brackets in array representation
file1.close()
#no. of samples
print(len(data))