import numpy as np
import datetime 
import matplotlib.pyplot as plt

## Year | Day | Hours | Minutes | Wind direction (degrees) 
Data = np.loadtxt('CR1000_Weather.csv', delimiter=",", skiprows=4, usecols=(2, 3,4,5,9))

ind = np.linspace(0, Data.shape[0]-1, 5000, dtype=np.intp)


year = int(Data[0,0])
days = int(Data[0,1])
hours = int(Data[0,2])
minutes = int(Data[0,3])

print(datetime.datetime(year, 1, 1))
print(datetime.timedelta(days - 1))

dt = datetime.timedelta(days=days-1, hours=hours, minutes=minutes)
print(datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1))
print(datetime.datetime(year, 1, 1) + dt)
print(Data[-1,])
print(dt.total_seconds())

tt = []

dt0 = datetime.timedelta(days=days-1, hours=hours, minutes=minutes)

for i in range(Data.shape[0]):
    days = int(Data[i,1])
    hours = int(Data[i,2])
    minutes = int(Data[i,3])
    dt = datetime.timedelta(days=days - 1, hours=hours, minutes=minutes)
    dt_diff = dt - dt0
    
    tt.append(dt_diff.total_seconds()/(60*60*24))
tt = np.array(tt)

ind = np.linspace(Data.shape[0]/2, Data.shape[0]-1, 1000, dtype=np.intp)

theta = (Data[:,-1]/360.)*2*np.pi

np.savetxt('wind1.txt', np.column_stack(( tt, Data[:,-1] )))




print(theta.min(), theta.max())

r = np.exp(0.09*tt)

plt.plot(tt[ind], theta[ind], '-')

#plt.plot(r[ind]*np.cos(theta[ind]), r[ind]*np.sin(theta[ind]), '-')
plt.show()
