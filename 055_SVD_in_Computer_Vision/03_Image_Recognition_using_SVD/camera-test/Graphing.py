import matplotlib.pyplot as plt
import numpy as np
import time


#plt.plot([ x1, x2, x3], [ y1, y2, y3] ,'ro') the ro makes scattered data and not connected lines
#plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^') first t declares variable, next t is the graph, 'r--' is red, 'bs' is blue, 'g^' is green
#t = np.arrange(0. ,5. , .5) makes evenly spaced dots of data
#plt.axis([x start, x max, y start, y min])
#plt.xlabel("Whats up") plt.ylabel("nothing much")
#plt.show() opens figure


plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()

points = [[1,3],[3,3]]
'''
plt.suptitle("hey whats up")
for cell in points:
    plt.plot(cell, 'bp')
plt.waitforbuttonpress()
plt.suptitle("i've changed")
plt.plot([0,1],[3.259 , 2.587])
plt.show()
'''

finalPoints = [[1,2],[6,3],[4,8],[8,7],[3,2]]
plt.suptitle("Base pictures plotted")
plt.scatter(*zip(*finalPoints),'r*') #base points are red stars. https://stackoverflow.com/questions/21519203/plotting-a-list-of-x-y-coordinates-in-python-matplotlib
plt.show()
plt.waitforbuttonpress()
x,y = 2,2
plt.scatter(x,y , 'gs') #observed image is representied by a green square
plt.suptitle("Oberved Point ")
plt.show()
