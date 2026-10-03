import numpy as np
import matplotlib.pyplot as plt


x=np.linspace(-5,5,100)
y=x**2


plt.plot(x,y)


plt.title("Parabola:y=x^2")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")



plt.show()