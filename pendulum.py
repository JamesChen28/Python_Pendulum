


import numpy as np
from scipy.integrate import odeint
from math import sin, cos, pi
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation



#### ode

# dx/dt = a * (y - x)
# dy/dt = x * (b - z) - y
# dz/dt = x * y - c * z

## without odeint

def move(Point, Steps, Coefs):
    x, y, z = Point
    a, b, c = Coefs
    
    dx = a * (y - x)
    dy = x * (b - z) - y
    dz = x * y - c * z
    
    next_x = x + dx * Steps
    next_y = y + dy * Steps
    next_z = z + dz * Steps
    
    return [next_x, next_y, next_z]

Coefs = [10, 28, 3]
t = np.arange(0, 30, 0.01)
P1 = [0, 1, 0]
P2 = [0, 1.01, 0]

P = P1
d = []
for v in t:
    P = move(P, 0.01, Coefs)
    d.append(P)
dnp1 = np.array(d)


P = P2
d = []
for v in t:
    P = move(P, 0.01, Coefs)
    d.append(P)
dnp2 = np.array(d)


# plot 3d

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(dnp1[:, 0], dnp1[:, 1], dnp1[:, 2], color = 'red')
ax.plot(dnp2[:, 0], dnp2[:, 1], dnp2[:, 2], color = 'blue')
plt.show()


## with odeint

def dmove(Point, Steps, Coefs):
    x, y, z = Point
    a, b, c = Coefs
    
    next_x = a * (y - x)
    next_y = x * (b - z) - y
    next_z = x * y - c * z
    
    return np.array([next_x, next_y, next_z])

t = np.arange(0, 30, 0.01)
P1 = odeint(dmove, (0, 1, 0), t, args = ([10, 28, 3],))
P2 = odeint(dmove, (0, 1.01, 0), t, args = ([10, 28, 3],))


# plot 3d

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(P1[:, 0], P1[:, 1], P1[:, 2], color = 'red')
ax.plot(P2[:, 0], P2[:, 1], P2[:, 2], color = 'blue')
ax.set_title('ODE', fontsize = 20)
ax.set_xlabel('x', fontsize = 20, rotation = 150)
ax.set_ylabel('y', fontsize = 20)
ax.set_zlabel('z', fontsize = 20)
plt.show()



#### simple pendulum

def simple_pendulum(Point, Time, Args):
    theta, v = Point
    g, l = Args
    
    dtheta = v
    dv = - g * sin(theta) / l
    
    return dtheta, dv


init_value = (1, 0)
t = np.arange(0, 30, 0.1)
g = 9.8
l = 1
args = (g, l)

P1 = odeint(simple_pendulum, init_value, t, args = (args, ))


# plot 3d

fig = plt.figure()
ax = Axes3D(fig)
ax.plot(P1[:, 0], P1[:, 1], t, color = 'red')
ax.set_title('Simple Pendulum', fontsize = 20)
ax.set_xlabel('Theta', fontsize = 20, rotation = 150)
ax.set_ylabel('v', fontsize = 20)
ax.set_zlabel('Time', fontsize = 20)
plt.show()


# plot animation

def init():
    ax.scatter(l * sin(P1[0, 0]/pi), l - l * cos(P1[0, 0]/pi))

def animate(i):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 2) 
    ax.scatter(l * sin(P1[i, 0]/pi), l - l * cos(P1[i, 0]/pi))
    ax.plot([0, l * sin(P1[i, 0]/pi)], [1, l - l * cos(P1[i, 0]/pi)], color = 'green')


fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 2)

ani = animation.FuncAnimation(fig, animate, frames = 300, interval = 10, init_func = init)

plt.show()



#### gravity

# use ode
def gravity(Point, Time, Args):
    y, v = Point
    g = Args
    
    dy = -v
    dv = g
    
    return dy, dv


init_value = (100, 0)
t = np.arange(0, 30, 0.1)
g = 9.8
args = (g)

P1 = odeint(gravity, init_value, t, args = (args, ))


# use physics

h = 100
g = 9.8
t = np.arange(0, 30, 0.1)
S = []
for i in range(len(t)):
    S.append(h - g * t[i] * t[i] / 2)



# plot animation

def animate(i):
    ax.clear()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 150) 
    #ax.scatter(0, P1[i, 0])
    ax.scatter(0, S[i])
    #print(i)
    ax.text(-0.5, 120, 'Time = {t}\n h = {S} '.format(t = t[i], S = S[i]))


fig, ax = plt.subplots()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 150)

ani = animation.FuncAnimation(fig, animate, frames = 3000, interval = 10)

plt.show()



#### double pendulum

def double_pendulum(Point, Time, Args):

    g, l1, l2, m1, m2 = Args
    
    theta1, theta2, v1, v2 = Point
    
    dtheta1 = v1
    dtheta2 = v2
    
    # eq of theta1
    coef1_1 = l1 * (m1 + m2)
    coef1_2 = l2 * m2 * cos(theta1 - theta2)
    coef1_3 = l2 * m2 * dtheta2 * dtheta2 * sin(theta1 - theta2) + (m1 + m2) * g * sin(theta1)
    
    # eq of theta2
    coef2_1 = l1 * cos(theta1 - theta2)
    coef2_2 = l2
    coef2_3 = -l1 * dtheta1 * dtheta1 * sin(theta1 - theta2) + g * sin(theta2)
    
    dv1, dv2 = np.linalg.solve([[coef1_1, coef2_1], [coef1_2, coef2_2]], [-coef1_3, -coef2_3])
    
    return np.array([dtheta1, dtheta2, dv1, dv2])


init_value = (1, 2, 1, 2)
t = np.arange(0, 30, 0.01)
g = 9.8
l1 = 1
l2 = 1
m1 = 1
m2 = 1
args = (g, l1, l2, m1, m2)

P1 = odeint(double_pendulum, init_value, t, args = (args, ))


# plot animation

theta1_array = P1[:, 0]
theta2_array = P1[:, 1]

x1 = l1 * np.sin(theta1_array)
y1 = -l1 * np.cos(theta1_array)
x2 = x1 + l2 * np.sin(theta2_array)
y2 = y1 - l2 * np.cos(theta2_array)
    
    
def animate(i):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3) 
    ax.scatter(x1[i], y1[i], color = 'red')
    ax.scatter(x2[i], y2[i], color = 'blue')
    ax.plot([0, x1[i]], [0, y1[i]], color = 'green')
    ax.plot([x1[i], x2[i]], [y1[i], y2[i]], color = 'green')
    print(i)
    ax.text(0, 2, 'Time = {t}\nx1 = {x1} \ny1 = {y1} \nx2 = {x2} \ny2 = {y2} ' \
            .format(t = t[i], x1 = x1[i], y1 = y1[i], x2 = x2[i], y2 = y2[i]))


fig, ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)

ani = animation.FuncAnimation(fig, animate, frames = 3000, interval = 10)

plt.show()





