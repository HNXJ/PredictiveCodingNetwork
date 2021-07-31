import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))
last_nums = 0.01


def animate1(i):
    global last_nums 
    last_nums = last_nums*0.3 + 0.3*np.sin(i/50 + np.pi/2)
    line.set_ydata(last_nums + 0.5*np.sin(x + i / 50))  # update the data.
    return line,


def animate2(i):
    global last_nums 
    last_nums = last_nums*0.8 + (np.random.rand(1)-0.5)*0.3
    line.set_ydata(last_nums + 0.5*np.sin(x + i / 50))  # update the data.
    return line,


def run1():
    ani = animation.FuncAnimation(fig, animate1, interval=20, blit=True, save_count=100)
    writer = animation.FFMpegWriter(
        fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("movie.mp4", writer=writer)
    
    plt.show()


def run2():
    ani = animation.FuncAnimation(fig, animate2, interval=20, blit=True, save_count=100)
    writer = animation.FFMpegWriter(
        fps=10, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("movie.mp4", writer=writer)
    
    plt.show()
    

mode = int(input("mode? (1/2) ->  "))
if mode == 1:
    run1()  
elif mode == 2:
    last_nums = 1
    run2()