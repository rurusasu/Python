# Flowchart
from math import *
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import *

param = sys.argv
iflag = int(param[1])  # flag for axis drawing
# 0: without axis and grid, 1: with axis and grid

# drawing area: w x h = 16 x 25 imaged A4 paper
xmin = -8
xmax = 8
ymin = 0
ymax = 25
dh = 0.7  # height of one row
A = []   # actural y-axis
B = []   # grid number in y-axis
for i in range(0, int(ymax//dh)+1):
    s = '{0:.1f}'.format(dh*i)
    A = A+[float(s)]
    B = B+[i]

fnameF = 'fig_flowchart.png'
fig = plt.figure()
ax1 = plt.subplot(111)
ax1 = plt.gca()
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymax, ymin])
aspect = (abs(ymax-ymin))/(abs(xmax-xmin))*abs(ax1.get_xlim()
                                               [1] - ax1.get_xlim()[0]) / abs(ax1.get_ylim()[1] - ax1.get_ylim()[0])


ax1.set_aspect(aspect)
if iflag == 0:
    plt.axis('off')
else:
    ax1.tick_params(labelsize=6)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(dh))
    plt.yticks(A, B)
    plt.grid(which='both', lw=0.3, color='#cccccc', linestyle='-')

# Store texts in list
a = []
a = a+['Initial setting']
a = a+['$i=0, t=0, EL=EL_{ini}$']
a = a+['Calculate $Vol[EL_{ini}], q_{out}[EL_{ini}]$']
a = a+['$i=i+1$ (time step increment)']
a = a+['Inflow setting']
a = a+['$\Delta t=t(i+1)-t(i)$']
a = a+['$Q_{in}=0.5*(q_{in}[t_i+\Delta t]+q_{in}[t_i])$']
a = a+['$h=0$ (zero set of level increment)']
a = a+['$h=h+\Delta h$ (level increment)']
a = a+['Iterative calculation']
a = a+['(1) Calculate $q_{out}[EL]$ and $q_{out}[EL+h]$']
a = a+['(2) $Q_{out}=0.5*(q_{out}[EL]+q_{out}[EL+\Delta t])$']
a = a+['(3) $\Delta S=(Q_{in}-Q_{out}) * \Delta t * 3600$']
a = a+['(4) $R[EL(t_i+\Delta t)]=Vol[EL(t_i)]+\Delta S$']
a = a+['(5) Calculate reservoir water level $elv$']
a = a+['    at reservoir volume $R[EL(t_i+\Delta t)]$']
a = a+['Converged?']
a = a+['$|(EL+h) - elv| < \epsilon$']
a = a+['Yes']
a = a+['No']
a = a+['Update equilibrium point data']
a = a+['$EL(t_i+\Delta t)=EL(t_i)+h$']
a = a+['$Vol[EL(t_i+\Delta t)]=R[EL(t_i+\Delta t)]$']
a = a+['$q_{out}[EL(t_i+\Delta t)]=q_{out}[EL(t_i)+h]$']
title = 'Basic Procedure of Calculation'

# Store coordinats of texts in list
iis = 3
ax = []
ay = []
ii = iis+0
ys = ii*dh
xs = 0
ax = ax+[xs]
ay = ay+[ys+0.5*dh]
ax = ax+[xs]
ay = ay+[ys+1.5*dh]
ax = ax+[xs]
ay = ay+[ys+2.5*dh]

ii = iis+4
ys = ii*dh
xs = 0
ax = ax+[xs]
ay = ay+[ys+0.5*dh]

ii = iis+6
ys = ii*dh
xs = 0
ax = ax+[xs]
ay = ay+[ys+0.5*dh]
ax = ax+[xs]
ay = ay+[ys+1.5*dh]
ax = ax+[xs]
ay = ay+[ys+2.5*dh]
ax = ax+[xs]
ay = ay+[ys+3.5*dh]

ii = iis+11
ys = ii*dh
xs = 0
ax = ax+[xs]
ay = ay+[ys+0.5*dh]

ii = iis+13
ys = ii*dh
xs = -4.5
ax = ax+[0]
ay = ay+[ys+0.5*dh]
ax = ax+[xs]
ay = ay+[ys+1.5*dh]
ax = ax+[xs]
ay = ay+[ys+2.5*dh]
ax = ax+[xs]
ay = ay+[ys+3.5*dh]
ax = ax+[xs]
ay = ay+[ys+4.5*dh]
ax = ax+[xs]
ay = ay+[ys+5.5*dh]
ax = ax+[xs]
ay = ay+[ys+6.5*dh]

ii = iis+22.5
ys = ii*dh
xs = 0
ax = ax+[xs]
ay = ay+[ys-0.45*dh]
ax = ax+[xs]
ay = ay+[ys+0.30*dh]
ax = ax+[0.7]
ay = ay+[ys+1.8*dh]
ax = ax+[-4.5]
ay = ay+[ys+0.5*dh]

ii = iis+25
ys = ii*dh
xs = 0
ax = ax+[0]
ay = ay+[ys+0.5*dh]
ax = ax+[xs]
ay = ay+[ys+1.5*dh]
ax = ax+[xs]
ay = ay+[ys+2.5*dh]
ax = ax+[xs]
ay = ay+[ys+3.5*dh]

# Store coordinates of tbox shapes in list
wx = 10
xs = -0.5*wx
xe = 0.5*wx
ii = iis+0
row = 1
ys = ii*dh
ye = ys+row*dh
po1t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
ii = iis+1
row = 2
ys = ii*dh
ye = ys+row*dh
po1a = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

ii = iis+4
row = 1
ys = ii*dh
ye = ys+row*dh
po2t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

ii = iis+6
row = 1
ys = ii*dh
ye = ys+row*dh
po3t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
ii = iis+7
row = 3
ys = ii*dh
ye = ys+row*dh
po3a = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

ii = iis+11
row = 1
ys = ii*dh
ye = ys+row*dh
po4t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

ii = iis+13
row = 1
ys = ii*dh
ye = ys+row*dh
po5t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
ii = iis+14
row = 6
ys = ii*dh
ye = ys+row*dh
po5a = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

ii = iis+22.5
row = 1.5
ys = ii*dh-row*dh
ye = ii*dh+row*dh
xsm = -4
xem = 4
po6t = [(xsm, ii*dh), (0, ys), (xem, ii*dh), (0, ye)]

ii = iis+25
row = 1
ys = ii*dh
ye = ys+row*dh
po7t = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]
ii = iis+26
row = 3
ys = ii*dh
ye = ys+row*dh
po7a = [(xs, ys), (xe, ys), (xe, ye), (xs, ye)]

# Draw box
llw = 0.5
for pox in [po1t, po2t, po3t, po4t, po5t, po6t, po7t]:
    poly = Polygon(pox, facecolor='#dddddd', edgecolor='#000000', lw=llw)
    ax1.add_patch(poly)
for pox in [po1a, po3a, po5a, po7a]:
    poly = Polygon(pox, facecolor='#eeeeee', edgecolor='#000000', lw=llw)
    ax1.add_patch(poly)

# Draw text
fsize = 5  # fontsize
for i in range(0, len(a)):
    if ax[i] == 0:
        plt.text(ax[i], ay[i], a[i], rotation=0,
                 ha='center', va='center', fontsize=fsize)
    else:
        plt.text(ax[i], ay[i], a[i], rotation=0,
                 ha='left', va='center', fontsize=fsize)
# Draw title
xs = 0
ys = iis*dh-1.0*dh
plt.text(xs, ys, title, rotation=0, ha='center',
         va='center', fontsize=fsize, fontweight='bold')

# Draw line
ii = iis + 3
lx1 = [0, 0]
ly1 = [ii*dh, (ii+1)*dh]
ii = iis + 5
lx2 = [0, 0]
ly2 = [ii*dh, (ii+1)*dh]
ii = iis+10
lx3 = [0, 0]
ly3 = [ii*dh, (ii+1)*dh]
ii = iis+12
lx4 = [0, 0]
ly4 = [ii*dh, (ii+1)*dh]
ii = iis+20
lx5 = [0, 0]
ly5 = [ii*dh, (ii+1)*dh]
ii = iis+24
lx6 = [0, 0]
ly6 = [ii*dh, (ii+1)*dh]
ii = iis+22.5
lx7 = [-4, -6, -6, -5]
ly7 = [ii*dh, ii*dh, (iis+11.5)*dh, (iis+11.5)*dh]
ii = iis+29
lx8 = [0, 0, -7, -7, -5]
ly8 = [ii*dh, (ii+1)*dh, (ii+1)*dh, (iis+4.5)*dh, (iis+4.5)*dh]
plt.plot(lx1, ly1, 'k-', lw=0.5)
plt.plot(lx2, ly2, 'k-', lw=0.5)
plt.plot(lx3, ly3, 'k-', lw=0.5)
plt.plot(lx4, ly4, 'k-', lw=0.5)
plt.plot(lx5, ly5, 'k-', lw=0.5)
plt.plot(lx6, ly6, 'k-', lw=0.5)
plt.plot(lx7, ly7, 'k-', lw=0.5)
plt.plot(lx8, ly8, 'k-', lw=0.5)

# Draw arrow
xx = []
yy = []
dx = []
dy = []
hl = 0.3
hw = 0.2
lww = 0.5
x1 = 0
ddx = 0
ddy = 0.01
ii = iis + 4
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis + 6
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis+11
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis+13
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis+21
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis+25
xx = xx+[x1]
yy = yy+[ii*dh-hl]
dx = dx+[ddx]
dy = dy+[ddy]
x1 = -5-hl
ddy = 0
ddx = 0.01
ii = iis+11.5
xx = xx+[x1]
yy = yy+[ii*dh]
dx = dx+[ddx]
dy = dy+[ddy]
ii = iis + 4.5
xx = xx+[x1]
yy = yy+[ii*dh]
dx = dx+[ddx]
dy = dy+[ddy]
for i in range(0, len(xx)):
    ax1.arrow(xx[i], yy[i], dx[i], dy[i], lw=lww,
              head_width=hw, head_length=hl, fc='k', ec='k')

plt.savefig(fnameF, dpi=200, bbox_inches="tight", pad_inches=0.2)
#plt.show()
