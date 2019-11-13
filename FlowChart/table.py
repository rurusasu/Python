# Table for Flowchart
from math import *
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import *

param=sys.argv
iflag=int(param[1]) # flag for axis drawing
# 0: without axis and grid, 1: with axis and grid

# drawing area: w x h = 16 x 25 imaged A4 paper
xmin=0
xmax=16
ymin=0
ymax=25
dh=0.7 # height of one row
A=[]   # actural y-axis
B=[]   # grid number in y-axis
for i in range(0,int(ymax//dh)+1):
    s='{0:.1f}'.format(dh*i)
    A=A+[float(s)]
    B=B+[i]

fnameF='fig_table.png'
fig = plt.figure()
ax1=plt.subplot(111)
ax1 = plt.gca()
ax1.set_xlim([xmin,xmax])
ax1.set_ylim([ymax,ymin])
aspect = (abs(ymax-ymin))/(abs(xmax-xmin))*abs(ax1.get_xlim()[1] - ax1.get_xlim()[0]) / abs(ax1.get_ylim()[1] - ax1.get_ylim()[0])
ax1.set_aspect(aspect)
if iflag==0: 
    plt.axis('off')
else:
    ax1.tick_params(labelsize=6)
    ax1.xaxis.set_major_locator(MultipleLocator(1))
    ax1.yaxis.set_major_locator(MultipleLocator(dh))
    plt.yticks(A,B)
    plt.grid(which='both',lw=0.3, color='#cccccc',linestyle='-')

############################################
iis=2
title='Basic Equation for Flood Routing'
text1='$\Delta S=(Q_i-Q_o)\cdot \Delta t$'
text2='$\Delta t$ : time interval'
text3='$\Delta S$ : storage accumulated during $\Delta t$'
text4='$Q_i$ : average inflow rate during $\Delta t$'
text5='$Q_o$ : average outflow rate during $\Delta t$'
xv0=0
fsize=5 # fontsize
xs=xv0; ys=iis*dh-0.5*dh; plt.text(xs,ys,title,rotation=0,ha='left',va='center',fontsize=fsize,fontweight='bold')
xs=xv0+2; ys=iis*dh+0.5*dh; plt.text(xs,ys,text1,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+1; ys=iis*dh+1.5*dh; plt.text(xs,ys,text2,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+1; ys=iis*dh+2.5*dh; plt.text(xs,ys,text3,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+1; ys=iis*dh+3.5*dh; plt.text(xs,ys,text4,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+1; ys=iis*dh+4.5*dh; plt.text(xs,ys,text5,rotation=0,ha='left',va='center',fontsize=fsize)


iis=8
xv0=0
xv1=15
llw=0.5  # line width
nt=1
na=3
ii=iis; ys=ii*dh; ye=ys+nt*dh
xs=xv0; xe=xv1; pot1=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
ii=iis+nt; ys=ii*dh; ye=ys+na*dh
xs=xv0; xe=xv1; poa1=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
poly=Polygon(pot1, facecolor='#dddddd', edgecolor='#777777',lw=llw)
ax1.add_patch(poly)
poly=Polygon(poa1, facecolor='#eeeeee', edgecolor='#777777',lw=llw)
ax1.add_patch(poly)
text1='Required Data for Analysis'
text2='1. Reservoir capacity curve (elevation - volume curve)'
text3='2. Inflow rate time history (hydrograph)'
text4='3. Outflow characteristic data (elevation - discharge curve)'
xs=0.5*(xv0+xv1); ys=iis*dh+0.5*dh; plt.text(xs,ys,text1,rotation=0,ha='center',va='center',fontsize=fsize)
xs=xv0+0.5; ys=iis*dh+1.5*dh; plt.text(xs,ys,text2,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+0.5; ys=iis*dh+2.5*dh; plt.text(xs,ys,text3,rotation=0,ha='left',va='center',fontsize=fsize)
xs=xv0+0.5; ys=iis*dh+3.5*dh; plt.text(xs,ys,text4,rotation=0,ha='left',va='center',fontsize=fsize)



# Store texts in list
iis=16
m=3   # number of columns
nt=1  # number of title rows
na=10 # number of contents rows
t = [[None for col in range(m)] for row in range(nt)]
a = [[None for col in range(m)] for row in range(na)]
j= 0;t[j][:]=['Variable'    ,'(Unit)'    ,'Description']
i= 0;a[i][:]=['$t$'         ,'(hour)'    ,'time']
i= 1;a[i][:]=['$EL(t)$'     ,'(EL.m)'    ,'reservoir water level']
i= 2;a[i][:]=['$q_{in}[t]$' ,'(m$^3$/s)' ,'inflow rate']
i= 3;a[i][:]=['$q_{out}[t]$','(m$^3$/s)' ,'outflow rate']
i= 4;a[i][:]=['$Vol[EL]$'   ,'(m$^3$)'   ,'reservoir volume at water level $EL$']
i= 5;a[i][:]=['$R[EL]$'     ,'(m$^3$)'   ,'work variable for reservoir volume']
i= 6;a[i][:]=['$\Delta S$'  ,'(m$^3$)'   ,'storage accumulated during $\Delta t$']
i= 7;a[i][:]=['$h$'         ,'(m)'       ,'water level increlent']
i= 8;a[i][:]=['$elv$'       ,'(EL.m)'    ,'work variable for reservoir water level']
i= 9;a[i][:]=['$\epsilon$'  ,'(m)'       ,'allowable error for convergence']
title='Description of Variables'

# Draw box
xv0= 0.0 # x-coordinate of vertical line_0
xv1= 2.5 # x-coordinate of vertical line_1
xv2= 5.0 # x-coordinate of vertical line_2
xv3=15.0 # x-coordinate of vertical line_3
llw=0.5  # line width
ii=iis; ys=ii*dh; ye=ys+nt*dh
xs=xv0; xe=xv1; pot1=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
xs=xv1; xe=xv2; pot2=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
xs=xv2; xe=xv3; pot3=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
ii=iis+nt; ys=ii*dh; ye=ys+na*dh
xs=xv0; xe=xv1; poa1=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
xs=xv1; xe=xv2; poa2=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
xs=xv2; xe=xv3; poa3=[(xs,ys),(xe,ys),(xe,ye),(xs,ye)]
for pox in [pot1,pot2,pot3]:
    poly=Polygon(pox, facecolor='#dddddd', edgecolor='#777777',lw=llw)
    ax1.add_patch(poly)
for pox in [poa1,poa2,poa3]:
    poly=Polygon(pox, facecolor='#eeeeee', edgecolor='#777777',lw=llw)
    ax1.add_patch(poly)

# Draw text
fsize=5 # fontsize
ii=iis; ys=ii*dh
for i in range(0,nt):
    xs=0.5*(xv0+xv1); j=0; plt.text(xs,ys+(0.5+i)*dh,t[i][j],rotation=0,ha='center',va='center',fontsize=fsize)
    xs=0.5*(xv1+xv2); j=1; plt.text(xs,ys+(0.5+i)*dh,t[i][j],rotation=0,ha='center',va='center',fontsize=fsize)
    xs=0.5*(xv2+xv3); j=2; plt.text(xs,ys+(0.5+i)*dh,t[i][j],rotation=0,ha='center',va='center',fontsize=fsize)
ii=iis+nt; ys=ii*dh
for i in range(0,na):
    xs=xv0+0.5; j=0; plt.text(xs,ys+(0.5+i)*dh,a[i][j],rotation=0,ha='left',va='center',fontsize=fsize)
    xs=xv1+0.5; j=1; plt.text(xs,ys+(0.5+i)*dh,a[i][j],rotation=0,ha='left',va='center',fontsize=fsize)
    xs=xv2+0.5; j=2; plt.text(xs,ys+(0.5+i)*dh,a[i][j],rotation=0,ha='left',va='center',fontsize=fsize)
# Draw title
xs=0.5*(xv0+xv3); ys=iis*dh-0.5*dh; plt.text(xs,ys,title,rotation=0,ha='center',va='center',fontsize=fsize,fontweight='bold')
############################################

plt.savefig(fnameF, dpi=200, bbox_inches="tight", pad_inches=0.2)
#plt.show()