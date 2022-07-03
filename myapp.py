import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#plt.ion()

#matplotlib.use('qt5agg')
import numpy as np

import pykrige.kriging_tools as kt
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

import panel as pn
from panel.interact import interact, interactive, fixed, interact_manual
import panel.widgets as pnw


def inter_plot(df, portee, scl=20, pep=None):
    OK = OrdinaryKriging(
    df['X'],
    df['Y'],
    df['Lux'],
    variogram_model="linear",
    verbose=False,
    enable_plotting=False,
    )
    
    #UK = UniversalKriging(
    #df['X'],
    #df['Y'],
    #df['Lux'],
    #variogram_model="linear",
    #verbose=False,
    #enable_plotting=False,
    #)
    
    pp = OK.variogram_model_parameters
    print(pp)
    if pep != None:
        pp[0]=pep
    if portee != None:
        pp[1]=portee
    
    OK.variogram_model_parameters = pp
    
    print("modified: ", OK.variogram_model_parameters)
    
    gridx = np.arange(df['X'].min()-scl, df['X'].max()+scl, scl)
    gridy = np.arange(df['Y'].min()-scl, df['Y'].max()+scl, scl)
    z, ss = OK.execute("grid", gridx, gridy)
    
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # the scatter plot:
    ax.imshow(z,  extent =[df['X'].min()-scl-scl/2, df['X'].max()+scl-scl/2, df['Y'].min()-scl-scl/2, df['Y'].max()+scl-scl/2], origin='lower'  )
    ax.scatter(df['X'],df['Y'],marker='+',color='r')

    # Set aspect of the main axes.
    #ax.set_aspect(1.)

    # create new axes on the right and on the top of the current axes
    divider = make_axes_locatable(ax)
    # below height and pad are in inches
    ax_histx = divider.append_axes("top", 1.2, pad=0.1, sharex=ax)
    ax_histy = divider.append_axes("right", 1.2, pad=0.1, sharey=ax)

    # make some labels invisible
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax_histx.plot(df['X'], df['Lux'], 'o-')
    ax_histy.plot(df['Lux'], df['Y'], 'o-')
    
    return fig

fle = open('L:\\data\\iod\\data4.csv')
colName = (fle.readline()[:-1]).split(',')
df = pd.DataFrame(columns=colName)
print(df.size)
for l in fle:
    s = l.split(',')
    #print(s)
    tstp = float(s[0])
    X = float(s[1])
    Y= float(s[2])
    rge = s[3][-2] #there is an eol character
    #print(X,Y,rge)
    e = 10**(-4+int(rge))
    lux = int(s[3][:-2])*e
    df=df.append({'timestamp':tstp,'X':X,'Y':Y,'Lux':lux},ignore_index=True)
fle.close()

scale  = pnw.FloatSlider(name='scl', value=100, start=10, end=500)
portee  = pnw.FloatSlider(name='portee', value=1000, start=0.5, end=20000)

reactive_interp = pn.bind(inter_plot, df, portee, scale, pep=None)

widgets   = pn.Column("<br>\n# Interpolation parameters", scale, portee)
interpane = pn.Row(reactive_interp, widgets)

interpane.servable()

