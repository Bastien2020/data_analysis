import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import param
import io
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

import holoviews as hv

#pn.extension()
hv.extension('bokeh')


def inter_plot_HV(df, portee, scl=20, pep=None):
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
    
    ds = hv.Dataset((gridx, gridy, z), ['x','y'], 'Light Intensity')
    sc = hv.Scatter((df['X'], df['Y'], df['Lux']),['x'], ['y', 'Light Intensity'])
    crvX = hv.Curve((df['X'],df['Lux']),'x','Light Intensity').opts(yaxis='left')
    crvY = hv.Curve((df['Y'],df['Lux']),'y','Light Intensity')
    
    #layout = (ds.to(hv.Image,['x','y']).opts(colorbar=True, cmap='gray')*sc<<crvY<<crvX)
    layout = ((crvX.opts(xaxis='top')+hv.Empty()+(ds.to(hv.Image,['x','y']).opts(colorbar=True, cmap='gray', tools=['hover'])*sc)+crvY.opts(yaxis='right',invert_axes=True)).cols(2))
    
    return layout


class DisplayLight(param.Parameterized):
    data = param.DataFrame()
    
    file_input = param.Parameter()
    file_selector = param.Parameter()
    
    fig = param.Parameter()
    portee  = param.Number(8.2, bounds=(0.5, 10000))
    scl  = param.Number(50, bounds=(0.1, 1000))
    
    #boundX = param.Range(default=(0,100), bounds=(0,100))
    #boundY = param.Range(default=(0,50), bounds=(0,50))
    #boundX = param.Range((0,100), bounds=(0,100))
    #boundY = param.Range((0,50), bounds=(0,50))
    
    #imgStade = hv.RGB.load_image("D:\\code\\IoD\\displayKonika\\stade.jpg", kdims=["stadeX","stadeY"]).opts(aspect='equal', invert_axes=True, )
    #boundingBox = hv.Bounds()
    
    def __init__(self, **params):
        self.param.file_input.default = pn.widgets.FileInput()
        self.param.file_selector.default = pn.widgets.FileSelector('.',file_pattern='*.csv')
        super().__init__(**params)
        #self.fig, ax = plt.subplots(figsize=(5.5, 5.5))
        self.fig= None
        
        imgStade1 = hv.RGB.load_image("stade.jpg", array=True)
        self.imgStade = hv.RGB(imgStade1.transpose((1,0,2)),bounds=(0,0,50,50*imgStade1.shape[1]/imgStade1.shape[0]), kdims=['stadeX','stadeY']).opts(aspect='equal')
        
        self.thet = 0 #rotation angle from anchor system related to stadium system
        self.scaleCoord = 0.001 #
        self.pos = np.array([0,0]) #position of first anchor in stadium coordinate
        
        bd = self.imgStade.bounds.lbrt()
        self.boundX = param.Range(default=(bd[0],bd[2]), bounds=(bd[0],bd[2]))
        self.boundY = param.Range(default=(bd[1],bd[3]), bounds=(bd[1],bd[3]))
        
        self.boundX = (bd[0],bd[2])
        self.boundY = (bd[1],bd[3])
        
        self.bnd = hv.Bounds((self.boundX[0],self.boundY[0],self.boundX[1],self.boundY[1]), kdims=['stadeX','stadY'])
                               
                               
    
    # convert data frome UWB related to the first anchor in mm to stadium coordinate in proportion of stadium width
    #we could implement a scaling and rotation matrix.
    def toStadiumCoordinate(self,datX, datY):
        matRot = np.array([[np.cos(self.thet), -np.sin(self.thet)], [np.sin(self.thet), np.cos(self.thet)]])
        r = matRot@np.array([datX,datY])
        r *= self.scaleCoord
        return r
      
    @pn.depends("file_selector.value", watch=True)
    def _readLocalFile(self):
        #try: 
          fo = open(self.file_selector.value[0], 'rb')
          strcsv=fo.read()
          fo.close()
          self.file_input.value = strcsv
        #except:
        #  print("error opening file", self.file_selector.value[0])
        #  return
          

    @pn.depends("file_input.value", watch=True)
    def _parse_file_input(self):
        value = self.file_input.value       
        print("barf")
        if value:
            self.test = True
            self.string_io = io.StringIO(value.decode("utf8"))
            data = pd.read_csv(self.string_io, dtype=float)
            #self.data = pd.read_csv(self.string_io, dtype=float)
            self.testplot=False
            
            stadCoord = self.toStadiumCoordinate(data['X'], data['Y'])
            
            data['stadeX'] = stadCoord[0,:]
            data['stadeY'] = stadCoord[1,:]
            
            self.boundX = (stadCoord[0,:].min(),stadCoord[0,:].max())
            self.boundY = (stadCoord[1,:].min(),stadCoord[1,:].max())
        
            self.bnd = hv.Bounds((self.boundX[0],self.boundY[0],self.boundX[1],self.boundY[1]), kdims=['stadeX','stadeY'])
            
            self.data = data
            
            self.get_plot()           
        else:
            print("error")

    @pn.depends('data', 'portee', 'scl', watch=True)
    def get_plot(self):
        #print(data)
        self.testplot=True
        self.fig = inter_plot_HV(self.data, self.portee, self.scl, pep=None)

    #@pn.depends('fig', watch=True)    
    def view(self):
        #return pn.Column(
        #    "## Upload and Plot Data",
        #    self.file_input,
        #    self.fig
        #)
        if self.fig==None:
            return self.imgStade
        else:
            stdPth = hv.Path({'stadeX': self.data['stadeX'], 'stadeY': self.data['stadeY'], 'Lux': self.data['Lux']}, kdims=['stadeX','stadeY'],vdims='Lux')
            return pn.Row((self.imgStade*self.bnd.opts(line_width=3, color='red')*stdPth).opts(width=200),self.fig)
            #return pn.Row(self.fig)


dspl = DisplayLight()

iodApp = pn.Column(pn.Row(dspl.file_input, pn.pane.Markdown("or select locally"), dspl.file_selector),dspl.param.scl, dspl.view)

iodApp.servable()


