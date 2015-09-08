"""
Helper module for displaying ROOT canvases in ipython notebooks on nersc

Usage example:
    # Save this file as rootnotes.py to your working directory.
    
    import rootnotes
    c1 = rootnotes.default_canvas()
    fun1 = TF1( 'fun1', 'abs(sin(x)/x)', 0, 10)
    c1.SetGridx()
    c1.SetGridy()
    fun1.Draw()
    c1

More examples: http://mazurov.github.io/webfest2013/

@author alexander.mazurov@cern.ch
@author andrey.ustyuzhanin@cern.ch
@date 2013-08-09
"""

import ROOT
ROOT.gROOT.SetBatch()

import tempfile
import os
import shutil
from IPython.core import display


def canvas(name="icanvas", size=(800, 600)):
    """Helper method for creating canvas"""
    #remove old temp files first
    clearCanvas()
    
    # Check if icanvas already exists
    if (os.path.exists('./tmpNB')==False):
        os.mkdir('./tmpNB')
    canvas = ROOT.gROOT.FindObject(name)
    assert len(size) == 2
    if canvas:
        return canvas
    else:
        return ROOT.TCanvas(name, name, size[0], size[1])


def default_canvas(name="icanvas", size=(800, 600)):
    """ depricated """
    return canvas(name=name, size=size)


def rtshow(canvas,size=(800,600)):
    #file = tempfile.NamedTemporaryFile(suffix=".png")
    #canvas.SaveAs(file.name)
    #ip_img = display.Image(filename=file.name, format='png', embed=True)
    #return ip_img._repr_png_()   
    #remove old temp files first
    clearCanvas()
    
    file = tempfile.NamedTemporaryFile(suffix=".pdf",dir='./tmpNB',delete=0)
    canvas.SaveAs(file.name)
    path=os.path.abspath('.')
    return PDF(file.name.partition(path+'/')[-1],size) #size=(600,400)

def _display_any(obj):
    #file = tempfile.NamedTemporaryFile(suffix=".png")
    #obj.Draw()
    #ROOT.gPad.SaveAs(file.name)
    #ip_img = display.Image(filename=file.name, format='png', embed=True)
    #return ip_img._repr_png_()
    file = tempfile.NamedTemporaryFile(suffix=".pdf",dir='./',delete=1)
    obj.Draw()
    ROOT.gPad.SaveAs(file.name)
    path=os.path.abspath('.')
    PDF(file.name.partition(path+'/')[-1]) #size=(600,400)

def clearCanvas():
    if (os.path.exists('./tmpNB')==True):
        shutil.rmtree('./tmpNB')
    
class PDF(object):
    def __init__(self, pdf, size=(200,200)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)

# register display function with PNG formatter:

