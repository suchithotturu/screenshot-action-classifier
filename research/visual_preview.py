"""Render a public synthetic fixture next to its direct pixel descriptors."""
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image,ImageOps
from skimage.feature import hog
from classifier.features import ROOT
from classifier.pixels import describe
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build():
 im=Image.open(ROOT/'demo/01-shirt.png').convert('RGB');f=describe(im)
 gray=cv2.cvtColor(np.asarray(ImageOps.pad(im,(128,256),method=Image.Resampling.BICUBIC,color=(127,127,127))),cv2.COLOR_RGB2GRAY)
 _,viz=hog(gray,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys',visualize=True)
 fig,axes=plt.subplots(1,4,figsize=(14,5.8),layout='constrained')
 axes[0].imshow(im);axes[0].set_title('Synthetic input');axes[0].axis('off')
 axes[1].imshow(viz,cmap='gray',vmin=0,vmax=max(.001,np.percentile(viz,99)));axes[1].set_title('HOG edge orientations');axes[1].axis('off')
 colors=['#bc5a5a','#4e9c8f','#4b70a1']
 for j,name in enumerate(['Hue','Saturation','Value']):
  axes[2].bar(np.arange(8)+j*9,f['color'][j*8:(j+1)*8],color=colors[j],label=name)
 axes[2].set_xticks([3.5,12.5,21.5],['Hue','Saturation','Value']);axes[2].set_ylim(0,1)
 axes[2].set_title('24 normalized color bins');axes[2].set_ylabel('Fraction of pixels')
 a=np.asarray(f['edges']).reshape(4,4);axes[3].imshow(a,cmap='Blues',vmin=0,vmax=max(.1,a.max()))
 for y in range(4):
  for x in range(4):axes[3].text(x,y,f'{a[y,x]:.1%}',ha='center',va='center')
 axes[3].set_title('16 regional edge densities');axes[3].set_xticks(range(4));axes[3].set_yticks(range(4))
 p=ROOT/'reports/visual-study';p.mkdir(parents=True,exist_ok=True);fig.savefig(p/'descriptor-preview.png',dpi=160);plt.close(fig)
 return p/'descriptor-preview.png'


if __name__=='__main__':print(build())
