import torch, os, pdb
from os import path as osp
from scipy import interpolate
from scipy import io
import matplotlib
import pandas
from pandas import DataFrame
import seaborn as sns
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

legendsize = 50
fontsize = 50
axis_size = 38
width, height, alpha = 6500, 4000, 1
dpi = 200

def figure_draw_v0(videos, ok_points, auc_previou, auc_after, save_path):
  
  figsize = width / float(dpi), height / float(dpi)

  datas = {'names':videos, 'CPM':auc_previou, 'CPM+PAM':auc_after, 'points': ok_points}
  dataframe = DataFrame(datas)
  print (dataframe)
  fig = plt.figure(figsize=figsize)
  sns.set(style='whitegrid')
  sns.set_color_codes('pastel')
  
  #sns.set_color_codes('muted')
  sns.barplot(x='names', y='CPM+PAM', data=dataframe, label='CPM+PAM', color='g', alpha=alpha, order=videos)
  sns.barplot(x='names', y='CPM', data=dataframe, label='CPM', color='r', alpha=alpha, order=videos)
  
  #sns.factorplot(x='names', y='points', data=dataframe, color='b', label='passed points', alpha=alpha, order=videos)
  plt.legend(ncol=2, loc='upper right', frameon=True, fontsize=legendsize)
  #sns.despine(left=True, bottom=True)
  #plt.title(title, fontsize=20)
  plt.xlabel('face size of each video clip on category A of 300-VW', fontsize=fontsize)
  plt.ylabel('AUC @ 0.08 (bar)', fontsize=fontsize)
  plt.ylim([0, 100])
  plt.xticks(fontsize=axis_size)
  plt.yticks(fontsize=axis_size)
  #fig.autofmt_xdate()
  ax = plt.gca()
  for tick in ax.get_xticklabels():
    tick.set_rotation(55)
  ax.set_yticks(np.arange(0, 101, 10))

  ax2 = ax.twinx()
  ax2.plot(datas['points'], linewidth=4)
  ax2.set_yticks(np.arange(0, 61, 6))
  ax2.set_ylabel('number of passed landmarks (blue line)', fontsize=fontsize)
  for tick in ax2.get_yticklabels():
    tick.set_fontsize(axis_size)

  sns.despine(ax=ax, right=True, left=False)
  sns.despine(ax=ax2, left=True, right=False)
  
  print ('Save into {}'.format(save_path))
  fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
  #plt.show()
  plt.close(fig)

def draw_resolution(resolutions, videos, auc_previou, auc_after, nme_previou, nme_after, ok_points, save_path):
  resolutions = np.array(resolutions)
  order_idx = np.argsort(resolutions)
  resolutions = resolutions[order_idx]
  auc_previou, auc_after = np.array(auc_previou), np.array(auc_after)
  nme_previou, nme_after = np.array(nme_previou), np.array(nme_after)
  ok_points = np.array(ok_points)
  auc_previou, auc_after = auc_previou[order_idx], auc_after[order_idx]
  ok_points = ok_points[order_idx]
  auc_gain = auc_after - auc_previou
  videos = videos[order_idx]

  videos = resolutions.tolist()
  videos = ['{}'.format(int(x)) for x in videos]
  
  figure_draw_v0(videos, ok_points, auc_previou, auc_after, save_path + '.pdf')
  #figure_draw_v2(videos, ok_points, auc_previou, auc_after, save_path + '.relative.pdf')
  
if __name__ == '__main__':
  this_dir = osp.dirname(osp.abspath(__file__))
  save_dir = osp.join(this_dir, 'cache_cvpr', 'curve')
  if not osp.isdir(save_dir): os.makedirs(save_dir)
  
  #resolutions = [221.89,190.92,215.94,180.10,145.10,199.98,151.51,137.09,134.73,105.08,126.87,93.53,122.05,146.11,141.25,106.03,106.82,56.06,100.32,94.59,94.71,109.55,169.08,113.52,140.89,155.48,107.89,107.76,71.17,139.15,118.38]
  resolutions = [221.89,190.92,215.94,180.10,145.10,199.98,151.51,137.09,134.73,104.08,126.87,93.53,122.05,146.11,141.25,105.03,106.82,56.06,100.32,94.59,95.71,109.55,169.08,113.52,140.89,155.48,108.19,107.76,71.17,139.15,118.38]
  auc_previou = [77.92, 69.39, 78.00, 67.93, 71.64, 68.15, 64.85, 56.04, 66.86, 57.57, 74.04, 38.84, 24.88, 56.79, 71.46, 66.59, 59.80, 40.94, 72.10, 66.67, 64.87, 47.64, 76.44, 68.79, 73.75, 73.69, 68.73, 71.74, 67.48, 63.07, 58.46]
  nme_previou = [1.77 , 2.45 , 1.77 , 2.57 , 2.27 , 2.55 , 4.27 , 13.12, 2.65 , 3.39 , 2.08 , 5.01 , 6.83 , 4.37 , 2.28 , 2.67 , 3.22 , 11.47, 2.23 , 2.67 , 2.84 , 4.22 , 1.88 ,2.50 , 2.10 , 2.15 , 2.51 , 2.26 , 2.60 , 3.29 , 3.39]
  assert len(resolutions) == 31 and len(auc_previou) == 31 and len(nme_previou) == 31
  
  auc_after = [80.35, 72.58, 79.59, 72.57, 76.50, 70.13, 69.32, 64.02, 68.99, 61.28, 76.24, 52.81, 47.86, 58.60, 75.97, 74.47, 70.20, 47.25, 74.35, 72.46, 68.60, 48.13, 81.96, 68.73, 74.68, 76.02, 73.79, 78.04, 71.31, 65.03, 60.34]
  nme_after = [1.57 , 2.19 , 1.65 , 2.19 , 1.88 , 2.39 , 3.37 , 16.16, 2.48 , 3.10 , 1.90 , 3.83 , 4.22 , 4.22 , 1.92 , 2.04 , 2.38 , 10.87, 2.05 , 2.20 , 2.57 , 4.15 , 1.44 , 2.50, 2.03 , 1.93 ,  2.10 , 1.76 , 2.30 , 2.99 , 3.20]
  assert len(auc_after) == 31 and len(nme_after) == 31

  ok_points = [47.3, 46.4, 46.7, 45.8, 46.4, \
               45.2, 42.5, 39.2, 38.8, 40.0, \
               38.1, 37.2, 21.8, 33.6, 42.8, \
               38.6, 41.3, 29.2, 39.0, 29.1, \
               23.7, 39.4, 39.7, 37.1, 42.2, \
               40.9, 37.8, 36.3, 40.7, 35.5, \
               33.3]
  videos = [114, 124, 125, 126, 150, 158, 401, 402, 505, 506, 507, 508, 509, 510, 511, 514, 515, 518, 519, 520, 521, 522, 524, 525, 537, 538, 540, 541, 546, 547, 548]
  videos = ['{}'.format(x) for x in videos]
  videos = np.array(videos, dtype=object)
  assert len(ok_points) == 31
  save_path = 'resolution'
  draw_resolution(resolutions, videos, auc_previou, auc_after, nme_previou, nme_after, ok_points, save_path)


