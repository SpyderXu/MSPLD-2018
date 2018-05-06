import pdb, os, sys
import os.path as osp
from collections import defaultdict
import cv2
import glob


class ImageNet(object):
  def __init__(self, name, cls, description):
    self.name = name
    self.cls  = cls
    self.description = description
    self.voc  = None
    self.coco = None

  def convert(self):
    string = '{:9s} {:4d} {:20s}'.format(self.name, self.cls, self.description)
    if self.voc is not None:
      string = string + ' {:10s}'.format(self.voc)
      if self.coco is not None:
        string = string + ' {:10s}'.format(self.coco)
  
  def load(self, string):
    alls = string.strip().split(' ')
    alls = [x for x in alls if x is not '']
    assert len(alls)>=3 and len(alls)<=5
    self.name = alls[0]
    self.cls  = int(alls[1])
    self.description = alls[2]
    if len(alls) > 3:
      self.voc = alls[3]
    if len(alls) > 4:
      self.coco = alls[4]

def load_txt(path):
  cfile = open(path, 'r')
  objects = []
  for line in cfile.readlines():
    x = ImageNet(None, None, None)
    x.load(line)
    objects.append(x)
  cfile.cloes()
  return objects
 
def load_meta(txt):
  cfile = open(txt, 'r')
  meta  = defaultdict(list)
  for line in cfile.readlines():
    line = line.strip()
    line.split(':')
    assert len(line) == 2, 'Line : {:}'.format(line)
    meta[line[0]] = line[1].split(', ')
  cfile.close()
  return meta

def main(in_file, out_file, idir):
  objects = load_txt(in_file)
  for obj in objects:
    cdir = osp.join(idir, obj.name)
    assert osp.isdir(cdir), 'Sub-dir {:} not find'.format(cdir)
    


if __name__ == '__main__':
  imagenet_dir = '~/datasets/ILSVRC2012/val'
  in_file = 'ImageNet-1000.txt'
  out_file = 'ImageNet-1000.new'
  coco = load_meta('ms-coco')
  pvoc = load_meta('pascal_voc')
  pdb.set_trace()
  main(in_file, out_file, imagenet_dir)
