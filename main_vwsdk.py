import math
from function import *
import argparse

parser = argparse.ArgumentParser(description='Set the parameters to operate VW-SDK')
parser.add_argument('--ar', default = 512, type = int, help = 'N of rows of the PIM array')
parser.add_argument('--ac', default = 512, type = int, help = 'N of columns of the PIM array')
parser.add_argument('--network', default = 'CNN8', type = str, help = 'Dataset = [VGG13, Resnet18]')
parser.add_argument('--mb', default = 1, type = int, help = 'Memory bit')
parser.add_argument('--wb', default = 1, type = int, help = 'Weight bit')
args = parser.parse_args()

array = [args.ar, args.ac]

network = args.network

if network == 'CNN8' :
  image = [40, 18, 18, 9, 7, 7, 5]
  kernel = [5, 3, 3, 3, 3, 3, 5]
  channel = [3, 24, 32, 32, 64, 64, 64, 256]
  stride = [1, 1, 1, 1, 1, 1, 1]

elif network == 'Resnet20' :
  image = [32, 32, 32, 16, 16, 16, 8, 8, 8] # 원래 마지막에는 2임
  kernel = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
  channel = [64, 64, 64, 128, 128, 128, 256, 256, 256]
  stride = [1, 1, 1, 1, 1, 1, 1, 1, 1]

# network_information(network, image, array, kernel, channel)
result(network, image, array, kernel, channel, stride, args.mb, args.wb)