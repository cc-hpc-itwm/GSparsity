import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model_search import Network
from architect import Architect
import utils_sparsenas

from ProxSGD_for_cell_search import ProxSGD


def main(args, filename):
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  RUN_ID = "lr_{}_edec_{}_rho_{}_rdec_{}_mu_{}_time_{}".format(args.learning_rate, args.epsilon_decay, args.rho, args.rho_decay, args.weight_decay, time.strftime("%Y%m%d-%H%M%S"))
  args.save = "{}-{}".format(args.save, RUN_ID)
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  CIFAR_CLASSES = 10
    
  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('CARME Slurm ID: {}'.format(os.environ['SLURM_PROCID']))
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()

  for cell_index, m in enumerate(model.cells):
        logging.info("cell index is {}".format(cell_index))
        for name, param in m.named_parameters():
            logging.info("op name is {}".format(name))
  exit()  

  model.load_state_dict(torch.load(filename))

  model_params = utils_sparsenas.group_model_params_by_stage(model, mu=args.weight_decay, reduce_cell_indices = [2, 5], num_edges=14, num_ops=7)
    
  print("finished")
    
  param_dict = {}
  reduce_cell_indices = [2, 5]
  stage_normal = 1
  stage_reduce = 1
  for cell_index, m in enumerate(model.cells):
        logging.info("cell index is {}".format(cell_index))
        
        for name, param in m.named_parameters():
            if "_ops" in name:
                if cell_index in reduce_cell_indices: #reduction cell
                    key_op = "stage_reduce_{}.{}".format(stage_reduce, name)
                else: #normal cell
                    key_op = "stage_normal_{}.{}".format(stage_normal, name)

                if key_op in param_dict:
#                     logging.info("key is in dictionary.")
                    param_dict[key_op].append(param)
                else:
#                     logging.info("key is NOT in dictionary!")
                    param_dict[key_op] = [param]

    #             logging.info("     original name:                {}".format(name))
    #             logging.info("reconstructed name: {}".format(key_op))
            else:
                logging.info(name)
        
        if cell_index in reduce_cell_indices:
            stage_normal += 1
            stage_reduce += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--epsilon_decay', type=float, default=0, help='init epsilon decay')
    parser.add_argument('--rho_decay', type=float, default=0, help='init rho decay')
    parser.add_argument('--rho', type=float, default=0.8, help='init rho')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
    parser.add_argument('--gamma', type=int, default=0, help='init gamma')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
#     parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
    parser.add_argument('--layers', type=int, default=8, help='total number of layers')
#     parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
#     parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
    parser.add_argument('--save', type=str, default='load_search_model', help='experiment name')
    parser.add_argument('--seed', type=int, default=2, help='random seed')
#     parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
#     parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
    parser.add_argument('--remark', type=str, default="no remark", help='remark about the experiment')
    parser.add_argument('--normalization', default=False, help='True for normalized regularization')
    args = parser.parse_args()
    
    file_pretrained_model = "search-for-cell-lr_0.001_edec_0_rho_0.8_rdec_0_mu_0.5_time_20201211-071230/final_weights_at_epoch_100"
    
    main(args, file_pretrained_model)   