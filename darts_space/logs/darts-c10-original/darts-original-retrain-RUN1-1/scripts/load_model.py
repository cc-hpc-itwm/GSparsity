import torch
import argparse
import torch.nn as nn
import genotypes
import utils_sparsenas

def main(args, model_folder, model_name):
  CIFAR_CLASSES = 10
    
  criterion = nn.CrossEntropyLoss()
    
  if args.layers == 20:
      from model import NetworkCIFAR as Network
      genotype = eval("genotypes.%s" % args.arch)
      model = Network(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  elif args.layers == 8:
      from model import Network
      model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)    

  model.load_state_dict(torch.load("{}/{}".format(model_folder, model_name)))

  utils_sparsenas.plot_individual_op_norm(model, model_folder+"/_testtest.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')    
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    args = parser.parse_args()
    
    model_folder = "darts-pruning/darts-pruning-lr_0.001_edec_0_rho_0.9_rdec_0_mu_0.05_time_20201204-105738"
    model_name = "full_last_weight_lr_0.001_edec_0_rdec_0_rho_0.9_mu_0.05"
    
    main(args, model_folder, model_name)