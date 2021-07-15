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
    
  model_params = utils_sparsenas.group_model_params_by_stage(model, network_search, mu=args.weight_decay)
  utils_sparsenas.plot_op_norm_across_stages(model_params, model_folder+"/_testtest_stage.png")

  model_params = utils_sparsenas.group_model_params_by_cell(model, network_search, mu=args.weight_decay)
  utils_sparsenas.plot_op_norm_across_cells(model_params, model_folder+"/_testtest_cell.png")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser("cifar")
    parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
    parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')    
    parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
    parser.add_argument('--layers', type=int, default=20, help='total number of layers')
    args = parser.parse_args()
    
    model_folder = "darts-original/darts-original-retrain-predefined-1"
    model_name = "full_weights"
    
    main(args, model_folder, model_name)