import sys
import argparse
import pickle
import os
from pathlib import Path
sys.path.append(os.environ['HOME']+'/half/genetic_algorithms/darts/cnn')

root_dir = os.getcwd()

local_path = root_dir 
sys.path.insert(0, local_path)

log_dir_default = local_path + "/log"
Path(log_dir_default).mkdir(parents = True, exist_ok = True)

HOME = os.environ['HOME']

from genetic.gene_evolution import GeneEvolution
from genetic import cellbody_templates
from genetic.cell_body import CellBody

from darts.cnn import train_search
from darts.cnn import train

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def check_objective_settings(objectives_list):
    sum_weights = 0 
    for objective in objectives_list:
        sum_weights += objective['weight']
        if objective['training_necessary'] == True and objective['mutation_relevant'] == True:
            raise AttributeError("This configuration is not possible!")
    if abs(sum_weights - 1) > 0.000001:
        raise ValueError("Sum of all weights must be one!")

def filter_mutation_relevant_objectives(objectives_list):
    relevant_objectives = []
    for objective in objectives_list:
        if objective['mutation_relevant'] == True and objective['weight'] > 0:
            objective_dict = {}
            objective_dict[objective['name']] = objective['range']
            relevant_objectives.append(objective_dict)
    return relevant_objectives


# Parse commandline arguments
parser = argparse.ArgumentParser(description='Evolving CNN structures using Genetic or DARTS Algorithm')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--log_dir', default=log_dir_default, help='directory for logging')

subparsers = parser.add_subparsers(dest='algo_flag')

# Parse commandline arguments for genetic algorithm
parser_genetic = subparsers.add_parser('genetic')
parser_genetic.add_argument('--number_of_offsprings', '-l', type=int, default=2, help='Num. of offsprings')
parser_genetic.add_argument('--number_of_parents', '-p', type=int, default=2,help='Num. of population')
parser_genetic.add_argument('--number_of_parents_initial_population', '-i', type=int, default=1,help ='Num of initial population')
parser_genetic.add_argument('--selection_policy', '-s', default='all_children',help='selected_children -  \
                                                                            all_children - \
                                                                            lemonade - \
                                                                            aging')
# parser_genetic.add_argument('--fitness_measure', '-m', default='accuracy',help='normalized_worst_objetive -  \
#                                                                            weighted_objectives - \
#                                                                            accuracy')                                                                      
parser_genetic.add_argument('--result_dir', default='./results', help='name of results folder')
parser_genetic.add_argument('--run_type', '-r', default='local', help='sequential: local - parallel: dart')
parser_genetic.add_argument('--nodefile', '-f', default=HOME+'/nodefile', help='nodefile')
parser_genetic.add_argument('--generations', '-g', default=1, type=int, help='nodefile')
parser_genetic.add_argument( '--use_gpu' # this parameter can be combined with "gpu" in DARTS
                           , '-u'
                           , type = str2bool
                           , nargs = '?'
                           , const = True
                           , default = True
                           , help='use GPUs if available'
                           )

# Parse commandline arguments for DARTS algorithm
parser_darts = subparsers.add_parser('darts')
parser_darts.add_argument('--data', type=str, default='../data', help='location of the data corpus')

# the floowing parameters could also be reused by the genetic algorithm
parser_darts.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser_darts.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser_darts.add_argument('--report_freq', type=float, default=200, help='report frequency')
parser_darts.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser_darts.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser_darts.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
########################################################################

parser_darts.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser_darts.add_argument('--layers', type=int, default=8, help='total number of layers')
parser_darts.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss')
parser_darts.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser_darts.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser_darts.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser_darts.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser_darts.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser_darts.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser_darts.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser_darts.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')

args = parser.parse_args()
print(args)

if args.algo_flag=='genetic':
    
    print("Logging to: " + args.log_dir)
    # Define cellbody specifications
    default_cellbody_spec = cellbody_templates.default_cellbody_spec()
    seed_cellbody_one_spec = cellbody_templates.seed_cellbody_one_spec()
    seed_cellbody_two_spec = cellbody_templates.seed_cellbody_two_spec()

    # optionally, augment with windowing
    WINDOW_FACTOR = None
    if WINDOW_FACTOR is not None and WINDOW_FACTOR != 0:
        seed_cellbody_one_spec = cellbody_templates.augment_windowing(seed_cellbody_one_spec, WINDOW_FACTOR)

    #build cellbodies
    default_cellbody = CellBody(default_cellbody_spec)
    seed_cellbody_one = CellBody(seed_cellbody_one_spec)
    seed_cellbody_two = CellBody(seed_cellbody_two_spec)

    seed_cellbodys = [seed_cellbody_one, seed_cellbody_two]

    # options for genetic selection strategies with weighted sum as fitness measure
    objectives_settings = []
    quantization_settings = []
    # specify here all possible objectives if unnecessary --> weight to zero
    # all objectives must be between zero and one; zero best
    # threshold: above this value improvment is irrelevant
    # desired value must be normalized between zero and one
    objectives_settings.extend( [ {'name': 'error', 'training_necessary': True, 'mutation_relevant': False, 'range': [0, 1], 'weight': 0.0, 'desired_value': 0 }
                                , {'name': 'number_parameters', 'training_necessary': False, 'mutation_relevant': True, 'range': [0, 100000], 'weight': 0.1, 'desired_value': 0 }
                                , {'name': 'non_detection_rate', 'training_necessary': True, 'mutation_relevant': False, 'range': [0, 1], 'weight': 0.3, 'desired_value': 0.1  }
                                , {'name': 'false_alarm_rate', 'training_necessary': True, 'mutation_relevant': False, 'range': [0, 1], 'weight': 0.3, 'desired_value': 0.2 }
                                , {'name': 'maximal_memory_consumption_layerwise', 'training_necessary': False, 'mutation_relevant': False, 'range': [100000, 2000000], 'weight': 0.1, 'desired_value': 0 }
                                , {'name': 'inference_speed_fpga', 'training_necessary': False, 'mutation_relevant': False, 'range': [0, 100000], 'weight': 0.1, 'desired_value': 0 }
                                , {'name': 'floating_point_operations', 'training_necessary': False, 'mutation_relevant': False, 'range': [0, 200000000], 'weight': 0.1, 'desired_value': 0}
                                , {'name': 'validation_loss', 'training_necessary': True, 'mutation_relevant': False, 'range': [0, 10], 'weight': 0.0, 'desired_value': 0 } #range for loss function ?
                                ]
                            )

    quantization_settings.extend( [{'name': 'weights_avg_bitwidth', 'range': [0, 32], 'desired_value': 2 }])
    #in case of genetic algorithms desired value is threshold, in case of lemoade_weighted selection target point
    check_objective_settings(objectives_settings)

    def check_mutation_settings(mutation_settings):
        possible_keys_mutation_settings = ['random_mutation', 'normal_mutation', 'active_mutation', 'inactive_mutation']
        necessary_keys_mutation_settings = ['variance', 'mutation_objectives']
        len_necessary_keys = len(necessary_keys_mutation_settings)
        counter_possible_keys = 0
        counter_necessary_keys = 0
        for setting in mutation_settings.keys():
            if setting in possible_keys_mutation_settings:
                counter_possible_keys += 1
            if setting in necessary_keys_mutation_settings:
                counter_necessary_keys +=1
        if not (counter_necessary_keys == len_necessary_keys and counter_possible_keys > 0):
            raise Exception("mutation settings have the wrong format!")
            
    mutation_settings = { 'random_mutation': [0.6, 1] 
                        , 'normal_mutation': [0.1, 0.7]
                        , 'active_mutation': [0, 0.4]
                        , 'inactive_mutation': [0, 0.1]
                        , 'variance': 0.1 #introduce some randomness for choosing mutation type
                        , 'mutation_objectives': filter_mutation_relevant_objectives(objectives_settings)
                        }
    # Check for wrong user input
    if args.selection_policy == 'lemonade' and args.number_of_offsprings < args.number_of_parents:
        raise Exception("In Lemonade, number of offsprings must be equal or greater than number_of_parents")

    # Execute evolution
    cgp = GeneEvolution ( default_cellbody
                        , seed_cellbodys
                        , working_dir = local_path
                        , root_dir = root_dir
                        , verbose = False
                        , number_of_epochs = args.epochs
                        , batchsize = args.batch_size
                        , learning_rate = args.learning_rate
                        , momentum = args.momentum
                        , dataset_name = 'half_dataset'
                        , number_of_offsprings = args.number_of_offsprings
                        , number_initial_population = args.number_of_parents_initial_population
                        , max_number_of_generations = args.generations
                        , number_population = args.number_of_parents
                        , selection_policy = args.selection_policy
                        , objectives_settings = objectives_settings
                        , quantization_settings = quantization_settings
                        , mutation_settings = mutation_settings
                        , run_type = args.run_type
                        , log_dir = args.log_dir
                        , result_dir = args.result_dir
                        , nodefile = args.nodefile
                        , use_gpu = args.use_gpu
                        )
    cgp.run()

    
elif args.algo_flag=='darts':
#     train.main(args)
    train_search.main(args)