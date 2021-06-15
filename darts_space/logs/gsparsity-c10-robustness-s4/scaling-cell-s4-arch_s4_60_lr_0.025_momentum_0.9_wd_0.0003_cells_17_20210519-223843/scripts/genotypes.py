from collections import namedtuple
from collections import OrderedDict

primitives_1 = OrderedDict([('primitives_normal', [['skip_connect',
                                                    'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'sep_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'skip_connect'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5'],
                                                    ['dil_conv_3x3',
                                                     'dil_conv_5x5']]),
                             ('primitives_reduct', [['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'dil_conv_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'sep_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['max_pool_3x3',
                                                     'avg_pool_3x3'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5'],
                                                    ['skip_connect',
                                                     'dil_conv_5x5']])])

primitives_2 = OrderedDict([('primitives_normal', 14 * [['skip_connect',
                                                         'sep_conv_3x3']]),
                            ('primitives_reduct', 14 * [['skip_connect',
                                                         'sep_conv_3x3']])])

# primitives_3 = OrderedDict([('primitives_normal', 14 * [['none',
#                                                          'skip_connect',
#                                                          'sep_conv_3x3']]),
#                             ('primitives_reduct', 14 * [['none',
#                                                          'skip_connect',
#                                                          'sep_conv_3x3']])])

primitives_4 = OrderedDict([('primitives_normal', 14 * [['noise',
                                                         'sep_conv_3x3']]),
                            ('primitives_reduct', 14 * [['noise',
                                                         'sep_conv_3x3']])])


primitives_darts = OrderedDict([('primitives_normal', 14 * [['max_pool_3x3',
                                                         'avg_pool_3x3',
                                                         'skip_connect',
                                                         'sep_conv_3x3',
                                                         'sep_conv_5x5',
                                                         'dil_conv_3x3',
                                                         'dil_conv_5x5']]),
                            ('primitives_reduct', 14 * [['max_pool_3x3',
                                                         'avg_pool_3x3',
                                                         'skip_connect',
                                                         'sep_conv_3x3',
                                                         'sep_conv_5x5',
                                                         'dil_conv_3x3',
                                                         'dil_conv_5x5']])
                           ])

spaces_dict = {
    's1': primitives_1,
    's2': primitives_2,
#     's3': primitives_3,
    's4': primitives_4,
    'darts': primitives_darts,
}


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

NASNet = Genotype(
  normal = [
    ('sep_conv_5x5', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 0),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 0),
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
  ],
  normal_concat = [2, 3, 4, 5, 6],
  reduce = [
    ('sep_conv_5x5', 1),
    ('sep_conv_7x7', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('sep_conv_5x5', 0),
    ('skip_connect', 3),
    ('avg_pool_3x3', 2),
    ('sep_conv_3x3', 2),
    ('max_pool_3x3', 1),
  ],
  reduce_concat = [4, 5, 6],
)
    
AmoebaNet = Genotype(
  normal = [
    ('avg_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('sep_conv_3x3', 0),
    ('sep_conv_5x5', 2),
    ('sep_conv_3x3', 0),
    ('avg_pool_3x3', 3),
    ('sep_conv_3x3', 1),
    ('skip_connect', 1),
    ('skip_connect', 0),
    ('avg_pool_3x3', 1),
    ],
  normal_concat = [4, 5, 6],
  reduce = [
    ('avg_pool_3x3', 0),
    ('sep_conv_3x3', 1),
    ('max_pool_3x3', 0),
    ('sep_conv_7x7', 2),
    ('sep_conv_7x7', 0),
    ('avg_pool_3x3', 1),
    ('max_pool_3x3', 0),
    ('max_pool_3x3', 1),
    ('conv_7x1_1x7', 0),
    ('sep_conv_3x3', 5),
  ],
  reduce_concat = [3, 4, 6]
)


"""The following genotypes are prestored in DARTS repository."""
DARTS_V1 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])

DARTS_V2 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])

DARTS = DARTS_V2


"""The following genotypes are obtained by rerunning darts_search."""
RUN1 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

RUN2 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('avg_pool_3x3', 1), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

RUN3 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

RUN1_cutout = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0)], reduce_concat=range(2, 6))

RUN2_cutout = Genotype(normal=[('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))

RUN3_cutout = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('skip_connect', 2)], reduce_concat=range(2, 6))

RUN2_cutout_seed = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=range(2, 6))

RUN3_cutout_seed = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('dil_conv_5x5', 1), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))

RUN2_seed = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('skip_connect', 2)], reduce_concat=range(2, 6))

RUN3_seed = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 2), ('skip_connect', 0), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))

DrNAS = Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

DrNAS_full = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('sep_conv_5x5', 3), ('sep_conv_5x5', 4), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))

