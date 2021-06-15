#     if args.scale_type == "stage":
#         pass
#     elif args.scale_type == "cell":
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_pretrained_False_time_20210101-092524/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_GC_0_time_20210217-111945/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_GC_0_time_20210211-203754/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_GC_0_time_20210217-210823/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_GC_0_time_20210217-210847/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_55_div_0.5_GC_0_time_20210213-201338/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_60_div_0.5_GC_0_time_20210213-201245/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_65_div_0.5_GC_0_time_20210213-201230/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_70_div_0.5_GC_0_time_20210213-201132/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_70_div_0.5_GC_0_time_20210217-105201/full_weights" # w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_70_div_0.5_GC_0_time_20210217-211036/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_70_div_0.5_GC_0_time_20210217-211054/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_75_div_0.5_GC_0_time_20210211-204016/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_75_div_0.5_GC_0_time_20210219-092418/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_80_div_0.5_GC_0_time_20210219-090357/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_85_div_0.5_GC_0_time_20210219-092509/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_90_div_0.5_GC_0_time_20210219-092311/full_weights" #w/o cutout in search
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_95_div_0.5_GC_0_time_20210219-092556/full_weights" #w/o cutout in search
        
#         """w/o cutout"""
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210117-212828/full_weights" 
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210128-135458/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210128-135603/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210211-120559/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210211-120623/full_weights"
#         """w/ cutout"""
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210120-111931/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210125-103025/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210125-103214/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210211-120447/full_weights"
#         model_to_discretize = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_12000_div_1_GC_0_time_20210211-120516/full_weights"

cell_half_mu_50 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_50_div_0.5_GC_0_time_20210217-210847" #w/o cutout in search

cell_mul_half_mu_5e_minus_3 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_0.005_mul_0.5_GC_0_time_20210320-212613"

cell_mul_half_mu_2e_minus_3 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_0.002_mul_0.5_GC_0_time_20210321-191032"

cell_mul_half_mu_2e_minus_3_RUN2 = "search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_0.002_mul_0.5_time_20210329-092526"

cell_mul_half_mu_2e_minus_3_RUN3 = "search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_0.002_mul_0.5_time_20210329-092609"

cell_mul_half_mu_3e_minus_3 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_0.003_mul_0.5_GC_0_time_20210321-191104"

cell_mul_half_mu_4e_minus_3 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_0.004_mul_0.5_GC_0_time_20210321-191125"

cell_mul_half_mu_1e0 = "search-for-cell/search-for-cell-lr_0.001_rho_0.8_mu_1_mul_0.5_GC_0_time_20210320-212435"

cell_half_mu_100 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_100.0_div_0.5_time_20210416-194217'

cell_half_mu_100_RUN2 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_100.0_div_0.5_time_20210424-234638'

cell_half_mu_3e_minus_3_bn_RUN2 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_0.003_mul_0.5_time_20210423-162638'

cell_half_mu_2e_minus_3_bn_RUN2 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_0.002_mul_0.5_time_20210423-151324'

cell_half_mu_125_minus_2_bn = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_1.25_none_0_time_20210425-094930'

cell_half_mu_1_bn = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_1.0_none_0_time_20210424-231154'

cell_none_mu_175_minus_2_bn = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_1.75_none_0_time_20210426-075501'

cell_div_half_mu_60_bn = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210424-235437'
cell_div_half_mu_60_bn_RUN2 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_60.0_div_0.5_time_20210502-195444'

cell_div_half_mu_70_bn = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_70.0_div_0.5_time_20210424-235539'

cell_none_mu_1p5 = 'search-for-cell/search-for-cell-lr_0.001_momentum_0.8_mu_1.5_none_0_time_20210426-075409'