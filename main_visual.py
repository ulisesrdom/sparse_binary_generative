# -*- coding: utf-8 -*-
import argparse
import visual_results_functions as vis_res_func

# python main_visual.py -IFOLDERS "" -OFOLDER "" -MAP_TYPE_PATCHES "bwr" -MAP_TYPE_BASIS "bwr" -ROWS_PATCHES 12 -COLS_PATCHES 12 -ROWS_BASIS 3 -COLS_BASIS 5 -MAX_SAMPLES 5000 -FS 50 -DPI 500 -LSTY "dotted" -COLORS "darkseagreen" -LL "Pop. counts" -N 18 -M 10 -S_ID1 0 -S_ID2 9950 -NLOCAL 2 -NBINS_PW 20 -EVERY_PW 50 -SHORT_O 1 -MODEL_T 4

# --------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# Read parameters ----------------------------------
# --------------------------------------------------


ap.add_argument("-IFOLDERS", "--IFOLDERS", required = True, help="String with comma separated input folders (where the pickle data was saved for each model).")
ap.add_argument("-OFOLDER", "--OFOLDER", required = True, help="Output folder for all the plots to be generated.")
ap.add_argument("-MAP_TYPE_PATCHES", "--MAP_TYPE_PATCHES", required = True, help="Color map to apply for the image patches ('hot', 'coolwarm', 'bwr').")
ap.add_argument("-MAP_TYPE_BASIS",   "--MAP_TYPE_BASIS", required = True, help="Color map to apply for the basis functions ('hot', 'coolwarm', 'bwr').")
ap.add_argument("-MAP_TYPE_HEATMAPS","--MAP_TYPE_HEATMAPS", required = True, help="Color map to apply for the heat maps ('hot', 'coolwarm', 'bwr', 'bone').")
ap.add_argument("-ROWS_PATCHES",     "--ROWS_PATCHES",  required = True, help="Number of rows for the image patch plot.")
ap.add_argument("-COLS_PATCHES",     "--COLS_PATCHES",  required = True, help="Number of columns for the image patch plot.")
ap.add_argument("-ROWS_BASIS",     "--ROWS_BASIS",  required = True, help="Number of rows for the basis functions plots.")
ap.add_argument("-COLS_BASIS",     "--COLS_BASIS",  required = True, help="Number of columns for the basis functions plots.")
ap.add_argument("-MAX_SAMPLES", "--MAX_SAMPLES", required = True, help="Maximum number of patch observations used.")
ap.add_argument("-FS",  "--FS", required = True, help="Frequency of saving parameter (used in main_model.py).")
ap.add_argument("-DPI", "--DPI", required = True, help="Dots per inch for the quality of the images.")
ap.add_argument("-LSTY", "--LINE_STYLES", required = True, help="String with comma separated line styles for each model in the model comparison plot.")
ap.add_argument("-COLORS", "--COLORS", required = True, help="String with comma separated colors for each model in the model comparison plot.")
ap.add_argument("-LL", "--LABEL_LEGENDS", required = True, help="String with comma separated label legends for each plotting case.")
ap.add_argument("-LP","--LABELS_PANELS", required = True, help="String with comma separated label legends for each panel.")
ap.add_argument("-VNL","--VALUES_NL", required = True, help="String with comma separated values for the inhibition effects plots (omega,bias).")
ap.add_argument("-N",   "--N",  required = True, help="Number of neurons in the population.")
ap.add_argument("-M",   "--M",  required = True, help="Number of samples used per observation.")
ap.add_argument("-S_ID1",   "--S_ID1",   required = True, help="First sample number to consider.")
ap.add_argument("-S_ID2",   "--S_ID2",   required = True, help="Up to which sample number to consider.")
ap.add_argument("-FUNC_M", "--FUNCTIONS_MASK", required=True, help="String with 7 comma separated binary values (0 or 1) to indicate a call to each plot function. The plot functions in order are: sample observations, log-average marginal likelihood, basis functions, similarity of basis functions matrix, histogram of population counts, heat maps, mean-field model time.")
ap.add_argument("-MODEL_T", "--MODEL_T", required = True, help="Dendritically modulated generative model (1), dendritically modulated mean-field generative model (2).")

args = vars(ap.parse_args())


IFOLDERS         = str(args['IFOLDERS']).split(',')
OFOLDER          = str(args['OFOLDER'])
MAP_TYPE_PATCHES = str(args['MAP_TYPE_PATCHES'])
MAP_TYPE_BASIS   = str(args['MAP_TYPE_BASIS'])
MAP_TYPE_HEATMAPS= str(args['MAP_TYPE_HEATMAPS'])
ROWS_PATCHES     = int(args['ROWS_PATCHES'])
COLS_PATCHES     = int(args['COLS_PATCHES'])
ROWS_BASIS       = int(args['ROWS_BASIS'])
COLS_BASIS       = int(args['COLS_BASIS'])
MAX_SAMPLES      = int(args['MAX_SAMPLES'])
FS               = int(args['FS'])
DPI              = int(args['DPI'])
LINE_STYLES      = str(args['LINE_STYLES']).split(',')
COLORS           = str(args['COLORS']).split(',')
LABEL_LEGENDS    = str(args['LABEL_LEGENDS']).split(',')
LABELS_PANELS    = str(args['LABELS_PANELS']).split(',')
VALUES_NL_str    = str(args['VALUES_NL']).split(',')
N                = int(args['N'])
M                = int(args['M'])
S_ID1            = int(args['S_ID1'])
S_ID2            = int(args['S_ID2'])
FUNC_M_str       = str(args['FUNCTIONS_MASK']).split(',')
FUNC_M           = []
for i in range(0,7):
   FUNC_M.append( int(FUNC_M_str[i]) )
MODEL_T          = int(args['MODEL_T'])

print("INPUT FOLDERS = {}".format(IFOLDERS))

# ------------------SUBSET OF USED IMAGE PATCHES ---------------------------------
if FUNC_M[0] == 1 :
   vis_res_func.plot_observations( IFOLDERS[0], OFOLDER, MAP_TYPE_PATCHES, ROWS_PATCHES,COLS_PATCHES )
# ------------------LOG-AVERAGE MARGINAL LIKELIHOOD COMPARISON BETWEEN MODELS ----
if FUNC_M[1] == 1 :
   vis_res_func.plot_log_average_marginal_likelihood_models( IFOLDERS, OFOLDER, MAX_SAMPLES,FS, DPI, LINE_STYLES, COLORS, LABEL_LEGENDS )
# ------------------BASIS FUNCTIONS DURING LEARNING ------------------------------
if FUNC_M[2] == 1 :
   vis_res_func.plot_basis_functions( IFOLDERS[0], OFOLDER, S_ID1, S_ID2, FS, MAP_TYPE_BASIS, ROWS_BASIS,COLS_BASIS )
# ------------------SIMILARITY MATRIX BETWEEN LEARNED BASIS FUNCTIONS ------------
if FUNC_M[3] == 1 :
   vis_res_func.plot_similarity_matrix( IFOLDERS[0], OFOLDER, S_ID2, MAP_TYPE_BASIS, DPI, N )
# ------------------HISTOGRAM OF BINARY POPULATION COUNTS ------------------------
if FUNC_M[4] == 1 :
   vis_res_func.plot_binary_pop_counts( IFOLDERS[0], OFOLDER, MAX_SAMPLES, LINE_STYLES, COLORS, LABEL_LEGENDS, FS, DPI, N, MODEL_T )
if FUNC_M[5] == 1 :
   vis_res_func.plot_heatmaps_fixed_inhibitions( OFOLDER, DPI, MAP_TYPE_HEATMAPS, LABELS_PANELS )
   vis_res_func.plot_inhibition_effects_hist_basis( OFOLDER, MAX_SAMPLES, LINE_STYLES, COLORS, MAP_TYPE_BASIS,LABELS_PANELS, FS, DPI, N, float(VALUES_NL_str[0]),float(VALUES_NL_str[1]),MODEL_T )
if MODEL_T == 2 and FUNC_M[6] == 1 :
   vis_res_func.plot_mf_time( IFOLDERS, OFOLDER, DPI, LINE_STYLES, COLORS, LABEL_LEGENDS )
