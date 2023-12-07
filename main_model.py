# -*- coding: utf-8 -*-
import numpy as np
import time
import argparse
import model_learning as ml

# --------------------------------------------------------------------------------
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# Read parameters ----------------------------------
# --------------------------------------------------
ap.add_argument("-im_f", "--image_file", required = True, help="Name of input mat image file that will be used for the image patches.")
ap.add_argument("-ou_f", "--output_folder", required = True, help="Output folder to store variables values and results.")
ap.add_argument("-BF_F", "--BASIS_FUNCTIONS_FILE", required = True, help="Name of basis functions file to load (NA for no file).")
ap.add_argument("-MP", "--MAX_PATCHES", required = True, help="Maximum number of patch samples to learn from.")
ap.add_argument("-SS", "--START_SAMPLE", required = True, help="Integer index of initial patch sample to start the learning from in [0,MAX_PATCHES-1].")
ap.add_argument("-PS", "--PATCH_SIZE", required = True, help="Side size for square patches.")
ap.add_argument("-M", "--M", required = True, help="Number of binary vector samples to consider.")
ap.add_argument("-N",  "--N",  required = True, help="Number of neurons to consider in the population equal to the number of basis functions.")
ap.add_argument("-NB", "--NB", required = True, help="Number of bins for the histograms of learnable parameters.")
ap.add_argument("-NLOCAL", "--NLOCAL", required = True, help="Number of local iterations for M-STEP for each observed sample.")
ap.add_argument("-FS", "--FREQUENCY_OF_SAVING", required = True, help="Number of iterations to wait before saving next state of variables.")
ap.add_argument("-ETA_V", "--ETA_VALUES", required = True, help="Comma separated learning rate values (basis functions, distal branch weight).")

ap.add_argument("-F_P", "--F_P", required = True, help="Comma separated initial values for the weights projecting at each proximal dendritic region.")
ap.add_argument("-F_I", "--F_I", required = True, help="Comma separated initial values for the C connections between the interneuron outputs and either distal or somatic regions.")
ap.add_argument("-TAU_I", "--TAU_I", required = True, help="Threshold for inhibiting SOM interneurons or fraction (in [0,1]) of inhibited connections for mean-field model.")
ap.add_argument("-SIGMA", "--SIGMA", required = True, help="Scalar value for the standard deviation.")
ap.add_argument("-C", "--C", required = True, help="Number of apical sub-branches for each gate.")
ap.add_argument("-OMEGA_NL", "--OMEGA_NL", required = True, help="Sensitivity parameter for the dendritic nonlinearity at the proximal region.")
ap.add_argument("-b_NL", "--b_NL", required = True, help="Bias parameter for the dendritic nonlinearity at the proximal region.")

ap.add_argument("-MODEL_T", "--MODEL_T", required = True, help="Generative model (1), generative mean-field model (2).")

args = vars(ap.parse_args())


IMG_FILE         = str(args['image_file'])
OUT_FOLDER       = str(args['output_folder'])
BASIS_F_FILE     = str(args['BASIS_FUNCTIONS_FILE'])
MAX_PATCHES      = int(args['MAX_PATCHES'])
START_SAMPLE     = int(args['START_SAMPLE'])
ps               = int(args['PATCH_SIZE'])
M                = int(args['M'])
N                = int(args['N'])
NB               = int(args['NB'])
NLOCAL           = int(args['NLOCAL'])
FREQ_SAVING      = int(args['FREQUENCY_OF_SAVING'])
ETA_VALS         = args['ETA_VALUES'].split(',')
ETA_PHI          = float(ETA_VALS[0])
ETA_F_I          = float(ETA_VALS[1])
F_P              = args['F_P'].split(',')
F_I              = args['F_I'].split(',')
TAU_I            = float(args['TAU_I'])
SIGMA            = float(args['SIGMA'])
C                = int(args['C'])
OMEGA_NL         = float(args['OMEGA_NL'])
b_NL             = float(args['b_NL'])
MODEL_T          = int(args['MODEL_T'])


if MODEL_T == 1 :
   # Learn dendritically modulated generative model for the given configuration, saving results in OUT_FOLDER folder
   ml.learn_gen_dendritic_model( IMG_FILE, OUT_FOLDER, BASIS_F_FILE, MAX_PATCHES, START_SAMPLE,\
                                 ps, M, N, NB, NLOCAL, FREQ_SAVING, ETA_PHI, ETA_F_I, float(F_P[0]), float(F_P[1]), F_I, TAU_I,\
                                 SIGMA, C, OMEGA_NL, b_NL )
else :
   start_time   = time.time()
   # Learn mean-field Plefka Unified Framework approximated model for the dendritically modulated generative model for the given configuration,
   # saving results in OUT_FOLDER folder
   ml.learn_gen_mf_pu_dendritic_model( IMG_FILE, OUT_FOLDER, BASIS_F_FILE, MAX_PATCHES, START_SAMPLE,\
                                       ps, M, N, NB, NLOCAL, FREQ_SAVING, ETA_PHI, ETA_F_I, float(F_P[0]), float(F_P[1]), F_I[0], TAU_I, \
                                       SIGMA, OMEGA_NL, b_NL )
   end_time     = time.time()
   elapsed_time_seconds = end_time - start_time
   elapsed_time_minutes = elapsed_time_seconds / 60.
   # Write the time to an output text file
   with open(OUT_FOLDER + '/ELAPSED_TIME.txt', 'w') as file:
      file.write(f'Time taken: {elapsed_time_minutes:.2f} minutes\n')
