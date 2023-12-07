# -*- coding: utf-8 -*-
import numpy as np
import random
import time
import argparse
import sys
import os
import model_functions as functions
#import functions_jax as functions_jax
import pickle as pk


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to preprocess image patches, obtain their dimension and initialize basis functions.
# Parameters:
# ---IMG_FILE : string value with the full path of the input mat image file where the
#               observed image patches are stored for the Bayesian learning.
# ---BASIS_F_FILE : string value with the name of the basis function file to load (when it
#               is required to start from a previous basis functions state); 'NA' for
#               no file.
# ---ps       : integer value with the side size for the square image patches.
# ---MAX_PATCHES : integer value with the maximum number of image patches to use for the
#               Bayesian learning (taken from the IMG_FILE mat file).
# ---sigma    : float value to use as Gaussian variance to initialize the basis functions.
# ---eps      : float value with small constant to avoid numerical division by zero.
# ---OUT_FOLDER : string value with the output folder, where the variable states are
#               required to be stored during the Bayesian learning.
# ---N        : integer value with the number of neurons to consider in the population, which
#               will also be equal to the number of basis functions.
# Returns:
# ---Y, PHI, d, where Y are the vectorized image patches (each one in a column), PHI are the
#    column-wise basis functions and d is the dimension of the image patches (and corresponding
#    basis functions).
def preprocess( IMG_FILE, BASIS_F_FILE, ps, MAX_PATCHES, sigma, eps, OUT_FOLDER, N ):
   
   # Read input data
   # --------------------------------------------------
   Y           = np.asarray(functions.readImagePatchesMAT(IMG_FILE,ps,MAX_PATCHES),dtype=np.float32)
   print("Total sampled patches = {}".format(Y.shape[1]))
   
   if Y.shape[1] > MAX_PATCHES :
      Y        = Y[:,0:MAX_PATCHES].copy()
   Y           = (Y - np.mean(Y,axis=0))
   Y           = Y /(np.linalg.norm(Y,axis=0) + eps)
   Y           = Y.copy(order='C')
   
   print("Data dimensions = {}".format(Y.shape))
   print("Data value range: [{},{}]".format(Y.min(),Y.max()))
   
   # Save used image patches
   pk.dump( Y , open(OUT_FOLDER+'/OBSERVATIONS.p','wb') )
   
   d           = Y.shape[0]
   
   if BASIS_F_FILE == 'NA' :
      np.random.seed( 111 )
      #PHI         = np.random.lognormal(mean=0.,sigma=sigma,size=(d,N))
      PHI         = np.random.normal(loc=0.,scale=sigma,size=(d,N))
   else:
      PHI         = np.asarray(  pk.load( open(BASIS_F_FILE,'rb') )  )
   
   # Initialization of data and parameters
   # --------------------------------------------------
   # Subtract the mean and normalize so that l2 norm is 1.0 per column
   PHI         = (PHI - np.mean(PHI,axis=0))
   #PHI         = PHI /(np.linalg.norm(PHI,axis=0))
   # Further average by number of total basis functions N
   # in order to start from a closer initial point where
   # the basis functions can reconstruct each observation
   PHI         = PHI / float(N)
   PHI         = np.asarray(PHI,dtype=np.float32)
   PHI         = PHI.copy(order='C')
   
   print("PHI range: [{},{}]".format(PHI.min(),PHI.max()))
   return Y, PHI, d


def learn_gen_dendritic_model( IMG_FILE, OUT_FOLDER, BASIS_F_FILE, MAX_PATCHES, START_SAMPLE,\
                                 ps, M, N, NB, NLOCAL, FREQ_SAVING, ETA_PHI, ETA_F_I, F_Q, F_NL, F_I_lst, TAU_I,\
                                 SIGMA, C, OMEGA_NL, b_NL ):
   # Preprocess patches and basis function initialization
   # ------------------------------------------------------
   eps         = 0.0001
   Y, PHI, d   = preprocess( IMG_FILE, BASIS_F_FILE, ps, int(1.5*MAX_PATCHES), SIGMA, eps, OUT_FOLDER, N )
   print("PHI range: [{},{}]".format(PHI.min(),PHI.max()))
   print("Data value range: [{},{}]".format(Y.min(),Y.max()))
   
   # Initialization of data and rest of parameters
   # -------------------------------------------------------
   F_I         = np.zeros((C,),dtype=np.float32)
   F_I         = F_I.copy(order='C')
   
   for c in range(0,C):
      F_I[c]   = float(F_I_lst[c])
   
   # Vectors to store Gibbs binary batch simulations
   X_prior     = np.zeros((M*N,),dtype=np.int32)
   X_prior     = X_prior.copy(order='C')
   X_post      = np.zeros((M*N,),dtype=np.int32)
   X_post      = X_post.copy(order='C')
   X_K_ACT     = np.zeros(( N+1,),dtype=np.int32)
   # Auxiliary variables to reduce computations
   Y_phi_i_s   = np.zeros((N,),dtype=np.float32)
   Y_phi_i_s   = Y_phi_i_s.copy(order='C')
   phi_i_2_2s  = np.zeros((N,),dtype=np.float32)
   phi_i_2_2s  = phi_i_2_2s.copy(order='C')
   sq_phi_i_2  = np.zeros((N,),dtype=np.float32)
   sq_phi_i_2  = sq_phi_i_2.copy(order='C')
   v           = np.zeros((d,),dtype=np.float32)
   v           = v.copy(order='C')
   
   s_phi_pos   = np.zeros((d,),dtype=np.float32)
   s_phi_pos   = s_phi_pos.copy(order='C')
   s_phi_pri   = np.zeros((d,),dtype=np.float32)
   s_phi_pri   = s_phi_pri.copy(order='C')
   sum_phi_pos = np.zeros((d,),dtype=np.float32)
   sum_phi_pos = sum_phi_pos.copy(order='C')
   
   sigma_2     = SIGMA * SIGMA
   
   # Lists to store values to report and variable states
   MARGINAL_LIKELIHOOD     = []
   ML                      = 0.
   
   # Initialize seed for random number generation
   # --------------------------------------------------
   functions.initRandom()   # <<<<<<<<<<<<<<<<<<<<<<<<<
   # --------------------------------------------------
   
   # --------------------------------------------------
   # Learn basis functions and parameters
   # using the EM framework
   # --------------------------------------------------
   for i in range(START_SAMPLE,MAX_PATCHES):
      y         = Y[:,i].copy(order='C')
      start_time= time.time()
      for iter_local in range(1,NLOCAL + 1):
         print("---Local iter {}. sample = {}...".format(iter_local,i))
         # Update auxiliary variables
         
         # PENDING TO PARALLELIZE
         for ii in range(0,N):
            Y_phi_i_s[ ii ]  = np.dot( y, PHI[:,ii] ) / (sigma_2)
            phi_i_2_2s[ ii ] = np.dot( PHI[:,ii], PHI[:,ii] ) / (2.0*sigma_2)
            v[:]             = np.sqrt( PHI[:,ii]**2 + eps )
            sq_phi_i_2[ ii ] = np.dot( v, v )
         
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------      
         # Compute posterior probabilities given
         # current parameter state (E-STEP)
         
         # Obtain M Gibbs samples for the prior
         print("Prior simulation...")
         # ---------------------------------------------------------------------------------
         # Burn-in period (discarded samples)
         functions.simulate_prior_dend( X_prior, PHI, sq_phi_i_2, s_phi_pri, 10, N,M, d,C, F_Q,F_NL,F_I, OMEGA_NL,b_NL, TAU_I, eps )
         # Simulated samples after burn-in period
         functions.simulate_prior_dend( X_prior, PHI, sq_phi_i_2, s_phi_pri,  1, N,M, d,C, F_Q,F_NL,F_I, OMEGA_NL,b_NL, TAU_I, eps )
         
         print("After prior simulation...")
         
         # Obtain M Gibbs samples for the posterior
         print("Posterior simulation...")
         # ---------------------------------------------------------------------------------
         # Burn-in period (discarded samples)
         functions.simulate_posterior_dend( X_post, PHI, Y_phi_i_s, phi_i_2_2s, sq_phi_i_2, s_phi_pos,sum_phi_pos, 10,N,M,d,C, F_Q,F_NL,F_I, OMEGA_NL,b_NL,TAU_I,eps,sigma_2 )
         # Simulated samples after burn-in period
         functions.simulate_posterior_dend( X_post, PHI, Y_phi_i_s, phi_i_2_2s, sq_phi_i_2, s_phi_pos,sum_phi_pos,  1,N,M,d,C, F_Q,F_NL,F_I, OMEGA_NL,b_NL,TAU_I,eps,sigma_2 )
         
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         # Compute new parameter state that maximizes complete data
         # log-likelihood (M-STEP). The E-STEP forms part of this
         # step.
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         
         # Gradient ascent for PHI
         if ETA_PHI > 0.0 :
            functions.gradient_ascent_PHI( y, X_post, X_prior, PHI, ETA_PHI,eps, d,N,M,C, F_I, TAU_I,sigma_2 )
         
         # Gradient ascent for F_I
         if ETA_F_I > 0.0 :
            functions.gradient_ascent_F_I( X_post, X_prior, PHI, ETA_F_I, N,M, C, d,F_I, TAU_I, eps )
         print("F_I = {}, SUM(X_post) ={}, SUM(X_prior)={}".format(F_I,np.sum(X_post), np.sum(X_prior) ))
         
         # Marginal likelihood approximation
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         ML = functions.marginal_likelihood_approx_single( y, PHI, X_prior, sigma_2, d,N,M, 0)
         print("ML = {}".format(ML))
      X_prior[:]  = np.zeros(( M*N,),dtype=np.int32)
      X_post[:]   = np.zeros(( M*N,),dtype=np.int32)
      s_phi_pri[:]= np.zeros((d,),dtype=np.float32)
      s_phi_pos[:]= np.zeros((d,),dtype=np.float32)
      
      print("Time per sample = {} seconds.".format(time.time() - start_time))
      print("--------------------------------------")
      #LEARNING RATE UPDATE FOR EACH PARAMETER
      #ETA_PHI  = ETA_PHI / (1.0 + (0.0000001)*np.float(i+1))
      ETA_F_I  = ETA_F_I / (1.0 + (0.00001)*float(i+1))
      
      #SAVE ALL VALUES
      # --------------------------------------------------------------
      if i % FREQ_SAVING == 0 :
         MARGINAL_LIKELIHOOD.append( ML )
         pk.dump( MARGINAL_LIKELIHOOD , open(OUT_FOLDER+'/MARGINAL_LIKELIHOOD_'+str(START_SAMPLE)+'_'+str(MAX_PATCHES)+'.p','wb') )
         pk.dump( F_I, open(OUT_FOLDER+'/F_I_SAMPLE_'+str(i)+'.p','wb') )
         #pk.dump( SIGMA_A, open(OUT_FOLDER+'/SIGMA_A_SAMPLE_'+str(i)+'.p','wb') )
         pk.dump( PHI , open(OUT_FOLDER+'/BASIS_FUNCTIONS_AT_SAMPLE_'+str(i)+'.p','wb') )
      # --------------------------------------------------------------
   
   print("EVALUATING OVER TESTING SET...")
   #print("---Generating prior samples...")
   NBLOCKS     = 100
   NS          = int(0.5*MAX_PATCHES)
   NS_B        = int( NS / NBLOCKS )
   ML_TOTAL    = 0.0
   X_K_ACT     = np.zeros(( N+1,),dtype=np.int32)
   X_prior     = np.zeros(( M*N,),dtype=np.int32)
   X_prior     = X_prior.copy(order='C')
   s_phi_pri   = np.zeros((d,),dtype=np.float32)
   s_phi_pri   = s_phi_pri.copy(order='C')
   
   print("---Evaluating average marginal likelihood approximation...")
   for bi in range(0,NBLOCKS):
     Ys             = Y[:,(MAX_PATCHES + (bi*NS_B)):(MAX_PATCHES + ((bi+1)*NS_B))].copy(order='C')
     for ii in range(0,NS_B):
        y[:]        = Ys[:,ii].copy(order='C')
        functions.simulate_prior_dend( X_prior, PHI, sq_phi_i_2, s_phi_pri,  1, N,M, d,C, F_Q,F_NL,F_I, OMEGA_NL,b_NL, TAU_I, eps )
        #ind_lst     = []
        for r in range(0,M):
           ind      = int( np.sum(X_prior[ (r*N):((r+1)*N) ]) )
           X_K_ACT[ ind ] = X_K_ACT[ ind ] + 1
        ML       = functions.marginal_likelihood_approx_single( y, PHI, X_prior, sigma_2, d,N,M, 0)
        ML_TOTAL = ML_TOTAL + ML
   ML_TOTAL   = ML_TOTAL / float( NBLOCKS * NS_B )
   print("AVERAGE UNNORMALIZED ML OVER {} TEST SETS OF SIZE {} = {}".format( NBLOCKS, NS_B, ML_TOTAL ))
   pk.dump( ML_TOTAL , open(OUT_FOLDER+'/AVG_MARGINAL_LIKELIHOOD_TEST_SET.p','wb') )
   pk.dump( X_K_ACT , open(OUT_FOLDER+'/X_PRIOR_K_ACTIVE.p','wb') )
   print("-----------------------------------------------------")
   
   return None


def learn_gen_mf_pu_dendritic_model( IMG_FILE, OUT_FOLDER, BASIS_F_FILE, MAX_PATCHES, START_SAMPLE,\
                                     ps, M, N, NB, NLOCAL, FREQ_SAVING, ETA_PHI, ETA_F_I, F_Q, F_NL, F_I_str,\
                                     TAU, SIGMA, OMEGA_NL, b_NL ):
   # Preprocess patches and basis function initialization
   # ------------------------------------------------------
   eps         = 0.0001
   Y, PHI, d   = preprocess( IMG_FILE, BASIS_F_FILE, ps, int(1.5*MAX_PATCHES), SIGMA, eps, OUT_FOLDER, N )
   print("PHI range: [{},{}]".format(PHI.min(),PHI.max()))
   print("Data value range: [{},{}]".format(Y.min(),Y.max()))
   
   # Initialization of data and rest of parameters
   # -------------------------------------------------------
   F_I         = float(F_I_str)
   
   # Vectors to store Gibbs binary batch simulations
   X_prior     = np.zeros((M*N,),dtype=np.int32)
   X_prior     = X_prior.copy(order='C')
   X_post      = np.zeros((M*N,),dtype=np.int32)
   X_post      = X_post.copy(order='C')
   X_K_ACT     = np.zeros(( N+1,),dtype=np.int32)
   # Sampling matrices for the basis functions
   U_E         = np.ones(( N,), dtype=np.int32)
   U_E         = U_E.copy(order='C')
   U_VS        = np.ones(( N,), dtype=np.int32)
   U_VS        = U_VS.copy(order='C')
   # Auxiliary variables to reduce computations
   Y_phi_i_s   = np.zeros((N,),dtype=np.float32)
   Y_phi_i_s   = Y_phi_i_s.copy(order='C')
   phi_i_2_2s  = np.zeros((N,),dtype=np.float32)
   phi_i_2_2s  = phi_i_2_2s.copy(order='C')
   sq_phi_i_2  = np.zeros((N,),dtype=np.float32)
   sq_phi_i_2  = sq_phi_i_2.copy(order='C')
   v           = np.zeros((d,),dtype=np.float32)
   v           = v.copy(order='C')
   
   theta_1     = np.zeros((N,),dtype=np.float32)
   theta_1     = theta_1.copy(order='C')
   theta_p_1   = np.zeros((N,),dtype=np.float32)
   theta_p_1   = theta_p_1.copy(order='C')
   eta_vect    = np.zeros((N,),dtype=np.float32)
   eta_vect    = eta_vect.copy(order='C')
   eta_p_vect  = np.zeros((N,),dtype=np.float32)
   eta_p_vect  = eta_p_vect.copy(order='C')
   
   sigma_2     = SIGMA * SIGMA
   
   # Lists to store values to report and variable states
   MARGINAL_LIKELIHOOD     = []
   ML                      = 0.
   
   # Initialize seed for random number generation
   # --------------------------------------------------
   functions.initRandom()   # <<<<<<<<<<<<<<<<<<<<<<<<<
   # --------------------------------------------------
   
   # Initialize random subsampling matrices
   # --------------------------------------------------
   K_ZEROS      = int( TAU * N )
   ind1       = np.arange(0,N)
   np.random.shuffle( ind1 )
   for k in range(0,K_ZEROS):
      U_E[ind1[k]] = 0
   ind2       = np.arange(0,N)
   np.random.shuffle( ind2 )
   for k in range(0,K_ZEROS):
      U_VS[ind2[k]]= 0
   # --------------------------------------------------
   # Learn basis functions and parameters
   # using the EM framework
   # --------------------------------------------------
   for i in range(START_SAMPLE,MAX_PATCHES):
      y         = Y[:,i].copy(order='C')
      start_time= time.time()
      for iter_local in range(1,NLOCAL + 1):
         print("---Local iter {}. sample = {}...".format(iter_local,i))
         # Update auxiliary variables
         
         # PENDING TO PARALLELIZE
         for ii in range(0,N):
            Y_phi_i_s[ ii ]  = np.dot( y, PHI[:,ii] ) / (sigma_2)
            phi_i_2_2s[ ii ] = np.dot( PHI[:,ii], PHI[:,ii] ) / (2.0*sigma_2)
            v[:]             = np.sqrt( PHI[:,ii]**2 + eps )
            sq_phi_i_2[ ii ] = np.dot( v, v )
         
         # Update canonical coordinates first-order vector for both prior and posterior
         functions.compute_theta_1( theta_1, theta_p_1, Y_phi_i_s, phi_i_2_2s, sq_phi_i_2, U_E, U_VS, N,M, d, F_NL,F_I, OMEGA_NL,b_NL, eps )
         
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------      
         
         # Obtain M Gibbs samples for the prior
         print("Prior simulation...")
         # ---------------------------------------------------------------------------------
         # Burn-in period (discarded samples)
         functions.simulate_ord1_can( X_prior, theta_p_1, 10, N, M )
         # Simulated samples after burn-in period
         functions.simulate_ord1_can( X_prior, theta_p_1,  1, N, M )
         
         print("After prior simulation...")
         
         # Obtain M Gibbs samples for the posterior
         print("Posterior simulation...")
         # ---------------------------------------------------------------------------------
         # Burn-in period (discarded samples)
         functions.simulate_ord1_can( X_post, theta_1, 10, N, M )
         # Simulated samples after burn-in period
         functions.simulate_ord1_can( X_post, theta_1,  1, N, M )
         
         # Estimate first-order mean-field vector with obtained samples (E-STEP)
         # ---------------------------------------------------------------------------------
         functions.compute_eta_1( eta_p_vect, X_prior, N, M )
         functions.compute_eta_1( eta_vect  , X_post , N, M )
         
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         # Compute new parameter state that maximizes complete data
         # log-likelihood (M-STEP). The E-STEP forms part of this
         # step.
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         
         # Gradient ascent for PHI
         if ETA_PHI > 0.0 :
            functions.gradient_ascent_mf_pu_PHI( y, eta_vect, eta_p_vect, PHI, U_E, U_VS, ETA_PHI,eps, d,N,M, F_I, sigma_2 )
         
         # Gradient ascent for F_I
         if ETA_F_I > 0.0 :
            F_I = functions.gradient_ascent_mf_pu_F_I( eta_vect, eta_p_vect, PHI, U_VS, ETA_F_I, N,M, d,F_I, eps )
         
         print("F_I = {}, SUM(X_post) ={}, SUM(X_prior)={}".format(F_I,np.sum(X_post), np.sum(X_prior) ))
         
         # Marginal likelihood approximation
         # ---------------------------------------------------------------------------------
         # ---------------------------------------------------------------------------------
         ML = functions.marginal_likelihood_approx_single( y, PHI, X_prior, sigma_2, d,N,M, 0)
         print("ML = {}".format(ML))
      X_prior[:]  = np.zeros(( M*N,),dtype=np.int32)
      X_post[:]   = np.zeros(( M*N,),dtype=np.int32)
      
      print("Time per sample = {} seconds.".format(time.time() - start_time))
      print("--------------------------------------")
      #LEARNING RATE UPDATE FOR EACH PARAMETER
      #ETA_PHI  = ETA_PHI / (1.0 + (0.0000001)*np.float(i+1))
      ETA_F_I  = ETA_F_I / (1.0 + (0.00001)*float(i+1))
      
      #SAVE ALL VALUES
      # --------------------------------------------------------------
      if i % FREQ_SAVING == 0 :
         MARGINAL_LIKELIHOOD.append( ML )
         pk.dump( MARGINAL_LIKELIHOOD , open(OUT_FOLDER+'/MARGINAL_LIKELIHOOD_'+str(START_SAMPLE)+'_'+str(MAX_PATCHES)+'.p','wb') )
         pk.dump( F_I, open(OUT_FOLDER+'/F_I_SAMPLE_'+str(i)+'.p','wb') )
         #pk.dump( SIGMA_A, open(OUT_FOLDER+'/SIGMA_A_SAMPLE_'+str(i)+'.p','wb') )
         pk.dump( PHI , open(OUT_FOLDER+'/BASIS_FUNCTIONS_AT_SAMPLE_'+str(i)+'.p','wb') )
      # --------------------------------------------------------------
   
   print("EVALUATING OVER TESTING SET...")
   #print("---Generating prior samples...")
   NBLOCKS     = 100
   NS          = int(0.5*MAX_PATCHES)
   NS_B        = int( NS / NBLOCKS )
   ML_TOTAL    = 0.0
   X_K_ACT     = np.zeros(( N+1,),dtype=np.int32)
   X_prior     = np.zeros(( M*N,),dtype=np.int32)
   X_prior     = X_prior.copy(order='C')
   
   print("---Evaluating average marginal likelihood approximation...")
   for bi in range(0,NBLOCKS):
     Ys             = Y[:,(MAX_PATCHES + (bi*NS_B)):(MAX_PATCHES + ((bi+1)*NS_B))].copy(order='C')
     for ii in range(0,NS_B):
        y[:]        = Ys[:,ii].copy(order='C')
        functions.simulate_ord1_can( X_prior, theta_p_1, 1, N, M )
        for r in range(0,M):
           ind      = int( np.sum(X_prior[ (r*N):((r+1)*N) ]) )
           X_K_ACT[ ind ] = X_K_ACT[ ind ] + 1
        ML       = functions.marginal_likelihood_approx_single( y, PHI, X_prior, sigma_2, d,N,M, 0)
        ML_TOTAL = ML_TOTAL + ML
   ML_TOTAL   = ML_TOTAL / float( NBLOCKS * NS_B )
   print("AVERAGE UNNORMALIZED ML OVER {} TEST SETS OF SIZE {} = {}".format( NBLOCKS, NS_B, ML_TOTAL ))
   pk.dump( ML_TOTAL , open(OUT_FOLDER+'/AVG_MARGINAL_LIKELIHOOD_TEST_SET.p','wb') )
   pk.dump( X_K_ACT , open(OUT_FOLDER+'/X_PRIOR_K_ACTIVE.p','wb') )
   print("-----------------------------------------------------")
   
   return None
