# -*- coding: utf-8 -*-
import numpy as np
import pickle as pk
import matplotlib
import matplotlib.pyplot as plt
import argparse
import glob
from pylab import *
from matplotlib import ticker
from matplotlib.gridspec import GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec

matplotlib.rc('xtick', labelsize=16)
matplotlib.rc('ytick', labelsize=16)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    
    if not ax:
        ax = plt.gca()
    
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    
    for spine in ax.spines.values():
        spine.set_edgecolor('green')
    
    # Create colorbar
    #cax = ax.figure.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    #cbar = ax.figure.colorbar(im, ax=cax, **cbar_kw)
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    tick_locator = ticker.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
        
    # Set the color of the outer edge of the heatmap
    #im.set_edgecolor('blue') 
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    
    # Turn spines off and create white grid.
    #for edge, spine in ax.spines.items():
    #    spine.set_visible(False)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    return im, cbar

def plot_heatmaps_fixed_inhibitions(OFOLDER, DPI, MAP_TYPE, LABELS):
   # --------------------------------------------------------------------------------------------------------------------
   # ----------------------------- PROCESS EACH MATRIX OF AVERAGE ML AS DISTAL INHIBITIONS     --------------------------
   # -----------------------------    VARY ACROSS BIAS AND SENSITIVITY NL PARAMETERS           --------------------------
   S_FOLD     = ['S075','S100','S150']
   F_I        = ['4','8','12','16','20']
   OMEGA      = ['6','12','18','24','30']
   B          = ['1','8','15']
   F_I__X__B  = []
   F_I__X__OME= []
   for i in range(0,3):
      F_I__X__B.append( np.zeros((5,3),dtype=np.float32) )
      F_I__X__OME.append( np.zeros((5,5),dtype=np.float32) )
   
   for s in S_FOLD:
    with open('EXP_'+s+'_12X12_DEND_MODEL.sh', 'r') as file:
      for line in file:
         # Process each line and append the result to the list
         line    = line.replace("python main_model.py", "").strip()
         # Split the line by the separator '-'
         parts   = line.split('-')
         out_fol = parts[2].split(" ")[1]
         eta_phi_str,eta_f_str = parts[12].split("  ")[1].split(",")
         eta_phi = float(eta_phi_str)
         eta_f   = float(eta_f_str)
         F_Q_str,F_NL_str = parts[13].split(" ")[1].split(",")
         F_Q     = float(F_Q_str)
         F_NL    = float(F_NL_str)
         F_I_str = parts[14].split(" ")[1].split(",")[0]
         F_I_    = float(F_I_str)
         sigma_  = float(parts[16].split(" ")[1])
         omega_  = float(parts[18].split("  ")[1])
         b_      = float(parts[19].split("  ")[1])
         if eta_f == 0.0 and eta_phi == 10.0 and F_Q == 100.0 and F_NL == 50.0 :
            AVG_TEST_ML = float( pk.load( open(out_fol + '/AVG_MARGINAL_LIKELIHOOD_TEST_SET.p','rb') ) )
            if sigma_ == 0.75 :
               if omega_ == 30 :
                  F_I__X__B[ 0 ][ int((F_I_ / 4.0)-1),int((b_-1)/7.0) ] = AVG_TEST_ML
               if b_ == 1.0 :
                  F_I__X__OME[ 0 ][ int((F_I_ / 4.0)-1),int((omega_/6.0)-1) ] = AVG_TEST_ML
            elif sigma_ == 1.0 :
               if omega_ == 30 :
                  F_I__X__B[ 1 ][ int((F_I_ / 4.0)-1),int((b_-1)/7.0) ] = AVG_TEST_ML
               if b_ == 1.0 :
                  F_I__X__OME[ 1 ][ int((F_I_ / 4.0)-1),int((omega_/6.0)-1) ] = AVG_TEST_ML
            else:
               if omega_ == 30 :
                  F_I__X__B[ 2 ][ int((F_I_ / 4.0)-1),int((b_-1)/7.0) ] = AVG_TEST_ML
               if b_ == 1.0 :
                  F_I__X__OME[ 2 ][ int((F_I_ / 4.0)-1),int((omega_/6.0)-1) ] = AVG_TEST_ML
   
   # Create a 2x3 subplot grid with an additional subplot for labels
   fig = plt.figure(figsize=(13, 8),dpi=DPI)
   gs = GridSpec(2, 4, width_ratios=[0.05, 1, 1, 1], wspace=0.5)
   # Row labels
   row_labels = LABELS
   for i in range(2):
       # Add subplot for labels
       ax_label = fig.add_subplot(gs[i, 0])
       ax_label.text(-0.1, 1.0, row_labels[i], rotation=0, ha='center', va='center', size=27)
       ax_label.axis('off')
   ax_row1   = []
   ax_row2   = []
   ax_row1.append( fig.add_subplot(gs[0, 1]) )
   ax_row1.append( fig.add_subplot(gs[0, 2]) )
   ax_row1.append( fig.add_subplot(gs[0, 3]) )
   ax_row2.append( fig.add_subplot(gs[1, 1]) )
   ax_row2.append( fig.add_subplot(gs[1, 2]) )
   ax_row2.append( fig.add_subplot(gs[1, 3]) )
   titles    = ['$\sigma=0.75$','$\sigma=1.0$','$\sigma=1.5$']
   
   # Plot heatmaps for first row
   # ----------------------------------------------------------------------------
   for ii in range(0,3):
      if ii == 1 :
         ax_row1[ii].set_xlabel('$b_{NL}$',fontsize=16)
      else:
         ax_row1[ii].set_xlabel(' ',fontsize=16)
      ax_row1[ii].set_title(titles[ii],fontsize=18)
      ax_row1[ii].xaxis.set_label_position('top')
      im,_ = heatmap(F_I__X__B[ii],F_I,B, ax=ax_row1[ii],cmap=MAP_TYPE, cbarlabel='')
      if ii == 0 :
         ax_row1[ii].set_ylabel(r'$\bar{f}_{I}$',fontsize=16)
      else:
         ax_row1[ii].set_ylabel('',fontsize=16)
         ax_row1[ii].set_yticklabels([])
   
   # Plot heatmaps for second row
   # ----------------------------------------------------------------------------
   for ii in range(0,3):
      if ii == 1 :
         ax_row2[ii].set_xlabel('$\omega_{NL}$',fontsize=16)
      else:
         ax_row2[ii].set_xlabel('',fontsize=16)
      ax_row2[ii].set_title('',loc='right')
      ax_row2[ii].xaxis.set_label_position('top')
      im,_ = heatmap(F_I__X__OME[ii],F_I,OMEGA, ax=ax_row2[ii], cmap=MAP_TYPE, cbarlabel='')
      if ii == 0 :
         ax_row2[ii].set_ylabel(r'$\bar{f}_{I}$',fontsize=16)
      else :
         ax_row2[ii].set_ylabel('',fontsize=16)
         ax_row2[ii].set_yticklabels([])
   plt.tight_layout()
   fig.savefig(OFOLDER+'/HEAT_MAP_AVG_ML_OMEGA_B_FI.png')
   return None

def plot_inhibition_effects_hist_basis( OFOLDER, MAX_SAMPLES, LINE_STYLES, COLORS, MAP_TYPE,LABEL_LEGENDS, FS, DPI, N, VALS1,VALS2,MODEL_T ):
   # Obtain parameters
   OMEGA_val    = VALS1
   B_val        = VALS2
   size_cols    = len(COLORS)
   size_lsty    = len(LINE_STYLES)
   
   # Search input folders corresponding to a slice over the pair (OMEGA_val, B_val)
   IFOLDERS     = []
   for s in ['S100']:
    with open('EXP_'+s+'_12X12_DEND_MODEL.sh', 'r') as file:
      for line in file:
         # Process each line and append the result to the list
         line    = line.replace("python main_model.py", "").strip()
         # Split the line by the separator '-'
         parts   = line.split('-')
         out_fol = parts[2].split(" ")[1]
         eta_phi_str,eta_f_str = parts[12].split("  ")[1].split(",")
         eta_phi = float(eta_phi_str)
         eta_f   = float(eta_f_str)
         F_Q_str,F_NL_str = parts[13].split(" ")[1].split(",")
         F_Q     = float(F_Q_str)
         F_NL    = float(F_NL_str)
         F_I_str = parts[14].split(" ")[1].split(",")[0]
         F_I_    = float(F_I_str)
         sigma_  = float(parts[16].split(" ")[1])
         omega_  = float(parts[18].split("  ")[1])
         b_      = float(parts[19].split("  ")[1])
         if eta_f == 0.0 and eta_phi == 10.0 and F_Q == 100.0 and F_NL == 50.0 and omega_ == OMEGA_val and b_ == B_val :
            IFOLDERS.append( out_fol )
   print("{} cases to be plotted ...".format(len(IFOLDERS)))
   # Create a 2x5 subplot grid with an additional subplot for labels
   fig = plt.figure(figsize=(18, 8),dpi=DPI)
   gs = GridSpec(2, 6, width_ratios=[0.1, 1, 1, 1, 1,1],height_ratios=[1,0.3], wspace=0.05, hspace=0.3)
   # Row labels
   row_labels = LABEL_LEGENDS
   for i in range(2):
       # Add subplot for labels
       ax_label = fig.add_subplot(gs[i, 0])
       ax_label.text(-4.0, 1.1, row_labels[i], rotation=0, ha='center', va='center', size=27)
       ax_label.axis('off')
   ax_row1   = []
   ax_row1.append( fig.add_subplot(gs[0, 1]) )
   ax_row1.append( fig.add_subplot(gs[0, 2]) )
   ax_row1.append( fig.add_subplot(gs[0, 3]) )
   ax_row1.append( fig.add_subplot(gs[0, 4]) )
   ax_row1.append( fig.add_subplot(gs[0, 5]) )
   ax_row2   = []
   ax_row2.append( fig.add_subplot(gs[1, 1]) )
   ax_row2.append( fig.add_subplot(gs[1, 2]) )
   ax_row2.append( fig.add_subplot(gs[1, 3]) )
   ax_row2.append( fig.add_subplot(gs[1, 4]) )
   ax_row2.append( fig.add_subplot(gs[1, 5]) )
   
   titles    = [r'$\bar{f}_{I}=4$',r'$\bar{f}_{I}=8$',r'$\bar{f}_{I}=12$',r'$\bar{f}_{I}=16$',r'$\bar{f}_{I}=20$']
   
   for i in range(0,5):
      # ---------------------------------------------------------------------------------------------
      # -----------Histogram ------------------------------------------------------------------------
      # Count values over which to plot the histogram
      K_points     = np.arange(0,N+1,dtype=np.int32)
      # Load data
      X_K_COUNTS   = np.asarray( pk.load( open(IFOLDERS[i]+'/X_PRIOR_K_ACTIVE.p','rb') ) , dtype=np.int32)
      F_I          = pk.load( open(IFOLDERS[i]+'/F_I_SAMPLE_'+str((MAX_SAMPLES-FS))+'.p','rb') )
      if MODEL_T > 1 :
         F_I_str= str(round(F_I,3))
      else:
         F_I_str= ''
         C      = F_I.shape[0]
         for c in range(0,C-1):
            F_I_str += (str(round(F_I[c],3))+',')
         F_I_str += (str(F_I[c]))
      ax_row1[i].plot( K_points , X_K_COUNTS, linestyle=LINE_STYLES[0], linewidth=5.0, color=COLORS[0], label='$f_{I}$='+str(F_I_str))
      #ax1.set_title("Histogram of population counts after learning",fontsize=20,pad=20)
      ax_row1[i].set_title(titles[i],fontsize=20,pad=20)
      if i == 2 :
         ax_row1[i].set_xlabel('$n$',fontsize=18)
      else:
         ax_row1[i].set_xlabel('',fontsize=18)
      if i == 0 :
         ax_row1[i].set_ylabel('Counts per bin',fontsize=18)
      else :
         ax_row1[i].set_ylabel('',fontsize=18)
         ax_row1[i].set_yticklabels([])
         #ax_row1[i].set_yticks([])
      ax_row1[i].grid()
      #ax_row1[i].legend(prop={'size': 16},loc='best')
      ax_row1[i].legend().remove()
      ax_row2[i].set_yticklabels([])
      ax_row2[i].set_yticks([])
      ax_row2[i].set_xticklabels([])
      ax_row2[i].set_xticks([])
      
      # ---------------------------------------------------------------------------------------------
      # -----------Basis functions ------------------------------------------------------------------
      PHI       = np.asarray(pk.load(open(IFOLDERS[i]+'/BASIS_FUNCTIONS_AT_SAMPLE_9950.p','rb')))
      d         = PHI.shape[0]
      sqrt_d    = int( np.sqrt( d ) )
      N         = PHI.shape[1]
      inner_gs  = GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[1, i+1], wspace=0.01, hspace=0.01)
      for row in range(2):
         for col in range(5):
            inner_ax     = fig.add_subplot(inner_gs[row, col])
            inner_ax.set_yticklabels([])
            inner_ax.set_yticks([])
            inner_ax.set_xticklabels([])
            inner_ax.set_xticks([])
            inner_ax.set_ylabel('',fontsize=18)
            inner_ax.set_xlabel('',fontsize=18)
            PATCH_BASIS  = PHI[:,(int(N/2) + (10*row) + 2*col) % N ]
            im           = inner_ax.imshow(PATCH_BASIS.reshape(sqrt_d,sqrt_d), cmap=MAP_TYPE, \
                                           interpolation='nearest',vmin=PHI.min(),vmax=PHI.max())
            im.set_clim([PHI.min(),PHI.max()])
   fig.savefig(OFOLDER+'/INHIB_EFFECTS_HIST_BASIS_OMEGA_'+str(OMEGA_val)+'_B_'+str(B_val)+'.png')
   
   return None

def plot_mf_time( IFOLDERS, OFOLDER, DPI, LINE_STYLES, COLORS, LABEL_LEGENDS ):
   N_VALS    = []
   T_VALS    = []
   for i in range(0,len(IFOLDERS)):
      parts  = IFOLDERS[i].strip().split("/")
      N_VALS.append( int( parts[1].replace("N","").strip() ) )
      with open(IFOLDERS[i]+'/ELAPSED_TIME.txt', 'r') as file:
         for line in file:
            line    = line.strip().split(" ")
            val     = float(line[2])
            print("N = {}, time = {} minutes".format(N_VALS[i],val))
            T_VALS.append( val )
   
   #Plot results
   # ------------------Time for the mean-field model -------------------
   fig  = plt.figure(figsize=(12,6),dpi=DPI)
   fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
   ax1  = fig.add_subplot(1,1,1)
   ax1.plot(N_VALS,T_VALS, linestyle=LINE_STYLES[0], marker='o', markersize=10, color=COLORS[0], label=LABEL_LEGENDS[0])
   ax1.set_xticks( N_VALS )
   ax1.set_title("CPU time for the mean-field model",fontsize=18)
   ax1.set_xlabel("N",fontsize=16)
   ax1.set_ylabel("Time (minutes)",fontsize=16)
   ax1.grid()
   ax1.legend(loc='best')
   ax1.grid()
   fig.savefig(OFOLDER+'/CPU_TIME_MF.png')
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot the log-average marginal likelihood as Bayesian learning evolved
# in one or more models.
# Parameters:
# ---IFOLDERS : list of strings, each with the value of the input folder where the
#               pickle file of marginal likelihood values for each model is stored.
# ---OFOLDER  : string value with the output folder, where the png image of
#               the plot is required to be stored.
# ---MAX_SAMPLES : integer value with the maximum number of data samples used
#               for Bayesian learning.
# ---FS       : integer value with the number of Bayesian learning iterations
#               that needed to occur before saving the next state of variables
#               in pickle files; one iteration corresponded to a single sample.
# ---DPI      : integer value with the dots per inch to use for the
#               quality of the png image to create.
# ---LINESTYLES : list of strings, each with the value of the linestyle argument
#               for the plot of the corresponding input model (see IFOLDERS).
# ---COLORS   : list of strings, each with the value of the color argument
#               for the plot of the corresponding input model (see IFOLDERS).
# ---LABELS   : list of strings, each with the value of the label legend argument
#               for the plot of the corresponding input model (see IFOLDERS).
# Returns:
# ---No return value. The image of the plot is stored in the OFOLDER output folder.
def plot_log_average_marginal_likelihood_models( IFOLDERS, OFOLDER, MAX_SAMPLES, FS, DPI,\
                                                 LINESTYLES, COLORS, LABELS ):
   MARGINAL_LIKELIHOOD      = []
   for i in range(0,len(IFOLDERS)):
      MARGINAL_LIKELIHOOD.append( np.asarray(pk.load(open(IFOLDERS[i]+'/MARGINAL_LIKELIHOOD_0_'+\
                                                 str(MAX_SAMPLES)+'.p','rb'))).reshape(-1,)      )
   
   N_SAMPLES                = np.arange(1,int(FS*MARGINAL_LIKELIHOOD[0].shape[0])+1,FS)
   LAVG_MARGINAL_LIKELIHOOD = []
   for i in range(0,len(IFOLDERS)):
      LAVG_MARGINAL_LIKELIHOOD.append( np.zeros((N_SAMPLES.shape[0],),dtype=np.float32) )
      for j in range(0,N_SAMPLES.shape[0]):
         if j < 10 :
            LAVG_MARGINAL_LIKELIHOOD[i][j] = np.log( MARGINAL_LIKELIHOOD[i][j] )
         else :
            LAVG_MARGINAL_LIKELIHOOD[i][j] = np.log( np.average( MARGINAL_LIKELIHOOD[i][(j-5):(j+1)] ) )
         #LAVG_MARGINAL_LIKELIHOOD[i][j] = np.log( np.average( MARGINAL_LIKELIHOOD[i][0:(j+1)] ) )
   
   #Plot results
   # ------------------LOG - AVERAGE MARGINAL LIKELIHOOD ---------------
   fig  = plt.figure(figsize=(12,6),dpi=DPI)
   fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
   ax1  = fig.add_subplot(1,1,1)
   for i in range(0,len(IFOLDERS)):
      ax1.plot(N_SAMPLES,LAVG_MARGINAL_LIKELIHOOD[i], linestyle=LINESTYLES[i],\
                         color=COLORS[i], label=LABELS[i])
   ax1.set_title("LOG(AVG. MARGINAL LIKELIHOOD) AS SAMPLES ARE OBSERVED",fontsize=18)
   ax1.set_xlabel("Number of samples (N)",fontsize=16)
   ax1.set_ylabel("Log(Average marginal likelihood)",fontsize=16)
   ax1.grid()
   ax1.legend(loc='best')
   ax1.grid()
   fig.savefig(OFOLDER+'/LOG_AVG_MARGINAL_LIKELIHOOD_MODELS.png')
   return None


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot observed image patches at a given range of ids used during
# Bayesian learning.
# Parameters:
# ---IFOLDER  : string value with the input folder, where the pickle file of
#               observed patches is stored.
# ---OFOLDER  : string value with the output folder, where the png image of
#               the plot is required to be stored.
# ---MAP_TYPE : string value with the type of color map to use for the
#               patches plot; acceptable values are 'hot', 'coolwarm' or 'bwr'.
# ---ROWS     : integer value with the number of rows for the subplots of the patches.
# ---COLS     : integer value with the number of columns for the subplots of patches.
# Returns:
# ---No return value. The image of the plot is stored in the OFOLDER output folder.
def plot_observations( IFOLDER, OFOLDER, MAP_TYPE, ROWS,COLS ):
   Y            = np.asarray(pk.load(open(IFOLDER+'/OBSERVATIONS.p','rb')),dtype=np.float32)
   ps           = int(np.sqrt(Y.shape[0]))
   plt.figure(figsize=(COLS, ROWS))
   for i in range(0,min(Y.shape[1],int(COLS*ROWS))):
      plt.subplot(ROWS, COLS, i + 1)
      plt.imshow(Y[:,i].reshape(ps,ps), cmap=MAP_TYPE, interpolation='nearest')
      plt.xticks(())
      plt.yticks(())
   plt.savefig( OFOLDER + '/PATCH_SAMPLES.png')
   return None


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot basis functions at a given range of observed image data patches
# during Bayesian learning.
# Parameters:
# ---IFOLDER  : string value with the input folder, where the pickle file of
#               basis functions is stored.
# ---OFOLDER  : string value with the output folder, where the png image of
#               the plot is required to be stored.
# ---S_ID_MIN : integer value that indicates the minimum id of the observed image patch
#               up to which the basis functions used for Bayesian learning.
# ---S_ID_MAX : integer value that indicates the maximum id of the observed image patch
#               up to which the basis functions used for Bayesian learning.
# ---FS       : integer value with the number of Bayesian learning iterations
#               that needed to occur before saving the next state of variables
#               in pickle files; one iteration corresponded to a single observation.
# ---MAP_TYPE : string value with the type of color map to use for the
#               basis functions plot; acceptable values are 'hot',
#               'coolwarm' or 'bwr'.
# ---ROWS     : integer value with the number of rows for the subplots of the
#               basis functions.
# ---COLS     : integer value with the number of columns for the subplots of
#               the basis functions.
# Returns:
# ---No return value. The image of the plot is stored in the OFOLDER output folder.
def plot_basis_functions( IFOLDER, OFOLDER, S_ID_MIN, S_ID_MAX, FS, MAP_TYPE, ROWS,COLS ):
   for s_i in range( S_ID_MIN, S_ID_MAX + 1, FS ):
      PHI       = np.asarray(pk.load(open(IFOLDER+'/BASIS_FUNCTIONS_AT_SAMPLE_'+\
                                     str(s_i)+'.p','rb')))
      d         = PHI.shape[0]
      sqrt_d    = int( np.sqrt( d ) )
      N         = PHI.shape[1]
      print("Observation {}: plotting {} basis functions.".format(s_i,N))
      fig, axes = plt.subplots(nrows=ROWS, ncols=COLS, figsize=(COLS, ROWS))
      sp_i      = 0
      for ax in axes.flat:
         ax.set_axis_off()
         PATCH_BASIS = PHI[:,sp_i]
         im          = ax.imshow(PATCH_BASIS.reshape(sqrt_d,sqrt_d), cmap=MAP_TYPE, \
                                 interpolation='nearest',vmin=PHI.min(),vmax=PHI.max())
         im.set_clim([PHI.min(),PHI.max()])
         sp_i        = sp_i + 1
      fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
      cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
      cbar = fig.colorbar(im, cax=cb_ax)
      # set the colorbar ticks and tick labels
      cbar.set_ticks([PHI.min(), 0.0, PHI.max()])
      cbar.ax.tick_params(labelsize=25)
      #cbar.set_ticklabels(['min', '0', 'max'])
      fig.savefig(OFOLDER + '/PHI_'+str(s_i)+'.png')
   
   return None


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to plot a similarity matrix between the learned basis functions.
# Parameters:
# ---IFOLDER  : string value with the input folder, where the pickle file of
#               basis functions is stored.
# ---OFOLDER  : string value with the output folder, where the png image of
#               the plot is required to be stored.
# ---S_ID     : integer value that indicates up to which number of sample
#               the basis functions used for Bayesian learning.
# ---MAP_TYPE : string value with the type of color map to use for the
#               similarity matrix plot; acceptable values are 'hot',
#               'coolwarm' or 'bwr'.
# ---DPI      : integer value with the dots per inch to use for the
#               quality of the png image to create.
# ---N        : integer value with the number of neurons in the population,
#               which corresponds to the number of basis functions.
# Returns:
# ---No return value. The image of the plot is stored in the OFOLDER output folder.
def plot_similarity_matrix( IFOLDER, OFOLDER, S_ID, MAP_TYPE, DPI, N ):
   PHI        = np.asarray(pk.load(open(IFOLDER+'/BASIS_FUNCTIONS_AT_SAMPLE_'+str(S_ID)+'.p','rb')))
   SIM_MATRIX = np.zeros((N,N),dtype=np.float32)
   for i in range(0,N):
    for j in range(0,N):
       d               = np.linalg.norm( PHI[:,i] ) * np.linalg.norm( PHI[:,j] ) #+ 1e-12
       if d > 0 :
          SIM_MATRIX[i,j] = np.dot( PHI[:,i], PHI[:,j] ) / d
       else :
          SIM_MATRIX[i,j] = 0.0
   
   fig  = plt.figure(figsize=(8,8),dpi=DPI)
   fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
   ax1  = fig.add_subplot(1,1,1)
   im   = ax1.imshow( SIM_MATRIX, cmap=MAP_TYPE, interpolation='none' )
   ax1.set_title("SIMILARITY BETWEEN BASIS ELEMENTS",fontsize=18)
   '''ax1.set_xlabel("BASIS ELEMENT ID",fontsize=16)
   ax1.set_ylabel("BASIS ELEMENT ID",fontsize=16)
   ax1.tick_params(axis='both', which='major', labelsize=10)
   ax1.set_xticks(np.arange(0,N))
   ax1.set_yticks(np.arange(0,N))'''
   from mpl_toolkits.axes_grid1 import make_axes_locatable
   div  = make_axes_locatable(ax1)
   cax  = div.append_axes('right', size='5%', pad=0.05)
   fig.colorbar( im, cax=cax, orientation='vertical')
   fig.savefig(OFOLDER+'/SIMILARITY_PHI_DOT_PHI_S'+str(S_ID)+'.png')
   return None

def plot_binary_pop_counts( IFOLDER, OFOLDER, MAX_SAMPLES, LINE_STYLES, COLORS, LABEL_LEGENDS, FS, DPI, N, MODEL_T ):
   # Obtain parameters
   
   size_cols    = len(COLORS)
   size_lsty    = len(LINE_STYLES)
   
   # Other variables
   eps          = 1e-15
   #delta_r      = 0.001
   #np.random.seed(1010) # set the seed for the pseudo-random numbers
   
   # Count values over which to plot the histogram
   K_points     = np.arange(0,N+1,dtype=np.int32)
   # Load data
   X_K_COUNTS   = np.asarray( pk.load( open(IFOLDER+'/X_PRIOR_K_ACTIVE.p','rb') ) , dtype=np.int32)
   F_I          = pk.load( open(IFOLDER+'/F_I_SAMPLE_'+str((MAX_SAMPLES-FS))+'.p','rb') )
   
   fig    = plt.figure(figsize=(12,6),dpi=DPI)
   fig.subplots_adjust(wspace = 0.6, hspace = 0.6)
   ax1    = fig.add_subplot(1,1,1)
   if MODEL_T > 1 :
      F_I_str= str(round(F_I,3))
   else:
      F_I_str= ''
      C      = F_I.shape[0]
      for c in range(0,C-1):
         F_I_str += (str(round(F_I[c],3))+',')
      F_I_str += (str(F_I[c]))
   ax1.plot( K_points , X_K_COUNTS, linestyle=LINE_STYLES[0], linewidth=5.0, color=COLORS[0], label='$f_{I}$='+str(F_I_str))
   ax1.set_title("Histogram of population counts after learning",fontsize=20,pad=20)
   ax1.set_xlabel('$n$ (number of active neurons)',fontsize=18)
   ax1.set_ylabel('Counts per bin',fontsize=18)
   ax1.grid()
   ax1.legend(prop={'size': 16},loc='best')
   #ax1.legend().remove()
   fig.savefig(OFOLDER+'/HIST_BINARY_POP_COUNTS_F_I'+F_I_str+'.png')
   
   return None