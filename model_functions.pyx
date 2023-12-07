import numpy as np
import random
from PIL import Image
from sklearn.feature_extraction import image
from scipy import ndimage
from scipy import fftpack
from scipy.io import loadmat
from libc.stdio cimport printf
from cython.parallel import prange
from cython.parallel cimport threadid
from cython import boundscheck, wraparound
from libc.time cimport time
from libc.math cimport pow,sqrt,exp,log,fabs,M_PI

cimport numpy as np
cimport cython


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Calls the initialization of the seed for random number generation for the c code functions.
# Parameters :
# ---No iput.
# Returns:
# ---No return value.
def initRandom():
   c_initRandom()

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function that reads an image file, converts it to grayscale and returns
# extracted patches as a matrix where each column in the matrix corresponds
# to an individual patch.
# Parameters:
# ---file_path: string with the name (including full path) of the image file.
# ---ps       : integer with side size for the square ps*ps patches.
# ---prop     : float value with proportion of available patches to extract
#               in [0,1].
# Returns:
# ---Matrix PATCHES.T of dimension (ps*ps, number of total patches).
def readImagePatches(str file_path, int ps, float prop):
   IMG     = Image.open(file_path).convert('L')
   IMG_ARR = np.asarray(IMG)
   PATCHES = image.extract_patches_2d(IMG_ARR,(ps,ps),max_patches=prop,random_state=111)
   PATCHES = np.reshape(PATCHES, (len(PATCHES), -1))
   return PATCHES.T

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function that reads the mat image data, converts it to grayscale and returns
# extracted patches as a matrix where each column in the matrix corresponds
# to an individual patch.
# Parameters:
# ---file_path: string with the name (including full path) of the image file.
# ---ps       : integer with side size for the square ps*ps patches.
# ---max_p    : integer value with the maximum patches to extract in total.
# Returns:
# ---Matrix PATCHES.T of dimension (ps*ps, number of total patches).
def readImagePatchesMAT(str file_path, int ps, int max_p):
   IMAGES  = loadmat(file_path)['IMAGES']
   N       = IMAGES.shape[2]
   max_p_i = int(max_p / N)
   # Define the cutoff frequency
   f0      = 200
   N_patch = 0
   for i in range(0,N):
      IMG_ARR = np.asarray( IMAGES[:,:,i] )
      # Apply forward Fourier transform
      img_fft = fftpack.fft2(IMG_ARR)
      img_fft = np.fft.fftshift(img_fft)
      # Compute the function for the exponential filter
      # in the frequency domain
      ny, nx = IMG_ARR.shape
      cy, cx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
      cy -= ny//2
      cx -= nx//2
      r = np.sqrt(cx**2 + cy**2)
      H = np.exp(-r**4 / ((f0)**4))
      # Apply the filter in the frequency domain
      img_fft_filtered = img_fft * H
      img_fft_filtered = np.fft.ifftshift(img_fft_filtered)
      # Apply the inverse Fourier transform to obtain the filtered image
      img_filtered = np.real(fftpack.ifft2(img_fft_filtered)) / float( ny * nx )
      IMG_ARR[:,:] = img_filtered[:,:]
      PATCHES = image.extract_patches_2d( IMG_ARR, (ps,ps), max_patches=max_p_i,
                                          random_state=111 )
      N_patch = N_patch + len(PATCHES)
   FULL_PATCHES = np.zeros((N_patch,ps*ps), dtype=np.float32)
   P_ind        = 0
   for i in range(0,N):
      IMG_ARR = np.asarray( IMAGES[:,:,i] )
      PATCHES = image.extract_patches_2d( IMG_ARR, (ps,ps), max_patches=max_p_i,
                                          random_state=111 )
      FULL_PATCHES[ (P_ind):(P_ind+len(PATCHES)), : ] = \
                             np.reshape( PATCHES, (len(PATCHES),-1) )
      P_ind   = P_ind + len(PATCHES)
   return FULL_PATCHES.T


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to obtain the marginal likelihood approximation by considering the expected value
# E[ p( y | X , theta ) ] and a corresponding approximation by the average value.
# Parameters:
# ---Ys       : float (d,NY) matrix with NY observation vectors.
# ---PHI      : float matrix with the PHI basis functions parameters.
# ---X        : integer array with M*NY simulated prior binary Gibbs samples, where each
#               sample has dimension N.
# ---sigma_2  : float value with the variance of the Gaussian likelihood.
# ---d        : integer value with the dimension of the observation vectors y.
# ---N        : integer value with the number of neurons in the population.
# ---NY       : integer value with the number of observed patches per update.
# ---M        : integer value with the number of Gibbs samples stored in X.
# ---NCONST   : integer value to indicate whether to apply Gaussian normalization (1) or not (0).
# Returns:
# ---float value with the marginal likelihood approximation.
def marginal_likelihood_approx( np.ndarray[float,ndim=2,mode="c"] Ys,\
                                np.ndarray[float,ndim=2,mode="c"] PHI,\
                                np.ndarray[int,ndim=1,mode="c"] X,\
                                float sigma_2, int d, int N, int NY, int M, int NCONST):
   cdef:
      int ky,r,samp
      float ML, t1
   cdef float[::1] arr_tmp = np.zeros((N,),dtype=np.float32)
   ML          = 0.0
   for r in range(0,NY*M):
       ky      = int( r / M )
       samp    = r * N
       arr_tmp = Ys[:,ky] - np.asarray( np.dot( PHI, X[(samp):(samp+N)] ) , dtype=np.float32)
       t1      = - np.dot( arr_tmp, arr_tmp )  / (2*sigma_2)
       t1      = np.exp( t1 )
       ML      = ML + ( t1 )
   # The gaussian normalization constant can be numerically prohibitive for large sigma and d
   if NCONST == 1 :
      return ( ML / ( float(M * NY) * (np.sqrt(2.0*np.pi*sigma_2)**d) ) )
   else :
      return ( ML / ( float(M * NY) ) )


# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to obtain the marginal likelihood approximation by considering the expected value
# E[ p( y | X , theta ) ] and a corresponding approximation by the average value.
# Parameters:
# ---y        : float d-dimensional observation vector.
# ---PHI      : float matrix with the PHI basis functions parameters.
# ---X        : integer array with M simulated prior binary Gibbs samples, where each
#               sample has dimension N.
# ---sigma_2  : float value with the variance of the Gaussian likelihood.
# ---d        : integer value with the dimension of the observation vector y.
# ---N        : integer value with the number of neurons in the population.
# ---M        : integer value with the number of Gibbs samples stored in X.
# ---NCONST   : integer value to indicate whether to apply Gaussian normalization (1) or not (0).
# Returns:
# ---float value with the marginal likelihood approximation.
def marginal_likelihood_approx_single( np.ndarray[float,ndim=1,mode="c"] y,\
                                       np.ndarray[float,ndim=2,mode="c"] PHI,\
                                       np.ndarray[int,ndim=1,mode="c"] X,\
                                       float sigma_2, int d, int N, int M, int NCONST):
   cdef:
      int r,samp
      float ML, t1
   cdef float[::1] arr_tmp = np.zeros((N,),dtype=np.float32)
   ML          = 0.0
   for r in range(0,M):
       samp    = r * N
       arr_tmp = y - np.asarray( np.dot( PHI, X[(samp):(samp+N)] ) , dtype=np.float32)
       t1      = - np.dot( arr_tmp, arr_tmp )  / (2*sigma_2)
       t1      = np.exp( t1 )
       ML      = ML + ( t1 )
   # The gaussian normalization constant can be numerically prohibitive for large sigma and d
   if NCONST == 1 :
      return ( ML / ( float(M) * (np.sqrt(2.0*np.pi*sigma_2)**d) ) )
   else :
      return ( ML / ( float(M) ) )

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def simulate_prior_dend( np.ndarray[int,ndim=1] X, \
                         np.ndarray[float,ndim=2] PHI,\
                         np.ndarray[float,ndim=1] sq_phi_i_2, \
                         np.ndarray[float,ndim=1] sum_phi, \
                         int NITE, int N, int M, int d, int C, \
                         float F_Q, float F_NL, \
                         np.ndarray[float,ndim=1] F_I, \
                         float OMEGA_NL, float b_NL, float TAU_I, float eps ):
   cdef:
      int r
   # Parallel simulation of each r-th sample
   for r in prange(0,M,nogil=True,schedule='static',num_threads=4):
      # Serial ordered Gibbs simulation of each prior sample
      c_compute_prior_dend( r, NITE, &X[0], &PHI[0,0], &sq_phi_i_2[0], &sum_phi[0],\
                            N, d, C, F_Q, F_NL, &F_I[0], OMEGA_NL, b_NL, TAU_I, eps )
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def simulate_posterior_dend( np.ndarray[int,ndim=1] X, \
                             np.ndarray[float,ndim=2] PHI,\
                             np.ndarray[float,ndim=1] Y_phi_i_s, \
                             np.ndarray[float,ndim=1] phi_i_2_2s, \
                             np.ndarray[float,ndim=1] sq_phi_i_2, \
                             np.ndarray[float,ndim=1] sum_phi_sq, \
                             np.ndarray[float,ndim=1] sum_phi, \
                             int NITE, int N, int M, int d, int C, \
                             float F_Q, float F_NL, \
                             np.ndarray[float,ndim=1] F_I, \
                             float OMEGA_NL, float b_NL, float TAU_I, float eps, float sigma_2 ):
   cdef:
      int r
   # Parallel simulation of each r-th sample
   for r in prange(0,M,nogil=True,schedule='static',num_threads=4):
      # Serial ordered computation of posterior
      c_compute_posterior_dend( r, NITE, &X[0], &PHI[0,0], &Y_phi_i_s[0], &phi_i_2_2s[0], &sq_phi_i_2[0],\
                                &sum_phi_sq[0], &sum_phi[0], N, d, C, F_Q, F_NL, &F_I[0], OMEGA_NL, b_NL, TAU_I, eps, sigma_2 )
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def compute_theta_1( np.ndarray[float,ndim=1] theta_1, np.ndarray[float,ndim=1] theta_p_1, \
                     np.ndarray[float,ndim=1] Y_phi_i_s, np.ndarray[float,ndim=1] phi_i_2_2s,\
                     np.ndarray[float,ndim=1] sq_phi_i_2, np.ndarray[int,ndim=1] U_E, \
                     np.ndarray[int,ndim=1] U_VS, int N, int M, int d, float F_NL, float F_I, \
                     float OMEGA_NL, float b_NL, float eps ):
   cdef:
      int k
      float term_I,term_NL,temp,t2N_2
   
   t2N_2             = 2.0 * float( N * N )
   # compute theta_0 at position zero
   temp              = exp( b_NL )
   term_NL           = -( F_I *  OMEGA_NL * temp ) / ( pow( 1.0 + temp, 2.0 ) )
   # compute rest of theta_1 vector
   for k in range(0,N):
      term_I         = - (F_I * sq_phi_i_2[ k ] * float( U_VS[ k ] ) ) / (t2N_2)
      theta_p_1[ k ] = term_I + term_NL
      theta_1[ k ]   = theta_p_1[ k ] + ( Y_phi_i_s[k] * float( U_E[ k ] ) ) - ( phi_i_2_2s[ k ] * float( U_E[ k ] ) )
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def simulate_ord1_can( np.ndarray[int,ndim=1] X, np.ndarray[float,ndim=1] theta_1,\
                       int NITE, int N, int M ):
   cdef:
      int r
   # Parallel simulation of each r-th sample
   for r in prange(0,M,nogil=True,schedule='static',num_threads=4):
      # Serial ordered Gibbs simulation of each prior sample
      c_compute_ord1_can( r, NITE, &X[0], &theta_1[0], N )
   
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
@boundscheck(False)
@wraparound(False)
def compute_eta_1( np.ndarray[float,ndim=1] eta_vect, np.ndarray[int,ndim=1] X, int N, int M ):
   cdef:
      float avg
      int k,r,samp
   # compute rest of theta_1 vector
   for k in range(0,N):
      avg            = 0.0
      for r in range(0,M):
         samp        = r*N
         avg         = avg + X[ samp + k ]
      eta_vect[ k ]  = avg / float(M)
   return None

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Function to compute a single step of gradient ascent for the basis weights parameters PHI.
# The individual terms of each gradient coordinate corresponding to the Gibbs samples are
# computed in a parallel for loop, where num_threads should be equal to the desired number
# of cores to use for the parallel computation.
# Parameters:
# ---Ys       : float matrix of size (d,NY) where the NY observations are stored.
# ---X_post   : integer array where M posterior binary Gibbs samples are stored, where
#               each sample has dimension N.
# ---PHI      : float matrix with the PHI basis functions parameters which need to be changed
#               according to a single gradient ascent step (averaged over NY patches).
# ---eta      : float value with the step size to be applied for the gradient ascent step.
# ---eps      : float value with small constant value for either column normalization of PHI
#               or for the corrector neurons distribution.
# ---d        : integer value with the dimension of each observation vector y.
# ---N        : integer value with the number of columns in PHI, which is the same as the
#               number of neurons in the population.
# ---NY       : integer value with the number of observed image patches per update.
# ---M        : integer value with the number of Gibbs samples stored in X_post.
# ---sigma_2  : float value with the variance of the likelihood model.
# Returns:
# ---No return value.
@boundscheck(False)
@wraparound(False)
def gradient_ascent_PHI( np.ndarray[float,ndim=1,mode="c"] y, \
                         np.ndarray[int,ndim=1,mode="c"] X_post, \
                         np.ndarray[int,ndim=1,mode="c"] X_prior, \
                         np.ndarray[float,ndim=2,mode="c"] PHI,\
                         float eta, float eps, int d, int N, int M, int C,\
                         np.ndarray[float,ndim=1,mode="c"] F_I, float TAU_I, float sigma_2 ):
   cdef:
      int u,v,r,j,k,samp,row
      float s1,s2,s3,t2N_2
   # Compute element-wise partial w.r.t. PHI
   cdef float[:, ::1] PART_PHI = np.zeros((d,N),dtype=np.float32)
   cdef float[::1]   BATCH_pri = np.zeros(( M*N,),dtype=np.float32)
   cdef float[::1]   BATCH_pos = np.zeros(( M*N,),dtype=np.float32)
   cdef float[::1]   B_s2phi_pr= np.zeros(( M*d,),dtype=np.float32)
   cdef float[::1]   B_s2phi_po= np.zeros(( M*d,),dtype=np.float32)
   cdef float[::1]   B_s_phi_po= np.zeros(( M*d,),dtype=np.float32)
   cdef float[::1]   B_s2phi2_1= np.zeros(( M,),dtype=np.float32)
   cdef float[::1]   B_s2phi2_2= np.zeros(( M,),dtype=np.float32)
   
   t2N_2      = 2.0 * float(N*N)
   # Parallel computation over the simulated (posterior) batch of M binary vectors
   for r in prange(0, M, nogil=True,schedule='static',num_threads=4):
      samp    = r*N
      row     = r*d
      for j in range(0,d):
        s1    = 0.0
        s2    = 0.0
        s3    = 0.0
        for k in range(0,N):
           s1 = s1 + (sqrt(PHI[j,k]*PHI[j,k] + eps)*float(X_prior[samp + k]))
           s2 = s2 + (sqrt(PHI[j,k]*PHI[j,k] + eps)*float(X_post[samp + k]))
           s3 = s3 + (PHI[j,k]*float(X_post[samp + k]))
        B_s_phi_po[ row + j ] = s3
        B_s2phi_pr[ row + j ] = s1
        B_s2phi_po[ row + j ] = s2
      s2      = 0.0
      s1      = 0.0
      for j in range(0,d):
         s2   = s2 + (B_s2phi_po[ row + j ] * B_s2phi_po[ row + j ])
         s1   = s1 + (B_s2phi_pr[ row + j ] * B_s2phi_pr[ row + j ])
      B_s2phi2_1[ r ] = s1 / t2N_2
      B_s2phi2_2[ r ] = s2 / t2N_2
   
   for u in range(0,d):
      # Parallel computation over the simulated (posterior) batch of M binary vectors
      for r in prange(0, M, nogil=True,schedule='static',num_threads=4):
         samp         = r*N
         for v in range(0,N):
            BATCH_pri[ samp + v ] = 0.0
            BATCH_pos[ samp + v ] = 0.0
         c_compute_partial_wrt_PHI_u( r, u, &BATCH_pri[0], &BATCH_pos[0], &y[0], &X_prior[0], \
                                      &X_post[0], &PHI[0,0], &B_s_phi_po[0], &B_s2phi_pr[0], &B_s2phi_po[0], \
                                      &B_s2phi2_1[0], &B_s2phi2_2[0], d,N,C, &F_I[0], TAU_I, eps, sigma_2 )
      # Collapse batch results into a sum over the batch
      for v in range(0,N):
         s1           = 0.0
         s2           = 0.0
         for r in range(0, M):
            s1        = s1 + BATCH_pri[ r * N + v ]
            s2        = s2 + BATCH_pos[ r * N + v ]
         # Finish computation of expected posterior value of partial of log-joint
         # by taking average and including constant division by squared sigma
         PART_PHI[u,v]= (s1 + s2) / ( (float(M)) )
         #printf("-----(v=%i), pri_part = %f, pos_part=%f\n", v, s1, s2)
   
   # Aplication of single-step gradient ascent
   for v in range(0,N):
    for u in range(0,d):
       PHI[u,v]     = PHI[u,v] + eta*PART_PHI[u,v]
   
   # Column normalization
   for v in range(0,N):
      s1            = 0.0
      for u in range(0,d):
         s1         = s1 + PHI[u,v]*PHI[u,v]
      s1    = np.sqrt( s1 )
      for u in range(0,d):
         PHI[u,v]   = PHI[u,v] / (s1 + eps)
   # Show norm of gradient
   s1       = 0.0
   for u in range(0,d):
      for v in range(0,N):
         s1 = s1 + PART_PHI[u,v]*PART_PHI[u,v]
   s1       = np.sqrt( s1 )
   printf("-----||GRADIENT_PHI||   = %.12f\n", s1)
   return None

@boundscheck(False)
@wraparound(False)
def gradient_ascent_mf_pu_PHI( np.ndarray[float,ndim=1,mode="c"] y, np.ndarray[float,ndim=1,mode="c"] eta_vect,\
                               np.ndarray[float,ndim=1,mode="c"] eta_p_vect, np.ndarray[float,ndim=2,mode="c"] PHI, \
                               np.ndarray[int,ndim=1,mode="c"] U_E, np.ndarray[int,ndim=1,mode="c"] U_VS, \
                               float eta, float eps, int d, int N, int M, float F_I, float sigma_2 ):
   cdef:
      int u,v,r,row
      float s,fI_N_2
   # Compute element-wise partial w.r.t. PHI
   cdef float[::1] PART_PHI   = np.zeros((d*N,),dtype=np.float32)
   
   fI_N_2    = F_I / (float(N * N))
   # Parallel computation to fill each row of partial matrix directly
   for u in prange(0, d, nogil=True,schedule='static',num_threads=4):
      c_compute_partial_mf_pu_wrt_PHI_u( u, &PART_PHI[0], &y[0], &eta_vect[0], &eta_p_vect[0],\
                                         &PHI[0,0], &U_E[0],&U_VS[0], N, fI_N_2,sigma_2, eps )
   # Aplication of single-step gradient ascent
   for u in range(0,d):
    row             = u*N
    for v in range(0,N):
       PHI[u,v]     = PHI[u,v] + eta*PART_PHI[ row + v]
   
   # Column normalization
   for v in range(0,N):
      s             = 0.0
      for u in range(0,d):
         s          = s  + PHI[u,v]*PHI[u,v]
      s     = np.sqrt( s )
      for u in range(0,d):
         PHI[u,v]   = PHI[u,v] / (s + eps)
   # Show norm of gradient
   s        = 0.0
   for u in range(0,d):
      row   = u*N
      for v in range(0,N):
         s  = s + PART_PHI[ row + v ]*PART_PHI[ row + v ]
   s        = np.sqrt( s )
   printf("-----||GRADIENT_PHI||   = %.12f\n", s)
   return None


@boundscheck(False)
@wraparound(False)
def gradient_ascent_F_I( np.ndarray[int,ndim=1,mode="c"] X_post,\
                         np.ndarray[int,ndim=1,mode="c"] X_prior,\
                         np.ndarray[float,ndim=2,mode="c"] PHI, \
                         float eta, int N, int M, int C, int d,\
                         np.ndarray[float,ndim=1,mode="c"] F_I, float TAU_I, float eps ):
   cdef:
      float avg_prior,avg_post,s1,s2,dot,t2N_2
      int r,j,k,ind,samp #,row
   cdef float[::1]   BATCH_pos  = np.zeros(( M,),dtype=np.float32)
   cdef float[::1]   BATCH_pri  = np.zeros(( M,),dtype=np.float32)
   t2N_2      = 2.0 * float(N*N)
   # Parallel computation over the simulated (posterior) batch of M binary vectors
   for r in prange(0, M, nogil=True,schedule='static',num_threads=4):
      samp           = r*N
      ind            = r*d
      # Prior term
      dot            = 0.0
      for j in range(0,d):
         #row                   = j*N
         s1                    = 0.0
         for k in range(0,N):
            s1                 = s1 + (sqrt( (PHI[ j, k ]*PHI[ j, k ]) + eps ) * float(X_prior[ samp + k ]) )
         dot                   = dot + (s1*s1)
      dot            = dot / t2N_2
      if dot > TAU_I :
         s2          = dot - TAU_I
      else :
         s2          = 0.0
      BATCH_pri[ r ] = s2
      # Posterior term
      dot            = 0.0
      for j in range(0,d):
         #row                   = j*N
         s1                    = 0.0
         for k in range(0,N):
            s1                 = s1 + (sqrt( (PHI[ j, k ]*PHI[ j, k ]) + eps ) * float(X_post[ samp + k ]) )
         dot                   = dot + (s1*s1)
      dot            = dot / t2N_2
      if dot > TAU_I :
         s2          = dot - TAU_I
      else :
         s2          = 0.0
      BATCH_pos[ r ] = s2
   avg_post          = 0.0
   avg_prior         = 0.0
   for r in range(0,M):
      avg_post       = avg_post + BATCH_pos[ r ]
      avg_prior      = avg_prior + BATCH_pri[ r ]
   avg_post          = avg_post / float(M)
   avg_prior         = avg_prior / float(M)
   printf("-----||GRAD_F_I||: ")
   for c in range(0,C):
      printf("%.12f, ", ( avg_prior - avg_post ) )
      F_I[ c ]       = F_I[ c ] + ( eta * ( avg_prior - avg_post ) )
   printf("\n")
   return None

@boundscheck(False)
@wraparound(False)
def gradient_ascent_mf_pu_F_I( np.ndarray[float,ndim=1,mode="c"] eta_vect, np.ndarray[float,ndim=1,mode="c"] eta_p_vect, \
                               np.ndarray[float,ndim=2,mode="c"] PHI, np.ndarray[int,ndim=1,mode="c"] U_VS, \
                               float eta, int N, int M, int d, float F_I, float eps ):
   cdef:
      float s_prior,s,s_post,temp,t2N_2
      int j,k
   t2N_2       = 2.0 * float(N * N)
   s_prior     = 0.0
   s_post      = 0.0
   for j in range(0,d):
      s        = 0.0
      for k in range(0,N):
         if U_VS[ k ] == 1 :
            s  = s + (sqrt( PHI[ j,k ]*PHI[ j,k ] + eps ) * eta_p_vect[ k ])
      s_prior  = s_prior + (s*s)
      s        = 0.0
      for k in range(0,N):
         if U_VS[ k ] == 1 :
            s  = s + (sqrt( PHI[ j,k ]*PHI[ j,k ] + eps ) * eta_vect[ k ])
      s_post   = s_post + (s*s)
   s_prior     = s_prior / t2N_2
   s_post      = s_post / t2N_2
   
   printf("-----||GRAD_F_I||: ")
   printf("%.12f, ", ( s_prior - s_post ) )
   printf("\n")
   return (F_I + (eta *( s_prior - s_post )))

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
# Declaration of all c code functions called from within this Cython file.
# The header function declaration file must be located in the extern from
# "path_to_c_header_file.h" defined below.
# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
cdef extern from "c/c_model_functions.h" nogil:
   void c_initRandom()
   
   void c_compute_prior_dend( int r, int NITE, int *X, float *PHI, float *sq_phi_i_2, float *sum_phi,\
                              int N, int d, int C, float F_Q, float F_NL, float *F_I,\
                              float OMEGA_NL, float b_NL, float TAU_I, float eps )
   void c_compute_posterior_dend( int r, int NITE, int *X, float *PHI, float *Y_phi_i_s, float *phi_i_2_2s, float *sq_phi_i_2, \
                                  float *sum_phi_sq, float *sum_phi,int N, int d, int C, float F_Q, float F_NL, float *F_I, \
                                  float OMEGA_NL, float b_NL, float TAU_I, float eps, float sigma_2 )
   void c_compute_ord1_can( int r, int NITE, int *X, float *theta_1, int N )  
   void c_compute_partial_wrt_PHI_u( int r, int u, float *BATCH_pri, float *BATCH_pos, float *y, int *X_prior, \
                                     int *X_post, float *PHI, float *B_s_phi_po, float *B_s2phi_pr, float *B_s2phi_po, \
                                     float *B_s2phi2_1, float *B_s2phi2_2, \
                                     int d,int N,int C, float *F_I, float TAU_I, float eps, float sigma_2 )
   void c_compute_partial_mf_pu_wrt_PHI_u( int u, float *PART_PHI, float *y, float *eta_vect, float *eta_p_vect, \
                                           float *PHI, int *U_E, int *U_VS, int N, float fI_N_2, float sigma_2, float eps )

# -----------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------
