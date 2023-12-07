#include "c_model_functions.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))


// -----------------------------------------------------------------------------------------------
// Initialize the seed for random number generation.
// Inputs :
//    void (no iput).
// Outputs:
//    void (no return value).
// -----------------------------------------------------------------------------------------------
void c_initRandom(){
   srand( 111 );
}

void c_compute_prior_dend( int r, int NITE, int *X, float *PHI, float *sq_phi_i_2, float *sum_phi,int N, int d, int C, float F_Q, float F_NL, float *F_I, float OMEGA_NL, float b_NL, float TAU_I, float eps ){
   int c,k,j,samp,ite,row;
   float P,q,log_ratio,G_I,G_Q,G_NL,LOG_h_diff;
   float temp0,temp1,s_phi_on,s_phi_off,s,tN_2;
   float sum_x,n_float_x,x_ratio_off;
   
   tN_2        = 2.0 * ((float)(N*N));
   samp        = r * N ;
   sum_x       = 0.0;
   for(k=0; k < N ; k++ ){
      sum_x    = sum_x + ((float)X[ samp + k ]);
   }
   
   for( ite = 1 ; ite <= NITE ; ite++ ){
      for( k=0; k < N ; k++ ){
         
		 n_float_x   = sum_x - ((float)X[ samp + k ]) ;
		 x_ratio_off = (n_float_x ) / ((float)N) ;
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute proximal terms -----------------------------
         G_Q         = F_Q * (n_float_x / ((float)(N*N))) ;
		 temp0       = (OMEGA_NL * x_ratio_off) + b_NL ;
		 temp1       = temp0 + (OMEGA_NL / ((float)N)) ;
		 G_NL        = F_NL * ( (1.0 / (1.0 + exp(temp1) )) - (1.0 / (1.0 + exp(temp0) )) ) ;
		 
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute inhibitory interneurons term ---------------
		 s_phi_off   = 0.0 ;
		 s_phi_on    = 0.0 ;
		 for(j = 0 ; j < d ; j++ ){
			row      = j*N ;
			temp1    = sum_phi[j] - (sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps )*((float)X[ samp + k ])) ;
            s_phi_off= s_phi_off + (temp1*temp1);
			s_phi_on = s_phi_on  + (temp1*temp1) + (2.0 * sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps ) * temp1);
		 }
		 s           = 0.0 ;
         for(j = 0 ; j<N ; j++){
            s        = s + (sq_phi_i_2[j]);
		 }
		 s_phi_off   = s_phi_off / tN_2 ;
		 s_phi_on    = (s_phi_on + s) / tN_2;
		 
		 if( s_phi_on > TAU_I ){
			 temp1   = s_phi_on - TAU_I ;
		 }else{
			 temp1   = 0.0 ;
		 }
         if( s_phi_off > TAU_I ){
			 temp0   = s_phi_off - TAU_I ;
		 }else{
			 temp0   = 0.0 ;
		 }
		 G_I         = 0.0 ;
         for( c=0; c<C ; c++){
		   G_I       = G_I - (F_I[c] * ( temp1 - temp0 ) );
         }
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute log of base measure function ---------------
		 LOG_h_diff  = log( (1.0 + n_float_x ) / ( ((float)N) - n_float_x ) ) ;
		 // ---------------------------------------------------------------------------
		 // ---------------------------------------------------------------------------
		 log_ratio   = G_Q + G_NL + G_I + LOG_h_diff;
		 
         P           = 1.0 / (  1.0 + exp(-log_ratio)  );
         q           = ( (float)rand() )  /  ( (float)RAND_MAX );
         if( q < P ){
			sum_x    = n_float_x + 1.0 ;
            if( X[ samp + k ] == 0 ){
               for( j = 0 ; j < d ; j++ ){
				  row          = j*N ;
                  sum_phi[ j ] = sum_phi[ j ] + sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps );
               }
			}
            X[ samp + k ] = 1;
         }else{
			sum_x    = n_float_x ;
            if( X[ samp + k ] == 1 ){
               for( j = 0 ; j < d ; j++ ){
				  row          = j*N ;
                  sum_phi[ j ] = sum_phi[ j ] - sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps );
               }
			}
            X[ samp + k ] = 0;
         }
      }
   }
}

void c_compute_posterior_dend( int r, int NITE, int *X, float *PHI, float *Y_phi_i_s, float *phi_i_2_2s, float *sq_phi_i_2, float *sum_phi_sq, float *sum_phi,int N, int d, int C, float F_Q, float F_NL, float *F_I, float OMEGA_NL, float b_NL, float TAU_I, float eps, float sigma_2 ){
   int c,k,j,samp,ite,row;
   float P,q,LR_likelihood,LR_prior,G_I,G_Q,G_NL,LOG_h_diff;
   float temp0,temp1,s_phi_on,s_phi_off,s,tN_2;
   float sum_x,n_float_x,x_ratio_off;
   
   tN_2        = 2.0 * ((float)(N*N));
   samp        = r * N ;
   sum_x       = 0.0;
   for(k=0; k < N ; k++ ){
      sum_x    = sum_x + ((float)X[ samp + k ]);
   }
   
   for( ite = 1 ; ite <= NITE ; ite++ ){
      for( k=0; k < N ; k++ ){
         
		 n_float_x   = sum_x - ((float)X[ samp + k ]) ;
		 x_ratio_off = (n_float_x ) / ((float)N) ;
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute proximal terms -----------------------------
         G_Q         = F_Q * (n_float_x / ((float)(N*N))) ;
		 temp0       = (OMEGA_NL * x_ratio_off) + b_NL ;
		 temp1       = temp0 + (OMEGA_NL / ((float)N)) ;
		 G_NL        = F_NL * ( (1.0 / (1.0 + exp(temp1) )) - (1.0 / (1.0 + exp(temp0) )) ) ;
		 
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute inhibitory interneurons term ---------------
		 s_phi_off   = 0.0 ;
		 s_phi_on    = 0.0 ;
		 for(j = 0 ; j < d ; j++ ){
			row      = j*N ;
			temp1    = sum_phi_sq[j] - (sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps )*((float)X[ samp + k ])) ;
            s_phi_off= s_phi_off + (temp1*temp1);
			s_phi_on = s_phi_on  + (temp1*temp1) + (2.0 * sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps ) * temp1);
		 }
		 s           = 0.0 ;
         for(j = 0 ; j<N ; j++){
            s        = s + (sq_phi_i_2[j]);
		 }
		 s_phi_off   = s_phi_off / tN_2 ;
		 s_phi_on    = (s_phi_on + s) / tN_2;
		 
		 if( s_phi_on > TAU_I ){
			 temp1   = s_phi_on - TAU_I ;
		 }else{
			 temp1   = 0.0 ;
		 }
         if( s_phi_off > TAU_I ){
			 temp0   = s_phi_off - TAU_I ;
		 }else{
			 temp0   = 0.0 ;
		 }
		 G_I         = 0.0 ;
         for( c=0; c<C ; c++){
		   G_I       = G_I - (F_I[c] * ( temp1 - temp0 ) );
         }
		 // ---------------------------------------------------------------------------
		 // ---------------------- Compute log of base measure function ---------------
		 LOG_h_diff  = log( (1.0 + n_float_x ) / ( ((float)N) - n_float_x ) ) ;
		 // ---------------------------------------------------------------------------
		 // ---------------------------------------------------------------------------
		 LR_prior    = G_Q + G_NL + G_I + LOG_h_diff ;
		 // Compute log-ratio of likelihood part
         // ------------------------------------------------------------------------------
		 s           = 0.0 ;
		 for( j = 0 ; j < d ; j++ ){
            s        = s + ( PHI[ j*N + k ] * sum_phi[ j ] / (sigma_2) ) ;
		 }
		 if( X[ samp + k ] == 1 ){
            s        = s - (2.0*phi_i_2_2s[ k ]) ;
		 }
		 LR_likelihood = Y_phi_i_s[ k ] - phi_i_2_2s[ k ] - s ;
		 
		 // ------------------------------------------------------------------------------
		 
         P           = 1.0 / (  1.0 + exp(-(LR_likelihood + LR_prior))  );
         q           = ( (float)rand() )  /  ( (float)RAND_MAX );
         if( q < P ){
			sum_x    = n_float_x + 1.0 ;
            if( X[ samp + k ] == 0 ){
               for( j = 0 ; j < d ; j++ ){
				  row          = j*N ;
				  sum_phi_sq[ j ]= sum_phi_sq[ j ] + sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps );
                  sum_phi[ j ] = sum_phi[ j ] + PHI[ row + k ] ;
               }
			}
            X[ samp + k ] = 1;
         }else{
			sum_x    = n_float_x ;
            if( X[ samp + k ] == 1 ){
               for( j = 0 ; j < d ; j++ ){
				  row          = j*N ;
                  sum_phi_sq[ j ]= sum_phi_sq[ j ] - sqrt( (PHI[ row + k ]*PHI[ row + k ]) + eps );
				  sum_phi[ j ] = sum_phi[ j ] - PHI[ row + k ] ;
               }
			}
            X[ samp + k ] = 0;
         }
      }
   }
}

void c_compute_ord1_can( int r, int NITE, int *X, float *theta_1, int N ){
   int k,samp,ite;
   float P,q,log_ratio,LOG_h_diff;
   float sum_x,n_float_x;
   
   samp        = r * N ;
   sum_x       = 0.0;
   for(k=0; k < N ; k++ ){
      sum_x    = sum_x + ((float)X[ samp + k ]);
   }
   
   for( ite = 1 ; ite <= NITE ; ite++ ){
      for( k=0; k < N ; k++ ){
         
		 n_float_x   = sum_x - ((float)X[ samp + k ]) ;
		 
		 // ---------------------- Compute log of base measure function ---------------
		 LOG_h_diff  = log( (1.0 + n_float_x ) / ( ((float)N) - n_float_x ) ) ;
		 // ---------------------------------------------------------------------------
		 
		 // ---------------------- Compute rest of canonical log-ratio ----------------
		 log_ratio   = theta_1[ k ] + LOG_h_diff;
		 // ---------------------------------------------------------------------------
		 
         P           = 1.0 / (  1.0 + exp(-log_ratio)  );
         q           = ( (float)rand() )  /  ( (float)RAND_MAX );
         if( q < P ){
			sum_x    = n_float_x + 1.0 ;
            X[ samp + k ] = 1;
         }else{
			sum_x    = n_float_x ;
            X[ samp + k ] = 0;
         }
      }
   }
}

// -----------------------------------------------------------------------------------------------
// Compute the r-th term of the partial derivative with respect to basis weights PHI[u,:],
// where r corresponds to the r-th posterior Gibbs sample X_post[r] in a simulated batch.
// Inputs :
//    r     (integer value with the index that identifies a posterior Gibbs sample stored in the
//           array X_post; the N-dimensional r-th sample in X_post is indexed as
//           X_post[ r * N ], X_post[ r * N + 1 ], ... , X_post[ r * N + (N-1)]).
//    ky    (integer value with the index of the current observation y starting from 0).
//    u     (integer value with the coordinate that identifies the row in PHI to be differentiated).
//    BATCH_res (pointer to float array where each of the r-th vectors of the partial is required
//           to be stored; each vector term is stored starting at BATCH_res[ r*N ]).
//    Ys    (pointer to flattened float array where the d x NY matrix values storing the
//           NY observations are stored).
//    X_post (pointer to integer array where the simulated posterior binary values are stored
//           for each neuron; X_post[ r * N *NY +  j ] stores the binary value of neuron j (out of
//           N neurons in the population) for the r-th Gibbs sample).
//    PHI   (pointer to flattened float array where the d x N matrix values of the basis weights
//           are stored).
//    N     (integer value with the size of the population of neurons).
//    NY    (integer value with the number of observed image patches per update).
//    eps   (float value with small constant value for the corrector neurons distribution).
// Outputs:
//    void  (no return value).
// -----------------------------------------------------------------------------------------------
void c_compute_partial_wrt_PHI_u( int r, int u, float *BATCH_pri, float *BATCH_pos, float *y, int *X_prior, int *X_post, float *PHI, float *B_s_phi_po, float *B_s2phi_pr, float *B_s2phi_po, float *B_s2phi2_1, float *B_s2phi2_2, int d,int N,int C, float *F_I, float TAU_I, float eps, float sigma_2 ){
   int v,row,samp,row2,c;
   float s,temp,y_u_s,G_I,tN_2;
   
   row            = u * N ;
   samp           = r * N ;
   row2           = r * d ;
   
   tN_2           = 2.0 * ((float)(N*N)) ;
   // -------------------------------------------------------------------------
   // ----------------LIKELIHOOD PART -----------------------------------------
   //s              = 0.0;
   //for( j = 0 ; j < N ; j++ ){
   //   s           = s + ( PHI[ row + j ] * X_post[ samp + j ] );
   //}
   y_u_s          = y[ u ]  - B_s_phi_po[ row2 + u ] ;
   
   // -------------------------------------------------------------------------
   // ----------------PRIOR PART ----------------------------------------------
   
   // Verify threshold for inhibitory interneuron input (posterior samples)
   if( B_s2phi2_2[ r ] > TAU_I ){
	  for( v = 0 ; v < N ; v++ ){
         temp     = sqrt( PHI[ row + v ]*PHI[ row + v ] + eps ) ;
		 s        = B_s2phi_po[ row2 + u ] - (temp*((float)X_post[ samp + v ])) ;
		 s        = (1.0 + (s / temp)) * ( ( 2.0 *PHI[ row + v ] / (tN_2)) *  ((float)X_post[ samp + v ]) ) ;
		 G_I      = 0.0 ;
         for( c=0; c < C ; c++ ){
            G_I   = G_I + (F_I[c] * s) ;
	     }
		 BATCH_pos[ samp + v ] = - G_I ;
	  }
   }
   // Verify threshold for inhibitory interneuron input (prior samples)
   if( B_s2phi2_1[ r ] > TAU_I ){
	  for( v = 0 ; v < N ; v++ ){
         temp     = sqrt( PHI[ row + v ]*PHI[ row + v ] + eps ) ;
		 s        = B_s2phi_pr[ row2 + u ] - (temp*((float)X_prior[ samp + v ])) ;
		 s        = (1.0 + (s / temp)) * ( ( 2.0 *PHI[ row + v ] / (tN_2)) *  ((float)X_prior[ samp + v ]) ) ;
		 G_I      = 0.0 ;
         for( c=0; c < C ; c++ ){
            G_I   = G_I + (F_I[c] * s) ;
	     }
		 BATCH_pri[ samp + v ] =  G_I ;
	  }
   }
   
   for( v = 0 ; v < N ; v++ ){
      BATCH_pos[ samp + v ] = BATCH_pos[ samp + v ] + ((y_u_s / sigma_2) * ((float)X_post[ samp + v ])) ;
   }
}

void c_compute_partial_mf_pu_wrt_PHI_u( int u, float *PART_PHI, float *y, float *eta_vect, float *eta_p_vect, float *PHI, int *U_E, int *U_VS, int N, float fI_N_2, float sigma_2, float eps ){
   int v,row;
   float s1,s2,s3,temp;
   
   row            = u * N ;
   // -------------------------------------------------------------------------
   // Pre-compute sums for both prior and posterior
   s1             = 0.0 ;
   s2             = 0.0 ;
   s3             = 0.0 ;
   for( v = 0 ; v < N ; v++ ){
	  // prior part
      if( U_VS[ v ] == 1 ){
         s1    = s1 + (sqrt( PHI[ row + v ]*PHI[ row + v ] + eps ) * eta_p_vect[ v ]);
      }
	  // posterior part
	  
	  // likelihood part
      if( U_E[ v ] == 1 ){
         s3    = s3 + (PHI[ row + v ] * eta_vect[ v ]) ;
      }
      // prior part
      if( U_VS[ v ] == 1 ){
         s2    = s2 + (sqrt( PHI[ row + v ]*PHI[ row + v ] + eps ) * eta_vect[ v ]);
      }
   }
   s1             = fI_N_2 * s1 ;
   s2             = fI_N_2 * s2 ;
   // -------------------------------------------------------------------------
   // Finish computation for each v-th position for both prior and posterior
   for( v = 0 ; v < N ; v++ ){
      PART_PHI[ row + v ]   = ( ( y[ u ] - s3 ) / sigma_2 )  * ( (float)(U_E[ v ] * eta_vect[ v ]) ) ;
	  temp                  = sqrt( PHI[ row + v ]*PHI[ row + v ] + eps ) ;
	  PART_PHI[ row + v]    = PART_PHI[ row + v] - (  (  ( fI_N_2 + (s2/temp)) * PHI[ row + v ]  )  * ( (float)(U_VS[ v ] * eta_vect[ v ])  ) );
	  PART_PHI[ row + v]    = PART_PHI[ row + v] + (  (  ( fI_N_2 + (s1/temp)) * PHI[ row + v ]  )  * ( (float)(U_VS[ v ] * eta_p_vect[ v ])) ) ;
   }
}
