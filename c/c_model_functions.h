// Declaration of all the c code functions for the computations involved
// at the learning stage of the Bayesian model.
// Each function is described in the file "c_model_functions.c".
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
# define PI           3.14159265358979323846
void c_initRandom();
void c_compute_prior_dend( int r, int NITE, int *X, float *PHI, float *sq_phi_i_2, float *sum_phi,int N, int d, int C, float F_Q, float F_NL, float *F_I, float OMEGA_NL, float b_NL, float TAU_I, float eps );
void c_compute_posterior_dend( int r, int NITE, int *X, float *PHI, float *Y_phi_i_s, float *phi_i_2_2s, float *sq_phi_i_2, float *sum_phi_sq, float *sum_phi,int N, int d, int C, float F_Q, float F_NL, float *F_I, float OMEGA_NL, float b_NL, float TAU_I, float eps, float sigma_2 );
void c_compute_ord1_can( int r, int NITE, int *X, float *theta_1, int N );
void c_compute_partial_wrt_PHI_u( int r, int u, float *BATCH_pri, float *BATCH_pos, float *y, int *X_prior, int *X_post, float *PHI, float *B_s_phi_po, float *B_s2phi_pr, float *B_s2phi_po, float *B_s2phi2_1, float *B_s2phi2_2, int d,int N,int C, float *F_I, float TAU_I, float eps, float sigma_2 );
void c_compute_partial_mf_pu_wrt_PHI_u( int u, float *PART_PHI, float *y, float *eta_vect, float *eta_p_vect, float *PHI, int *U_E, int *U_VS, int N, float fI_N_2, float sigma_2, float eps );
// -------------------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------------------
