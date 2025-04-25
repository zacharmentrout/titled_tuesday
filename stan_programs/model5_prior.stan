functions {
  
  // Log probability density function over cut points
  // induced by a Dirichlet probability density function
  // over baseline probabilities and a latent logistic
  // density function.
  real induced_dirichlet_lpdf(vector c, vector alpha) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);

    // Log Jacobian correction
    real logJ = 0;
    for (k in 1:(K - 1)) {
      if (c[k] >= 0)
        logJ += -c[k] - 2 * log(1 + exp(-c[k]));
      else
        logJ += +c[k] - 2 * log(1 + exp(+c[k]));
    }

    return dirichlet_lpdf(p | alpha) + logJ;
  }
  
  // Derive cut points from baseline probabilities
  // and latent logistic density function.
  vector derived_cut_points(vector p) {
    int K = num_elements(p);
    vector[K - 1] c;

    real cum_sum = 0;
    for (k in 1:(K - 1)) {
      cum_sum += p[k];
      c[k] = logit(cum_sum);
    }

    return c;
  }
  
  // Ordinal pseudo-random number generator assuming
  // a latent standard logistic density function.
  vector shifted_derived_ordinal_probs(vector c, real gamma) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c - gamma);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return p;
  }

}

data {
  int<lower=1> N_games; // Number of games

  // Matchups
  int<lower=1> N_players;
  int<lower=1> K; 

  array[N_games] int<lower=1, upper=N_players> white_idxs;
  array[N_games] int<lower=1, upper=N_players> black_idxs;
  
  vector[N_players] delta_elo; // elo diff vs. elo of player idx 1
  
  simplex[K] rho; // ordinal prob locations
  real<lower=0> tau; // ordinal prob concentration
  
  array[N_games] int<lower=1, upper=K> y; // Observed game outcomes
  
}


generated quantities {
  // player skill
  vector[N_players - 1] gamma_free; // free parameters
  vector[N_players] gamma;
  
  // interior cutpoints
  ordered[K - 1] cut_points;
  
  // predicted ordinal outcomes
  array[N_games] int<lower=1, upper=K> y_pred;
  
  vector[K] ordinal_probs;

  // Matchup-specific discrimination effects (non-centered parameterization)
  array[N_players] vector[N_players] d_matchup_ncp;
  
  // Hyperparameters for discrimination effects
  real mu_d;                        // Global mean discrimination
  real<lower=0> tau_d;              // Global scale for discrimination
  vector<lower=0>[N_players] tau_player;  // Player-specific scales
  cholesky_factor_corr[N_players] L_player;  // Correlation matrix for player similarities
  
// Priors for discrimination parameters
  for (i in 1:N_players) {
    for (j in 1:N_players) {
      d_matchup_ncp[i][j] = normal_rng(0, 1);
    }
  }
  
  mu_d = normal_rng(0, 0.5);        // Prior centered at 0
  tau_d = abs(normal_rng(0, 1/2.57));       // Half-normal for global scale
  
  for (i in 1:N_players) {
    tau_player[i] = abs(normal_rng(0, 0.5/2.57));  // Half-normal for player scales
  }
  
  // LKJ prior for correlation matrix
  L_player = lkj_corr_cholesky_rng(N_players, 5.0);
  
  
  
  
  // Matchup discrimination matrix
  array[N_players] vector[N_players] d_matchup;
  
  {
    // Apply the multivariate normal structure to create correlated rows
    matrix[N_players, N_players] L_cov = diag_pre_multiply(tau_player, L_player);
    
    
    for (i in 1:N_players) {
        d_matchup[i] = mu_d + tau_d * (L_cov * d_matchup_ncp[i]);
    }
  }

  // simulate
  ordinal_probs = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points = derived_cut_points(ordinal_probs);
  
  for (n in 1:(N_players-1)) {
    gamma_free[n] = normal_rng(delta_elo[n], 1/2.32);
  }
  
  gamma = append_row([0]', gamma_free);
 
  for (n in 1:N_games) {
    int w = white_idxs[n];
    int b = black_idxs[n];
    
    real scaled_diff = (1 + d_matchup[w, b]) * (gamma[w] - gamma[b]);

    y_pred[n] = ordered_logistic_rng(scaled_diff, cut_points);
  }

}

