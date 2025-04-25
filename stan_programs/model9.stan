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
    real logJ = sum(-abs(c) - 2 * log1p_exp(-abs(c)));

    return dirichlet_lpdf(p | alpha) + logJ;
  }
  
  // Derive cut points from baseline probabilities
  // and latent logistic density function.
  vector derived_cut_points(vector p) {
    int K = num_elements(p);
    vector[K - 1] c;
    c = logit(cumulative_sum(p[1:(K-1)]));
    return c;
  }
  
  // Ordinal pseudo-random number generator assuming
  // a latent standard logistic density function.
  vector derived_ordinal_probs(vector c) {
    int K = num_elements(c) + 1;
    vector[K - 1] Pi = inv_logit(c);
    vector[K] p = append_row(Pi, [1]') - append_row([0]', Pi);
    return p;
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
  
  vector[N_players-1] delta_elo; // elo diff vs. elo of player idx 1
  
  simplex[K] rho; // ordinal prob locations
  real<lower=0> tau; // ordinal prob concentration
  
  array[N_games] int<lower=1, upper=K> y; // Observed game outcomes
  
}

parameters {
  ordered[K - 1] cut_points; // Interior cut points

  real beta; // linear elo coefficient
  real<lower=0> tau_alpha; // linear elo coef population scale
  vector[2] mu_alpha; // linear elo coef population mean
  
  real<lower=-1, upper=1> cor_wb; // Correlation between white and black skills
  
  // Non-centered raw parameters (standard normal)
  matrix[N_players-1, 2] eta;
}

transformed parameters {
  // Player-specific skill differences for white and black
  matrix[N_players-1, 2] alpha;
  
  // More direct parameterization of multivariate normal
  for (n in 1:(N_players-1)) {
    // Common component (representing correlation)
    real common = eta[n, 1] * tau_alpha;
    
    // White skill = common component + white-specific component
    alpha[n, 1] = mu_alpha[1] + common;
    
    // Black skill = common component (weighted by correlation) + black-specific component
    alpha[n, 2] = mu_alpha[2] + cor_wb * common + tau_alpha * sqrt(1 - square(cor_wb)) * eta[n, 2];
  }
  
    
  // Vectorized version using gamma_free approach
  matrix[N_players-1, 2] gamma_free;
  gamma_free[, 1] = beta * delta_elo + alpha[, 1]; // White skills
  gamma_free[, 2] = beta * delta_elo + alpha[, 2]; // Black skills
  
  // matrix[N_players, 2] gamma = append_row([0, 0]', gamma_free);
  matrix[N_players, 2] gamma = append_row(to_row_vector([0, 0]), gamma_free);
}

model {
  // linear skill relationship with first elo
  beta ~ normal(0.004, 0.004 / 2.57);
  
  // skill diff from baseline
  mu_alpha ~ normal(0,1);
  tau_alpha ~ normal(0,3/2.57);
  
  to_vector(eta) ~ normal(0, 1);
  
  cor_wb ~ uniform(0.5, 1); // Alternatively: cor_wb ~ lkj_corr_cholesky(2);

  // Prior model
  cut_points ~ induced_dirichlet(rho/tau + rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(gamma[white_idxs, 1] - gamma[black_idxs, 2], cut_points);
}

generated quantities {
  array[N_games] int<lower=1, upper=K> y_pred;
  vector[K] ordinal_probs;
  vector[K] ordinal_probs_prior;
  ordered[K-1] cut_points_prior;
  
  array[N_players] real mean_outcome_player_white_pred
    = rep_array(0, N_players);
  array[N_players] real mean_outcome_player_black_pred
    = rep_array(0, N_players);
    
  array[N_players] real C_W = rep_array(0, N_players);
  array[N_players] real C_B = rep_array(0, N_players);

  // predicted outcomes
  for (n in 1:N_games) {
    real delta_w = 0;
    real delta_b = 0;
    int c_w = white_idxs[n];
    int c_b = black_idxs[n];
    
    y_pred[n] = ordered_logistic_rng(gamma[white_idxs[n], 1] - gamma[black_idxs[n], 2], cut_points);
    
    C_W[c_w] += 1;
    delta_w = y_pred[n] - mean_outcome_player_white_pred[c_w];
    mean_outcome_player_white_pred[c_w] += delta_w / C_W[c_w];
    
    C_B[c_b] += 1;
    delta_b = y_pred[n] - mean_outcome_player_black_pred[c_b];
    mean_outcome_player_black_pred[c_b] += delta_b / C_B[c_b];
  }
  
  // ordinal probs from cut points
  ordinal_probs = derived_ordinal_probs(cut_points);
  
  ordinal_probs_prior = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points_prior = derived_cut_points(ordinal_probs_prior);
}

