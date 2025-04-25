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
  
  vector ordinal_probs_elo_davidson(real z, real k) {
    real eps = 0.5*z / 400;
    real denom = pow(10, eps) + pow(10, -eps) + k;
    vector[3] p = rep_vector(0, 3);
    
    p[1] = pow(10, -eps) / denom;
    p[2] = k / denom;
    p[3] = pow(10, eps) / denom;
    
    return p;
  }

}

data {
  int<lower=1> N_games; // Number of games
  int<lower=1> N_players; 
  int<lower=1> K;
  array[N_games] int<lower=1, upper=K> y; // Observed game outcomes
  
  vector[N_games] z; // elo difference white v. black
  
  array[N_games] int<lower=1, upper=N_players> white_idxs;
  array[N_games] int<lower=1, upper=N_players> black_idxs;
  
}

parameters {
  real<lower=0> kappa; // draw parameter
}


model {
  // Uniform prior distribution of Kappa

  for (n in 1:N_games) {
    real eps = 0.5*z[n] / 400;
    if (y[n] == 1) 
      target += -eps * log(10) - log(pow(10, eps) + pow(10, -eps) + kappa);       
    else if (y[n] == 2) 
      target += log(kappa) -     log(pow(10, eps) + pow(10, -eps) + kappa);         
    else
      target += eps * log(10) -  log(pow(10, eps) + pow(10, -eps) + kappa); 
  }
}

generated quantities {
  array[N_games] int<lower=1, upper=K> y_pred;

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
    real eps = 0.5*z[n] / 400;
    
    y_pred[n] = categorical_rng(ordinal_probs_elo_davidson(z[n], kappa));
    
    C_W[c_w] += 1;
    delta_w = y_pred[n] - mean_outcome_player_white_pred[c_w];
    mean_outcome_player_white_pred[c_w] += delta_w / C_W[c_w];
    
    C_B[c_b] += 1;
    delta_b = y_pred[n] - mean_outcome_player_black_pred[c_b];
    mean_outcome_player_black_pred[c_b] += delta_b / C_B[c_b];
  }
  
}

