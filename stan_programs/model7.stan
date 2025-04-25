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
  
  vector[N_players] delta_elo; // elo diff vs. elo of player idx 1
  
  simplex[K] rho; // ordinal prob locations
  real<lower=0> tau; // ordinal prob concentration
  
  array[N_games] int<lower=1, upper=K> y; // Observed game outcomes
}

parameters {
  ordered[K - 1] cut_points; // Interior cut points
  
  real<lower=0> tau_alpha; // linear elo coef population scale
  real mu_alpha; // linear elo coef population mean
  vector[N_players-1] eta_alpha;
  
  real mu_phi;
  real<lower=0> tau_phi;
  
  vector[N_players-1] eta_phi;

  real beta; // linear elo coefficient
}

transformed parameters {
  vector[N_players - 1] alpha = mu_alpha + eta_alpha * tau_alpha;
  vector[N_players - 1] phi_free = exp(mu_phi + eta_phi * tau_phi);
  vector[N_players - 1] gamma_free;
  
  for (n in 1:(N_players-1)) {
    gamma_free[n] = beta * delta_elo[n] + alpha[n];
  }

  vector[N_players] gamma = append_row([0]', gamma_free);
  vector[N_players] phi = append_row([1]', phi_free);
}

model {
  beta ~ normal(0.004, 0.004 / 2.57);
  
  // Skill difference from initial elo
  eta_alpha ~ normal(0, 1);
  mu_alpha ~ normal(0, 1);
  tau_alpha ~ normal(0, 3/2.57);
  
  // Discrimination black
  eta_phi ~ normal(0, 1);
  mu_phi ~ normal(0, 0.15 / 2.32);
  tau_phi ~ normal(0, 0.15 / 2.57);
  

  // Prior model
  cut_points ~ induced_dirichlet(rho/tau + rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(phi[black_idxs] .* (gamma[white_idxs] - gamma[black_idxs]), cut_points);
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
  
  array[N_players] real mean_outcome_player_pred = rep_array(0, N_players);
    
  array[N_players] real C_W = rep_array(0, N_players);
  array[N_players] real C_B = rep_array(0, N_players);
  array[N_players] real C_P = rep_array(0, N_players);

  // predicted outcomes
  for (n in 1:N_games) {
    real delta_w = 0;
    real delta_b = 0;
    real delta_p_w = 0;
    real delta_p_b = 0;
    int c_w = white_idxs[n];
    int c_b = black_idxs[n];
    
    y_pred[n] = ordered_logistic_rng(phi[black_idxs[n]] * (gamma[white_idxs[n]] - gamma[black_idxs[n]]), cut_points);
    
    C_W[c_w] += 1;
    delta_w = y_pred[n] - mean_outcome_player_white_pred[c_w];
    mean_outcome_player_white_pred[c_w] += delta_w / C_W[c_w];
    
    C_B[c_b] += 1;
    delta_b = y_pred[n] - mean_outcome_player_black_pred[c_b];
    mean_outcome_player_black_pred[c_b] += delta_b / C_B[c_b];
    
    C_P[c_w] += 1;
    C_P[c_b] += 1;
    delta_p_w = y_pred[n] - mean_outcome_player_pred[c_w];
    delta_p_b = (4-y_pred[n]) - mean_outcome_player_pred[c_b];
    
    mean_outcome_player_pred[c_w] += delta_p_w / C_P[c_w];
    mean_outcome_player_pred[c_b] += delta_p_b / C_P[c_b];

    
  }
  
  // ordinal probs from cut points
  ordinal_probs = derived_ordinal_probs(cut_points);
  
  ordinal_probs_prior = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points_prior = derived_cut_points(ordinal_probs_prior);
}

