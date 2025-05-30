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
  vector[N_players - 1] gamma_white_free; // player skill with white
  vector[N_players - 1] gamma_black_free; // player skill with black
  ordered[K - 1] cut_points; // Interior cut points
}

transformed parameters {
  vector[N_players] gamma_white = append_row([0]', gamma_white_free);
  vector[N_players] gamma_black = append_row([0]', gamma_black_free);

}

model {
  for (n in 1:(N_players-1)) {
    gamma_white_free[n] ~ normal(delta_elo[n], 2/2.32);
    gamma_black_free[n] ~ normal(delta_elo[n], 2/2.32);
  }

  // Prior model
  cut_points ~ induced_dirichlet(rho/tau + rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(gamma_white[white_idxs] - gamma_black[black_idxs], cut_points);
}

generated quantities {
  array[N_games] int<lower=1, upper=K> y_pred;
  vector[N_players] relative_skill_white_black;
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
    
    y_pred[n] = ordered_logistic_rng(gamma_white[white_idxs[n]] - gamma_black[black_idxs[n]], cut_points);

    C_W[c_w] += 1;
    delta_w = y_pred[n] - mean_outcome_player_white_pred[c_w];
    mean_outcome_player_white_pred[c_w] += delta_w / C_W[c_w];
    
    C_B[c_b] += 1;
    delta_b = y_pred[n] - mean_outcome_player_black_pred[c_b];
    mean_outcome_player_black_pred[c_b] += delta_b / C_B[c_b];
  }

  // relative skill playing with white vs. black
  for (n in 1:N_players) {
    relative_skill_white_black[n] = gamma_white[n] - gamma_black[n];
  }
  
  // ordinal probs from cut points
  ordinal_probs = derived_ordinal_probs(cut_points);
  
  ordinal_probs_prior = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points_prior = derived_cut_points(ordinal_probs_prior);

}

