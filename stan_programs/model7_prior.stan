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
  
  int<lower=1> N_delta_elo_grid;
  vector[N_delta_elo_grid] delta_elo_grid; // grid of elo differences to check
}


generated quantities {
  vector[N_players - 1] gamma_free; // player skill with white
  ordered[K - 1] cut_points; // Interior cut points
  array[N_games] int<lower=1, upper=K> y_pred;
  array[N_delta_elo_grid] int<lower=1, upper=K> y_pred_elo_grid;
  simplex[K] ordinal_probs_grid[N_delta_elo_grid];
  vector[K] ordinal_probs;
  
  real<lower=0> tau_alpha; // linear elo coef population scale
  real mu_alpha; // linear elo coef population mean
  vector[N_players-1] eta;
  vector[N_players-1] alpha; // skill difference

  real beta; // linear elo coefficient
  
  vector[N_players] gamma;

  ordinal_probs = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points = derived_cut_points(ordinal_probs);
  
  beta = normal_rng(0.004, 0.004 / 2.57);
  for (n in 1:(N_players - 1)) {
    eta[n] = normal_rng(0,1);
  }
  mu_alpha = normal_rng(0,1);
  tau_alpha = abs(normal_rng(0,3/2.57));
  
  for (n in 1:(N_players - 1)) {
    alpha[n] = mu_alpha + eta[n] * tau_alpha;
  }

  for (n in 1:(N_players-1)) {
    gamma_free[n] = beta * delta_elo[n] + alpha[n];
  }
  
  gamma = append_row([0]', gamma_free);

  for (n in 1:N_games)
    y_pred[n] = ordered_logistic_rng(gamma[white_idxs[n]] - gamma[black_idxs[n]], cut_points);
    
  for (d in 1:N_delta_elo_grid) {
      y_pred_elo_grid[d] = ordered_logistic_rng(beta*delta_elo_grid[d], cut_points);
      ordinal_probs_grid[d] = shifted_derived_ordinal_probs(cut_points, beta*delta_elo_grid[d]);
  }
}

