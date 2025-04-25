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

// parameters {
//   vector[N_players - 1] gamma_white_free; // player skill with white
//   vector[N_players - 1] gamma_black_free; // player skill with black
//   ordered[K - 1] cut_points; // Interior cut points
// }

// transformed parameters {
//   // Relative skills for all teams
//   // vector[N_players] gamma_white_free = tau_gamma_white * eta_white + mu_gamma_white;
//   // vector[N_players] gamma_black_free = tau_gamma_black * eta_black + mu_gamma_black;
// 
//   vector[N_players] gamma_white = append_row([0]', gamma_white_free);
//   vector[N_players] gamma_black = append_row([0]', gamma_black_free);
//   
// }

// model {
//   for (n in 1:(N_players-1)) {
//     gamma_white_free[n] ~ normal(delta_elo[n], 2/2.32)
//     gamma_black_free[n] ~ normal(delta_elo[n], 2/2.32)
//   }
//   
//   // Prior model
//   cut_points ~ induced_dirichlet(rep_vector(1, K));
// 
//   // Observational model
//   y ~ ordered_logistic(gamma_white[white_idxs] - gamma_black[black_idxs], cut_points);
// }

generated quantities {
  vector[N_players - 1] gamma_white_free; // player skill with white
  vector[N_players - 1] gamma_black_free; // player skill with black
  ordered[K - 1] cut_points; // Interior cut points
  array[N_games] int<lower=1, upper=K> y_pred;
  vector[K] ordinal_probs;
  
  vector[N_players] gamma_white;
  vector[N_players] gamma_black;

  ordinal_probs = dirichlet_rng(rho/tau + rep_vector(1, K));
  cut_points = derived_cut_points(ordinal_probs);
  
  for (n in 1:(N_players-1)) {
    gamma_white_free[n] = normal_rng(delta_elo[n], 2/2.32);
    gamma_black_free[n] = normal_rng(delta_elo[n], 2/2.32);
  }
  
  gamma_white = append_row([0]', gamma_white_free);
  gamma_black = append_row([0]', gamma_black_free);
  
  for (n in 1:N_games)
    y_pred[n] = ordered_logistic_rng(gamma_white[white_idxs[n]] - gamma_black[black_idxs[n]], cut_points);
}

