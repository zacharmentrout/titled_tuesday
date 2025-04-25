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
  
  int<lower=0, upper=N_players> N_players_ncp;          // Number of noncentered individuals
  int<lower=1, upper=N_players> ncp_idx[N_players_ncp]; // Index of noncentered individuals
  
  int<lower=0, upper=N_players> N_players_cp;           // Number of centered individuals
  int<lower=1, upper=N_players> cp_idx[N_players_cp];   // Index of noncentered individuals

  array[N_games] int<lower=1, upper=N_players> white_idxs;
  array[N_games] int<lower=1, upper=N_players> black_idxs;
  
  vector[N_players-1] delta_elo; // elo diff vs. elo of player idx 1
  
  simplex[K] rho; // ordinal prob locations
  real<lower=0> tau; // ordinal prob concentration
  
  array[N_games] int<lower=1, upper=K> y; // Observed game outcomes
  
  
}

parameters {
  ordered[K - 1] cut_points; // Interior cut points
  
  real<lower=0> tau_alpha; // linear elo coef population scale
  real mu_alpha; // linear elo coef population mean

  vector[N_players_ncp] eta_ncp;  // Non-centered individual parameters
  vector[N_players_cp]  eta_cp;   // Ccentered individual parameters

  real beta; // linear elo coefficient
}

transformed parameters {
  // Skill Diff. from Baseline
  vector[N_players - 1] alpha;
  alpha[ncp_idx] = mu_alpha + eta_ncp * tau_alpha;
  alpha[cp_idx] = eta_cp;
  
  // vector[N_players - 1] gamma_free;
  
  // for (n in 1:(N_players-1)) {
  //   gamma_free[n] = beta * delta_elo[n] + alpha[n];
  // }
  // With this:
  vector[N_players - 1] gamma_free = beta * delta_elo + alpha;

  vector[N_players] gamma = append_row([0]', gamma_free);
}

model {
  // linear skill relationship with first elo
  beta ~ normal(0.004, 0.004 / 2.57);
  
  // skill diff from baseline
  mu_alpha ~ normal(0,1);
  tau_alpha ~ normal(0,3/2.57);
  
  eta_ncp ~ normal(0,1);
  eta_cp ~ normal(mu_alpha, tau_alpha);
  
  // Prior model
  cut_points ~ induced_dirichlet(rho/tau + rep_vector(1, K));

  // Observational model
  y ~ ordered_logistic(gamma[white_idxs] - gamma[black_idxs], cut_points);
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
    
    y_pred[n] = ordered_logistic_rng(gamma[white_idxs[n]] - gamma[black_idxs[n]], cut_points);
    
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

