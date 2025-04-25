
############################################################
# Configure Graphics
############################################################

c_light <- c("#DCBCBC")
c_light_highlight <- c("#C79999")
c_mid <- c("#B97C7C")
c_mid_highlight <- c("#A25050")
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")

c_light_teal <- c("#6B8E8E")
c_mid_teal <- c("#487575")
c_dark_teal <- c("#1D4F4F")

par(family="serif", las=1, bty="l", 
    cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 5))

library(colormap)
library(scales)
nom_colors <- c("#DCBCBC", "#C79999", "#B97C7C", "#A25050", "#8F2727", "#7C0000")

############################################################
# Working directory, libraries
############################################################

setwd('~/Documents/git/titled_tuesday/')

par(family="serif", las=1, bty="l",
    cex.axis=1, cex.lab=1, cex.main=1,
    xaxs="i", yaxs="i", mar = c(5, 5, 3, 1))

library(rstan)
library(stringr)
library(colormap)

############################################################
# Set up Stan
############################################################
rstan_options(auto_write = TRUE)            # Cache compiled Stan programs
options(mc.cores = parallel::detectCores()) # Parallelize chains
parallel:::setDefaultClusterOptions(setup_strategy = "sequential")

util <- new.env()
source('mcmc_analysis_tools_rstan.R', local=util)
source('mcmc_visualization_tools.R', local=util)
source('functions.R', local=util)

# Load data
games_init <- read.csv(unz('data/tt_2021.zip', "tt_2021.csv"))
games_prior <-
  rbind(read.csv(unz('data/tt_2020.zip', 'tt_2020.csv')),
        read.csv(unz('data/tt_2019.zip', 'tt_2019.csv')),
        read.csv(unz('data/tt_2018.zip', 'tt_2018.csv')),
        read.csv(unz('data/tt_2017.zip', 'tt_2017.csv'))
        )

clean_data <- util$clean_chess_data(games_init, max_tournament_idx = 8)
games <- clean_data$games
players <- clean_data$players
tournaments_players <- clean_data$tournaments_players
games_long <- clean_data$games_long

tournaments_players <- merge(tournaments_players, players[,c('player_idx', 'first_elo')], by = 'player_idx')

util$plot_line_hist(tournament_players$first)


unique_tournaments <- unique(games[,c('tournament_idx', 'tournament_date', 'Event')])
unique_tournaments[order(unique_tournaments$tournament_idx),]

player1_elo <- players$first_elo[players$player_idx==1]
delta_elo <- (players$first_elo[2:nrow(players)] - player1_elo)
diff_elo_white_v_black <- games$white_first_elo - games$black_first_elo

#################
# Inform prior
#################
games_prior_close_elo <- games_prior[abs(games_prior$WhiteElo - games_prior$BlackElo) < 10 , ]
table(games_prior_close_elo$Result) / nrow(games_prior_close_elo)
nrow(games_prior_close_elo)

games_prior_far_elo <- games_prior[games_prior$WhiteElo - games_prior$BlackElo < -500 , ]
table(games_prior_far_elo$Result) / nrow(games_prior_far_elo)
nrow(games_prior_far_elo)


n <- 1000
c <- c(-0.5,0.5)
shift <- -0.5
x <- util$rordered_logistic(1,c, shift)

rho <- c(0.44, 0.12, 0.44)
tau <- 0.05
p_sample <- util$rdirichlet(n, rho, tau)
cp_sample <- t(apply(p_sample, 1, function(x) {util$ordinal_probs_to_cutpoints(x)}))
elo_diff <- 10
shift <- elo_diff / 200 
outcomes_sample <- sapply(1:n, function(x) { util$rordered_logistic(1, cp_sample[x,], shift) })
probs_sample_white_wins <- sapply(1:n, function(x) { util$probs_ordered_logistic(1, cp_sample[x,], shift) })[3,]
mean(probs_sample_white_wins)
scores_sample <- c(0, 0.5, 1)[outcomes_sample]
table(outcomes_sample)/length(outcomes_sample)
mean(scores_sample)
util$elo1(elo_diff)

hist(p_sample[,2])
util$my_q(p_sample[,2])
mean(p_sample[,2])


alt <- util$clean_chess_data(games_init)
games_alt <- alt$games
players_alt <- alt$players

players_alt[1,]


dim(games_alt)
dim(games)

games[games$tournament_round_idx == 1,][1:10,]

sum(games$player_black_total_pre_game_tournament_score)
sum(games_alt$black_player_total_pre_game_tournament_score)
#################################
# EDA
#################################
par(mfrow=c(1, 1), mar=c(5, 5, 2, 1))

mean_outcome_player_white <-
  sapply(1:length(unique(games$white_player_idx)),
         function(c) mean(games$outcome_ordinal[games$white_player_idx == c]))
util$plot_line_hist(mean_outcome_player_white,
                    0, 3.5, 0.25,
                    xlab="White Player-wise Average Game Outcomes")

mean_outcome_player_white[is.nan(mean_outcome_player_white)] <- 0

mean_outcome_player_black <-
  sapply(1:length(unique(games$black_player_idx)),
         function(c) mean(games$outcome_ordinal[games$black_player_idx == c]))
util$plot_line_hist(mean_outcome_player_black,
                    0, 3.5, 0.25,
                    xlab="Black Player-wise Average Game Outcomes")
mean_outcome_player_black[is.nan(mean_outcome_player_black)] <- 0


mean_outcome_tournament_round <-
  sapply(1:length(unique(games$tournament_round_idx)),
         function(c) mean(games$outcome_ordinal[games$tournament_round_idx == c]))
util$plot_line_hist(mean_outcome_tournament_round,
                    0, 3.5, 0.25,
                    xlab="Tournament-Round-level Average Game Outcomes")


# Multiple statistics at once
summary_stats <- do.call(rbind, 
                         by(games, 
                            list(games$tournament_idx, games$player), 
                            function(x) {
                              data.frame(
                                count = nrow(x),
                                sum_value = sum(x$value),
                                distinct_ids = length(unique(x$id))
                              )
                            }))

#################################
# Mod data
#################################
delta_elo_grid <- c(0, -100, -200, -300, -400, -500, -600, -700, -800, -900, -1000) 

players_free <- players[2:nrow(players),]
thresh <- 1000
ncp_idx <- which(players_free$games <= thresh)
cp_idx <- which(players_free$games > thresh)

length(ncp_idx)
length(cp_idx)

mod_data <- list(
  "N_games" = nrow(games),
  "N_players" = nrow(players),
  "K" = 3,
  "white_idxs" = games$white_player_idx,
  "black_idxs" = games$black_player_idx,
  "delta_elo" = delta_elo,
  "rho" = c(0.44, 0.12, 0.44),
  "tau" = 0.05,
  "y" = games$outcome_ordinal,
  "delta_elo_grid" = delta_elo_grid,
  "N_delta_elo_grid" = length(delta_elo_grid),
  "ncp_idx" = array(ncp_idx),
  "cp_idx" = array(cp_idx),
  "N_players_ncp" = length(ncp_idx),
  "N_players_cp" = length(cp_idx)
)

#################################
# Prior samples
#################################
mod_prior <- stan(file=file.path("stan_programs/model8_prior.stan"), 
                           data=mod_data, seed=78100028, algorithm="Fixed_param",
                           iter=1000, chains=1, warmup=0
)

#extract_mod_prior <- extract(mod_prior)

# The diagnostics for all parameter expectands are clear.
mod_samples_prior <- util$extract_expectand_vals(mod_prior)

# Prior predictive
par(mfrow=c(1, 1))
pred_names <- grep('', names(mod_samples_prior), value=TRUE)
util$plot_hist_quantiles(samples = mod_samples_prior,bin_min = 0.5, bin_max=3.5, bin_delta=1,
                         val_name_prefix = "y_pred", 
                         xlab = "Outcome")


for (i in 1:10) {
  pred_grid_names <- grep(paste0('ordinal_probs_grid\\[',i, ','), names(mod_samples_prior), value=TRUE)
  util$plot_expectand_pushforward(mod_samples_prior[[pred_grid_names[1]]], 45, flim=c(0.8,1.1),
                                  col=util$c_light,
                                  border="#DDDDDDDD",
                                  main = paste0('elo diff = ', delta_elo_grid[i])
                                  )

  util$plot_expectand_pushforward(mod_samples_prior[[pred_grid_names[2]]], 45,# flim=c(0,1),
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD", add=T)

  # util$plot_expectand_pushforward(mod_samples_prior[[pred_grid_names[3]]], 45, flim=c(0,1),
  #                                 col=util$c_dark_teal,
  #                                 border="#DDDDDDDD",
  #                                 add=T)
}

mean(mod_samples_prior[["ordinal_probs_grid[10,1]"]] > 0.999)


#################################
# Fit model
#################################
# rm(games_prior)
# rm(games_init)
# rm(cp_sample)
# rm(games_black)
# rm(games_white)
# rm(games_long)
# rm(games_prior_close_elo)
# rm(games_prior_far_elo)
# rm(graph)
# rm(unique_tournaments)

mod <- stan(file=file.path("stan_programs/model8c.stan"), 
              data=mod_data, warmup=1000, iter=2000,
            control = list(adapt_delta = 0.8)
)

# All Hamiltonian Monte Carlo diagnostics are clear.
diagnostics <- util$extract_hmc_diagnostics(mod)
util$check_all_hmc_diagnostics(diagnostics)

#extract_mod <- extract(mod)


# The diagnostics for all parameter expectands are clear.
mod_samples <- util$extract_expectand_vals(mod)
names <- c(
  paste0('gamma_free[', 1:(nrow(players)-1),']'),
  paste0('cut_points[',1:2,']'),
  paste0('alpha[',1:(nrow(players)-1), ']'),
  'mu_alpha',
  'tau_alpha'
)
  
base_samples <- util$filter_expectands(mod_samples, names)
util$check_all_expectand_diagnostics(base_samples)

# Posterior retrodictive check
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_hist_quantiles(samples = mod_samples, bin_min = 0.5, bin_max = 3.5, bin_delta = 1,
                         val_name_prefix = "y_pred", baseline_values = mod_data$y,
                         xlab = "Outcome")


# Cut points
util$plot_cut_point_overlay(mod_samples, 'cut_points[',
                       flim=c(-1, 1), fname='Interior Cut Points', ylim=c(0,20))


util$plot_cut_point_post_vs_prior_overlay(mod_samples, prefix = 'cut_points[', prefix_prior = 'cut_points_prior[',
                            flim=c(-1.5, 1.5), fname='Interior Cut Points', ylim=c(0,15))


# Posterior ordinal probs vs. prior
util$plot_ordinal_post_vs_prior_overlay(mod_samples, prefix = 'ordinal_probs[', prefix_prior = 'ordinal_probs_prior[',
                                          flim=c(0,1), fname='Ordinal Probs', ylim=c(0,50))



##########
# Posterior vs. Prior
##########
# Beta
par(mfrow=c(1,1))
util$plot_expectand_pushforward(mod_samples[["beta"]], 25,
                                display_name="Beta", flim = c(-0.001,0.008))
xs <- seq(-0.001, 0.008, 0.0001)
ys <- dnorm(xs,0.004, 0.004/2.57)
lines(xs, ys, lwd=2, col=c_light_teal)

# Population Mean
par(mfrow=c(1,1))
util$plot_expectand_pushforward(mod_samples[["mu_alpha"]], 25,
                                display_name="Population Mean", flim = c(-3,3))
xs <- seq(-3, 3, 0.01)
ys <- dnorm(xs,0, 1)
lines(xs, ys, lwd=2, col=c_light_teal)

# Population Scale
par(mfrow=c(1,1))
util$plot_expectand_pushforward(mod_samples[["tau_alpha"]], 25,
                                display_name="Population Scale", flim = c(0,3))
xs <- seq(0, 3, 0.01)
ys <- abs(dnorm(xs,0, 3/2.57))
lines(xs, ys, lwd=2, col=c_light_teal)


# Tournaments
par(mfrow=c(4,2), mar=c(5, 5, 1, 1))
for (c in 1:8) {
  names <- sapply(which(games$tournament_idx == c),
                      function(n) paste0('y_pred[', n, ']'))

  
  filtered_samples <- util$filter_expectands(mod_samples, names)
  outcomes <- games$outcome_ordinal[games$tournament_idx == c]
  
  util$plot_hist_quantiles(filtered_samples, 'y_pred',
                           0.5, 3.5, 1,
                           baseline_values=outcomes,
                           xlab="Outcome",
                           main=paste('Tournament', c))
}


# Rounds
for (c in 1:11) {
  if(c %% 2 == 1) {
    par(mfrow=c(1,2), mar=c(5, 5, 1, 1))
  }
  
  names <- sapply(which(games$Round == c),
                  function(n) paste0('y_pred[', n, ']'))
  
  
  filtered_samples <- util$filter_expectands(mod_samples, names)
  outcomes <- games$outcome_ordinal[games$Round == c]
  
  util$plot_hist_quantiles(filtered_samples, 'y_pred',
                           0.5, 3.5, 1,
                           baseline_values=outcomes,
                           xlab="Outcome",
                           main=paste('Round', c),
                           display_ylim=c(0,4000))
}


# Player skill
par(mfrow=c(1, 1), mar=c(5, 5, 1, 1))

names <- sapply(1:mod_data$N_players,
                function(i) paste0('gamma[', i, ']'))
util$plot_disc_pushforward_quantiles(mod_samples,
                                     names,
                                     #names,#[sample(1:mod_data$N_players,100)],
                                     xlab="Player",
                                     ylab="Relative Skill")


names <- sapply(1:(mod_data$N_players-1),
                function(i) paste0('alpha[', i, ']'))
util$plot_disc_pushforward_quantiles(mod_samples,
                                     #names[1:100], 
                                     names,#[sample(1:mod_data$N_players,100)],
                                     xlab="Player",
                                     ylab="Relative Skill Diff. from Baseline")
abline(h=0)


# A few players
par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))
for (c in c(2, 1241, 687, 63)) {
  names_white <- sapply(which(mod_data$white_idxs == c),
                        function(n) paste0('y_pred[', n, ']'))
  
  names_black <- sapply(which(mod_data$black_idxs == c),
                        function(n) paste0('y_pred[', n, ']'))
  
  filtered_samples_white <- util$filter_expectands(mod_samples, names_white)
  filtered_samples_black <- util$filter_expectands(mod_samples, names_black)
  
  outcomes_white <- mod_data$y[mod_data$white_idxs == c]
  outcomes_black <- mod_data$y[mod_data$black_idxs == c]
  
  util$plot_hist_quantiles(filtered_samples_white, 'y_pred',
                           0.5, 3.5, 1,
                           baseline_values=outcomes_white,
                           xlab="Outcome",
                           main=paste('Player', c, '- White'))
  
  util$plot_hist_quantiles(filtered_samples_black, 'y_pred',
                           0.5, 3.5, 1,
                           baseline_values=outcomes_black,
                           xlab="Outcome",
                           main=paste('Player', c, '- Black'))
}

# Player-wise
par(mfrow=c(2, 1), mar=c(5, 5, 1, 1))

null_white <- which(mean_outcome_player_white == 0)
pred_names_white <- paste0('mean_outcome_player_white_pred[',setdiff(1:nrow(players), null_white),']')
util$plot_hist_quantiles(mod_samples[pred_names_white], 'mean_outcome_player_white_pred',
                         0, 3, 0.5,
                         baseline_values=mean_outcome_player_white[-null_white],
                         xlab="Player-wise White Average Outcomes")

null_black <- which(mean_outcome_player_black == 0)
pred_names_black <- paste0('mean_outcome_player_black_pred[',setdiff(1:nrow(players), null_black),']')
util$plot_hist_quantiles(mod_samples[pred_names_black], 'mean_outcome_player_black_pred',
                         0, 3, 0.5,
                         baseline_values=mean_outcome_player_black,
                         xlab="Player-wise Black Average Outcomes")


# tournaments so far
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$white_player_prior_distinct_tournaments, baseline_values = games$outcome_ordinal,bin_min = 0, bin_max=10,bin_delta = 1,display_ylim = c(0,2.5))
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$black_player_prior_distinct_tournaments, baseline_values = games$outcome_ordinal,bin_min = 0, bin_max=10,bin_delta = 1,display_ylim = c(0,2.5))

util$plot_conditional_mean_quantiles(mod_samples,
                                     pred_names,
                                     obs_xs = games$player_white_total_pre_game_tournament_score - games$player_black_total_pre_game_tournament_score,
                                     baseline_values = games$outcome_ordinal,
                                     display_ylim = c(0,3.5))

# sub <- which(abs(games$player_white_total_pre_game_tournament_score - games$player_black_total_pre_game_tournament_score) < 0.5)
# pred_names <- paste0('y_pred[', sub,']' )
# util$plot_hist_quantiles(mod_samples[pred_names],val_name_prefix = 'y_pred', baseline_values = games$outcome_ordinal[sub])


# experience difference
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$player_white_prior_cumulative_tt_games - games$player_black_prior_cumulative_tt_games, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.0))

# tournament-round index
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$tournament_round_idx, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.0))


# starting elo 
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$player_white_first_elo, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.5))
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$player_white_first_elo - games$player_black_first_elo, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.5))

# total games
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$count_player_white_total_games_dataset + games$count_player_black_total_games_dataset, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.5))

# up-to-date elo
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$WhiteElo - games$BlackElo, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.5))
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$WhiteElo - games$BlackElo,
                                     baseline_values = games$outcome_ordinal, residual = T)

util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = (games$WhiteElo + games$BlackElo) - (games$player_white_first_elo + games$player_black_first_elo),
                                     baseline_values = games$outcome_ordinal, residual = T)

# Round
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs =games$Round, baseline_values = games$outcome_ordinal, display_ylim = c(0,3.5))


# residuals
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$outcome_ordinal, baseline_values = games$outcome_ordinal,bin_min = 0.5, bin_max=3.5,bin_delta = 1,display_ylim = c(0,4))
util$plot_conditional_mean_quantiles(mod_samples, pred_names,obs_xs = games$outcome_ordinal, baseline_values = games$outcome_ordinal,bin_min = 0.5, bin_max=3.5,bin_delta = 1,residual = T)


# pre-game tournament score white vs. black
util$plot_conditional_mean_quantiles(mod_samples,
                                     pred_names,
                                     obs_xs = games$player_white_total_pre_game_tournament_score - games$player_black_total_pre_game_tournament_score,
                                     baseline_values = games$outcome_ordinal,
                                     display_ylim = c(0,3.5))


# prior and posterior variance
names <- paste0('gamma[',1:nrow(players),']')
prior_means_gamma <- delta_elo
posterior_means_gamma <- sapply(names, function(i) { mean(mod_samples[[i]]) })
posterior_sds_gamma <- sapply(names, function(i) {sd(mod_samples[[i]])})
contraction <- 1 - posterior_sds_gamma / (1/2.32)
mean_gamma_shift <- (posterior_means_gamma - prior_means_gamma) / posterior_sds_gamma

par(mfrow=c(1,1))
plot(contraction, mean_gamma_shift, col=c("#8F272720"), lwd=2, pch=16, cex=0.8, main="Skill - Contraction vs. Mean Shift",
     xlim=c(0, 1), xlab="Posterior Contraction", ylim=c(-5, 5), ylab="Posterior Mean Shift")



mean_gamma_shift
players[players$player_idx==1241,]




