
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


#################################
# Mod data
#################################

mod_data <- list(
  "N_games" = nrow(games),
  "K" = 3,
  "y" = games$outcome_ordinal,
  "z" = games$WhiteElo - games$BlackElo,
  "N_players" = nrow(players),
  "white_idxs" = games$white_player_idx,
  "black_idxs" = games$black_player_idx
)

#################################
# Fit model
#################################
mod <- stan(file=file.path("stan_programs/elo_davidson.stan"), 
            data=mod_data, warmup=1000, iter=2000,
            control = list(adapt_delta = 0.8)
              
)

map_fit <- optimizing(stan_model("stan_programs/elo_davidson.stan"), data = mod_data)
map_estimates <- as.list(map_fit$par)
print(map_estimates[['kappa']])
pred_names <- grep('y_pred',names(map_estimates), value=T)
y_pred_map <- map_estimates[pred_names]

# All Hamiltonian Monte Carlo diagnostics are clear.
diagnostics <- util$extract_hmc_diagnostics(mod)
util$check_all_hmc_diagnostics(diagnostics)

#extract_mod <- extract(mod)

# The diagnostics for all parameter expectands are clear.
mod_samples <- util$extract_expectand_vals(mod)
names <- c(
  'kappa'
)
  
base_samples <- util$filter_expectands(mod_samples, names)
util$check_all_expectand_diagnostics(base_samples)

# Posterior retrodictive check
par(mfrow=c(1, 1))
pred_names <- grep('y_pred', names(mod_samples), value=TRUE)
util$plot_hist_quantiles(samples = mod_samples, bin_min = 0.5, bin_max = 3.5, bin_delta = 1,
                         val_name_prefix = "y_pred", baseline_values = mod_data$y,
                         xlab = "Outcome")


util$plot_line_hists(mod_data$y, as.numeric(y_pred_map), bin_min = 0.5, bin_max = 3.5, bin_delta=1)

##########
# Posterior vs. Prior
##########
# kappa
par(mfrow=c(1,1))
util$plot_expectand_pushforward(mod_samples[["kappa"]], 25,
                                display_name="Kappa", flim = c(0, 1))
xs <- seq(0, 1, 0.01)
ys <- dunif(xs,0, 1)
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




