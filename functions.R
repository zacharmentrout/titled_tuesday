
build_adj_matrix <- function(N_items, N_comparisons, idx1, idx2) {
  if (min(idx1) < 1 | max(idx1) > N_items) {
    stop("Out of bounds idx1 values.")
  }
  if (min(idx2) < 1 | max(idx2) > N_items) {
    stop("Out of bounds idx2 values.")
  }
  
  adj <- matrix(0, nrow=N_items, ncol=N_items)
  
  # Compute directed edges
  for (n in 1:N_comparisons) {
    adj[idx1[n], idx2[n]] <- adj[idx1[n], idx2[n]] + 1
  }
  
  # Compute symmetric, undirected edges
  for (p1 in 1:(N_items - 1)) {
    for (p2 in (p1 + 1):N_items) {
      N1 <- adj[p1, p2]
      N2 <- adj[p2, p1]
      adj[p1, p2] <- N1 + N2
      adj[p2, p1] <- N1 + N2
    }
  }
  
  (adj)
}

elo1 <- function(x) {
  1 / (1 + 10^(-x/400))
}

# test function 
elo2 <- function(x) {
  1 / (1+ exp(-x/173.7178))
}



derived_cut_points <- function(p) {
  K <- length(p)
  c <- rep(NA, length.out=K-1)
  cum_sum <- 0
  for (k in 1:(K-1)) {
    cum_sum <- cum_sum + p[k]
    c[k] <- logit(cum_sum)
  }
  return(c)
}

rdirichlet <- function(n, rho, tau) {
  alpha <- rho / tau + 1
  rBeta2009::rdirichlet(n, alpha)
}


logit <- function(p) {
  if (any(p <= 0 | p >= 1)) stop("Probabilities must be strictly between 0 and 1.")
  log(p / (1 - p))
}

my_q <- function(x) {
  quantile(x, c(0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
}

ordinal_probs_to_cutpoints <- function(probs) {
  # Ensure the probabilities are cumulative
  cum_probs <- cumsum(probs)
  
  # Exclude the last value since the cumulative probability for the final category should be 1
  if (!all.equal(tail(cum_probs, 1), 1)) stop("Probabilities must sum to 1.")
  
  # Compute cutpoints using the logit transformation
  cutpoints <- qlogis(cum_probs[-length(cum_probs)])
  
  return(cutpoints)
}


rordered_logistic <- function(n, cutpoints, shift) {
  # Ensure cutpoints are sorted
  cutpoints <- sort(cutpoints)
  
  # Sample uniform values and map to categorical outcomes
  u <- runif(n)
  
  # Compute cumulative probabilities based on the logistic CDF
  probs <- plogis(cutpoints - shift)
  
  # Compute valid category probabilities
  probs <- c(probs[1], diff(probs), 1 - tail(probs, 1))
  
  # Sample from the categorical distribution
  categories <- sample(seq_along(probs), size = n, replace = TRUE, prob = probs)
  
  return(categories)
  #probs
}

probs_ordered_logistic <- function(n, cutpoints, shift) {
  # Ensure cutpoints are sorted
  cutpoints <- sort(cutpoints)
  
  # Sample uniform values and map to categorical outcomes
  u <- runif(n)
  
  # Compute cumulative probabilities based on the logistic CDF
  probs <- plogis(cutpoints - shift)
  
  # Compute valid category probabilities
  probs <- c(probs[1], diff(probs), 1 - tail(probs, 1))
  
  # Sample from the categorical distribution
  categories <- sample(seq_along(probs), size = n, replace = TRUE, prob = probs)
  
  return(probs)
}


plot_cut_point_overlay <- function(expectand_vals_list, prefix,
                                   flim, fname, ylim, main=NULL) {
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim, main=main)
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
}


plot_cut_point_post_vs_prior_overlay <- function(expectand_vals_list, prefix, prefix_prior,
                                   flim, fname, ylim, main=NULL) {
  # posterior
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim, main=main)
  
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  # prior
  name <- paste0(prefix_prior, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark_teal, border="#DDDDDDDD",
                                  ylim=ylim,
                                  add=TRUE)
  
  name <- paste0(prefix_prior, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)  
  
}

plot_ordinal_post_vs_prior_overlay <- function(expectand_vals_list, prefix, prefix_prior,
                                                 flim, fname, ylim, main=NULL) {
  # prior
  name <- paste0(prefix_prior, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark_teal, border="#DDDDDDDD",
                                  ylim=ylim,
                                  main=main)
  
  name <- paste0(prefix_prior, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)  
  
  name <- paste0(prefix_prior, 3, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_light_teal,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  # posterior
  name <- paste0(prefix, 1, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim, display_name=fname,
                                  col=util$c_dark, border="#DDDDDDDD",
                                  ylim=ylim,
                                  add=TRUE)
  
  name <- paste0(prefix, 2, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_mid_highlight,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  
  name <- paste0(prefix, 3, ']')
  util$plot_expectand_pushforward(expectand_vals_list[[name]],
                                  45, flim=flim,
                                  col=util$c_light,
                                  border="#DDDDDDDD",
                                  add=TRUE)
  

  
}




clean_chess_data <- function(games, max_tournament_idx=8) {
  # 1. Create basic indices and clean dates
  games$tournament_date <- gsub('\\.', '-', games$Date)
  unique_dates <- sort(unique(games$tournament_date))
  games$tournament_idx <- match(games$tournament_date, unique_dates)
  
  # Sort by round and tournament for consistent ordering
  games <- games[order(games$Round, games$tournament_idx), ]
  games$tournament_round_idx <- as.integer(factor(interaction(games$Round, games$tournament_idx)))
  
  # Filter to first 8 tournaments if needed
  games <- games[games$tournament_idx <= max_tournament_idx, ]
  games <- games[order(games$tournament_round_idx), ]
  games$game_idx <- seq_len(nrow(games))
  
  # 2. Create outcome mappings
  score_map_white <- c("1-0" = 1, "0-1" = 0, "1/2-1/2" = 0.5)
  score_map_black <- c("1-0" = 0, "0-1" = 1, "1/2-1/2" = 0.5)
  score_map_ordinal <- c("0-1" = 1, "1/2-1/2" = 2, "1-0" = 3)
  
  games$white_score <- score_map_white[games$Result]
  games$black_score <- score_map_black[games$Result]
  games$outcome_ordinal <- score_map_ordinal[games$Result]
  
  # 3. Create player-level dataset first (combines white and black games)
  temp_players <- data.frame(
    player = unique(c(games$White, games$Black)),
    stringsAsFactors = FALSE
  )
  
  # Add initial Elo for each player (use first appearance)
  player_elos <- data.frame(
    player = c(games$White, games$Black),
    elo = c(games$WhiteElo, games$BlackElo),
    stringsAsFactors = FALSE
  )
  player_elos <- player_elos[!duplicated(player_elos$player), ]
  
  # Merge and sort by Elo descending
  temp_players <- merge(temp_players, player_elos, by = "player")
  temp_players <- temp_players[order(-temp_players$elo, temp_players$player), ]
  
  # Assign player indices in descending Elo order
  temp_players$player_idx <- seq_len(nrow(temp_players))
  
  # Create the mapping
  player_idx_map <- setNames(temp_players$player_idx, temp_players$player)
  
  # 4. Create a common ID for games to help with sequencing
  # Include tournament_idx to avoid duplicate IDs from same-day tournaments
  games$game_id <- paste0(games$tournament_idx, '-', games$Date, '-', sprintf("%02d", games$Round))
  
  # 5. Create player-game data (long format) - temporarily needed for calculations
  white_data <- data.frame(
    player = games$White,
    player_idx = player_idx_map[games$White],
    player_color = "White",
    game_idx = games$game_idx,
    game_id = games$game_id,
    tournament_idx = games$tournament_idx,
    tournament_round_idx = games$tournament_round_idx,
    Date = games$Date,
    Round = games$Round,
    Elo = games$WhiteElo,
    opponent = games$Black,
    opponent_idx = player_idx_map[games$Black],
    opponent_Elo = games$BlackElo,
    score = games$white_score,
    Result = games$Result,
    stringsAsFactors = FALSE
  )
  
  black_data <- data.frame(
    player = games$Black,
    player_idx = player_idx_map[games$Black],
    player_color = "Black",
    game_idx = games$game_idx,
    game_id = games$game_id,
    tournament_idx = games$tournament_idx,
    tournament_round_idx = games$tournament_round_idx,
    Date = games$Date,
    Round = games$Round,
    Elo = games$BlackElo,
    opponent = games$White,
    opponent_idx = player_idx_map[games$White],
    opponent_Elo = games$WhiteElo,
    score = games$black_score,
    Result = games$Result,
    stringsAsFactors = FALSE
  )
  
  games_long <- rbind(white_data, black_data)
  games_long <- games_long[order(games_long$player_idx, games_long$tournament_idx, games_long$game_id), ]
  
  # 6. Calculate first game stats and player metrics
  games_long$game_num <- ave(
    rep(1, nrow(games_long)), 
    games_long$player_idx, 
    FUN = cumsum
  )
  
  # First game info per player
  player_first_games <- games_long[games_long$game_num == 1, ]
  player_first_games <- player_first_games[order(-player_first_games$Elo, player_first_games$player), ]
  
  # 7. Calculate tournament scores and cumulative games
  games_long$player_total_pre_game_tournament_score <- ave(
    games_long$score, 
    games_long$player_idx, 
    games_long$tournament_idx,
    FUN = function(x) c(0, cumsum(x)[-length(x)])
  )
  
  games_long$player_total_post_game_tournament_score <- 
    games_long$player_total_pre_game_tournament_score + games_long$score
  
  games_long$player_prior_cumulative_games <- ave(
    rep(1, nrow(games_long)), 
    games_long$player_idx,
    FUN = function(x) c(0, cumsum(x)[-length(x)])
  )
  

  tournament_counts <- aggregate(
    tournament_idx ~ player_idx, 
    data = games_long, 
    FUN = function(x) length(unique(x))
  )
  names(tournament_counts)[2] <- "total_tournaments"
  
  # Create a data frame of player-tournament combinations
  player_tournaments <- unique(games_long[, c("player_idx", "tournament_idx")])
  player_tournaments <- player_tournaments[order(player_tournaments$player_idx, player_tournaments$tournament_idx), ]
  
  # Add sequence number for each tournament per player
  player_tournaments$tournament_seq <- ave(
    rep(1, nrow(player_tournaments)), 
    player_tournaments$player_idx, 
    FUN = seq_along
  )
  
  # Now for each game, get the tournament sequence number
  games_long <- merge(
    games_long, 
    player_tournaments, 
    by = c("player_idx", "tournament_idx"), 
    all.x = TRUE
  )
  
  # Prior tournaments is sequence number minus 1
  games_long$player_prior_distinct_tournaments <- games_long$tournament_seq - 1
  
  
  # 7.1 Create tournament-player table
  
  # Multiple statistics at once
  tournament_player_stats <- as.data.frame(do.call(rbind, 
                           by(games_long, 
                              list(games_long$tournament_idx, games_long$player_idx), 
                              function(x) {
                                data.frame(
                                  tournament_idx = x$tournament_idx[1],  # This preserves the g1 value
                                  player_idx = x$player_idx[1],  # This preserves the g2 value
                                  count_rounds = length(unique(x$Round)),
                                  first_round = min(x$Round),
                                  last_round = max(x$Round),
                                  tournament_score = sum(x$score)
                                )
                              })))
  
  
  tournament_player_stats$late_join <- tournament_player_stats$first_round > 1
  tournament_player_stats$drop_out <- tournament_player_stats$last_round < 11
  tournament_player_stats$irregular_participation <- tournament_player_stats$count_rounds < 11 & tournament_player_stats$last_round == 11 & tournament_player_stats$first_round == 1
  
  # 8. Create complete player table with all player-level metrics
  # Create aggregate functions to calculate total games, wins, losses, draws
  player_stats <- aggregate(
    cbind(
      games = 1,
      wins = score == 1,
      draws = score == 0.5,
      losses = score == 0
    ) ~ player_idx + player,
    data = games_long,
    FUN = sum
  )
  
  # Add first game information
  player_data <- merge(
    player_stats,
    player_first_games[, c("player_idx", "Date", "tournament_idx", "Elo", "game_id")],
    by = "player_idx"
  )
  
  names(player_data)[names(player_data) == "Date"] <- "first_game_date"
  names(player_data)[names(player_data) == "tournament_idx"] <- "first_tournament_idx"
  names(player_data)[names(player_data) == "Elo"] <- "first_elo"
  names(player_data)[names(player_data) == "game_id"] <- "first_game_id"
  
  # Calculate win percentage
  player_data$win_percentage <- player_data$wins / player_data$games * 100
  
  # Add last game information
  last_games <- by(
    games_long,
    games_long$player_idx,
    function(x) x[which.max(x$game_num), c("Date", "tournament_idx", "Elo", "game_id")]
  )
  
  last_game_data <- do.call(rbind, last_games)
  last_game_data <- data.frame(
    player_idx = as.integer(row.names(last_game_data)),
    last_game_date = last_game_data$Date,
    last_tournament_idx = last_game_data$tournament_idx,
    last_elo = last_game_data$Elo,
    last_game_id = last_game_data$game_id,
    stringsAsFactors = FALSE
  )
  
  player_data <- merge(player_data, last_game_data, by = "player_idx")
  
  # Calculate Elo change
  player_data$elo_change <- player_data$last_elo - player_data$first_elo
  
  # 9. Add player indices to games table
  games$white_player_idx <- player_idx_map[games$White]
  games$black_player_idx <- player_idx_map[games$Black]
  
  # 10. Join player-level information back to games
  # Prepare white and black specific game data
  white_game_data <- games_long[games_long$player_color == "White", 
                                c("game_idx", "player_idx", "player_total_pre_game_tournament_score", 
                                  "player_total_post_game_tournament_score", "player_prior_cumulative_games",
                                  "player_prior_distinct_tournaments")]
  names(white_game_data)[-c(1,2)] <- paste0("white_", names(white_game_data)[-c(1,2)])
  
  black_game_data <- games_long[games_long$player_color == "Black", 
                                c("game_idx", "player_idx", "player_total_pre_game_tournament_score", 
                                  "player_total_post_game_tournament_score", "player_prior_cumulative_games",
                                  "player_prior_distinct_tournaments")]
  names(black_game_data)[-c(1,2)] <- paste0("black_", names(black_game_data)[-c(1,2)])
  
  # Add white-specific data
  games <- merge(games, white_game_data, 
                 by.x = c("game_idx", "white_player_idx"), 
                 by.y = c("game_idx", "player_idx"))
  
  # Add black-specific data
  games <- merge(games, black_game_data, 
                 by.x = c("game_idx", "black_player_idx"), 
                 by.y = c("game_idx", "player_idx"))
  
  # Add first Elo and total games for each player
  games <- merge(games, player_data[, c("player_idx", "first_elo", "games")], 
                 by.x = "white_player_idx", by.y = "player_idx")
  names(games)[ncol(games) - 1] <- "white_first_elo"
  names(games)[ncol(games)] <- "white_total_games"
  
  games <- merge(games, player_data[, c("player_idx", "first_elo", "games")], 
                 by.x = "black_player_idx", by.y = "player_idx")
  names(games)[ncol(games) - 1] <- "black_first_elo"
  names(games)[ncol(games)] <- "black_total_games"
  
  # Ensure proper ordering
  games <- games[order(games$tournament_round_idx, games$game_idx), ]
  player_data <- player_data[order(player_data$player_idx), ]
  
  # Return only the final needed data frames
  return(list(games = games, players = player_data, tournaments_players = tournament_player_stats, games_long = games_long))
}





