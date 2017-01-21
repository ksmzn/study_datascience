library(stringr)
library(rvest)
library(dplyr)
library(purrr)
library(lazyeval)
library(pforeach)

# 関数定義
extract_playtimes <-function(match_html, cls, section){
  node_cls <- str_c(cls, " table")
  playtimes_df <- match_html %>% 
    html_nodes(node_cls) %>%
    html_table(fill=TRUE) %>%
    .[[1]] %>%
    .[c(1, 2, 4, 6)] %>%
    set_names(c("Position", "Number", "PlayerName", "Apps")) %>%
    filter(row_number()!=12) %>% 
    as_data_frame %>%
    mutate_(Apps = lazyeval::interp(~ as.integer(v1), v1 = as.name("Apps")),
            Number = lazyeval::interp(~ as.integer(v2), v2 = as.name("Number"))) %>% 
    mutate(Section = section)
  return(playtimes_df)
}

create_section_playtimes <- function(game, section_number){
  game_html <- read_html(game)
  game_teams <- game_html %>% 
    html_nodes(".preview_name table") %>% 
    html_table() %>% 
    map("X2")
  game_hometeam <- extract_playtimes(game_html, ".myTeamBox", section_number)
  game_awayteam <- extract_playtimes(game_html, ".enemyBox", section_number)
  game_teams_playtimes <- list()
  game_teams_playtimes[[game_teams[[1]]]] <- game_hometeam
  game_teams_playtimes[[game_teams[[2]]]] <- game_awayteam
  return(game_teams_playtimes)
}

# 基本のURL
base_url <- 'http://www.football-lab.jp'


# 全節へのリンクを取得
first_url <- 'http://www.football-lab.jp/y-fm/report/'
first_html <- read_html(first_url)
sections <- first_html %>% html_nodes("div.linkPast a") %>%
  html_attr("href") %>% 
  str_c(base_url, .)

# チーム名を取得
teams <- first_html %>%
  html_nodes(".header_clublist a") %>%
  html_attr("title")

# 各節各試合の出場時間を取得
section <- sections[2]
base_html <- read_html(section)

## 各節の対戦カードへのリンクを取得
games <- base_html %>% html_nodes("div.cardtab a") %>%
  html_attr("href") %>% 
  str_c(base_url, .)

## 各試合の情報を取得
game_section_playtimes <- npforeach(game = games, i = 1:length(games), .multicombine = T, .combine = c)({
  cat(i, ":", Sys.time(), "\n")
  Sys.sleep(1)
  create_section_playtimes(game)
})

game2_section_playtimes <- npforeach(game = games, i = 1:length(games), .multicombine = T, .combine = c)({
  cat(i, ":", Sys.time(), "\n")
  Sys.sleep(1)
  create_section_playtimes(game)
})
# teams_playtimes <- npforeach(section = sections, i = 1:length(sections), .combine = list)({
teams_playtimes <- npforeach(section = sections[1:3], i = 1:length(sections[1:3]), .combine = list)({
  base_html <- read_html(section)
  
  ## 各節の対戦カードへのリンクを取得
  games <- base_html %>% html_nodes("div.cardtab a") %>%
    html_attr("href") %>% 
    str_c(base_url, .)
  
  ## 各試合の情報を取得
  game_section_playtimes <- npforeach(game = games, j = 1:length(games), .multicombine = T, .combine = c)({
    cat(i, ":", j, ":", Sys.time(), "\n")
    Sys.sleep(1)
    create_section_playtimes(game, i)
  })
  game_section_playtimes
})
teams_playtimes[[2]]
teams_playtimes_all <- teams_playtimes[[1]]
teams_J1 <- names(teams_playtimes_all)
for(i in 2:length(teams_playtimes)){
  for(team in teams){
    if(team %in% teams_J1){
      teams_playtimes_all[[team]] <- teams_playtimes_all[[team]] %>%
        bind_rows(teams_playtimes[[i]][[team]])
    } else {
      next
    }
  }
}
teams_playtimes_all

team_playtimes$`横浜Ｆ・マリノス` %>% View
team_playtimes <- game1_section_playtimes
team <- teams[19]
team_playtimes <- list()
team_playtimes[team] <- game1_section_playtimes[team]
# team_playtimes[team] <- team_playtimes[team] %>%
for(team in teams) 
team_playtimes[[team]] %>%
  full_join(game2_section_playtimes[[team]] %>% select(-Position), by=c("Number", "PlayerName"))

game1_section_playtimes[[team]] %>% 
  bind_rows(game2_section_playtimes[[team]])

names(game_section_playtimes) %>% sort()

# transfer
url <- 'http://www.football-lab.jp/y-fm/'
team_html <- read_html(url)
# team_html_sjis <- team_html %>% iconv(from = "UTF-8")
team_html %>% html_table()
team_players <- team_html %>% 
  html_node('#sorTable') %>% 
  html_table()
team_players <- team_players %>% 
  set_names(c("Position", "Number", "PlayerName", "Games", "Starting", "Apps", "Attack", "Pass", "Dribble", "Cross", "Shoot", "Defense", "Save")) %>% 
  as_data_frame()

transfer_url <- 'http://www.jleague.jp/special/transfer/2017winter/'
transfer_html <- read_html(transfer_url)

transfer_team <- transfer_html %>% 
  html_node('#team-yokohamafm') %>% 
  html_node('.nyc2colRight') %>% 
  html_node('.dataTable') %>% 
  html_table()

transfer_name <- transfer_team %>% 
  as_data_frame %>% 
  rename(PlayerName = X1) %>% 
  slice(2:n()) %>% 
  mutate(PlayerName = str_replace(PlayerName, '［.*', '')) %>% 
  mutate(transfer_flg = 1)
transfer_name

team_players %>% 
  left_join(transfer_name, by = "PlayerName") %>% 
  mutate(AppsRate=Apps / sum(Apps)) %>% 
  arrange(desc(AppsRate)) %>% View
  group_by(transfer_flg) %>% 
  summarise(Time=sum(Apps)) %>% 
  ungroup()
  
  View
