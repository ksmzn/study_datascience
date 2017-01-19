install.packages("curl")
install.packages("rvest")
install.packages("stringr")
install.packages("stringi")
library(stringr)
library(rvest)
library(dplyr)

base_url <- 'http://www.football-lab.jp'

# 全節へのリンクを取得
first_url <- 'http://www.football-lab.jp/y-fm/report/'
base_html <- read_html(first_url)
sections <- base_html %>% html_nodes("div.linkPast a") %>%
  html_attr("href") %>% 
  str_c(base_url, .)
  # html_structure()

# 各節の対戦カードへのリンクを取得
matches <- base_html %>% html_nodes("div.cardtab a") %>%
  html_attr("href") %>% 
  str_c(base_url, .)

url <- 'http://www.football-lab.jp/y-fm/'
team_html <- read_html(url)
team_html_sjis <- team_html %>% iconv(from = "UTF-8")
team_html %>% html_table()
team_players <- team_html %>% 
  html_node('#sorTable') %>% 
  html_table()
team_players <- team_players %>% 
  magrittr::set_colnames(c("Position", "Number", "PlayerName", "Games", "Starting", "Apps", "Attack", "Pass", "Dribble", "Cross", "Shoot", "Defense", "Save")) %>% 
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
