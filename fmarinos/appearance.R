install.packages("curl")
install.packages("rvest")
install.packages("stringr")
install.packages("stringi")
library(stringr)
library(rvest)
library(dplyr)

url <- 'http://www.football-lab.jp/y-fm/'
team_html <- read_html(url)
team_html_sjis <- team_html %>% iconv(from = "UTF-8")
team_html %>% html_table()
team_players <- team_html %>% 
  html_node('#sorTable') %>% 
  html_table()
colnames(team_players) <- c("Position", "Number", "Name", "Games", "Starting", "Apps", "Attack", "Pass", "Dribble", "Cross", "Shoot", "Defense", "Save")
team_players <- team_players %>% 
  as_data_frame()

transfer_url <- 'http://www.jleague.jp/special/transfer/2017winter/'
transfer_html <- read_html(transfer_url)

transfer_team <- transfer_html %>% 
  html_node('#team-yokohamafm') %>% 
  html_node('.nyc2colRight') %>% 
  html_node('.dataTable') %>% 
  html_table()
colnames(transfer_team) <- C
transfer_team %>% 
  as_data_frame %>% 
  rename(Name = X1)1
j１jjjjjjjjjj
  mutate()
