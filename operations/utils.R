library(tidyverse)


get_data <- function(dname){
  all_files <- list.files(path=dname, pattern=".csv", all.files=FALSE,
                          full.names=TRUE)
  
  full_data  <- read_csv(all_files[1])
  
  for(fname in all_files[2:length(all_files)]){
    tmp_data  <- read_csv(fname)
    full_data <- full_data %>% 
      bind_rows(tmp_data)
  }
  
  return(full_data)
}

