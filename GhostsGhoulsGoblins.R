library(vroom)
library(tidyverse)
library(dplyr)
library(patchwork)
library(tidymodels)
library(glmnet)
library(stacks)
library(embed)
library(discrim)
library(naivebayes)
library(kknn)
library(kernlab)
library(themis)

missingtrain <- vroom("trainWithMissingValues.csv")%>% 
  mutate_at('id', as.character) %>%
  mutate_at('color', as.factor) %>%
  mutate_at('type', as.factor) %>%
  select(-id)

ggg.test <- vroom("gggtest.csv")%>% 
  mutate_at('id', as.factor) %>%
  select(-id)

ggg.train <- vroom("gggtrain.csv")%>% 
  mutate_at('id', as.factor)%>%
  select(-id)

my_recipe <- recipe(type~bone_length + rotting_flesh + hair_length + has_soul + color , data = missingtrain) %>%
  step_impute_mean(bone_length) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_mean(hair_length) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = missingtrain)

View(baked)
rmse_vec(ggg.train[is.na(missingtrain)], baked[is.na(missingtrain)])
