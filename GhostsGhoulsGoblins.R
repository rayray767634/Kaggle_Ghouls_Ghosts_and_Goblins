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

ggg.test <- vroom("gggtest.csv")


ggg.train <- vroom("gggtrain.csv")

my_recipe <- recipe(type~bone_length + rotting_flesh + hair_length + has_soul + color , data = missingtrain) %>%
  step_impute_mean(bone_length) %>%
  step_impute_mean(rotting_flesh) %>%
  step_impute_mean(hair_length) 

prep <- prep(my_recipe)
baked <- bake(prep, new_data = missingtrain)

View(baked)
rmse_vec(ggg.train[is.na(missingtrain)], baked[is.na(missingtrain)])

my_recipe <- recipe(type~bone_length + rotting_flesh + hair_length + has_soul + color,data = ggg.train) %>%
  step_mutate_at(color, fn = factor)


prep <- prep(my_recipe)
baked <- bake(prep, new_data = ggg.train)

# naive bayes

## nb model 
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

# Tune smoothness and Laplace here
# Set up grid of tuning values
tuning_grid_nb <- grid_regular(Laplace(),
                               smoothness(),
                               levels = 5)

# Set up K-fold CV
folds <- vfold_cv(ggg.train, v = 5, repeats = 1)
# Run the CV
CV_results_nb <-nb_wf  %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_nb,
            metrics = metric_set(accuracy))

# Find best tuning parameters
bestTune_nb <- CV_results_nb %>%
  select_best("accuracy")

# Finalize workflow and predict
final_wf_nb <- 
  nb_wf %>%
  finalize_workflow(bestTune_nb) %>%
  fit(data = ggg.train)

# predict
ggg_nb_predictions <- predict(final_wf_nb,
                                 new_data = ggg.test,
                                 type = "class") %>%
  bind_cols(., ggg.test) %>% #Bind predictions with test data
  select(id, .pred_class) %>% #Just keep id and pred_1
  rename(type=.pred_class) #rename pred1 to type (for submission to Kaggle)

vroom_write(x=ggg_nb_predictions, file="./GGGNBPreds.csv", delim=",")
