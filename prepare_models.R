# ML Exercise --------------------------------------------------------------------------------

# Author: JAD, DHEIM
# Description: This project is designed to explore ML techniques
# Resources: https://www.tidymodels.org/packages/

library(readxl)
library(lubridate)
library(tidyverse)
library(tidymodels)
library(ggcorrplot)   # correlation matrices in ggplot format
library(skimr)        # summaries of data
library(polycor)      # correlation matrices with factor variables: computing colleation matrices before passing to ggcorrplot
library(labelr)
library(glmnet)       # for model engine
library(tidytext)     # for cleaning produkt
library(stopwords)    # for cleaning produkt (get_stopwords call)
library(progress)     # progress bar

rm(list = ls())

# Remark ----
# Debugging pipe %>%: stick magrittr::debug_pipe(), debug(`%>%`) and undebug(`%>%`) in order to enter
# debugging mode at each pipe location; If you want to debug a certain function (and not only the piped
# value) you can regularily use debugonce(function)



# Read in data --------------------------------------------------------------------------------

kredz <- readRDS(file = "./data/kredz.rds") %>% drop_na(pd)

# sample!
kredz <- initial_split(kredz, 1/30) %>% magrittr::debug_pipe() %>% training() # rm --------------------------------------------------------------------------------

# Prepare labels
sheets <- list(produkt = "produkt", vars = "vars")
labels <- map(sheets, ~readxl::read_excel(path = "./ML exercise/data-raw/labels.xlsx", 
                                          sheet = .x))

labelr::labels$set(labels)



# Product categories --------------------------------------------------------------------------------

# See product_mapping.R for a first try to extract product categories...
# As of now, there are among others the following problems:
# - fest hypothek und hypothek fest -> order matters in script
# - plural -> singular
# - franz -> deutsch
# - separate words festhypothek -> fest + hypothek
# - use tidytext!!
# See product_categories.R for potential starting point(s)



# Variable exploring --------------------------------------------------------------------------------

# See script variable_exploring.R



# Preprocessing --------------------------------------------------------------------------------

# Initial cleaning ----
# It seems not efficient to perform all the preprocessing steps within the recipe itself since not
# the whole dplyr functionality is accessible (see remarks below).

# Clean produkt
kredz <-
  kredz %>%
  mutate(id = row_number()) %>% 
  unnest_tokens(word, produkt, token = "words", drop = F, strip_numeric = T) %>%
  anti_join(get_stopwords("de")) %>%
  anti_join(get_stopwords("fr")) %>%
  group_by(id, produkt) %>%
  mutate(produkt = str_c(word, collapse = "_")) %>%
  ungroup() %>%
  distinct(across(-word))

kredz %>% count(produkt) %>% arrange(desc(n))

kredz <-
  kredz %>%
  # see codebook: orange highlighted variables
  select(datum, bank_id, pd, einzelkredit, bet, befristet, raten, gedeckt, zis, komm, kanton, zis_cap, produkt, noga) %>%
  mutate(bank_id = as.factor(bank_id),
         across(where(is.character), ~as.factor(.x)),
         across(where(is.logical), ~as.factor(.x)),
         across(where(lubridate::is.Date), ~as.factor(.x)))

vapply(kredz, function(x) mean(!is.na(x)), numeric(1))


# Understand variables with help of "Variablen Beschrieb Kreditzinsstatistik.xlsx"
# Variable selection; type setting (factor, indicator, etc.); standardization;  rm zero variance; rm multicollinearity, ....
# Some variables can't easily be generated in a step (group_by -> aggregate step is missing, etc.). This is a little strange, since
# I would like to apply all preprocessing steps directly to the raw data without manipulating it outside of automatically reproducible steps!
# ONE WAY OF DEALING: write a preprocess() function which you can apply before init the recipe.
help(package = "recipes")

# Training and testing data
data_split <- initial_split(kredz, prop = 3/4)
train_data <- training(data_split)
test_data <- testing(data_split)

# Initialize recipe
# A recipe is a description of what steps should be applied to a data set in order to get it ready for data analysis.
# ATTENTION: The ordering of the steps matters!!! See article on recipes -> ordering of steps
base_rec <-
  # A note on the formula: if you want to specify interactions you can use the step_interact() withouth the need to
  # specify the interaction in the formula. Also the formula can be updated when you bundle everything together in a
  # a workflow, using update_formula(). Note that the formula method is used here to declare the variables, their roles 
  # and nothing else. If you use inline functions (e.g. log) it will complain. These types of operations can be added later!
  recipe(pd ~ ., data = head(train_data)) %>%
  # impute missing values
  step_meanimpute(all_numeric()) %>%
  step_modeimpute(all_nominal()) %>% 
  step_unknown(all_nominal()) %>%
  # create dummies (for factor vars)
  step_dummy(all_nominal()) %>%
  # center and scale (for numeric vars)
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  # zero variance
  step_zv(all_numeric()) %>%
  # handle mutlicollinearity
  step_corr(all_numeric())

# check
# The type col would still shows the old variable type which is a little confusing
base_rec
base_rec %>% summary()

# Estimate parameters: For a recipe with at least one preprocessing operation,
# estimate the required parameters from a training set that can be later applied to other data sets.
trained_rec <- prep(base_rec, training = train_data, verbose = T)
trained_rec

# Get tibbles: For a recipe with at least one preprocessing operation that has been trained by prep.recipe(), 
# apply the computations to new data
# ATTENTION: Do not assign these to train_data (or should I???) -> in the workflow below we need the unprepped data!
train_baked <- bake(trained_rec, new_data = train_data) # or should we actually use juice if we don't supply new data?

vapply(train_baked, function(x) mean(!is.na(x)), numeric(1)) %>% unique()

# alternatively: juice(trained_rec)
test_baked <- bake(trained_rec, new_data = test_data)



# Modelling --------------------------------------------------------------------------------

# - specify models (formula, engine)
# - set up validation routine and tuning for hyperparameters
# - create workflow (applicable to each model <- maybe loop over models... or compare Tidytuesday: organize in data.frame)
# - evaluate (create final model and final workflow)
# - check out TidyTuesday Tidymodels NN with keras...

# Trivial case
# Probabilites of default are most often close to 0 -> always predict 0
trivial <- tibble(truth = kredz$pd, pred = 0) %>% drop_na()
yardstick::rmse(trivial, truth, pred)

# Creating folds for CV for tuning hyperparameters
kfolds <- rsample::vfold_cv(train_data, v = 3, repeats = 1) # applied to our train data
# already creates the folds
kfolds

# extract first data.frame which model is trained
analysis(kfolds$splits[[1]])
# extract first data.frame which model is assessed
assessment(kfolds$splits[[1]])

# Tuning models
lm <-
  parsnip::linear_reg(
    # penalty refers to regularization
    penalty = tune(),
    # we can use lasso or ridge regression (1 corresponds to lasso)
    mixture = 1) %>%
  # since we are using a penalty, we have to use a different engine than "lm"
  set_engine("glmnet") %>%
  # actually, this can also be passed as argument in linear_reg and is default...
  set_mode("regression")
lm


# Remark: ----
# This is a form of regression, that constrains/ regularizes or shrinks the coefficient 
# estimates towards zero. In other words, this technique discourages learning a more
# complex or flexible model, so as to avoid the risk of overfitting.
# https://towardsdatascience.com/regularization-in-machine-learning-76441ddcf99a




# TODO ---------------------------------------------------------------------------
# extend modelling with: rf, nn, svn, ...




# Workflow ----
# Bundle together the recipe (preprocessing and the model specification)
# Time and individual fixed effects; Recall recipe formula != model formula
# I create a unique workflow for each modelling procedure (where in this initial
# case the workflows only differ in the recipe).
rec_ti <- base_rec
lm_ti_wf <- workflow() %>%
  add_recipe(rec_ti) %>% 
  add_model(lm) # if no recipe exists you could specify an add_formula() call here...
lm_ti_wf

# time fixed effects only
# update recipe, alternatively one could update the model formula
rec_t <- base_rec %>% step_rm(contains("bank_id"))
lm_t_wf <- workflow() %>%
  add_recipe(rec_t) %>%
  add_model(lm, formula = pd ~ .)

# individual fixed effects only
rec_i <- base_rec %>% step_rm(contains("datum"))
lm_i_wf <- workflow() %>%
  add_recipe(rec_i) %>%
  add_model(lm, formula = pd ~ .)

# Organize workflows in data.frame
# you could even build the model_frame at an earlier stage, passing formulas, models, recipes, etc. individually
model_frame <- tibble(desc = c("time + individual", "time", "individual"),
                      rec = list(rec_ti, rec_t, rec_i),
                      model = list(lm, lm, lm),
                      wf = list(lm_ti_wf, lm_t_wf, lm_i_wf))

# We now have collected all the ingredients in the data.frame and can iterate (the power of tidymodels)!



# Helper functions --------------------------------------------------------------------------------

# Tune ----
# apply cv strategy on workflow
tuner <- function(wf, grid = 5) {
  
  pb$tick()
  
  tuned <-
    # if no tuning would have been specified you could use fit_resamples() call here...
    tune_grid(wf,
              # this is our folds object from above
              resamples = kfolds,
              grid = grid)
  
  # In case of errors you can inspect the cause by calling
  # tuned$.notes[[1]] %>% pull(.notes)
  
  return(tuned)

}



# Evaluate ----
# Could be extended: adding model and recipe as arguments (currently taken from enclosing scope) ...
finalizer <- function(tuned, model, rec, metric = "rmse") {
  
  pb$tick()
  
  # Select best
  best <- tuned %>% select_best(metric)
  
  # Finalize
  # Final model: give the actual model its parameters back
  final_model <- finalize_model(model, best)
  
  # Final workflow
  final_wf <- workflow() %>%
    add_recipe(rec) %>%
    add_model(final_model)
  
  return(final_wf)
  
}



# Get coefficients ----
retriever <- function(final_wf) {
  
  pb$tick()

  coefs <- final_wf %>% fit(train_data) %>% pull_workflow_fit() %>% tidy()
 
  return(coefs)
  
}



# Iterate --------------------------------------------------------------------------------

n_funcs <- 3

# Progress bar
pb <- progress_bar$new(format = "  looping [:bar] :elapsedfull",
                       total = n_funcs * nrow(model_frame), clear = FALSE, width = 100)

# If warnings during tuner are issued then this is because of the step_corr where at least one var
# in the training data of the split is a constant ...
model_frame <- model_frame %>%
  rowwise() %>%
  mutate(tuned = list(tuner(wf)),
         final_wf = list(finalizer(tuned, model, rec)),
         coefs = list(retriever(final_wf))) %>%
  ungroup()

# Stop progress bar
pb$terminate()


# Compare --------------------------------------------------------------------------------

# The power of tidy data!!
model_frame %>%
  mutate(metrics_collected = map(tuned, ~collect_metrics(.x))) %>%
  unnest(metrics_collected) %>%
  filter(.metric == "rmse") %>%
  group_by(desc) %>% mutate(penalty = row_number()) %>% ungroup() %>%
  ggplot(aes(x = fct_inorder(factor(penalty)), y = mean, group = desc, col = desc)) +
  geom_line() +
  geom_point(shape = 2) +
  theme_bw() +
  scale_color_manual(values = c("green", "blue", "red")) +
  xlab("penalty grid")



# Winner winner --------------------------------------------------------------------------------

# Evaluate the test sets using the final model ----
winner_winner <- 
  model_frame %>%
  mutate(metrics_collected = map(tuned, ~collect_metrics(.x))) %>%
  unnest(metrics_collected) %>%
  filter(.metric == "rmse") %>% filter(mean == min(mean)) %>% 
  pull(desc) %>% 
  unique()
winner_winner

winner_winner <- model_frame %>% filter(desc == winner_winner) %>% pull(final_wf) %>% .[[1]]
winner_ff <- winner_winner %>% last_fit(data_split)
winner_ff %>% collect_metrics()

# ... you can use the workflow for fitting with lm_ti_final_wf %>% fit(data) ...

# Everything is organized tidy in a data.frame
# f.ex. collect_predictions yields the following df which we can leverage for (not such a useful) plot
winner_ff %>%
  collect_predictions() %>%
  mutate(id = factor(row_number())) %>%
  select(id, .pred, pd) %>%
  gather(key, value, -id) %>%
  mutate(key = ifelse(key == "pd", "truth", "predicted")) %>%
  ggplot(aes(x = id, y = value, group = key, col = key)) +
  geom_line(alpha = 0.5) +
  scale_color_manual(values = c("orange", "grey")) +
  theme_classic() +
  theme(axis.text.x = element_blank(),
        axis.ticks.x = element_blank())

# Density plot
winner_ff %>% 
  collect_predictions() %>%
  select(.pred, pd) %>%
  gather(key, value) %>%
  ggplot(aes(x = value, col = key)) +
  geom_density() +
  theme_bw() +
  scale_colour_manual(values = c("blue", "green"))

# Randomly select 100 obs
winner_ff %>%
  collect_predictions() %>%
  mutate(id = factor(row_number())) %>%
  select(id, .pred, pd) %>%
  sample_n(100) %>%
  gather(key, value, -id) %>%
  mutate(key = ifelse(key == "pd", "truth", "predicted")) %>%
  ggplot(aes(x = id, y = value, group = key, col = key)) +
  geom_line() +
  scale_color_manual(values = c("orange", "grey")) +
  theme_bw() +
  theme(axis.text.x = element_blank())

# Could be interesting to analyse the fit at bank level ... -> include id role in recipe (bank_name)?


# Take away --------------------------------------------------------------------------------

# So we've seen that the actual modelling procedure / setup is a little more complex and cumbersome
# but it forces you to be very explicit about your routine and you have many options to
# choose from. Once the "skeletton" is set up, it is really conveniant to evaluate the model
# and pull the best fit - create a finalized workflow which is applicable to new data.
# Also, the tidymodels framework is made with the idea of comparing many different models.
# The framework enables you to leverage powerful engines s.a. keras. However, getting used to the
# universe of functions and especially debugging is a pain!
