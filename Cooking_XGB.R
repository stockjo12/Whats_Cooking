### SET UP ###
#Downloading libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(textrecipes)
library(vroom)
library(jsonlite)
library(xgboost)
library(beepr)

#Bringing in Data
setwd("~/Library/CloudStorage/OneDrive-BrighamYoungUniversity/STAT 348/Coding/Whats_Cooking")
train <- read_file("train.json") |>
  fromJSON() |>
  unnest(ingredients) |>
  group_by(id, cuisine) |>
  summarise(ingredients = paste(ingredients, collapse = " "), .groups = "drop") 
test <- read_file("test.json") |>
  fromJSON() |>
  unnest(ingredients) |>
  group_by(id) |>
  summarise(ingredients = paste(ingredients, collapse = " "), .groups = "drop")

### FEATURE ENGINEERING ###
#Making Original Recipe
cooking_recipe <- recipe(cuisine ~ ingredients, data = train) |>
  step_tokenize(ingredients) |>
  step_tfidf(ingredients) 
prep <- prep(cooking_recipe)
juiced <- juice(prep)

### WORK FLOW ###
#Naive Bayes Model
#Defining Model
xgb_model <- boost_tree(
  mode = "classification",
  trees = 100,
  tree_depth = 6,
  learn_rate = 0.1
) |>
  set_engine("xgboost")

#Creating Workflows
xgb_wf <- workflow() |>
  add_recipe(cooking_recipe) |>
  add_model(xgb_model)

### FIT AND PREDICT ###
#Finalizing Workflow
final_xwf <- fit(xgb_wf, data = train)
beepr::beep()

### SUBMISSION ###
#Making Predictions
xgb_pred <- predict(final_xwf, new_data = test, type = "class")

#Formatting Predictions for Kaggle
kaggle_xgb <- xgb_pred |>
  transmute(
    id = test$id,
    cuisine = .pred_class
  )

#Saving CSV File
vroom_write(kaggle_xgb, file = "./XGB_Test.csv", delim = ",")
