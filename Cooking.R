### SET UP ###
#Downloading libraries
library(tidyverse)
library(tidymodels)
library(tidytext)
library(beepr)
library(vroom)
library(janitor)
library(doParallel)
library(jsonlite)
library(stringr)

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

#Setting Up Ingredient Categories
ingredient_categories <- list(
  vegetables = c("tomato", "onion", "garlic", "spinach", "lettuce", "pepper"),
  proteins   = c("chicken", "beef", "tofu", "egg", "pork"),
  spices     = c("basil", "cumin", "oregano", "chili", "paprika"),
  dairy      = c("milk", "cheese", "butter", "yogurt"),
  grains     = c("rice", "pasta", "flour"),
  sauces     = c("olive oil", "soy sauce", "vinegar"))

#Setting Up Rare Categories
ingredients_long <- train |>
  unnest_tokens(word, ingredients)
rare_words <- ingredients_long |>
  count(cuisine, word) |>
  group_by(word) |>
  mutate(total = sum(n),
         prop = n / total) |>
  filter(prop > 0.8) |>
  pull(word)

#Creating New Features
train <- train |>
  rowwise() |>
  mutate(
    num_vegetables = sum(str_detect(ingredients, ingredient_categories$vegetables)),
    num_proteins   = sum(str_detect(ingredients, ingredient_categories$proteins)),
    num_spices     = sum(str_detect(ingredients, ingredient_categories$spices)),
    num_dairy      = sum(str_detect(ingredients, ingredient_categories$dairy)),
    num_grains     = sum(str_detect(ingredients, ingredient_categories$grains)),
    num_sauces     = sum(str_detect(ingredients, ingredient_categories$sauces)),
    num_ingredients = str_count(ingredients, "\\S+"), #Total Ingredients
    rare_count = sum(str_detect(ingredients, rare_words))) |> #Rare Ingredients
  ungroup()

#Using TF-IDF on the Training Dataset
tfidf <- train |>
  unnest_tokens(words, ingredients) |>
  count(id, cuisine, words) |>
  bind_tf_idf(term = words, document = id, n = n)

### FEATURE ENGINEERING ###
#Making Recipe


### WORK FLOW ###
# X Model
#Defining Model


#Creating Workflows


### CROSS VALIDATION ###
#Defining Grids of Values


#Splitting Data


#Run Cross Validations


#Find Best Tuning Parameters


#Finalizing Workflow


### SUBMISSION ###
#Making Predictions


#Formatting Predictions for Kaggle


#Saving CSV File

