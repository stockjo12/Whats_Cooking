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
library(discrim)
library(textrecipes)

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
  bind_tf_idf(term = words, document = id, n = n) |>
  select(id, words, tf_idf) |>
  pivot_wider(names_from = words, values_from = tf_idf, values_fill = 0) |>
  left_join(train |> select(id), by = "id")

### FEATURE ENGINEERING ###
#Making Original Recipe
cooking_recipe <- recipe(cuisine ~ ingredients, data = train) |>
  step_tokenize(ingredients) |>
  step_tfidf(ingredients) 
# prep <- prep(cooking_recipe)
# juiced <- juice(prep)

### WORK FLOW ###
#Naive Bayes Model
#Defining Model
bayes_model <- naive_Bayes(Laplace = tune(),
                           smoothness = tune()) |>
  set_engine("naivebayes") |>
  set_mode("classification")

#Creating Workflows
bayes_wf <- workflow() |>
  add_recipe(cooking_recipe) |>
  add_model(bayes_model)

### CROSS VALIDATION ###
#Defining Grids of Values
bayes_grid <- grid_regular(Laplace(range = c(0, 2)),
                           smoothness(range = c(0.01, 1)),
                           levels = 3) #3 for Testing; 5 for Results

#Splitting Data
bayes_folds <- vfold_cv(train,
                        v = 5,
                        repeats = 1) #1 for Testing; 3 for Results

#Run Cross Validations
bayes_results <- bayes_wf |>
  tune_grid(resamples = bayes_folds,
            grid = bayes_grid,
            metrics = metric_set(mn_log_loss))
beepr::beep()

#Find Best Tuning Parameters
bayes_best <- bayes_results |>
  select_best(metric = "mn_log_loss")

#Finalizing Workflow
final_bwf <- bayes_wf |>
  finalize_workflow(bayes_best) |>
  fit(data = train)

### SUBMISSION ###
#Making Predictions
bayes_pred <- predict(final_bwf, new_data = test, type = "class")

#Formatting Predictions for Kaggle
kaggle_bayes <- bayes_pred |>
  transmute(
    id = test$id,
    cuisine = .pred_class
  )

#Saving CSV File
vroom_write(kaggle_bayes, file = "./Bayes_Test.csv", delim = ",")
