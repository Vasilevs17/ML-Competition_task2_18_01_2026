# ML Competition 2: Food Recommendation from Fridge Data

This repository contains a compact machine learning solution for a recommendation competition. The task is to rank candidate dishes for each user and return the top-5 most relevant dishes.

The project is based on a learning-to-rank style formulation: each candidate dish is converted into a separate training row, the model estimates how suitable this dish is for the current user/context, and the final submission is built by sorting candidates by predicted probability.

## Task

For each query from the test set, the model receives several candidate dishes and must generate five recommendations:

```text
query_id, rec_1, rec_2, rec_3, rec_4, rec_5
```

The available data describes:

- user preferences;
- candidate dishes;
- dish categories, tags, calories and allergens;
- current context such as day, meal slot, number of guests, diet mode and fridge load.

## Repository structure

```text
ML-competition-2/
├── README.md
├── requirements.txt
├── .gitignore
├── zip-2_task.zip
└── src/
    └── train_catboost_recommender.py
```

The archive `zip-2_task.zip` contains the original task files. After unpacking it, the training script expects the following CSV files:

```text
train_holodilnik.csv
test_holodilnik.csv
users.csv
dishes.csv
```

## Approach

The solution uses CatBoost as a strong baseline for tabular recommendation data. Instead of training a model directly to output five dishes, the task is transformed into binary classification:

- positive examples are the dishes that were actually selected;
- negative examples are sampled from the remaining candidate dishes;
- the model predicts the probability that a candidate dish is the correct recommendation;
- candidates are sorted by this probability;
- the top-5 unique dishes are written to the submission file.

## Feature engineering

The model combines several groups of features:

- **Context features**: user id, day, meal slot, hangover level, number of guests, diet mode and fridge load.
- **Dish features**: dish id, category, calories and spiciness.
- **User preference features**: light food preference, sweet tooth flag, coffee addiction and trust in microwave food.
- **Candidate position**: the original candidate index inside the candidate list.
- **Tag matching features**: overlap between user-liked tags and dish tags.
- **Negative preference features**: overlap between disliked user tags and dish tags.
- **Allergy feature**: whether the dish contains allergens that match user allergies.

This keeps the model interpretable: the recommendation is not based only on dish id, but also on how well the dish matches user preferences and the current situation.

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Unpack the task archive so that the CSV files are available in the repository root, or place them in another folder and pass it through `--data-dir`.

Run the script:

```bash
python src/train_catboost_recommender.py --data-dir . --output submission.csv
```

For CPU training:

```bash
python src/train_catboost_recommender.py --data-dir . --output submission.csv --task-type CPU
```

For GPU training:

```bash
python src/train_catboost_recommender.py --data-dir . --output submission.csv --task-type GPU
```

The script creates:

```text
submission.csv
```

## Main parameters

The script supports the following arguments:

```text
--data-dir       folder with CSV files
--output         path to generated submission.csv
--seed           random seed
--task-type      CPU or GPU
--neg-per-pos    number of negative candidate dishes sampled per positive example
```

Example:

```bash
python src/train_catboost_recommender.py --data-dir data/raw --output submission.csv --task-type CPU --neg-per-pos 9
```

## Output

The final submission has the following columns:

```text
query_id, rec_1, rec_2, rec_3, rec_4, rec_5
```

Each row contains five unique recommended dishes for one query.

## Main technologies

- Python
- pandas
- NumPy
- CatBoost

## Notes

The repository is intentionally focused on a single competition solution. The code is organized to make the full pipeline easy to inspect: loading data, building candidate rows, adding user/dish/tag features, training CatBoost and generating top-5 recommendations.