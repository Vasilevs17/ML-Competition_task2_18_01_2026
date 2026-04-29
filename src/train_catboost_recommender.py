"""Train a CatBoost recommender for the fridge food recommendation task.

The task is treated as candidate ranking. For each training query, the selected
dish becomes a positive example and several non-selected candidate dishes become
negative examples. CatBoost then estimates the relevance probability for every
candidate dish in the test set, and the script returns top-5 recommendations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DEFAULT_CONTEXT_COLUMNS = [
    "user_id",
    "day",
    "meal_slot",
    "hangover_level",
    "guests_count",
    "diet_mode",
    "fridge_load_pct",
]

USER_FEATURE_COLUMNS = [
    "prefers_light_food",
    "sweet_tooth",
    "coffee_addict",
    "microwave_trust",
]

DISH_FEATURE_COLUMNS = ["category", "calories", "spicy"]


def split_tags(value: object) -> set[str]:
    """Convert a pipe-separated tag string to a set of tags."""
    if pd.isna(value) or value == "":
        return set()
    return {tag for tag in str(value).split("|") if tag}


def get_candidate_columns(df: pd.DataFrame) -> list[str]:
    """Return candidate dish columns from the competition data."""
    candidate_columns = [column for column in df.columns if column.startswith("cand_")]
    if not candidate_columns:
        raise ValueError("No candidate columns found. Expected columns starting with 'cand_'.")
    return candidate_columns


def add_user_and_dish_features(
    frame: pd.DataFrame,
    users: pd.DataFrame,
    dishes: pd.DataFrame,
) -> pd.DataFrame:
    """Attach user-level, dish-level and tag-overlap features."""
    result = frame.copy()

    users_indexed = users.set_index("user_id")
    dishes_indexed = dishes.set_index("dish_id")

    for column in USER_FEATURE_COLUMNS:
        result[column] = result["user_id"].map(users_indexed[column])

    for column in DISH_FEATURE_COLUMNS:
        result[column] = result["dish_id"].map(dishes_indexed[column])

    result["spicy"] = result["spicy"].fillna(0).astype(int)

    dish_tags = dishes_indexed["tags"].fillna("").apply(split_tags).to_dict()
    dish_allergens = dishes_indexed["allergen_tags"].fillna("").apply(split_tags).to_dict()
    user_liked_tags = users_indexed["liked_tags"].fillna("").apply(split_tags).to_dict()
    user_disliked_tags = users_indexed["disliked_tags"].fillna("").apply(split_tags).to_dict()
    user_allergies = users_indexed["allergies"].fillna("").apply(split_tags).to_dict()

    user_ids = result["user_id"].to_numpy()
    dish_ids = result["dish_id"].to_numpy()

    result["liked_overlap"] = [
        len(user_liked_tags.get(user_id, set()) & dish_tags.get(dish_id, set()))
        for user_id, dish_id in zip(user_ids, dish_ids)
    ]
    result["disliked_overlap"] = [
        len(user_disliked_tags.get(user_id, set()) & dish_tags.get(dish_id, set()))
        for user_id, dish_id in zip(user_ids, dish_ids)
    ]
    result["allergy_hit"] = [
        int(bool(user_allergies.get(user_id, set()) & dish_allergens.get(dish_id, set())))
        for user_id, dish_id in zip(user_ids, dish_ids)
    ]

    return result


def build_training_candidates(
    train: pd.DataFrame,
    users: pd.DataFrame,
    dishes: pd.DataFrame,
    candidate_columns: list[str],
    seed: int,
    negatives_per_positive: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build binary classification rows from candidate dishes."""
    train_candidates = train[candidate_columns].to_numpy(dtype=np.int32)
    target = train["target_dish_id"].to_numpy(dtype=np.int32)
    n_candidates = len(candidate_columns)
    n_events = len(train)

    negatives_per_positive = max(1, min(negatives_per_positive, n_candidates - 1))
    rng = np.random.default_rng(seed)

    positive_rows = np.arange(n_events)
    positive_candidate_index = np.argmax(train_candidates == target[:, None], axis=1)

    negative_rows = np.repeat(positive_rows, negatives_per_positive)
    negative_choices = np.array(
        [
            rng.choice(n_candidates - 1, size=negatives_per_positive, replace=False)
            for _ in range(n_events)
        ],
        dtype=np.int32,
    )
    negative_choices = np.where(
        negative_choices >= positive_candidate_index[:, None],
        negative_choices + 1,
        negative_choices,
    )
    negative_candidate_index = negative_choices.reshape(-1)

    all_rows = np.concatenate([positive_rows, negative_rows])
    all_candidate_indices = np.concatenate([positive_candidate_index, negative_candidate_index])
    labels = np.concatenate(
        [
            np.ones(n_events, dtype=np.int8),
            np.zeros(n_events * negatives_per_positive, dtype=np.int8),
        ]
    )
    dish_ids = train_candidates[all_rows, all_candidate_indices]

    context = train[DEFAULT_CONTEXT_COLUMNS].to_numpy()
    context_repeated = context[all_rows]

    features = pd.DataFrame(context_repeated, columns=DEFAULT_CONTEXT_COLUMNS)
    features["dish_id"] = dish_ids
    features["candidate_index"] = all_candidate_indices
    features = add_user_and_dish_features(features, users, dishes)

    return features, labels


def build_test_candidates(
    test: pd.DataFrame,
    users: pd.DataFrame,
    dishes: pd.DataFrame,
    candidate_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build candidate rows for every test query."""
    test_candidates = test[candidate_columns].to_numpy(dtype=np.int32)
    n_candidates = len(candidate_columns)

    dish_ids = test_candidates.reshape(-1)
    context = test[DEFAULT_CONTEXT_COLUMNS].to_numpy()
    context_repeated = np.repeat(context, n_candidates, axis=0)
    candidate_indices = np.tile(np.arange(n_candidates), len(test))

    features = pd.DataFrame(context_repeated, columns=DEFAULT_CONTEXT_COLUMNS)
    features["dish_id"] = dish_ids
    features["candidate_index"] = candidate_indices
    features = add_user_and_dish_features(features, users, dishes)

    return features, test_candidates


def get_categorical_feature_indices(features: pd.DataFrame) -> list[int]:
    """Return CatBoost categorical feature indices."""
    categorical_columns = [
        "user_id",
        "meal_slot",
        "dish_id",
        "category",
        "candidate_index",
    ]
    return [features.columns.get_loc(column) for column in categorical_columns]


def train_model(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    task_type: str,
) -> CatBoostClassifier:
    """Train CatBoost on a shuffled holdout split."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_index = int(len(indices) * 0.9)
    train_indices = indices[:split_index]
    valid_indices = indices[split_index:]

    X_train = X.iloc[train_indices]
    y_train = y[train_indices]
    X_valid = X.iloc[valid_indices]
    y_valid = y[valid_indices]

    model = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        verbose=50,
        od_type="Iter",
        od_wait=50,
        task_type=task_type,
        allow_writing_files=False,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_valid, y_valid),
        cat_features=get_categorical_feature_indices(X),
    )
    return model


def make_top5_submission(
    test: pd.DataFrame,
    test_candidates: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Create the final top-5 recommendation submission file."""
    n_queries, n_candidates = test_candidates.shape
    probability_matrix = probabilities.reshape(n_queries, n_candidates)

    recommendations: list[list[int]] = []
    for row_number, row in test.iterrows():
        candidate_list = test_candidates[row_number]
        scores = probability_matrix[row_number]
        sorted_indices = np.argsort(-scores)

        top_dishes: list[int] = []
        for candidate_index in sorted_indices:
            dish_id = int(candidate_list[candidate_index])
            if dish_id not in top_dishes:
                top_dishes.append(dish_id)
            if len(top_dishes) == 5:
                break

        if len(top_dishes) < 5:
            for dish_id in candidate_list:
                dish_id = int(dish_id)
                if dish_id not in top_dishes:
                    top_dishes.append(dish_id)
                if len(top_dishes) == 5:
                    break

        recommendations.append([int(row["query_id"]), *top_dishes])

    submission = pd.DataFrame(
        recommendations,
        columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"],
    )
    submission.to_csv(output_path, index=False)


def run_pipeline(
    data_dir: Path,
    output_path: Path,
    seed: int,
    task_type: str,
    negatives_per_positive: int,
) -> None:
    """Run the full recommendation pipeline."""
    print("Loading data...")
    train = pd.read_csv(data_dir / "train_holodilnik.csv")
    test = pd.read_csv(data_dir / "test_holodilnik.csv")
    users = pd.read_csv(data_dir / "users.csv")
    dishes = pd.read_csv(data_dir / "dishes.csv")

    candidate_columns = get_candidate_columns(train)
    print(f"Candidate columns: {len(candidate_columns)}")

    print("Building training candidates...")
    X_train, y_train = build_training_candidates(
        train=train,
        users=users,
        dishes=dishes,
        candidate_columns=candidate_columns,
        seed=seed,
        negatives_per_positive=negatives_per_positive,
    )

    print("Training CatBoost model...")
    model = train_model(X_train, y_train, seed=seed, task_type=task_type)

    print("Building test candidates...")
    X_test, test_candidates = build_test_candidates(
        test=test,
        users=users,
        dishes=dishes,
        candidate_columns=candidate_columns,
    )

    print("Predicting candidate relevance...")
    probabilities = model.predict_proba(X_test)[:, 1]

    print("Creating top-5 submission...")
    make_top5_submission(
        test=test,
        test_candidates=test_candidates,
        probabilities=probabilities,
        output_path=output_path,
    )
    print(f"Saved submission to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a CatBoost model and create top-5 dish recommendations.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory with train_holodilnik.csv, test_holodilnik.csv, users.csv and dishes.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission.csv"),
        help="Path to the generated submission file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--task-type",
        choices=["CPU", "GPU"],
        default="CPU",
        help="CatBoost training backend.",
    )
    parser.add_argument(
        "--neg-per-pos",
        type=int,
        default=9,
        help="Number of negative candidate dishes sampled per positive example.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_dir=args.data_dir,
        output_path=args.output,
        seed=args.seed,
        task_type=args.task_type,
        negatives_per_positive=args.neg_per_pos,
    )


if __name__ == "__main__":
    main()
