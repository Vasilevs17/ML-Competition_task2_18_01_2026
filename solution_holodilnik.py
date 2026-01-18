#!/usr/bin/env python3
import os
import sys
import subprocess
import importlib.util

if importlib.util.find_spec("catboost") is None:
    print("Устанавливаю catboost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "catboost"])

import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

DATA_DIR = os.getenv("DATA_DIR", ".")
TRAIN_FILE = os.getenv("TRAIN_FILE", "train_holodilnik.csv")
TEST_FILE = os.getenv("TEST_FILE", "test_holodilnik.csv")
USERS_FILE = os.getenv("USERS_FILE", "users.csv")
DISHES_FILE = os.getenv("DISHES_FILE", "dishes.csv")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "submission.csv")
SEED = int(os.getenv("SEED", "42"))
USE_GPU = os.getenv("USE_GPU", "1")
NEG_PER_POS = int(os.getenv("NEG_PER_POS", "9"))


def main() -> None:
    print("Загружаю данные...")
    train = pd.read_csv(os.path.join(DATA_DIR, TRAIN_FILE))
    test = pd.read_csv(os.path.join(DATA_DIR, TEST_FILE))
    users = pd.read_csv(os.path.join(DATA_DIR, USERS_FILE))
    dishes = pd.read_csv(os.path.join(DATA_DIR, DISHES_FILE))

    cand_cols = [col for col in train.columns if col.startswith("cand_")]
    n_cand = len(cand_cols)

    print("Готовлю кандидатов для обучения...")
    train_cands = train[cand_cols].to_numpy(dtype=np.int32)
    target = train["target_dish_id"].to_numpy(dtype=np.int32)

    context_cols = [
        "user_id",
        "day",
        "meal_slot",
        "hangover_level",
        "guests_count",
        "diet_mode",
        "fridge_load_pct",
    ]
    context = train[context_cols].to_numpy()

    rng = np.random.default_rng(SEED)
    total_events = len(train)
    neg_per_pos = max(1, min(NEG_PER_POS, n_cand - 1))

    pos_rows = np.arange(total_events)
    pos_cand_idx = np.argmax(train_cands == target[:, None], axis=1)

    neg_rows = np.repeat(pos_rows, neg_per_pos)
    neg_choices = np.array(
        [rng.choice(n_cand - 1, size=neg_per_pos, replace=False) for _ in range(total_events)],
        dtype=np.int32,
    )
    neg_choices = np.where(neg_choices >= pos_cand_idx[:, None], neg_choices + 1, neg_choices)
    neg_cand_idx = neg_choices.reshape(-1)

    all_rows = np.concatenate([pos_rows, neg_rows])
    all_cand_idx = np.concatenate([pos_cand_idx, neg_cand_idx])
    labels = np.concatenate(
        [np.ones(total_events, dtype=np.int8), np.zeros(total_events * neg_per_pos, dtype=np.int8)]
    )
    dish_ids = train_cands[all_rows, all_cand_idx]

    context_rep = context[all_rows]

    X_train = pd.DataFrame(context_rep, columns=context_cols)
    X_train["dish_id"] = dish_ids
    X_train["cand_idx"] = all_cand_idx

    print("Добавляю признаки пользователей и блюд...")
    users_features = users.set_index("user_id")[
        ["prefers_light_food", "sweet_tooth", "coffee_addict", "microwave_trust"]
    ]
    dishes_features = dishes.set_index("dish_id")[["category", "calories", "spicy"]]

    for col in users_features.columns:
        X_train[col] = X_train["user_id"].map(users_features[col])
    for col in dishes_features.columns:
        X_train[col] = X_train["dish_id"].map(dishes_features[col])

    X_train["spicy"] = X_train["spicy"].fillna(0).astype(int)

    print("Добавляю признаки тегов и аллергенов...")
    dishes["tags"] = dishes["tags"].fillna("")
    dishes["allergen_tags"] = dishes["allergen_tags"].fillna("")
    users["liked_tags"] = users["liked_tags"].fillna("")
    users["disliked_tags"] = users["disliked_tags"].fillna("")
    users["allergies"] = users["allergies"].fillna("")

    dish_tags = dishes.set_index("dish_id")["tags"].apply(lambda v: set(v.split("|")) if v else set()).to_dict()
    dish_allergens = dishes.set_index("dish_id")["allergen_tags"].apply(
        lambda v: set(v.split("|")) if v else set()
    ).to_dict()
    user_liked = users.set_index("user_id")["liked_tags"].apply(lambda v: set(v.split("|")) if v else set()).to_dict()
    user_disliked = users.set_index("user_id")["disliked_tags"].apply(
        lambda v: set(v.split("|")) if v else set()
    ).to_dict()
    user_allergies = users.set_index("user_id")["allergies"].apply(lambda v: set(v.split("|")) if v else set()).to_dict()

    train_user_ids = X_train["user_id"].to_numpy()
    train_dish_ids = X_train["dish_id"].to_numpy()
    X_train["liked_overlap"] = [
        len(user_liked.get(u, set()) & dish_tags.get(d, set()))
        for u, d in zip(train_user_ids, train_dish_ids)
    ]
    X_train["disliked_overlap"] = [
        len(user_disliked.get(u, set()) & dish_tags.get(d, set()))
        for u, d in zip(train_user_ids, train_dish_ids)
    ]
    X_train["allergy_hit"] = [
        1 if (user_allergies.get(u, set()) & dish_allergens.get(d, set())) else 0
        for u, d in zip(train_user_ids, train_dish_ids)
    ]

    print("Готовлю обучающие/валидационные выборки...")
    idx = np.arange(len(X_train))
    rng.shuffle(idx)
    split = int(len(idx) * 0.9)
    train_idx = idx[:split]
    valid_idx = idx[split:]

    X_tr = X_train.iloc[train_idx]
    y_tr = labels[train_idx]
    X_val = X_train.iloc[valid_idx]
    y_val = labels[valid_idx]

    cat_features = [
        X_train.columns.get_loc("user_id"),
        X_train.columns.get_loc("meal_slot"),
        X_train.columns.get_loc("dish_id"),
        X_train.columns.get_loc("category"),
        X_train.columns.get_loc("cand_idx"),
    ]

    print("Обучаю модель CatBoost...")
    task_type = "GPU" if USE_GPU == "1" else "CPU"
    model = CatBoostClassifier(
        iterations=400,
        depth=8,
        learning_rate=0.1,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=50,
        od_type="Iter",
        od_wait=50,
        task_type=task_type,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_features)

    print("Готовлю кандидатов для теста...")
    test_cands = test[cand_cols].to_numpy()
    test_dish_ids = test_cands.reshape(-1)

    test_context = test[context_cols].to_numpy()
    test_context_rep = np.repeat(test_context, n_cand, axis=0)
    test_cand_idx = np.tile(np.arange(n_cand), len(test))

    X_test = pd.DataFrame(test_context_rep, columns=context_cols)
    X_test["dish_id"] = test_dish_ids
    X_test["cand_idx"] = test_cand_idx

    for col in users_features.columns:
        X_test[col] = X_test["user_id"].map(users_features[col])
    for col in dishes_features.columns:
        X_test[col] = X_test["dish_id"].map(dishes_features[col])
    X_test["spicy"] = X_test["spicy"].fillna(0).astype(int)

    test_user_ids = X_test["user_id"].to_numpy()
    test_dish_ids = X_test["dish_id"].to_numpy()
    X_test["liked_overlap"] = [
        len(user_liked.get(u, set()) & dish_tags.get(d, set()))
        for u, d in zip(test_user_ids, test_dish_ids)
    ]
    X_test["disliked_overlap"] = [
        len(user_disliked.get(u, set()) & dish_tags.get(d, set()))
        for u, d in zip(test_user_ids, test_dish_ids)
    ]
    X_test["allergy_hit"] = [
        1 if (user_allergies.get(u, set()) & dish_allergens.get(d, set())) else 0
        for u, d in zip(test_user_ids, test_dish_ids)
    ]

    print("Считаю предсказания...")
    preds = model.predict_proba(X_test)[:, 1]
    preds_matrix = preds.reshape(len(test), n_cand)

    print("Формирую топ-5 рекомендаций...")
    recs = []
    for i, row in test.iterrows():
        cand_list = test_cands[i]
        scores = preds_matrix[i]
        order = np.argsort(-scores)
        top = []
        for idx_c in order:
            dish_id = int(cand_list[idx_c])
            if dish_id not in top:
                top.append(dish_id)
            if len(top) == 5:
                break
        if len(top) < 5:
            for dish_id in cand_list:
                dish_id = int(dish_id)
                if dish_id not in top:
                    top.append(dish_id)
                if len(top) == 5:
                    break
        recs.append([int(row["query_id"]) ] + top)

    result = pd.DataFrame(recs, columns=["query_id", "rec_1", "rec_2", "rec_3", "rec_4", "rec_5"])
    result.to_csv(os.path.join(DATA_DIR, OUTPUT_FILE), index=False)
    print(f"Готово! Файл сохранён: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
