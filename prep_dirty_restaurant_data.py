#!/usr/bin/env python3
"""Prepare the Kaggle "Restaurant Sales - Dirty Data for Cleaning Training" dataset
for the Restaurant Intelligence Engine.

Reads the first CSV in `data/raw_client/*.csv`, cleans and reconstructs values,
and writes `data/client_sales.csv` and `data/client_menu.csv`.

Run:
    python prep_dirty_restaurant_data.py
"""
from pathlib import Path
import glob
import sys
import random

import numpy as np
import pandas as pd


RAW_GLOB = "data/raw_client/*.csv"
CLIENT_SALES_PATH = Path("data/client_sales.csv")
CLIENT_MENU_PATH = Path("data/client_menu.csv")


def find_first_csv():
    files = glob.glob(RAW_GLOB)
    if not files:
        raise FileNotFoundError(f"No CSV files found matching {RAW_GLOB}")
    return Path(sorted(files)[0])


def map_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Expected raw column names (exact in source schema)
    expected = {
        "Order ID": "order_id",
        "Customer ID": "customer_id",
        "Category": "category",
        "Item": "item",
        "Price": "price",
        "Quantity": "quantity",
        "Order Total": "order_total",
        "Order Date": "order_date",
        "Payment Method": "payment_method",
    }

    # Build a mapping from actual df columns to expected snake_case names
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    for raw_name, target in expected.items():
        key = raw_name.lower()
        if key in cols_lower:
            col_map[cols_lower[key]] = target
        else:
            # try a loose match (strip spaces and lower)
            found = None
            for c in df.columns:
                if c.strip().lower() == key:
                    found = c
                    break
            if found:
                col_map[found] = target
            else:
                # missing column will be created with NaN later if needed
                pass

    df = df.rename(columns=col_map)
    return df


def reconstruct_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Convert types
    df["order_date"] = pd.to_datetime(df.get("order_date"), errors="coerce")
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce")
    df["quantity"] = pd.to_numeric(df.get("quantity"), errors="coerce")
    df["order_total"] = pd.to_numeric(df.get("order_total"), errors="coerce")

    # Reconstruct missing values where possible
    # If price is NaN but quantity and order_total present: price = order_total / quantity
    mask_price_na = df["price"].isna() & df["quantity"].notna() & df["order_total"].notna() & (df["quantity"] != 0)
    if mask_price_na.any():
        df.loc[mask_price_na, "price"] = df.loc[mask_price_na, "order_total"] / df.loc[mask_price_na, "quantity"]

    # If quantity is NaN but price and order_total present: quantity = order_total / price
    mask_qty_na = df["quantity"].isna() & df["price"].notna() & df["order_total"].notna() & (df["price"] != 0)
    if mask_qty_na.any():
        df.loc[mask_qty_na, "quantity"] = df.loc[mask_qty_na, "order_total"] / df.loc[mask_qty_na, "price"]

    # If order_total is NaN but price and quantity are present: order_total = price * quantity
    mask_total_na = df["order_total"].isna() & df["price"].notna() & df["quantity"].notna()
    if mask_total_na.any():
        df.loc[mask_total_na, "order_total"] = df.loc[mask_total_na, "price"] * df.loc[mask_total_na, "quantity"]

    # Drop rows where ANY of these are still missing: item, category, price, quantity, order_date
    # Normalize item/category to strings and treat empty strings as missing
    df["item"] = df.get("item").astype(object)
    df["category"] = df.get("category").astype(object)

    def empty_or_nan(x):
        if pd.isna(x):
            return True
        try:
            s = str(x).strip()
            return s == "" or s.lower() == "nan"
        except Exception:
            return False

    mask_missing = (
        df["item"].apply(empty_or_nan)
        | df["category"].apply(empty_or_nan)
        | df["price"].isna()
        | df["quantity"].isna()
        | df["order_date"].isna()
    )

    df_clean = df.loc[~mask_missing].copy()

    # Cast quantity to int (rounding)
    df_clean["quantity"] = df_clean["quantity"].round().astype(int)

    return df_clean


def build_client_sales(df_clean: pd.DataFrame) -> pd.DataFrame:
    sales = pd.DataFrame()
    sales["order_datetime"] = df_clean["order_date"]
    sales["item_name"] = df_clean["item"].astype(str)
    sales["category"] = df_clean["category"].astype(str)
    sales["qty"] = df_clean["quantity"].astype(int)
    sales["price"] = df_clean["price"].astype(float).round(2)

    sales = sales.sort_values("order_datetime").reset_index(drop=True)
    return sales


def build_client_menu(df_clean: pd.DataFrame, seed: int = 2025) -> pd.DataFrame:
    # Group by item + category and compute median price
    grouped = (
        df_clean.groupby([df_clean["item"].astype(str), df_clean["category"].astype(str)])
        ["price"].median()
        .reset_index()
    )
    grouped.columns = ["item_name", "category", "sell_price"]

    rng = random.Random(seed)

    def pick_cost_pct(cat: str) -> float:
        c = (cat or "").lower()
        if "starter" in c:
            return rng.uniform(0.25, 0.32)
        elif "main" in c:
            return rng.uniform(0.28, 0.36)
        elif "dessert" in c:
            return rng.uniform(0.20, 0.30)
        elif "drink" in c:
            return rng.uniform(0.15, 0.25)
        elif "side" in c:
            return rng.uniform(0.25, 0.35)
        else:
            return rng.uniform(0.28, 0.35)

    grouped["cost_pct"] = grouped["category"].apply(pick_cost_pct)
    grouped["cost_per_unit"] = (grouped["sell_price"] * grouped["cost_pct"]).round(2)
    grouped["gp_per_unit"] = (grouped["sell_price"] - grouped["cost_per_unit"]).round(2)

    menu = grouped[["item_name", "category", "sell_price", "cost_per_unit", "gp_per_unit"]]
    return menu


def main():
    try:
        csv_path = find_first_csv()
    except Exception as e:
        print("Error:", e)
        sys.exit(1)

    print(f"Loading raw CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path, dtype=str)
    total_raw = len(df_raw)

    # map columns to expected snake_case
    df_mapped = map_raw_columns(df_raw)

    df_clean = reconstruct_and_clean(df_mapped)
    total_clean = len(df_clean)

    # Build client sales
    sales = build_client_sales(df_clean)
    CLIENT_SALES_PATH.parent.mkdir(parents=True, exist_ok=True)
    sales.to_csv(CLIENT_SALES_PATH, index=False)

    # Build client menu
    menu = build_client_menu(df_clean)
    CLIENT_MENU_PATH.parent.mkdir(parents=True, exist_ok=True)
    menu.to_csv(CLIENT_MENU_PATH, index=False)

    # Summary
    print("\n=== Summary ===")
    print(f"Total raw rows loaded: {total_raw}")
    print(f"Total rows after cleaning: {total_clean}")
    print(f"Number of unique menu items: {len(menu)}")
    if not sales["order_datetime"].isna().all():
        print(f"Order date range: {sales['order_datetime'].min()} -> {sales['order_datetime'].max()}")

    print("\nExample client_sales (head):")
    print(sales.head().to_string(index=False))

    print("\nExample client_menu (head):")
    print(menu.head().to_string(index=False))


if __name__ == "__main__":
    main()
