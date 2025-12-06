#!/usr/bin/env python3
"""Prepare local Kaggle restaurant sales CSV for the Restaurant Intelligence Engine.

This script DOES NOT download anything. It expects a CSV to already exist
in `data/raw_kaggle/` (for example created by `kaggle datasets download ...`).

Output:
  - data/client_sales.csv
  - data/client_menu.csv

Usage:
  python prep_kaggle_restaurant_data.py
"""
from pathlib import Path
import sys
import re
import math
import random
import glob
import pandas as pd


RAW_DIR = Path("data/raw_kaggle")
CLIENT_SALES = Path("data/client_sales.csv")
CLIENT_MENU = Path("data/client_menu.csv")


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLIENT_SALES.parent.mkdir(parents=True, exist_ok=True)


def clean_col_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def find_first_csv(dirpath: Path):
    files = sorted(glob.glob(str(dirpath / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {dirpath}. Please unzip the Kaggle dataset there.")
    return Path(files[0])


def guess_column(cols, keys):
    cols_l = [c.lower() for c in cols]
    for k in keys:
        for orig, low in zip(cols, cols_l):
            if k in low:
                return orig
    return None


def map_and_clean(df_raw: pd.DataFrame) -> pd.DataFrame:
    # preserve original columns list for diagnostics
    orig_cols = list(df_raw.columns)
    # clean names for heuristic matching
    cleaned_map = {c: clean_col_name(c) for c in orig_cols}
    df_raw = df_raw.rename(columns=cleaned_map)

    cols = list(df_raw.columns)

    dt_col = guess_column(cols, ["order", "date", "datetime", "timestamp", "time"]) or guess_column(cols, ["txn", "transaction"]) 
    item_col = guess_column(cols, ["item", "product", "dish", "menu", "description", "name"]) 
    cat_col = guess_column(cols, ["category", "cat", "course", "type"]) 
    qty_col = guess_column(cols, ["qty", "quantity", "units", "count"]) 
    price_col = guess_column(cols, ["price", "amount", "unit_price", "total", "sale", "price_each"]) 

    if not dt_col:
        raise RuntimeError(f"Could not locate a datetime column. Columns found: {cols}")
    if not item_col:
        raise RuntimeError(f"Could not locate an item column. Columns found: {cols}")
    if not price_col:
        raise RuntimeError(f"Could not locate a price column. Columns found: {cols}")

    # create working df with expected column names
    df = pd.DataFrame()
    df["order_datetime"] = pd.to_datetime(df_raw[dt_col], errors="coerce")
    df["item_name"] = df_raw[item_col].astype(str)
    if cat_col:
        df["category"] = df_raw[cat_col].astype(str)
    else:
        df["category"] = "Mains"
    if qty_col:
        df["qty"] = pd.to_numeric(df_raw[qty_col], errors="coerce").fillna(1)
    else:
        df["qty"] = 1
    df["price"] = pd.to_numeric(df_raw[price_col], errors="coerce")

    if df["order_datetime"].isna().all():
        raise RuntimeError("Parsed `order_datetime` are all NaT; please check the source CSV date column format.")

    # If price looks like a line total, convert to unit price where qty > 0
    mask_qty_gt1 = df["qty"] > 1
    if mask_qty_gt1.any():
        # Use median unit price heuristic
        unit_candidates = (df.loc[mask_qty_gt1, "price"] / df.loc[mask_qty_gt1, "qty"]).replace([pd.NA, pd.NaT], None)
        if not unit_candidates.dropna().empty:
            median_unit = unit_candidates.median()
            if pd.notna(median_unit) and median_unit > 0:
                df.loc[mask_qty_gt1, "price"] = df.loc[mask_qty_gt1, "price"] / df.loc[mask_qty_gt1, "qty"]

    # final safe types and ordering
    df["qty"] = df["qty"].astype(int)
    df["price"] = df["price"].round(2)

    df = df[["order_datetime", "item_name", "category", "qty", "price"]]
    return df


def build_menu(df: pd.DataFrame, seed: int = 2025) -> pd.DataFrame:
    menu = df[["item_name", "category", "price"]].drop_duplicates().copy()
    menu["category"] = menu["category"].fillna("Mains")

    ranges = {
        "starters": (0.25, 0.32),
        "mains": (0.28, 0.36),
        "desserts": (0.20, 0.30),
        "sides": (0.25, 0.35),
        "drinks": (0.15, 0.25),
    }

    rng = random.Random(seed)

    def pick_pct(cat: str) -> float:
        c = str(cat).lower()
        for k, (lo, hi) in ranges.items():
            if k in c:
                return rng.uniform(lo, hi)
        # fallback: Mains range
        return rng.uniform(0.28, 0.36)

    menu["cost_pct"] = menu["category"].apply(pick_pct)
    menu["sell_price"] = menu["price"].astype(float).round(2)
    menu["cost_per_unit"] = (menu["sell_price"] * menu["cost_pct"]).round(2)
    menu["gp_per_unit"] = (menu["sell_price"] - menu["cost_per_unit"]).round(2)

    menu = menu[["item_name", "category", "sell_price", "cost_per_unit", "gp_per_unit"]]
    return menu


def main():
    ensure_dirs()
    try:
        csv_path = find_first_csv(RAW_DIR)
    except Exception as e:
        print("Error locating CSV:", e)
        sys.exit(1)

    print(f"Loading raw CSV: {csv_path}")
    df_raw = pd.read_csv(csv_path, dtype=str)

    try:
        df = map_and_clean(df_raw)
    except Exception as e:
        print("Failed to map/clean source CSV:", e)
        print("Columns available:", list(df_raw.columns))
        sys.exit(1)

    df.to_csv(CLIENT_SALES, index=False)
    print(f"Wrote cleaned sales to: {CLIENT_SALES}")

    menu = build_menu(df)
    menu.to_csv(CLIENT_MENU, index=False)
    print(f"Wrote generated menu to: {CLIENT_MENU}")

    print("\nSummary:")
    print(f"Sales rows: {len(df):,}")
    print(f"Menu items: {len(menu):,}")
    try:
        print(f"Date range: {df['order_datetime'].min()} -> {df['order_datetime'].max()}")
    except Exception:
        pass

    print("\nExample sales rows:")
    print(df.head().to_string(index=False))

    print("\nExample menu rows:")
    print(menu.head().to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Kaggle Restaurant Data Preparation Script

Downloads the Kaggle restaurant sales dataset, cleans it, and generates
compatible CSV files for the Restaurant Intelligence Engine.

Usage:
    python prep_kaggle_restaurant_data.py

Requirements:
    - kaggle API credentials configured (~/.kaggle/kaggle.json)
    - pandas, numpy installed
"""

import os
import sys
import subprocess
import zipfile
import json
from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import numpy as np


# Configuration
KAGGLE_DATASET_ID = "salikhussaini/restaurant-sales-transactions"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_KAGGLE_DIR = DATA_DIR / "raw_kaggle"
CLIENT_SALES_PATH = DATA_DIR / "client_sales.csv"
CLIENT_MENU_PATH = DATA_DIR / "client_menu.csv"

# Cost percentage ranges by category (realistic COGS percentages)
COST_PCT_RANGES = {
    "starters": (0.25, 0.32),
    "mains": (0.28, 0.36),
    "desserts": (0.20, 0.30),
    "sides": (0.25, 0.35),
    "drinks": (0.15, 0.25),
}


def normalize_column_name(col: str) -> str:
    """Convert column name to snake_case."""
    # lowercase
    col = col.lower()
    # remove spaces and special chars, replace with underscore
    col = re.sub(r"[^\w]+", "_", col)
    # remove leading/trailing underscores
    col = col.strip("_")
    return col


def download_kaggle_dataset() -> None:
    """Download the Kaggle dataset using kaggle CLI."""
    print(f"\n[1/5] Downloading Kaggle dataset: {KAGGLE_DATASET_ID}")
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET_ID, "-p", str(DATA_DIR)],
            check=True,
            capture_output=True,
            text=True,
        )
        print("✓ Dataset downloaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download dataset: {e.stderr}")
        print("\nEnsure Kaggle API credentials are configured at ~/.kaggle/kaggle.json")
        print("See: https://www.kaggle.com/settings/account")
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install it with: pip install kaggle")
        sys.exit(1)


def unzip_dataset() -> Path:
    """Unzip the downloaded dataset and return path to CSV file."""
    print(f"\n[2/5] Unzipping dataset into {RAW_KAGGLE_DIR}")
    
    # Find the zip file in DATA_DIR
    zip_files = list(DATA_DIR.glob("*.zip"))
    if not zip_files:
        print("✗ No .zip file found in data/ directory.")
        sys.exit(1)
    
    zip_path = zip_files[0]
    print(f"  Found: {zip_path.name}")
    
    # Create raw_kaggle directory
    RAW_KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Extract
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(RAW_KAGGLE_DIR)
        print(f"✓ Extracted to {RAW_KAGGLE_DIR}")
        
        # Find CSV file
        csv_files = list(RAW_KAGGLE_DIR.glob("**/*.csv"))
        if not csv_files:
            print("✗ No CSV file found in extracted data.")
            sys.exit(1)
        
        csv_path = csv_files[0]
        print(f"  Found CSV: {csv_path.name}")
        return csv_path
    except Exception as e:
        print(f"✗ Failed to extract: {e}")
        sys.exit(1)


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Load CSV and clean column names."""
    print(f"\n[3/5] Loading and cleaning data from {csv_path.name}")
    try:
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Clean column names
        df.columns = [normalize_column_name(col) for col in df.columns]
        print(f"  Cleaned column names: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        sys.exit(1)


def standardize_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize the sales DataFrame to match expected columns.
    Attempts to auto-detect and map columns.
    """
    print(f"\n[4/5] Standardizing sales data to client format")
    
    # Auto-detect and map columns
    df_std = pd.DataFrame()
    
    # Map order_datetime (look for date/time column)
    datetime_cols = [c for c in df.columns if any(x in c for x in ["date", "time", "order", "created"])]
    if datetime_cols:
        datetime_col = datetime_cols[0]
        df_std["order_datetime"] = pd.to_datetime(df[datetime_col], errors="coerce")
        print(f"  Mapped order_datetime <- {datetime_col}")
    else:
        print("  Warning: no datetime column detected, using index as datetime.")
        df_std["order_datetime"] = pd.date_range(start="2024-01-01", periods=len(df), freq="1H")
    
    # Map item_name (look for product, item, dish, menu_item)
    item_cols = [c for c in df.columns if any(x in c for x in ["item", "product", "dish", "menu", "name"])]
    if item_cols:
        item_col = item_cols[0]
        df_std["item_name"] = df[item_col].astype(str).str.strip()
        print(f"  Mapped item_name <- {item_col}")
    else:
        print("  Warning: no item column detected.")
        df_std["item_name"] = "Unknown Item"
    
    # Map category (look for category, type, cuisine)
    cat_cols = [c for c in df.columns if any(x in c for x in ["category", "type", "cuisine", "class"])]
    if cat_cols:
        cat_col = cat_cols[0]
        df_std["category"] = df[cat_col].astype(str).str.strip()
        print(f"  Mapped category <- {cat_col}")
    else:
        print("  Warning: no category column detected, defaulting to 'Other'.")
        df_std["category"] = "Other"
    
    # Map qty (look for quantity, qty, units, count)
    qty_cols = [c for c in df.columns if any(x in c for x in ["qty", "quantity", "units", "count", "amount"])]
    if qty_cols:
        qty_col = qty_cols[0]
        df_std["qty"] = pd.to_numeric(df[qty_col], errors="coerce").fillna(1).astype(int)
        print(f"  Mapped qty <- {qty_col}")
    else:
        print("  Warning: no quantity column detected, defaulting to 1.")
        df_std["qty"] = 1
    
    # Map price (look for price, amount, cost, total, revenue, sales)
    price_cols = [c for c in df.columns if any(x in c for x in ["price", "amount", "cost", "total", "revenue", "sales"])]
    if price_cols:
        price_col = price_cols[0]
        df_std["price"] = pd.to_numeric(df[price_col], errors="coerce").fillna(0)
        print(f"  Mapped price <- {price_col}")
    else:
        print("  Warning: no price column detected, defaulting to 0.")
        df_std["price"] = 0.0
    
    # Drop rows with missing critical fields
    df_std = df_std.dropna(subset=["order_datetime", "item_name"])
    print(f"  After cleaning: {len(df_std)} rows")
    
    return df_std


def generate_menu(df_sales: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-generate a menu from unique items in sales data.
    Assign realistic COGS based on category.
    """
    print(f"\n[5/5] Generating menu from unique items")
    
    # Extract unique items
    menu = df_sales.groupby(["item_name", "category"])["price"].agg(["mean", "count"]).reset_index()
    menu.columns = ["item_name", "category", "price", "times_ordered"]
    menu = menu.sort_values("times_ordered", ascending=False).reset_index(drop=True)
    
    print(f"  Found {len(menu)} unique items")
    
    # Assign cost_per_unit based on category
    np.random.seed(42)  # reproducibility
    
    def assign_cost(row):
        cat_lower = row["category"].lower()
        # Find matching category range
        cost_range = None
        for key, val_range in COST_PCT_RANGES.items():
            if key in cat_lower:
                cost_range = val_range
                break
        
        # Default if no match
        if cost_range is None:
            cost_range = (0.25, 0.35)
        
        # Random cost percentage within range
        cost_pct = np.random.uniform(*cost_range)
        cost_per_unit = row["price"] * cost_pct
        
        return pd.Series({
            "cost_per_unit": round(cost_per_unit, 2),
            "cost_pct": round(cost_pct, 3),
        })
    
    cost_data = menu.apply(assign_cost, axis=1)
    menu = pd.concat([menu, cost_data], axis=1)
    
    # Calculate GP per unit
    menu["gp_per_unit"] = (menu["price"] - menu["cost_per_unit"]).round(2)
    
    # Reorder columns
    menu = menu[["item_name", "category", "price", "cost_per_unit", "gp_per_unit", "cost_pct"]]
    
    print(f"  Menu ready with {len(menu)} items")
    print(f"\n  Category breakdown:")
    for cat, count in menu["category"].value_counts().items():
        print(f"    - {cat}: {count} items")
    
    return menu


def save_outputs(df_sales: pd.DataFrame, df_menu: pd.DataFrame) -> None:
    """Save cleaned sales and generated menu to CSV files."""
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save sales
    df_sales.to_csv(CLIENT_SALES_PATH, index=False)
    print(f"\n✓ Saved {len(df_sales)} sales rows to {CLIENT_SALES_PATH}")
    
    # Save menu
    df_menu.to_csv(CLIENT_MENU_PATH, index=False)
    print(f"✓ Saved {len(df_menu)} menu items to {CLIENT_MENU_PATH}")


def print_summary(df_sales: pd.DataFrame, df_menu: pd.DataFrame) -> None:
    """Print a summary of the prepared data."""
    print("\n" + "=" * 70)
    print("KAGGLE RESTAURANT DATA PREPARATION — SUMMARY")
    print("=" * 70)
    
    print(f"\nMenu:")
    print(f"  Unique items: {len(df_menu)}")
    print(f"  Categories: {', '.join(df_menu['category'].unique())}")
    print(f"  Price range: £{df_menu['price'].min():.2f} – £{df_menu['price'].max():.2f}")
    print(f"  Avg COGS %: {df_menu['cost_pct'].mean()*100:.1f}%")
    
    print(f"\nSales:")
    print(f"  Total transactions: {len(df_sales)}")
    print(f"  Date range: {df_sales['order_datetime'].min()} to {df_sales['order_datetime'].max()}")
    print(f"  Total revenue: £{(df_sales['price'] * df_sales['qty']).sum():.2f}")
    
    print(f"\nOutput files:")
    print(f"  ✓ {CLIENT_MENU_PATH}")
    print(f"  ✓ {CLIENT_SALES_PATH}")
    
    print(f"\nNext steps:")
    print(f"  1. Verify the CSV files in {DATA_DIR}")
    print(f"  2. Update run_client.py paths if needed (currently configured for these files)")
    print(f"  3. Run: python run_client.py")
    
    print(f"\nMenu preview (first 5 items):")
    print(df_menu[["item_name", "category", "price", "cost_pct", "gp_per_unit"]].head().to_string(index=False))
    
    print(f"\nSales preview (first 5 transactions):")
    print(df_sales[["order_datetime", "item_name", "category", "qty", "price"]].head().to_string(index=False))
    
    print("\n" + "=" * 70)


def main():
    """Main orchestration function."""
    print("\n" + "=" * 70)
    print("RESTAURANT DATA PREPARATION FROM KAGGLE")
    print("=" * 70)
    
    try:
        # 1. Download
        download_kaggle_dataset()
        
        # 2. Unzip and find CSV
        csv_path = unzip_dataset()
        
        # 3. Load and clean
        df_raw = load_and_clean_data(csv_path)
        
        # 4. Standardize to expected schema
        df_sales = standardize_sales_data(df_raw)
        
        # 5. Generate menu
        df_menu = generate_menu(df_sales)
        
        # 6. Save outputs
        save_outputs(df_sales, df_menu)
        
        # 7. Print summary
        print_summary(df_sales, df_menu)
        
        print("\n✓ Data preparation complete!")
        print("  You can now run: python run_client.py\n")
        return 0
    
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
