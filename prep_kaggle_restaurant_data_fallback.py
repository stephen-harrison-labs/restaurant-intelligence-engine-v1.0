#!/usr/bin/env python3
"""
Fallback: Generate Synthetic Client Restaurant Data

If the Kaggle dataset download fails, this script generates synthetic
client CSV files (client_menu.csv and client_sales.csv) that are
compatible with the Restaurant Intelligence Engine.

Usage:
    python prep_kaggle_restaurant_data.py
    (or this fallback will be invoked if Kaggle fails)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CLIENT_MENU_PATH = DATA_DIR / "client_menu.csv"
CLIENT_SALES_PATH = DATA_DIR / "client_sales.csv"

# Realistic restaurant menu data
MENU_DATA = {
    "item_name": [
        "Grilled Salmon", "Beef Steak", "Chicken Pasta", "Vegetable Risotto",
        "Prawn Curry", "Duck Confit", "Lamb Chops", "Margherita Pizza",
        "Caesar Salad", "Tomato Soup", "Garlic Bread", "Calamari Rings",
        "Chocolate Cake", "Tiramisu", "Ice Cream Trio", "Lemon Tart",
        "Espresso", "Cappuccino", "Red Wine", "Craft Beer", "Sparkling Water"
    ],
    "category": [
        "Mains", "Mains", "Mains", "Mains",
        "Mains", "Mains", "Mains", "Mains",
        "Starters", "Starters", "Starters", "Starters",
        "Desserts", "Desserts", "Desserts", "Desserts",
        "Drinks", "Drinks", "Drinks", "Drinks", "Drinks"
    ],
    "price": [
        28.50, 35.00, 18.50, 16.00,
        24.00, 32.00, 38.00, 14.00,
        12.50, 8.00, 5.50, 13.00,
        8.50, 7.50, 10.00, 6.50,
        3.50, 4.50, 8.00, 6.50, 3.00
    ]
}

# Cost percentages by category
COST_PCT = {
    "Mains": 0.32,
    "Starters": 0.28,
    "Desserts": 0.25,
    "Drinks": 0.20,
}


def generate_menu() -> pd.DataFrame:
    """Generate menu DataFrame."""
    df_menu = pd.DataFrame(MENU_DATA)
    
    # Assign cost_per_unit
    df_menu["cost_pct"] = df_menu["category"].map(COST_PCT)
    df_menu["cost_per_unit"] = (df_menu["price"] * df_menu["cost_pct"]).round(2)
    df_menu["gp_per_unit"] = (df_menu["price"] - df_menu["cost_per_unit"]).round(2)
    
    df_menu = df_menu[["item_name", "category", "price", "cost_per_unit", "gp_per_unit", "cost_pct"]]
    return df_menu


def generate_sales(menu: pd.DataFrame, num_days: int = 90, transactions_per_day: int = 50) -> pd.DataFrame:
    """Generate synthetic sales transactions."""
    np.random.seed(42)
    
    # Generate random transactions
    transactions = []
    start_date = datetime(2024, 1, 1)
    
    for day_offset in range(num_days):
        order_date = start_date + timedelta(days=day_offset)
        
        # 5-10 transactions per day
        num_transactions = np.random.randint(5, 15)
        
        for _ in range(num_transactions):
            # Random time during business hours (11:00-22:00)
            hour = np.random.randint(11, 22)
            minute = np.random.randint(0, 60)
            order_datetime = order_date.replace(hour=hour, minute=minute)
            
            # Random item(s) ordered
            num_items = np.random.randint(1, 4)
            for _ in range(num_items):
                item = menu.sample(1).iloc[0]
                qty = np.random.randint(1, 3)
                
                transactions.append({
                    "order_datetime": order_datetime,
                    "item_name": item["item_name"],
                    "category": item["category"],
                    "qty": qty,
                    "price": item["price"],
                })
    
    df_sales = pd.DataFrame(transactions)
    df_sales["order_datetime"] = pd.to_datetime(df_sales["order_datetime"])
    return df_sales.sort_values("order_datetime").reset_index(drop=True)


def main():
    """Generate and save synthetic client data."""
    print("\n" + "=" * 70)
    print("SYNTHETIC CLIENT RESTAURANT DATA GENERATOR")
    print("=" * 70)
    print("\n(Kaggle dataset download failed; generating synthetic data instead.)\n")
    
    # Create data directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate menu
    print("[1/2] Generating menu...")
    df_menu = generate_menu()
    df_menu.to_csv(CLIENT_MENU_PATH, index=False)
    print(f"  ✓ Saved {len(df_menu)} menu items to {CLIENT_MENU_PATH}")
    
    # Generate sales
    print("[2/2] Generating sales transactions...")
    df_sales = generate_sales(df_menu, num_days=90)
    df_sales.to_csv(CLIENT_SALES_PATH, index=False)
    print(f"  ✓ Saved {len(df_sales)} transactions to {CLIENT_SALES_PATH}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMenu:")
    print(f"  Items: {len(df_menu)}")
    print(f"  Categories: {', '.join(df_menu['category'].unique())}")
    print(f"  Price range: £{df_menu['price'].min():.2f} – £{df_menu['price'].max():.2f}")
    
    print(f"\nSales:")
    print(f"  Transactions: {len(df_sales)}")
    print(f"  Date range: {df_sales['order_datetime'].min()} to {df_sales['order_datetime'].max()}")
    total_revenue = (df_sales["price"] * df_sales["qty"]).sum()
    print(f"  Total revenue: £{total_revenue:.2f}")
    
    print(f"\nOutput files:")
    print(f"  ✓ {CLIENT_MENU_PATH}")
    print(f"  ✓ {CLIENT_SALES_PATH}")
    
    print(f"\nNext: Run the client analysis with:")
    print(f"  python run_client.py")
    
    print("\n" + "=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
