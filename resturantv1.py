# %% [markdown]
# # Restaurant Intelligence Engine – v2.0 (Full Demo)
#
# This notebook builds a **full demo** for your consulting offer:
#
# - Synthetic OR client data
# - Menu + GP + category analysis
# - Waste & waste-adjusted GP
# - Staff performance (synthetic demo)
# - Bookings / day-of-week pattern (synthetic demo)
# - Pricing & cost scenarios
# - Visual charts (matplotlib)
# - A big ChatGPT export block to turn into a polished PDF report
#
# Use synthetic mode for your **example PDF**.
# Use client mode when a real restaurant sends you their data.


# %%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import insights_module
from difflib import get_close_matches

pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# Import adjustText for smart label positioning (prevents overlapping labels)
try:
    from adjustText import adjust_text
    ADJUST_TEXT_AVAILABLE = True
except ImportError:
    ADJUST_TEXT_AVAILABLE = False
    print("⚠️  WARNING: adjustText not installed. Menu engineering labels may overlap.")
    print("   Install with: pip install adjustText")

# Provide a safe `display()` for non-interactive runs (falls back to print)
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(obj):
        # For DataFrames, show the head/repr to keep output readable
        try:
            if hasattr(obj, "to_string"):
                print(obj.to_string())
            else:
                print(obj)
        except Exception:
            print(obj)

# %% [markdown]
# ## 1. CONFIG – Master Settings
#
# - Adjust these for your demo or per-client.
# - DATA_SOURCE: "synthetic" for demo, "client" for real data.


# %%
CONFIG = {
    "currency": "£",
    "random_seed": 42,

    # Synthetic data sizes
    "n_menu_items": 40,
    "n_orders": 50000,              # Realistic annual volume for mid-sized restaurant
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",

    # Synthetic cost % ranges by category
    "category_cost_pct_ranges": {
        "Mains": (0.28, 0.36),
        "Starters": (0.25, 0.32),
        "Desserts": (0.18, 0.35),      # WIDER variance: simple puddings vs premium desserts
        "Sides": (0.25, 0.35),
        "Drinks": (0.15, 0.25),
    },

    # Menu engineering thresholds (median split for better quadrant distribution)
    "high_volume_quantile": 0.5,
    "high_margin_quantile": 0.5,

    # Price elasticity assumption
    # -1.0 = unit elasticity (1% volume drop per 1% price increase)
    # -0.3 = inelastic (customers less sensitive to price changes)
    # This affects all pricing scenario calculations
    "price_elasticity_assumption": -1.0,

    # Scenario definitions
    "scenario_price_increase_premium": 0.08,   # +8% premium
    "scenario_price_decrease_puzzles": -0.10,  # -10% puzzles
    "scenario_mains_price_increase": 0.05,     # +5% mains
    "scenario_cost_inflation": 0.05,           # +5% ingredient costs
    
    # Category matching keywords (flexible for different client data)
    "mains_category_keywords": ["Main"],       # Matches "Mains", "Main Dishes", "Main Courses", etc.

    # Demo / report meta
    "period_label": "Jan–Dec 2024",
    "restaurant_name": "The Heritage Kitchen",
    "engine_version": "v1.1-phase1",  # Track which analysis version generated this report

    # Synthetic staff & bookings
    "n_staff": 12,                  # More realistic staffing
    "n_bookings": 6000,             # ~16 bookings per day average

    # Real data file paths (fill when using client mode)
    "client_menu_path": "data/client_menu.csv",
    "client_sales_path": "data/client_sales.csv",
    "client_waste_path": None,   # e.g. "data/client_waste.csv"
}

np.random.seed(CONFIG["random_seed"])

# "synthetic" for your example PDF
# "client" when you have real files + paths set above
DATA_SOURCE = "client"

# %% [markdown]
# ## 2. Synthetic Data Generators
#
# These create a realistic demo restaurant:
# - Menu
# - Orders
# - Waste
# - Staff
# - Bookings


# %%
CATEGORIES = ["Starters", "Mains", "Desserts", "Sides", "Drinks"]


def generate_synthetic_menu(config: dict) -> pd.DataFrame:
    """
    Generates a realistic 40-item restaurant menu using a handcrafted,
    consultant-quality list of dishes. Prices and cost % are assigned
    realistically per category.

    This creates a PERFECT demo dataset for your example PDF.
    """

    # -----------------------------
    # 1. Handcrafted Menu Structure
    # -----------------------------
    starters = [
        "Garlic Bread with Herbs",
        "Crispy Calamari with Lemon Aioli",
        "Tomato & Basil Soup",
        "Chicken Wings (BBQ or Hot)",
        "Smoked Salmon & Capers",
        "Burrata with Cherry Tomatoes",
        "Beef Carpaccio",
        "Halloumi Fries with Sweet Chili",
    ]

    mains = [
        "Chargrilled Ribeye Steak",
        "Pan-Seared Sea Bass",
        "Chicken Alfredo",
        "Classic Beef Burger",
        "Vegan Buddha Bowl",
        "Spaghetti Carbonara",
        "Lamb Shank with Red Wine Jus",
        "Buttermilk Fried Chicken",
        "Wild Mushroom Risotto",
        "Pork Belly with Apple Glaze",
        "Margherita Pizza",
        "Thai Green Curry (Chicken or Veg)",
    ]

    desserts = [
        "Sticky Toffee Pudding",
        "Chocolate Brownie & Ice Cream",
        "Lemon Tart",
        "Cheesecake of the Day",
        "Vanilla Ice Cream Trio",
    ]

    sides = [
        "Fries",
        "Onion Rings",
        "Side Salad",
        "Sweet Potato Fries",
        "Mac & Cheese",
    ]

    drinks = [
        "Coca-Cola",
        "Diet Cola",
        "Elderflower Fizz",
        "House White Wine (175ml)",
        "House Red Wine (175ml)",
        "IPA Pint",
        "Lager Pint",
        "Mojito",
        "Flat White",
        "Americano",
    ]

    # Flatten into one list
    full_menu = (
        [(name, "Starters") for name in starters] +
        [(name, "Mains") for name in mains] +
        [(name, "Desserts") for name in desserts] +
        [(name, "Sides") for name in sides] +
        [(name, "Drinks") for name in drinks]
    )

    df_menu = pd.DataFrame(full_menu, columns=["item_name", "category"])
    df_menu["item_id"] = range(1, len(df_menu) + 1)

    # -----------------------------
    # 2. Assign Realistic Price Ranges
    # -----------------------------
    price_ranges = {
        "Starters": (5.50, 8.50),
        "Mains": (11.00, 24.00),
        "Desserts": (4.50, 8.50),  # Wider variance: simple puddings to premium desserts
        "Sides": (3.50, 6.00),
        "Drinks": (2.50, 9.50),
    }

    prices = []
    for cat in df_menu["category"]:
        low, high = price_ranges[cat]
        prices.append(round(np.random.uniform(low, high), 2))

    df_menu["sell_price"] = prices

    # -----------------------------
    # 3. Assign Cost % by Category
    # -----------------------------
    cost_ranges = config["category_cost_pct_ranges"]
    cost_pct_list = []
    for idx, row in df_menu.iterrows():
        low, high = cost_ranges[row["category"]]
        cost_pct_list.append(np.random.uniform(low, high))

    df_menu["cost_pct"] = cost_pct_list
    df_menu["cost_per_unit"] = (df_menu["sell_price"] * df_menu["cost_pct"]).round(2)
    df_menu["gp_per_unit"] = (df_menu["sell_price"] - df_menu["cost_per_unit"]).round(2)
    df_menu["gp_pct"] = df_menu["gp_per_unit"] / df_menu["sell_price"]

    # -----------------------------
    # 4. MANUAL Engineering for Storytelling
    # -----------------------------

    # ⭐ PREMIUM HERO ITEM (high GP, high volume potential)
    ribeye_idx = df_menu.index[df_menu["item_name"] == "Chargrilled Ribeye Steak"][0]
    df_menu.loc[ribeye_idx, "sell_price"] = 21.50
    df_menu.loc[ribeye_idx, "cost_pct"] = 0.32
    df_menu.loc[ribeye_idx, "cost_per_unit"] = round(21.50 * 0.32, 2)
    df_menu.loc[ribeye_idx, "gp_per_unit"] = round(21.50 - df_menu.loc[ribeye_idx, "cost_per_unit"], 2)
    df_menu.loc[ribeye_idx, "gp_pct"] = df_menu.loc[ribeye_idx, "gp_per_unit"] / 21.50

    # ⭐ PUZZLE ITEM (high GP, but low natural volume)
    sea_bass_idx = df_menu.index[df_menu["item_name"] == "Pan-Seared Sea Bass"][0]
    df_menu.loc[sea_bass_idx, "sell_price"] = 22.00
    df_menu.loc[sea_bass_idx, "cost_pct"] = 0.25  # intentionally super strong
    df_menu.loc[sea_bass_idx, "cost_per_unit"] = round(22.00 * 0.25, 2)
    df_menu.loc[sea_bass_idx, "gp_per_unit"] = round(22.00 - df_menu.loc[sea_bass_idx, "cost_per_unit"], 2)
    df_menu.loc[sea_bass_idx, "gp_pct"] = df_menu.loc[sea_bass_idx, "gp_per_unit"] / 22.00

    # 🚫 LOSS LEADER #1 (low margin to drive traffic)
    veg_bowl_idx = df_menu.index[df_menu["item_name"] == "Vegan Buddha Bowl"][0]
    df_menu.loc[veg_bowl_idx, "sell_price"] = 12.00
    df_menu.loc[veg_bowl_idx, "cost_pct"] = 0.65  # Loss leader - 35% GP
    df_menu.loc[veg_bowl_idx, "cost_per_unit"] = round(12.00 * 0.65, 2)
    df_menu.loc[veg_bowl_idx, "gp_per_unit"] = round(12.00 - df_menu.loc[veg_bowl_idx, "cost_per_unit"], 2)
    df_menu.loc[veg_bowl_idx, "gp_pct"] = df_menu.loc[veg_bowl_idx, "gp_per_unit"] / 12.00

    # 🚫 ULTRA-LOSS LEADER #2 (kids menu strategy - drive family traffic)
    margherita_idx = df_menu.index[df_menu["item_name"] == "Margherita Pizza"][0]
    df_menu.loc[margherita_idx, "sell_price"] = 9.95
    df_menu.loc[margherita_idx, "cost_pct"] = 0.78  # Ultra-loss leader - 22% GP
    df_menu.loc[margherita_idx, "cost_per_unit"] = round(9.95 * 0.78, 2)
    df_menu.loc[margherita_idx, "gp_per_unit"] = round(9.95 - df_menu.loc[margherita_idx, "cost_per_unit"], 2)
    df_menu.loc[margherita_idx, "gp_pct"] = df_menu.loc[margherita_idx, "gp_per_unit"] / 9.95

    # 💰 SUPER HIGH MARGIN DRINK (realistic 92% GP)
    mojito_idx = df_menu.index[df_menu["item_name"] == "Mojito"][0]
    df_menu.loc[mojito_idx, "cost_pct"] = 0.08  # Drinks can be 90%+ GP
    mojito_cost = round(df_menu.loc[mojito_idx, "sell_price"] * 0.08, 2)
    df_menu.loc[mojito_idx, "cost_per_unit"] = mojito_cost
    df_menu.loc[mojito_idx, "gp_per_unit"] = round(df_menu.loc[mojito_idx, "sell_price"] - mojito_cost, 2)
    df_menu.loc[mojito_idx, "gp_pct"] = df_menu.loc[mojito_idx, "gp_per_unit"] / df_menu.loc[mojito_idx, "sell_price"]

    # 🍰 HIGH-WASTE DESSERT
    stp_idx = df_menu.index[df_menu["item_name"] == "Sticky Toffee Pudding"][0]
    df_menu.loc[stp_idx, "cost_pct"] = 0.45  # desserts often waste more + higher cost ratio
    stp_cost = round(df_menu.loc[stp_idx, "sell_price"] * 0.45, 2)
    df_menu.loc[stp_idx, "cost_per_unit"] = stp_cost
    df_menu.loc[stp_idx, "gp_per_unit"] = round(df_menu.loc[stp_idx, "sell_price"] - stp_cost, 2)
    df_menu.loc[stp_idx, "gp_pct"] = df_menu.loc[stp_idx, "gp_per_unit"] / df_menu.loc[stp_idx, "sell_price"]

    return df_menu



def generate_synthetic_orders(config: dict, menu_df: pd.DataFrame) -> pd.DataFrame:
    n_orders = config["n_orders"]
    date_range = pd.date_range(config["start_date"], config["end_date"], freq="H")

    # Popularity weights with HIGH variance (create mega-hits & slow-movers)
    # Boost drinks to achieve 20-30% revenue share (higher popularity)
    base_weights = []
    for _, row in menu_df.iterrows():
        cat = row["category"]
        if cat == "Mains":
            w = np.random.uniform(0.5, 3.5)  # Wider range for variety
        elif cat == "Drinks":
            w = np.random.uniform(0.8, 4.0)  # BOOSTED - drinks should be 20-30% revenue
        elif cat == "Starters":
            w = np.random.uniform(0.3, 2.0)  # More variance
        elif cat == "Desserts":
            w = np.random.uniform(0.2, 1.8)  # Some slow-movers
        else:  # Sides
            w = np.random.uniform(0.2, 1.5)  # Wider range
        base_weights.append(w)

    weights = np.array(base_weights)
    weights = weights / weights.sum()

    # Add realistic seasonality: Christmas peak, summer dip, spring/autumn moderate
    # Month weights: Jan(0.85), Feb(0.80), Mar(0.95), Apr(1.00), May(1.10), Jun(1.15),
    #                Jul(0.90), Aug(0.85), Sep(1.05), Oct(1.10), Nov(1.20), Dec(1.45)
    month_multipliers = {
        1: 0.85, 2: 0.80, 3: 0.95, 4: 1.00, 5: 1.10, 6: 1.15,
        7: 0.90, 8: 0.85, 9: 1.05, 10: 1.10, 11: 1.20, 12: 1.45
    }
    
    # Day-of-week multipliers: Weekend boost (Fri/Sat 1.4x busier)
    dow_multipliers = {
        0: 1.0,  # Monday
        1: 1.0,  # Tuesday
        2: 1.0,  # Wednesday
        3: 1.05, # Thursday
        4: 1.4,  # Friday (weekend starts)
        5: 1.4,  # Saturday
        6: 1.1,  # Sunday
    }
    
    # Allocate orders by month with seasonal variation
    orders_by_month = []
    for month in range(1, 13):
        month_orders = int(n_orders * month_multipliers[month] / sum(month_multipliers.values()))
        month_start = pd.Timestamp(f"2024-{month:02d}-01")
        if month == 12:
            month_end = pd.Timestamp("2024-12-31 23:59:59")
        else:
            month_end = pd.Timestamp(f"2024-{month+1:02d}-01") - pd.Timedelta(seconds=1)
        
        # Generate all potential timestamps
        month_dates_all = pd.date_range(month_start, month_end, freq="h")
        
        # Filter to service hours: 11-14 (lunch 4hrs) and 17-21 (dinner 5hrs) = 75% of orders
        service_dates = month_dates_all[
            ((month_dates_all.hour >= 11) & (month_dates_all.hour <= 14)) |  # Lunch
            ((month_dates_all.hour >= 17) & (month_dates_all.hour <= 21))    # Dinner
        ]
        # Other hours for remaining 25%
        other_dates = month_dates_all[
            ((month_dates_all.hour >= 9) & (month_dates_all.hour < 11)) |
            ((month_dates_all.hour > 14) & (month_dates_all.hour < 17)) |
            ((month_dates_all.hour > 21) & (month_dates_all.hour <= 23))
        ]
        
        # Allocate 75% to service periods, 25% to other
        service_orders = int(month_orders * 0.75)
        other_orders = month_orders - service_orders
        
        # Apply weekend boost AND peak hour weighting (12-1pm lunch, 7-8pm dinner)
        if len(service_dates) > 0:
            # Combine day-of-week and hour-of-day multipliers
            peak_hour_multipliers = {11: 0.9, 12: 1.3, 13: 1.3, 14: 0.8,  # Lunch peak 12-1pm
                                     17: 0.9, 18: 1.0, 19: 1.4, 20: 1.3, 21: 0.8}  # Dinner peak 7-8pm
            combined_weights = []
            for ts in service_dates:
                dow_mult = dow_multipliers[pd.Timestamp(ts).dayofweek]
                hour_mult = peak_hour_multipliers.get(ts.hour, 1.0)
                combined_weights.append(dow_mult * hour_mult)
            service_dow_weights = np.array(combined_weights)
            service_dow_weights = service_dow_weights / service_dow_weights.sum()
            service_timestamps = np.random.choice(service_dates, size=service_orders, p=service_dow_weights, replace=True)
        else:
            service_timestamps = []
            
        if len(other_dates) > 0 and other_orders > 0:
            other_dow_weights = np.array([dow_multipliers[pd.Timestamp(ts).dayofweek] for ts in other_dates])
            other_dow_weights = other_dow_weights / other_dow_weights.sum()
            other_timestamps = np.random.choice(other_dates, size=other_orders, p=other_dow_weights, replace=True)
        else:
            other_timestamps = []
            
        final_timestamps = np.concatenate([service_timestamps, other_timestamps])
        
        month_item_ids = np.random.choice(menu_df["item_id"], size=month_orders, p=weights)
        month_qty = np.random.choice([1, 1, 1, 2, 2, 3], size=month_orders, p=[0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
        
        orders_by_month.append(pd.DataFrame({
            "order_datetime": final_timestamps,
            "item_id": month_item_ids,
            "qty": month_qty,
        }))
    
    df_orders = pd.concat(orders_by_month, ignore_index=True)
    df_orders["order_line_id"] = range(1, len(df_orders) + 1)
    df_orders = df_orders[["order_line_id", "order_datetime", "item_id", "qty"]]
    df_orders = df_orders.sort_values("order_datetime").reset_index(drop=True)
    df_orders["order_line_id"] = range(1, len(df_orders) + 1)

    return df_orders


def generate_synthetic_waste(menu_df: pd.DataFrame) -> pd.DataFrame:
    """
    Synthetic waste per item:
    - More waste on perishable mains/desserts.
    - Less on drinks.
    - Realistic annual waste: 4-6% of revenue
    - Added variance to prevent identical waste amounts
    """
    df = menu_df[["item_id", "category", "cost_per_unit"]].copy()
    # Realistic waste rates targeting 4-5% of revenue (annual totals)
    lam_map = {
        "Mains": 350,        # Very high waste on perishable proteins
        "Starters": 200,     # High waste on fresh ingredients
        "Desserts": 280,     # High waste (prep spoilage, short shelf life)
        "Sides": 150,        # Medium waste
        "Drinks": 40,        # Low waste (long shelf life, spillage only)
    }
    lam = df["category"].map(lam_map).fillna(100)
    df["waste_qty"] = np.random.poisson(lam=lam)
    
    # Add item-level variance (±15%) to prevent identical amounts
    variance = np.random.uniform(0.85, 1.15, size=len(df))
    df["waste_qty"] = (df["waste_qty"] * variance).round().astype(int)
    
    df["waste_qty"] = df["waste_qty"].clip(lower=0)
    df["waste_cost"] = df["waste_qty"] * df["cost_per_unit"]
    return df[["item_id", "waste_qty", "waste_cost"]]


def generate_synthetic_staff(config: dict) -> pd.DataFrame:
    n_staff = config.get("n_staff", 8)
    roles = ["Server", "Server", "Server", "Head Waiter",
             "Bartender", "Bartender", "Supervisor", "Manager"]

    if n_staff <= len(roles):
        staff_roles = roles[:n_staff]
    else:
        staff_roles = (roles * ((n_staff // len(roles)) + 1))[:n_staff]

    # Expanded name pool to avoid duplicates
    names = [
        "Oliver", "Amelia", "Jack", "Sophia", "Liam", "Isla", 
        "Noah", "Mia", "Ethan", "Ava", "Charlotte", "James",
        "Emily", "George", "Poppy", "Harry", "Freya", "Oscar"
    ]
    # Ensure unique names
    if n_staff <= len(names):
        staff_names = names[:n_staff]
    else:
        # If we need more names than available, add suffixes
        staff_names = names[:n_staff]
        for i in range(n_staff - len(names)):
            staff_names.append(f"{names[i % len(names)]} {chr(65 + i // len(names))}")

    base_rates = []
    for r in staff_roles:
        if r in ["Manager", "Supervisor"]:
            base_rates.append(np.random.uniform(13, 18))
        elif r in ["Head Waiter", "Bartender"]:
            base_rates.append(np.random.uniform(11, 15))
        else:
            base_rates.append(np.random.uniform(10, 13))

    staff_df = pd.DataFrame({
        "staff_id": range(1, n_staff + 1),
        "staff_name": staff_names,
        "role": staff_roles,
        "hourly_rate": [round(x, 2) for x in base_rates],
    })
    return staff_df


def assign_staff_to_orders(orders_df: pd.DataFrame, staff_df: pd.DataFrame) -> pd.DataFrame:
    df_orders = orders_df.copy()
    staff_ids = staff_df["staff_id"].values
    weights = np.linspace(1.5, 0.5, len(staff_ids))
    weights = weights / weights.sum()
    assigned_staff = np.random.choice(staff_ids, size=len(df_orders), p=weights)
    df_orders["staff_id"] = assigned_staff
    return df_orders


def generate_synthetic_bookings(config: dict) -> pd.DataFrame:
    n_bookings = config.get("n_bookings", 1200)
    date_range = pd.date_range(config["start_date"], config["end_date"], freq="H")
    base_times = np.random.choice(date_range, size=n_bookings)

    df = pd.DataFrame({
        "booking_id": range(1, n_bookings + 1),
        "booking_datetime": base_times,
    })
    df["day_of_week"] = df["booking_datetime"].dt.day_name()

    dow_weights = {
        "Monday": 0.06,
        "Tuesday": 0.08,
        "Wednesday": 0.11,
        "Thursday": 0.15,
        "Friday": 0.22,
        "Saturday": 0.25,
        "Sunday": 0.13,
    }
    probs = df["day_of_week"].map(dow_weights)
    probs = probs / probs.sum()
    sampled_idx = np.random.choice(df.index, size=n_bookings, p=probs)
    df = df.loc[sampled_idx].reset_index(drop=True)
    df["booking_id"] = range(1, len(df) + 1)

    df["covers"] = np.random.choice(
        [1, 2, 2, 2, 3, 4, 5, 6],
        size=len(df),
        p=[0.07, 0.35, 0.15, 0.1, 0.15, 0.1, 0.05, 0.03]
    )

    sources = ["Phone", "Online", "Walk-in", "Third-party App"]
    source_probs = [0.25, 0.4, 0.25, 0.1]
    df["source"] = np.random.choice(sources, size=len(df), p=source_probs)

    status = ["Arrived", "Arrived", "Arrived", "No-show", "Cancelled"]
    status_probs = [0.7, 0.15, 0.05, 0.05, 0.05]
    df["status"] = np.random.choice(status, size=len(df), p=status_probs)

    return df


def generate_synthetic_restaurant(config: dict):
    """
    Full synthetic dataset for your demo:
    - menu_df
    - orders_df (with staff_id)
    - waste_df
    - staff_df
    - bookings_df
    """
    menu_df = generate_synthetic_menu(config)
    staff_df = generate_synthetic_staff(config)
    orders_df = generate_synthetic_orders(config, menu_df)
    orders_df = assign_staff_to_orders(orders_df, staff_df)
    waste_df = generate_synthetic_waste(menu_df)
    bookings_df = generate_synthetic_bookings(config)
    return menu_df, orders_df, waste_df, staff_df, bookings_df

# %% [markdown]
# ## 3. Real Data Loaders (Menu, Sales, Waste)
#
# Use these when `DATA_SOURCE = "client"` and you have CSV/Excel exports.


# %%
REQUIRED_MENU_COLUMNS = ["item_name", "category", "sell_price", "cost_per_unit"]
REQUIRED_SALES_COLUMNS = ["order_datetime", "item_name", "qty"]

DEFAULT_MENU_COLUMN_MAP = {
    "Item": "item_name",
    "Item Name": "item_name",
    "Menu Item": "item_name",
    "Category": "category",
    "Menu Category": "category",
    "Sell Price": "sell_price",
    "Price": "sell_price",
    "Cost": "cost_per_unit",
    "Recipe Cost": "cost_per_unit",
}

DEFAULT_SALES_COLUMN_MAP = {
    "Order Date": "order_datetime",
    "Date": "order_datetime",
    "DateTime": "order_datetime",
    "Item": "item_name",
    "Item Name": "item_name",
    "Quantity": "qty",
    "Qty": "qty",
}

DEFAULT_WASTE_COLUMN_MAP = {
    "Item": "item_name",
    "Item Name": "item_name",
    "Menu Item": "item_name",
    "Waste Qty": "waste_qty",
    "Waste Quantity": "waste_qty",
    "Qty": "waste_qty",
}

# Category normalization mapping (handles common variations)
CATEGORY_NORMALIZATION_MAP = {
    "main": "Mains",
    "mains": "Mains",
    "main dishes": "Mains",
    "main dish": "Mains",
    "entrees": "Mains",
    "entree": "Mains",
    "starter": "Starters",
    "starters": "Starters",
    "appetizer": "Starters",
    "appetizers": "Starters",
    "dessert": "Desserts",
    "desserts": "Desserts",
    "sweet": "Desserts",
    "sweets": "Desserts",
    "side": "Sides",
    "sides": "Sides",
    "side dishes": "Sides",
    "side dish": "Sides",
    "drink": "Drinks",
    "drinks": "Drinks",
    "beverage": "Drinks",
    "beverages": "Drinks",
}

def normalize_category_name(category: str) -> str:
    """Normalize category names to standard format with typo tolerance."""
    if pd.isna(category) or not category:
        return "Uncategorized"
    
    # Strip whitespace and lowercase for matching
    clean = str(category).strip().lower()
    
    # Try exact match first
    if clean in CATEGORY_NORMALIZATION_MAP:
        return CATEGORY_NORMALIZATION_MAP[clean]
    
    # Try fuzzy matching for typos (e.g., "mainss" -> "Mains")
    from difflib import get_close_matches
    matches = get_close_matches(clean, CATEGORY_NORMALIZATION_MAP.keys(), n=1, cutoff=0.8)
    if matches:
        return CATEGORY_NORMALIZATION_MAP[matches[0]]
    
    # If no match, return title-cased original (preserve unique categories)
    return category.strip().title()


def _read_any_table(path: str) -> pd.DataFrame:
    """Read CSV/Excel with robust encoding handling and error reporting."""
    if path.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    else:
        # Try UTF-8 with BOM first (handles Excel exports)
        try:
            df = pd.read_csv(path, encoding='utf-8-sig', on_bad_lines='warn')
            return df
        except UnicodeDecodeError:
            # Fallback to latin-1 (handles legacy Windows files)
            print(f"⚠️  UTF-8 decode failed, trying latin-1 encoding for {path}")
            df = pd.read_csv(path, encoding='latin-1', on_bad_lines='warn')
            return df
        except Exception as e:
            raise ValueError(f"Failed to read {path}: {str(e)}")


def load_client_menu_and_sales(config: dict,
                               menu_column_map: dict = None,
                               sales_column_map: dict = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    menu_path = config["client_menu_path"]
    sales_path = config["client_sales_path"]
    if not menu_path or not sales_path:
        raise ValueError("Set CONFIG['client_menu_path'] and CONFIG['client_sales_path'].")

    menu_raw = _read_any_table(menu_path)
    sales_raw = _read_any_table(sales_path)

    menu_map = DEFAULT_MENU_COLUMN_MAP.copy()
    if menu_column_map:
        menu_map.update(menu_column_map)

    sales_map = DEFAULT_SALES_COLUMN_MAP.copy()
    if sales_column_map:
        sales_map.update(sales_column_map)

    menu_df = menu_raw.rename(columns={k: v for k, v in menu_map.items() if k in menu_raw.columns})
    sales_df = sales_raw.rename(columns={k: v for k, v in sales_map.items() if k in sales_raw.columns})

    missing_menu = [c for c in REQUIRED_MENU_COLUMNS if c not in menu_df.columns]
    missing_sales = [c for c in REQUIRED_SALES_COLUMNS if c not in sales_df.columns]
    if missing_menu:
        available = list(menu_raw.columns)
        raise ValueError(
            f"❌ CRITICAL: Menu file missing required columns: {missing_menu}\n"
            f"Available columns: {available}\n"
            f"Please check column names match exactly or add to DEFAULT_MENU_COLUMN_MAP"
        )
    if missing_sales:
        available = list(sales_raw.columns)
        raise ValueError(
            f"❌ CRITICAL: Sales file missing required columns: {missing_sales}\n"
            f"Available columns: {available}\n"
            f"Please check column names match exactly or add to DEFAULT_SALES_COLUMN_MAP"
        )

    menu_df = menu_df.copy()
    
    # Normalize categories to fix case/typo issues
    menu_df["category"] = menu_df["category"].apply(normalize_category_name)
    menu_df["item_name"] = menu_df["item_name"].astype(str).str.strip()
    
    # Check for empty categories BEFORE proceeding
    empty_cats = (menu_df["category"] == "Uncategorized").sum()
    if empty_cats > 0:
        raise ValueError(
            f"❌ CRITICAL: {empty_cats} menu items have missing/empty category.\n"
            f"All items must have a valid category for analysis."
        )

    # Track original row count for quality reporting
    original_menu_rows = len(menu_df)
    
    for col in ["sell_price", "cost_per_unit"]:
        # Strip all common currency symbols, not just CONFIG currency
        menu_df[col] = (
            menu_df[col]
            .astype(str)
            .str.replace(r"[£$€¥USD GBP EUR]", "", regex=True)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
        # Track conversion failures
        original_vals = menu_df[col].copy()
        menu_df[col] = pd.to_numeric(menu_df[col], errors="coerce")
        failed_conversions = menu_df[col].isna().sum() - original_vals.isna().sum()
        if failed_conversions > 0:
            print(f"⚠️  WARNING: {failed_conversions} rows had invalid {col} values (set to NaN)")

    # Track dropped rows
    before_drop = len(menu_df)
    menu_df = menu_df.dropna(subset=["sell_price", "cost_per_unit"])
    dropped_rows = before_drop - len(menu_df)
    if dropped_rows > 0:
        pct_dropped = (dropped_rows / original_menu_rows) * 100
        if pct_dropped > 5:
            raise ValueError(
                f"❌ CRITICAL: Dropped {dropped_rows} menu items ({pct_dropped:.1f}%) due to missing price/cost.\n"
                f"This is too many rows to proceed safely. Please fix source data."
            )
        else:
            print(f"⚠️  WARNING: Dropped {dropped_rows} menu items ({pct_dropped:.1f}%) with missing price/cost")
    # C1.2 + H1.2: Check for duplicate item names BEFORE assigning IDs
    duplicates = menu_df[menu_df.duplicated(subset=["item_name"], keep=False)]
    if len(duplicates) > 0:
        dup_names = duplicates["item_name"].unique()
        raise ValueError(
            f"❌ CRITICAL: {len(dup_names)} duplicate item names found in menu: {list(dup_names)[:5]}\n"
            f"Each item must have a unique name. Please fix source data or merge duplicate entries."
        )
    
    menu_df["item_id"] = range(1, len(menu_df) + 1)

    # C2.1: Division by zero protection for GP calculations
    menu_df["gp_per_unit"] = (menu_df["sell_price"] - menu_df["cost_per_unit"]).round(2)
    menu_df["gp_pct"] = np.where(
        menu_df["sell_price"] > 0,
        menu_df["gp_per_unit"] / menu_df["sell_price"],
        0.0
    )
    menu_df["cost_pct"] = np.where(
        menu_df["sell_price"] > 0,
        menu_df["cost_per_unit"] / menu_df["sell_price"],
        0.0
    )

    # H1.3: Track date parsing failures
    original_sales_rows = len(sales_df)
    sales_df["order_datetime"] = pd.to_datetime(sales_df["order_datetime"], errors="coerce")
    unparseable_dates = sales_df["order_datetime"].isna().sum()
    
    if unparseable_dates > 0:
        pct_failed = (unparseable_dates / original_sales_rows) * 100
        print(f"⚠️  WARNING: {unparseable_dates} rows ({pct_failed:.1f}%) have unparseable dates")
        if pct_failed > 5:
            raise ValueError(
                f"❌ CRITICAL: {pct_failed:.1f}% of dates failed to parse. This is too high to proceed.\n"
                f"Check date format in sales file. Expected formats: YYYY-MM-DD, DD/MM/YYYY, MM/DD/YYYY"
            )
    
    sales_df = sales_df.dropna(subset=["order_datetime"])

    sales_df["qty"] = pd.to_numeric(sales_df["qty"], errors="coerce").fillna(0).astype(int)
    sales_df = sales_df[sales_df["qty"] > 0]

    # C2.5 + S9.1: CRITICAL FIX - Use LEFT join to preserve all sales, audit unmatched items
    before_merge_count = len(sales_df)
    sales_df = sales_df.merge(
        menu_df[["item_id", "item_name"]],
        on="item_name",
        how="left",  # CHANGED FROM INNER - preserves all sales
        indicator=True  # Track merge quality
    )
    
    # Audit merge quality
    unmatched_sales = sales_df[sales_df["_merge"] == "left_only"]
    if len(unmatched_sales) > 0:
        pct_unmatched = (len(unmatched_sales) / before_merge_count) * 100
        unmatched_items = unmatched_sales["item_name"].value_counts().head(10)
        
        print(f"⚠️  WARNING: {len(unmatched_sales)} sales rows ({pct_unmatched:.1f}%) not found in menu:")
        for item, count in unmatched_items.items():
            print(f"    • {item}: {count} orders")
        
        if pct_unmatched > 20:
            raise ValueError(
                f"❌ CRITICAL: {pct_unmatched:.1f}% of sales don't match menu items.\n"
                f"This suggests:  1) Menu file is incomplete, or\n"
                f"              2) Item names don't match between files (check spelling/case), or\n"
                f"              3) Sales data includes removed/seasonal items\n"
                f"Please fix before proceeding."
            )
        else:
            # Create "Unknown" category for unmatched items
            max_id = menu_df["item_id"].max()
            unknown_items = unmatched_sales["item_name"].unique()
            for i, unknown_item in enumerate(unknown_items):
                sales_df.loc[sales_df["item_name"] == unknown_item, "item_id"] = max_id + i + 1
            print(f"    → Created temporary IDs for {len(unknown_items)} unknown items")
    
    sales_df = sales_df.drop(columns=["_merge"])
    sales_df = sales_df.reset_index(drop=True)
    sales_df["order_line_id"] = range(1, len(sales_df) + 1)

    return menu_df, sales_df


def load_client_waste(config: dict,
                      menu_df: pd.DataFrame,
                      waste_column_map: dict = None) -> pd.DataFrame:
    path = config["client_waste_path"]
    if not path:
        return pd.DataFrame(columns=["item_id", "waste_qty", "waste_cost"])

    waste_raw = _read_any_table(path)
    waste_map = DEFAULT_WASTE_COLUMN_MAP.copy()
    if waste_column_map:
        waste_map.update(waste_column_map)

    waste_df = waste_raw.rename(columns={k: v for k, v in waste_map.items() if k in waste_raw.columns})

    if "item_name" not in waste_df.columns or "waste_qty" not in waste_df.columns:
        raise ValueError("Waste file must have item name + waste_qty columns (after mapping).")

    waste_df["item_name"] = waste_df["item_name"].astype(str).str.strip()
    waste_df["waste_qty"] = pd.to_numeric(waste_df["waste_qty"], errors="coerce").fillna(0)

    # C1.7: CRITICAL FIX - Use LEFT join to preserve all waste records, audit unmatched
    before_merge = len(waste_df)
    waste_df = waste_df.merge(
        menu_df[["item_id", "item_name", "cost_per_unit"]],
        on="item_name",
        how="left",  # CHANGED FROM INNER - preserves all waste records
        indicator=True
    )
    
    # Audit merge quality
    unmatched_waste = waste_df[waste_df["_merge"] == "left_only"]
    if len(unmatched_waste) > 0:
        pct_unmatched = (len(unmatched_waste) / before_merge) * 100
        unmatched_items = unmatched_waste["item_name"].unique()
        print(f"⚠️  WARNING: {len(unmatched_waste)} waste records ({pct_unmatched:.1f}%) not found in menu:")
        for item in unmatched_items[:5]:
            print(f"    • {item}")
        print("    These waste records will be excluded from analysis (no cost data available)")
    
    waste_df = waste_df.drop(columns=["_merge"])
    waste_df = waste_df.dropna(subset=["item_id", "cost_per_unit"])  # Drop unmatched records
    waste_df["waste_cost"] = waste_df["waste_qty"] * waste_df["cost_per_unit"]

    return waste_df[["item_id", "waste_qty", "waste_cost"]]

# %% [markdown]
# ## 4. Load Data (Synthetic or Client)
# NOTE: Data loading is now handled ONLY by run_full_analysis_v2() function below.
# The module-level code that was here has been removed to avoid loading synthetic
# data at import time.


# %% [markdown]
# *** OLD MODULE-LEVEL DATA LOADING CODE REMOVED ***
# Data loading now happens exclusively within run_full_analysis_v2(data_source=...)
# This ensures that callers can choose "synthetic" vs "client" mode dynamically.

# %% [markdown]
# ## 4.5. Data Validation
#
# Validates client data quality before analysis to catch common issues early.


# %%
def validate_client_data(menu_df: pd.DataFrame, 
                        orders_df: pd.DataFrame, 
                        waste_df: pd.DataFrame = None,
                        config: dict = None) -> dict:
    """
    Validates client data for common issues that would break analysis or produce incorrect results.
    
    Returns:
        {
            "valid": bool,
            "errors": [list of critical issues that prevent analysis],
            "warnings": [list of issues that may affect quality but won't break],
            "summary": {dict of key metrics}
        }
    """
    errors = []
    warnings = []
    summary = {}
    
    # --- MENU VALIDATION ---
    if menu_df is None or menu_df.empty:
        errors.append("CRITICAL: menu_df is empty or None")
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": summary}
    
    # Required columns
    required_menu_cols = ["item_id", "item_name", "category", "sell_price"]
    missing_cols = [col for col in required_menu_cols if col not in menu_df.columns]
    if missing_cols:
        errors.append(f"CRITICAL: Menu missing required columns: {missing_cols}")
    
    # Check for cost data
    if "cost_per_unit" not in menu_df.columns and "cost_pct" not in menu_df.columns:
        warnings.append("WARNING: No cost data (cost_per_unit or cost_pct) - GP calculations will be inaccurate")
    
    # Category validation
    null_categories = menu_df["category"].isna().sum()
    if null_categories > 0:
        errors.append(f"CRITICAL: {null_categories} menu items have null/empty category")
    
    empty_categories = (menu_df["category"].astype(str).str.strip() == "").sum()
    if empty_categories > 0:
        errors.append(f"CRITICAL: {empty_categories} menu items have empty category")
    
    # Price validation
    if "sell_price" in menu_df.columns:
        zero_price = (menu_df["sell_price"] <= 0).sum()
        if zero_price > 0:
            warnings.append(f"WARNING: {zero_price} items have zero or negative price")
        
        summary["menu_items"] = len(menu_df)
        summary["avg_price"] = menu_df["sell_price"].mean()
        summary["price_range"] = f"{menu_df['sell_price'].min():.2f} - {menu_df['sell_price'].max():.2f}"
    
    # Cost validation (if present)
    if "cost_per_unit" in menu_df.columns:
        negative_gp = ((menu_df["cost_per_unit"] > menu_df["sell_price"]) & 
                       (menu_df["sell_price"] > 0)).sum()
        if negative_gp > 0:
            warnings.append(f"WARNING: {negative_gp} items have cost > price (negative GP)")
    
    # Duplicate item names
    duplicates = menu_df["item_name"].duplicated().sum()
    if duplicates > 0:
        warnings.append(f"WARNING: {duplicates} duplicate item names found")
    
    # --- ORDERS VALIDATION ---
    if orders_df is None or orders_df.empty:
        errors.append("CRITICAL: orders_df is empty or None")
        return {"valid": False, "errors": errors, "warnings": warnings, "summary": summary}
    
    # Required columns
    required_order_cols = ["order_datetime", "item_id", "qty"]
    missing_order_cols = [col for col in required_order_cols if col not in orders_df.columns]
    if missing_order_cols:
        errors.append(f"CRITICAL: Orders missing required columns: {missing_order_cols}")
    
    # Order volume check
    order_count = len(orders_df)
    summary["total_orders"] = order_count
    if order_count < 100:
        warnings.append(f"WARNING: Only {order_count} orders - need 100+ for meaningful analysis")
    elif order_count < 500:
        warnings.append(f"INFO: {order_count} orders - recommend 500+ for robust insights")
    
    # Date validation
    if "order_datetime" in orders_df.columns:
        try:
            orders_dt = pd.to_datetime(orders_df["order_datetime"], errors='coerce')
            null_dates = orders_dt.isna().sum()
            if null_dates > 0:
                warnings.append(f"WARNING: {null_dates} orders have invalid dates")
            
            if null_dates < len(orders_df):
                summary["date_range"] = f"{orders_dt.min()} to {orders_dt.max()}"
                days_span = (orders_dt.max() - orders_dt.min()).days
                summary["days_of_data"] = days_span
                if days_span < 30:
                    warnings.append(f"WARNING: Only {days_span} days of data - recommend 30+ days")
        except Exception as e:
            errors.append(f"CRITICAL: Cannot parse order_datetime: {e}")
    
    # Quantity validation
    if "qty" in orders_df.columns:
        negative_qty = (orders_df["qty"] < 0).sum()
        if negative_qty > 0:
            warnings.append(f"WARNING: {negative_qty} orders with negative quantity (returns?)")
        
        zero_qty = (orders_df["qty"] == 0).sum()
        if zero_qty > 0:
            warnings.append(f"WARNING: {zero_qty} orders with zero quantity")
    
    # Item_id matching
    if "item_id" in orders_df.columns and "item_id" in menu_df.columns:
        order_items = set(orders_df["item_id"].unique())
        menu_items = set(menu_df["item_id"].unique())
        orphan_orders = order_items - menu_items
        if orphan_orders:
            orphan_count = orders_df[orders_df["item_id"].isin(orphan_orders)].shape[0]
            warnings.append(f"WARNING: {len(orphan_orders)} item_ids in orders not found in menu ({orphan_count} order lines)")
    
    # --- WASTE VALIDATION (if provided) ---
    if waste_df is not None and not waste_df.empty:
        summary["has_waste_data"] = True
        if "waste_cost" in waste_df.columns:
            total_waste = waste_df["waste_cost"].sum()
            summary["total_waste"] = f"{total_waste:,.2f}"
    else:
        summary["has_waste_data"] = False
        warnings.append("INFO: No waste data provided - analysis will use waste=0")
    
    # --- CATEGORY VALIDATION ---
    if "category" in menu_df.columns:
        categories = menu_df["category"].dropna().unique()
        summary["categories"] = list(categories)
        summary["category_count"] = len(categories)
        
        # Check for mains category match
        if config:
            mains_keywords = config.get("mains_category_keywords", ["Main"])
            pattern = "|".join([k.strip() for k in mains_keywords if k.strip()])
            mains_match = menu_df["category"].astype(str).str.contains(pattern, case=False, na=False).sum()
            if mains_match == 0:
                warnings.append(f"WARNING: No categories match mains keywords {mains_keywords} - Scenario C won't work")
            else:
                summary["mains_items_matched"] = mains_match
    
    # Final validation result
    valid = len(errors) == 0
    
    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "summary": summary
    }


# %% [markdown]
# ## 5. Core Menu Performance + Waste-Adjusted GP


# %%
def build_menu_performance(menu_df: pd.DataFrame,
                           orders_df: pd.DataFrame,
                           waste_df: pd.DataFrame,
                           config: dict):
    # Aggregate sales
    sales = (
        orders_df
        .groupby("item_id")["qty"]
        .sum()
        .rename("units_sold")
        .reset_index()
    )

    df = menu_df.merge(sales, on="item_id", how="left")
    df["units_sold"] = df["units_sold"].fillna(0).astype(int)

    # Merge waste
    df = df.merge(waste_df, on="item_id", how="left")
    df["waste_qty"] = df["waste_qty"].fillna(0)
    df["waste_cost"] = df["waste_cost"].fillna(0)

    # Core revenue & GP
    df["revenue"] = df["units_sold"] * df["sell_price"]
    df["gross_profit"] = df["units_sold"] * df["gp_per_unit"]

    # Waste-adjusted GP
    df["gp_after_waste"] = df["gross_profit"] - df["waste_cost"]
    
    # C2.1: Division-by-zero protection
    df["gp_pct"] = np.where(
        df["sell_price"] > 0,
        df["gp_per_unit"] / df["sell_price"],
        0.0
    )
    df["gp_pct_after_waste"] = np.where(
        df["revenue"] > 0,
        df["gp_after_waste"] / df["revenue"],
        0.0,
    )

    total_gp = df["gross_profit"].sum()
    total_gp_after_waste = df["gp_after_waste"].sum()
    total_rev = df["revenue"].sum()
    total_waste_cost = df["waste_cost"].sum()

    df["margin_contribution_pct"] = np.where(
        total_gp > 0, df["gross_profit"] / total_gp, 0.0
    )
    df["waste_contribution_pct"] = np.where(
        total_waste_cost > 0, df["waste_cost"] / total_waste_cost, 0.0
    )

    # Category stats
    cat_stats = (
        df.groupby("category")
        .agg(
            cat_total_revenue=("revenue", "sum"),
            cat_total_gp=("gross_profit", "sum"),
            cat_total_gp_after_waste=("gp_after_waste", "sum"),
            cat_total_waste=("waste_cost", "sum"),
            cat_avg_gp_pct=("gp_pct", "mean"),
            cat_units_sold=("units_sold", "sum"),
        )
        .reset_index()
    )
    cat_stats["cat_gp_pct"] = np.where(
        cat_stats["cat_total_revenue"] > 0,
        cat_stats["cat_total_gp"] / cat_stats["cat_total_revenue"],
        0.0,
    )
    cat_stats["cat_gp_pct_after_waste"] = np.where(
        cat_stats["cat_total_revenue"] > 0,
        cat_stats["cat_total_gp_after_waste"] / cat_stats["cat_total_revenue"],
        0.0,
    )

    df = df.merge(
        cat_stats[["category", "cat_avg_gp_pct", "cat_gp_pct", "cat_gp_pct_after_waste"]],
        on="category",
        how="left",
    )

    df["gp_vs_cat_pct_points"] = (df["gp_pct"] - df["cat_avg_gp_pct"]) * 100

    # C3.1: Size-aware thresholds for menu engineering
    menu_size = len(df)
    if menu_size < 30:  # Small restaurant
        high_volume_quantile = 0.70  # Top 30% (e.g., top 9 of 30 items)
        high_margin_quantile = 0.70
        print(f"ℹ️  Small menu ({menu_size} items): Using top 30% thresholds for Stars")
    elif menu_size < 80:  # Medium restaurant
        high_volume_quantile = 0.60  # Top 40% (e.g., top 32 of 80 items)
        high_margin_quantile = 0.60
        print(f"ℹ️  Medium menu ({menu_size} items): Using top 40% thresholds for Stars")
    else:  # Large restaurant
        high_volume_quantile = config.get("high_volume_quantile", 0.50)  # Top 50%
        high_margin_quantile = config.get("high_margin_quantile", 0.50)
        print(f"ℹ️  Large menu ({menu_size} items): Using top 50% thresholds for Stars")

    # Thresholds
    vol_threshold = df["units_sold"].quantile(high_volume_quantile)
    gp_threshold = df["gp_pct"].quantile(high_margin_quantile)

    df["is_high_volume"] = df["units_sold"] >= vol_threshold
    df["is_high_margin"] = df["gp_pct"] >= gp_threshold

    # Menu engineering
    def classify_item(row):
        hv = row["is_high_volume"]
        hm = row["is_high_margin"]
        if hv and hm:
            return "Star"
        elif hv and not hm:
            return "Plowhorse"
        elif (not hv) and hm:
            return "Puzzle"
        else:
            return "Dog"

    df["menu_engineering_class"] = df.apply(classify_item, axis=1)

    # Consultant tags (including waste)
    tags = []
    for _, row in df.iterrows():
        item_tags = []

        if row["sell_price"] > df["sell_price"].quantile(0.8) and row["gp_pct"] > df["gp_pct"].quantile(0.5):
            item_tags.append("Strategic Anchor")

        if row["gp_vs_cat_pct_points"] > 3 and row["is_high_volume"]:
            item_tags.append("Underpriced Driver")

        if (row["units_sold"] < df["units_sold"].quantile(0.3)) and (row["gp_vs_cat_pct_points"] < 1):
            item_tags.append("Overpriced Risk")

        if (row["gp_vs_cat_pct_points"] > 5) and (row["units_sold"] < df["units_sold"].quantile(0.4)):
            item_tags.append("Premium Opportunity")

        if row["waste_cost"] > 0 and row["waste_cost"] > 0.15 * max(row["gross_profit"], 1):
            item_tags.append("High Waste Risk")

        if row["waste_cost"] > df["waste_cost"].quantile(0.8):
            item_tags.append("Top Waste Contributor")

        tags.append(", ".join(item_tags) if item_tags else "")

    df["consultant_tags"] = tags

    df = df.sort_values("gross_profit", ascending=False).reset_index(drop=True)

    summary = {
        "total_revenue": total_rev,
        "total_gp": total_gp,
        "total_gp_before_waste": total_gp,  # Alias for consistency
        "total_gp_after_waste": total_gp_after_waste,
        "avg_gp_pct": (total_gp / total_rev) if total_rev > 0 else 0.0,
        "avg_gp_pct_before_waste": (total_gp / total_rev) if total_rev > 0 else 0.0,  # Alias
        "avg_gp_pct_after_waste": (total_gp_after_waste / total_rev) if total_rev > 0 else 0.0,
        "total_waste_cost": total_waste_cost,
        "total_units_sold": int(df["units_sold"].sum()),
        "days_of_data": 0,  # Will be filled from diagnostics in insight graph builder
    }

    cat_summary = (
        df.groupby("category")
        .agg(
            total_revenue=("revenue", "sum"),
            total_gp=("gross_profit", "sum"),
            total_gp_after_waste=("gp_after_waste", "sum"),
            total_waste=("waste_cost", "sum"),
            units_sold=("units_sold", "sum"),
        )
        .reset_index()
    )
    cat_summary["gp_pct"] = np.where(
        cat_summary["total_revenue"] > 0,
        cat_summary["total_gp"] / cat_summary["total_revenue"],
        0.0,
    )
    cat_summary["gp_pct_after_waste"] = np.where(
        cat_summary["total_revenue"] > 0,
        cat_summary["total_gp_after_waste"] / cat_summary["total_revenue"],
        0.0,
    )

    return df, summary, cat_summary


# *** OLD MODULE-LEVEL ANALYSIS CODE REMOVED ***
# All analysis now happens within run_full_analysis_v2() function.
# This function-level code was run at module import time and caused issues.

# %% [markdown]
# ## 6. Staff Performance Analysis


# %%
def build_staff_performance(orders_df: pd.DataFrame,
                            staff_df: pd.DataFrame,
                            perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates revenue & GP per staff member based on orders they handled.
    GP here is before waste (that's fine for demo).
    """
    if staff_df.empty or "staff_id" not in orders_df.columns:
        return pd.DataFrame(columns=[
            "staff_name", "role", "orders_handled", "units_sold",
            "revenue", "gp", "est_hours_worked",
            "revenue_per_hour", "gp_per_hour"
        ])

    order_items = orders_df.merge(
        perf_df[["item_id", "sell_price", "gp_per_unit"]],
        on="item_id",
        how="left"
    )
    order_items["line_revenue"] = order_items["qty"] * order_items["sell_price"]
    order_items["line_gp"] = order_items["qty"] * order_items["gp_per_unit"]

    staff_agg = (
        order_items
        .groupby("staff_id")
        .agg(
            orders_handled=("order_line_id", "count"),
            units_sold=("qty", "sum"),
            revenue=("line_revenue", "sum"),
            gp=("line_gp", "sum"),
        )
        .reset_index()
    )

    staff_perf = staff_agg.merge(staff_df, on="staff_id", how="left")

    # Realistic annual hours: 48 weeks * 40 hours = ~1920 hours per full-time staff
    # Distribute proportionally based on orders handled
    annual_hours_per_staff = 1920
    total_staff_hours = len(staff_df) * annual_hours_per_staff
    if staff_perf["orders_handled"].sum() > 0:
        staff_perf["est_hours_worked"] = total_staff_hours * (
            staff_perf["orders_handled"] / staff_perf["orders_handled"].sum()
        )
    else:
        staff_perf["est_hours_worked"] = 0.0

    staff_perf["revenue_per_hour"] = np.where(
        staff_perf["est_hours_worked"] > 0,
        staff_perf["revenue"] / staff_perf["est_hours_worked"],
        0.0,
    )
    staff_perf["gp_per_hour"] = np.where(
        staff_perf["est_hours_worked"] > 0,
        staff_perf["gp"] / staff_perf["est_hours_worked"],
        0.0,
    )

    staff_perf = staff_perf[
        ["staff_name", "role", "orders_handled", "units_sold",
         "revenue", "gp", "est_hours_worked",
         "revenue_per_hour", "gp_per_hour"]
    ].sort_values("gp_per_hour", ascending=False)

    return staff_perf


# NOTE: Module-level exploratory code removed to allow proper module import
# All data loading and analysis now happens inside run_full_analysis_v2()
# This section previously called: build_staff_performance(), build_booking_summary(), etc.
# at module import time, preventing dynamic data_source switching.

# %% [markdown]
# ## 7. Booking / Day-of-Week Analysis


# %%
def build_booking_summary(bookings_df: pd.DataFrame) -> pd.DataFrame:
    if bookings_df.empty:
        return pd.DataFrame(columns=[
            "day_of_week", "booking_count", "total_covers",
            "avg_covers_per_booking", "no_show_rate_pct"
        ])

    df = bookings_df.copy()
    df["day_of_week"] = df["booking_datetime"].dt.day_name()

    by_dow = (
        df.groupby("day_of_week")
        .agg(
            booking_count=("booking_id", "count"),
            total_covers=("covers", "sum"),
            no_show_count=("status", lambda x: (x == "No-show").sum() + (x == "Cancelled").sum())
        )
        .reset_index()
    )
    by_dow["avg_covers_per_booking"] = by_dow["total_covers"] / by_dow["booking_count"]
    by_dow["no_show_rate_pct"] = np.where(
        by_dow["booking_count"] > 0,
        by_dow["no_show_count"] / by_dow["booking_count"] * 100,
        0.0,
    )

    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow["dow_sort"] = by_dow["day_of_week"].apply(lambda x: dow_order.index(x) if x in dow_order else 7)
    by_dow = by_dow.sort_values("dow_sort").drop(columns=["dow_sort"])

    return by_dow


# NOTE: Module-level call removed. booking_summary_df is now computed inside run_full_analysis_v2()
# booking_summary_df = build_booking_summary(bookings_df)

# print("Booking summary (by day of week):")
# display(booking_summary_df)

# %% [markdown]
# ## 8. Pricing & Cost Scenarios


# %%
def apply_price_change_scenario(perf_df: pd.DataFrame,
                                price_change_filter: pd.Series,
                                price_change_pct: float,
                                elasticity: float) -> dict:
    df = perf_df.copy()

    base_rev = df["revenue"].sum()
    base_gp = df["gp_after_waste"].sum()

    df["scenario_sell_price"] = df["sell_price"]
    df.loc[price_change_filter, "scenario_sell_price"] = df.loc[price_change_filter, "sell_price"] * (1 + price_change_pct)

    price_ratio = df["scenario_sell_price"] / df["sell_price"]
    df["scenario_units_sold"] = (df["units_sold"] * (price_ratio ** elasticity)).round().astype(int)
    df["scenario_units_sold"] = df["scenario_units_sold"].clip(lower=0)

    df["scenario_revenue"] = df["scenario_units_sold"] * df["scenario_sell_price"]

    # keep waste ratio per unit the same for demo
    waste_ratio = np.where(df["units_sold"] > 0, df["waste_qty"] / df["units_sold"], 0.0)
    df["scenario_waste_qty"] = (df["scenario_units_sold"] * waste_ratio).round()
    df["scenario_waste_cost"] = df["scenario_waste_qty"] * df["cost_per_unit"]

    df["scenario_gross_profit"] = df["scenario_units_sold"] * (df["scenario_sell_price"] - df["cost_per_unit"])
    df["scenario_gp_after_waste"] = df["scenario_gross_profit"] - df["scenario_waste_cost"]

    scen_rev = df["scenario_revenue"].sum()
    scen_gp = df["scenario_gp_after_waste"].sum()

    return {
        "label": "",
        "base_revenue": base_rev,
        "base_gp_after_waste": base_gp,
        "scenario_revenue": scen_rev,
        "scenario_gp_after_waste": scen_gp,
        "delta_revenue": scen_rev - base_rev,
        "delta_gp_after_waste": scen_gp - base_gp,
        "detail_df": df,
    }


def apply_cost_inflation_scenario(perf_df: pd.DataFrame,
                                  cost_inflation_pct: float) -> dict:
    df = perf_df.copy()

    base_rev = df["revenue"].sum()
    base_gp = df["gp_after_waste"].sum()

    df["inflated_cost_per_unit"] = df["cost_per_unit"] * (1 + cost_inflation_pct)
    df["inflated_gp_per_unit"] = df["sell_price"] - df["inflated_cost_per_unit"]
    df["inflated_gross_profit"] = df["inflated_gp_per_unit"] * df["units_sold"]
    df["inflated_gp_after_waste"] = df["inflated_gross_profit"] - df["waste_cost"]

    scen_rev = base_rev
    scen_gp = df["inflated_gp_after_waste"].sum()

    return {
        "label": "",
        "base_revenue": base_rev,
        "base_gp_after_waste": base_gp,
        "scenario_revenue": scen_rev,
        "scenario_gp_after_waste": scen_gp,
        "delta_revenue": scen_rev - base_rev,
        "delta_gp_after_waste": scen_gp - base_gp,
        "detail_df": df,
    }


def run_scenarios(perf_df: pd.DataFrame, config: dict) -> dict:
    elasticity = config["price_elasticity_assumption"]
    scenarios = {}

    premium_filter = (
        (perf_df["menu_engineering_class"] == "Star") |
        (perf_df["consultant_tags"].str.contains("Strategic Anchor", na=False))
    )
    scenarios["A_premium_up_8"] = apply_price_change_scenario(
        perf_df,
        price_change_filter=premium_filter,
        price_change_pct=config["scenario_price_increase_premium"],
        elasticity=elasticity,
    )
    scenarios["A_premium_up_8"]["label"] = "Increase price of premium (Stars + Strategic Anchors) by 8%"

    puzzle_filter = perf_df["menu_engineering_class"] == "Puzzle"
    scenarios["B_puzzles_down_10"] = apply_price_change_scenario(
        perf_df,
        price_change_filter=puzzle_filter,
        price_change_pct=config["scenario_price_decrease_puzzles"],
        elasticity=elasticity,
    )
    scenarios["B_puzzles_down_10"]["label"] = "Reduce price of Puzzle items by 10% to stimulate volume"

    # Scenario C: Use keyword matching for mains category (flexible for different datasets)
    mains_keywords = config.get("mains_category_keywords", ["Main"])
    pattern = "|".join([k.strip() for k in mains_keywords if k.strip()])
    if pattern:
        mains_filter = perf_df["category"].astype(str).str.contains(pattern, case=False, na=False)
    else:
        # Fallback if keywords list is empty
        mains_filter = perf_df["category"].astype(str).str.contains("Main", case=False, na=False)
    
    scenarios["C_mains_up_5"] = apply_price_change_scenario(
        perf_df,
        price_change_filter=mains_filter,
        price_change_pct=config["scenario_mains_price_increase"],
        elasticity=elasticity,
    )
    
    # Dynamic label showing what was actually matched
    price_increase_pct = config["scenario_mains_price_increase"] * 100
    keywords_text = ", ".join([f'"{k}"' for k in mains_keywords])
    scenarios["C_mains_up_5"]["label"] = f"Increase prices by {price_increase_pct:.0f}% for mains categories matching: {keywords_text}"

    scenarios["D_cost_inflation_5"] = apply_cost_inflation_scenario(
        perf_df,
        cost_inflation_pct=config["scenario_cost_inflation"],
    )
    scenarios["D_cost_inflation_5"]["label"] = "Simulate 5% ingredient cost inflation (no menu price changes)"

    return scenarios


# NOTE: Module-level call removed. scenarios is now computed inside run_full_analysis_v2()
# scenarios = run_scenarios(perf_df, CONFIG)

# print("Scenario GP-after-waste deltas:")
# for key, scen in scenarios.items():
#     print(f"{key} ({scen['label']}): {CONFIG['currency']}{scen['delta_gp_after_waste']:,.2f}")

# %% [markdown]
# ## 9. Insight Generation (Opportunities & Risks)


# %%
def generate_opportunities(perf_df: pd.DataFrame,
                           cat_summary_df: pd.DataFrame,
                           summary_metrics: dict,
                           booking_summary_df: pd.DataFrame,
                           config: dict) -> list:
    opps = []

    # Underpriced Drivers
    drivers = perf_df[perf_df["consultant_tags"].str.contains("Underpriced Driver", na=False)]
    drivers = drivers.sort_values("gross_profit", ascending=False)
    for _, row in drivers.head(5).iterrows():
        opps.append(
            f"Consider a controlled price increase on **{row['item_name']}** "
            f"({row['category']}). It sells {row['units_sold']} units and "
            f"runs approximately {row['gp_vs_cat_pct_points']:.1f} percentage points "
            f"above the average GP% for {row['category'].lower()}."
        )

    # Puzzles with Premium Opportunity
    puzzles = perf_df[
        (perf_df["menu_engineering_class"] == "Puzzle") &
        (perf_df["consultant_tags"].str.contains("Premium Opportunity", na=False))
    ].sort_values("gross_profit", ascending=False)
    for _, row in puzzles.head(5).iterrows():
        opps.append(
            f"**{row['item_name']}** is a high-margin, low-volume Puzzle. "
            f"Its GP% is {row['gp_pct']*100:.1f}% and it contributes "
            f"{row['margin_contribution_pct']*100:.2f}% of total menu GP. "
            f"Highlighting this dish on the menu and briefing the team to recommend it "
            f"could lift its volume without resorting to discounting."
        )

    # Plowhorses
    plowhorses = perf_df[perf_df["menu_engineering_class"] == "Plowhorse"].sort_values("units_sold", ascending=False)
    for _, row in plowhorses.head(5).iterrows():
        opps.append(
            f"**{row['item_name']}** is a Plowhorse – very popular ({row['units_sold']} units) "
            f"but with modest GP% at {row['gp_pct']*100:.1f}%. Review portion size and trimmings, "
            f"and consider a small price increase or menu re-positioning to protect margin."
        )

    # Category-level opportunities (waste-adjusted GP)
    overall_gp_after_waste = summary_metrics["avg_gp_pct_after_waste"]
    weaker_cats = cat_summary_df[cat_summary_df["gp_pct_after_waste"] < overall_gp_after_waste].sort_values("total_revenue", ascending=False)
    for _, row in weaker_cats.head(3).iterrows():
        opps.append(
            f"The **{row['category']}** category delivers {row['total_revenue'] / summary_metrics['total_revenue'] * 100:.1f}% "
            f"of total revenue but runs below the overall waste-adjusted GP%. Focused pricing and portion work "
            f"in this section could release additional profit."
        )

    # Waste improvement opportunities
    high_waste = perf_df.sort_values("waste_cost", ascending=False).head(5)
    for _, row in high_waste.iterrows():
        if row["waste_cost"] > 0:
            opps.append(
                f"**{row['item_name']}** incurs approximately {config['currency']}{row['waste_cost']:,.2f} in waste cost. "
                f"Tightening prep levels, reviewing holding times, or adjusting menu positioning could reduce this leakage."
            )

    # Booking opportunities (if any data)
    # e.g. building stronger mid-week trade
    # We'll simply note that if Fri/Sat dominate, mid-week can be targeted.
    if not booking_summary_df.empty:
        # identify strongest and weakest day by covers
        max_row = booking_summary_df.loc[booking_summary_df["total_covers"].idxmax()]
        min_row = booking_summary_df.loc[booking_summary_df["total_covers"].idxmin()]
        opps.append(
            f"**{max_row['day_of_week']}** is the strongest day by covers, while **{min_row['day_of_week']}** is the weakest. "
            f"Targeted offers or set menus on quieter days could rebalance demand without discounting peak periods."
        )

    seen = set()
    unique_opps = []
    for o in opps:
        if o not in seen:
            unique_opps.append(o)
            seen.add(o)

    return unique_opps[:10]


def generate_risks(perf_df: pd.DataFrame,
                   cat_summary_df: pd.DataFrame,
                   summary_metrics: dict,
                   booking_summary_df: pd.DataFrame,
                   config: dict) -> list:
    risks = []

    # Overpriced Risks
    over_risks = perf_df[perf_df["consultant_tags"].str.contains("Overpriced Risk", na=False)]
    over_risks = over_risks.sort_values("units_sold", ascending=True)
    for _, row in over_risks.head(5).iterrows():
        risks.append(
            f"**{row['item_name']}** appears overpriced relative to demand. It sells "
            f"only {row['units_sold']} units and its GP% ({row['gp_pct']*100:.1f}%) "
            f"is not materially better than the {row['category']} average."
        )

    # True Dogs (after waste)
    vol_low = perf_df["units_sold"] <= perf_df["units_sold"].quantile(0.3)
    gp_low = perf_df["gp_pct_after_waste"] <= perf_df["gp_pct_after_waste"].quantile(0.4)
    dogs = perf_df[vol_low & gp_low].sort_values("gp_after_waste", ascending=True)
    for _, row in dogs.head(5).iterrows():
        risks.append(
            f"**{row['item_name']}** is a low-volume, low-margin item even after accounting for waste, "
            f"contributing only {row['margin_contribution_pct']*100:.2f}% of total GP. It is a candidate "
            f"for recipe redesign, re-pricing, or removal."
        )

    # Waste-heavy items
    waste_heavy = perf_df[perf_df["consultant_tags"].str.contains("Top Waste Contributor", na=False)].sort_values("waste_cost", ascending=False)
    for _, row in waste_heavy.head(5).iterrows():
        risks.append(
            f"**{row['item_name']}** is among the highest waste contributors, with "
            f"{config['currency']}{row['waste_cost']:,.2f} written off in the period. "
            f"Without intervention, this directly erodes the profit contribution of the dish."
        )

    # Category complexity risk
    total_rev = summary_metrics["total_revenue"]
    small_cats = cat_summary_df[cat_summary_df["total_revenue"] < 0.08 * total_rev]
    for _, row in small_cats.iterrows():
        risks.append(
            f"The **{row['category']}** section accounts for only "
            f"{row['total_revenue'] / total_rev * 100:.1f}% of revenue. If it is also operationally complex "
            f"(prep/ingredients), it may not justify its current footprint on the menu."
        )

    seen = set()
    unique_risks = []
    for r in risks:
        if r not in seen:
            unique_risks.append(r)
            seen.add(r)

    return unique_risks[:10]


# NOTE: Module-level calls removed. opportunities and risks are now computed inside run_full_analysis_v2()
# opportunities = generate_opportunities(perf_df, cat_summary_df, summary_metrics, CONFIG)
# risks = generate_risks(perf_df, cat_summary_df, summary_metrics, CONFIG)

# print("Sample opportunities:")
# for o in opportunities[:3]:
#     print("-", o)

# print("\nSample risks:")
# for r in risks[:3]:
#     print("-", r)

# %% [markdown]
# ## 10. Visual Charts (Demo-Grade)
#
# These are for your own use and for screenshots in your example PDF.


# %%
def plot_category_revenue_gp(cat_summary_df: pd.DataFrame, config: dict):
    if cat_summary_df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(cat_summary_df))
    ax1.bar(x, cat_summary_df["total_revenue"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(cat_summary_df["category"], rotation=45, ha="right")
    ax1.set_ylabel(f"Revenue ({config['currency']})")
    ax1.set_title("Category Revenue")

    plt.tight_layout()
    plt.show()

    fig, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(x, cat_summary_df["gp_pct_after_waste"] * 100)
    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_summary_df["category"], rotation=45, ha="right")
    ax2.set_ylabel("GP% (after waste)")
    ax2.set_title("Category GP% After Waste")

    plt.tight_layout()
    plt.show()


def plot_menu_engineering(perf_df: pd.DataFrame):
    if perf_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    x = perf_df["units_sold"]
    y = perf_df["gp_pct"] * 100
    ax.scatter(x, y)
    ax.set_xlabel("Units Sold")
    ax.set_ylabel("GP%")
    ax.set_title("Menu Engineering – Volume vs GP%")

    # draw median lines
    ax.axvline(perf_df["units_sold"].median(), linestyle="--")
    ax.axhline((perf_df["gp_pct"] * 100).median(), linestyle="--")

    plt.tight_layout()
    plt.show()


def plot_top_waste(perf_df: pd.DataFrame, config: dict, top_n: int = 10):
    top_waste = perf_df.sort_values("waste_cost", ascending=False).head(top_n)
    if top_waste.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_waste["item_name"], top_waste["waste_cost"])
    ax.invert_yaxis()
    ax.set_xlabel(f"Waste Cost ({config['currency']})")
    ax.set_title("Top Waste Items")
    plt.tight_layout()
    plt.show()


def plot_booking_covers(booking_summary_df: pd.DataFrame):
    if booking_summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(booking_summary_df))
    ax.bar(x, booking_summary_df["total_covers"])
    ax.set_xticks(x)
    ax.set_xticklabels(booking_summary_df["day_of_week"], rotation=45, ha="right")
    ax.set_ylabel("Total Covers")
    ax.set_title("Covers by Day of Week")
    plt.tight_layout()
    plt.show()


def plot_scenario_gp_changes(scenarios: dict, config: dict):
    if not scenarios:
        return
    labels = []
    deltas = []
    for key, scen in scenarios.items():
        labels.append(key)
        deltas.append(scen["delta_gp_after_waste"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, deltas)
    ax.set_ylabel(f"GP Change ({config['currency']})")
    ax.set_title("Scenario GP Change (After Waste)")
    plt.tight_layout()
    plt.show()


def save_all_charts(results: dict, output_dir: str, config: dict = CONFIG) -> None:
    """
    Save all analysis charts as PNG files in the output directory.

    Args:
        results: Dictionary returned by run_full_analysis_v2().
        output_dir: Directory where PNG files will be saved.
        config: Configuration dictionary (defaults to module CONFIG).
    """
    os.makedirs(output_dir, exist_ok=True)

    cat_summary_df = results["cat_summary_df"]
    perf_df = results["perf_df"]
    booking_summary_df = results["booking_summary_df"]
    scenarios = results["scenarios"]

    # Chart 1: Category Revenue and GP%
    if not cat_summary_df.empty:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        x = np.arange(len(cat_summary_df))
        
        # Color palette for categories
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
        bar_colors = colors[:len(cat_summary_df)]
        
        # Revenue bars
        bars = ax1.bar(x, cat_summary_df["total_revenue"], color=bar_colors, alpha=0.8, edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_summary_df["category"], rotation=45, ha="right", fontsize=10)
        ax1.set_ylabel(f"Revenue ({config['currency']})", fontsize=11, fontweight='bold')
        ax1.set_title("Category Performance: Revenue & GP%", fontsize=13, fontweight='bold', pad=20)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{config["currency"]}{height:,.0f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add GP% line on secondary axis
        ax2 = ax1.twinx()
        line = ax2.plot(x, cat_summary_df["gp_pct"] * 100, 
                        color='darkred', marker='o', linewidth=2.5, markersize=8, label='GP%')
        ax2.set_ylabel("GP%", fontsize=11, fontweight='bold', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        
        # Add GP% value labels
        for i, (xi, gp) in enumerate(zip(x, cat_summary_df["gp_pct"] * 100)):
            ax2.text(xi, gp + 2, f'{gp:.1f}%', ha='center', fontsize=9, 
                    fontweight='bold', color='darkred')
        
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "category_revenue_and_gp.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    # Chart 2: Menu Engineering (Volume vs GP%)
    if not perf_df.empty:
        fig, ax = plt.subplots(figsize=(12, 9))
        x = perf_df["units_sold"]
        y = perf_df["gp_pct"] * 100
        
        # Color by quadrant with refined palette
        colors = []
        for _, row in perf_df.iterrows():
            quad = row.get("menu_engineering_class", "")
            if quad == "Star":
                colors.append("#FFD700")  # Gold
            elif quad == "Plowhorse":
                colors.append("#2E7D32")  # Dark green
            elif quad == "Puzzle":
                colors.append("#F57C00")  # Dark orange
            elif quad == "Dog":
                colors.append("#C62828")  # Dark red
            else:
                colors.append("#757575")  # Gray
        
        # Scatter plot with refined styling
        ax.scatter(x, y, c=colors, s=120, alpha=0.7, edgecolors="white", linewidths=1.5, zorder=3)
        
        # Add item labels using adjustText for automatic overlap prevention
        texts = []
        for idx, row in perf_df.iterrows():
            item_name = row["item_name"]
            # Smart truncation - preserve readability
            if len(item_name) > 22:
                item_name = item_name[:19] + "..."
            
            txt = ax.text(
                row["units_sold"], 
                row["gp_pct"] * 100,
                item_name,
                fontsize=8,
                fontweight="500",
                alpha=0.9,
                zorder=4
            )
            texts.append(txt)
        
        # Apply adjustText to prevent overlaps (if available)
        if ADJUST_TEXT_AVAILABLE and len(texts) > 0:
            adjust_text(
                texts,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.6),
                expand_points=(1.2, 1.3),
                expand_text=(1.1, 1.2),
                force_points=(0.3, 0.5),
                force_text=(0.3, 0.5),
                ax=ax
            )
        
        # Refined axis styling
        ax.set_xlabel("Units Sold (Volume)", fontsize=12, fontweight="600")
        ax.set_ylabel("Gross Profit %", fontsize=12, fontweight="600")
        ax.set_title("Menu Engineering Matrix – Volume vs Profitability", 
                    fontsize=14, fontweight="bold", pad=20)
        
        # Quadrant lines with refined styling
        x_med = perf_df["units_sold"].median()
        y_med = (perf_df["gp_pct"] * 100).median()
        ax.axvline(x_med, linestyle="--", color="#424242", linewidth=1.5, alpha=0.6, zorder=1)
        ax.axhline(y_med, linestyle="--", color="#424242", linewidth=1.5, alpha=0.6, zorder=1)
        
        # Add quadrant labels with refined positioning
        x_max, y_max = x.max(), y.max()
        x_min, y_min = x.min(), y.min()
        ax.text(x_med + (x_max - x_med) * 0.5, y_med + (y_max - y_med) * 0.9, 
                "STARS", fontsize=11, fontweight="bold", ha="center", 
                color="#B8860B", alpha=0.8, zorder=2)
        ax.text(x_med - (x_med - x_min) * 0.5, y_med + (y_max - y_med) * 0.9, 
                "PUZZLES", fontsize=11, fontweight="bold", ha="center", 
                color="#D84315", alpha=0.8, zorder=2)
        ax.text(x_med + (x_max - x_med) * 0.5, y_med - (y_med - y_min) * 0.9, 
                "PLOWHORSES", fontsize=11, fontweight="bold", ha="center", 
                color="#1B5E20", alpha=0.8, zorder=2)
        ax.text(x_med - (x_med - x_min) * 0.5, y_med - (y_med - y_min) * 0.9, 
                "DOGS", fontsize=11, fontweight="bold", ha="center", 
                color="#B71C1C", alpha=0.8, zorder=2)
        
        # Clean grid for professional look
        ax.grid(True, alpha=0.2, linestyle="-", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "menu_engineering.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    # Chart 3: Top Waste Items
    if not perf_df.empty:
        top_waste = perf_df.sort_values("waste_cost", ascending=False).head(10)
        if not top_waste.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Color gradient from dark red (highest) to light coral (lowest)
            colors = plt.cm.Reds(np.linspace(0.8, 0.4, len(top_waste)))
            
            bars = ax.barh(top_waste["item_name"], top_waste["waste_cost"], 
                          color=colors, edgecolor='darkred', linewidth=0.8, alpha=0.85)
            ax.invert_yaxis()
            
            # Add value labels INSIDE bars (right-aligned) to prevent overlap
            for i, (idx, row) in enumerate(top_waste.iterrows()):
                waste_val = row["waste_cost"]
                # Place label inside bar at 95% of bar width
                label_x = waste_val * 0.97
                ax.text(label_x, i, f'{config["currency"]}{waste_val:,.0f} ',
                       va='center', ha='right', fontsize=10, 
                       fontweight='bold', color='white',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='darkred', alpha=0.7, edgecolor='none'))
            
            ax.set_xlabel(f"Annual Waste Cost ({config['currency']})", fontsize=12, fontweight='bold')
            ax.set_title("Top 10 Waste Items by Cost", fontsize=14, fontweight='bold', pad=20)
            ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
            ax.set_axisbelow(True)
            
            # Add total waste annotation (positioned bottom-right to avoid overlap)
            total_waste = top_waste["waste_cost"].sum()
            overall_waste = perf_df["waste_cost"].sum()
            pct_of_total = (total_waste / overall_waste * 100) if overall_waste > 0 else 0
            
            ax.text(0.98, 0.03, 
                   f'Top 10: {config["currency"]}{total_waste:,.0f}\n({pct_of_total:.1f}% of total waste)',
                   transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                   fontweight='600',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4B5', 
                            alpha=0.9, edgecolor='darkred', linewidth=1.5))
            
            # Add margin to prevent cutoff
            ax.margins(x=0.15)
            
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "top_waste_items.png"),
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

    # Chart 4: Booking Covers by Day
    if not booking_summary_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(booking_summary_df))
        
        # Color weekends differently (Friday, Saturday, Sunday)
        weekend_days = ['Friday', 'Saturday', 'Sunday']
        colors = ['#FF6B6B' if day in weekend_days else '#4ECDC4' 
                 for day in booking_summary_df["day_of_week"]]
        
        bars = ax.bar(x, booking_summary_df["total_covers"], color=colors, 
                     edgecolor='black', linewidth=1, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(booking_summary_df["day_of_week"], rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Total Covers", fontsize=11, fontweight='bold')
        ax.set_title("Booking Covers by Day of Week", fontsize=13, fontweight='bold', pad=15)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', edgecolor='black', label='Weekday'),
            Patch(facecolor='#FF6B6B', edgecolor='black', label='Weekend')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "booking_covers_by_day.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()

    # Chart 5: Scenario GP Changes
    if scenarios:
        labels = []
        deltas = []
        for key, scen in scenarios.items():
            labels.append(key)
            deltas.append(scen["delta_gp_after_waste"])

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Color bars: green for positive, red for negative
        colors = ['#28a745' if d >= 0 else '#dc3545' for d in deltas]
        
        bars = ax.bar(labels, deltas, color=colors, edgecolor='black', linewidth=1, alpha=0.8)
        
        # Add value labels on bars
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            label_y = height if height >= 0 else height
            va = 'bottom' if height >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{config["currency"]}{delta:,.0f}',
                   ha='center', va=va, fontsize=9, fontweight='bold')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel(f"GP Change ({config['currency']})", fontsize=11, fontweight='bold')
        ax.set_title("Scenario Analysis: GP Impact (After Waste)", fontsize=13, fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Rotate x-labels if needed
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "scenario_gp_changes.png"),
            bbox_inches="tight",
            dpi=150,
        )
        plt.close()


def export_results_to_excel(results: dict, path: str) -> None:
    """
    Write key result tables into a multi-sheet Excel file.

    Args:
        results: Dictionary returned by run_full_analysis_v2().
        path: File path where the Excel workbook will be saved.
    """
    with pd.ExcelWriter(path) as writer:
        # Helper to safely write DataFrames when present and non-empty
        def _write_if_df(key: str, sheet_name: str):
            df = results.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        _write_if_df("perf_df", "Menu_Performance")
        _write_if_df("cat_summary_df", "Category_Summary")
        _write_if_df("staff_perf_df", "Staff_Performance")
        _write_if_df("booking_summary_df", "Booking_Summary")
        _write_if_df("waste_df", "Waste")

        # Write Data Quality diagnostics (if present)
        dq = results.get("data_quality_diagnostics")
        dq_notes = results.get("data_quality_notes")
        if dq and isinstance(dq, dict):
            # Convert diagnostics dict to DataFrame for a sheet
            try:
                dq_df = pd.DataFrame(list(dq.items()), columns=["metric", "value"])
                dq_df.to_excel(writer, sheet_name="Data_Quality", index=False)
            except Exception:
                # intentionally silent: file write should not raise here
                pass
        if dq_notes and isinstance(dq_notes, (list, tuple)):
            try:
                notes_df = pd.DataFrame({"note": list(dq_notes)})
                notes_df.to_excel(writer, sheet_name="Data_Quality_Notes", index=False)
            except Exception:
                pass

        # Optionally write time-based summaries
        time_summary = results.get("time_summary")
        if time_summary and isinstance(time_summary, dict):
            monthly = time_summary.get("monthly")
            dow = time_summary.get("dow")
            dayparts = time_summary.get("dayparts")
            if isinstance(monthly, pd.DataFrame) and not monthly.empty:
                try:
                    monthly.to_excel(writer, sheet_name="Time_Monthly", index=False)
                except Exception:
                    pass
            if isinstance(dow, pd.DataFrame) and not dow.empty:
                try:
                    dow.to_excel(writer, sheet_name="Time_DayOfWeek", index=False)
                except Exception:
                    pass
            if isinstance(dayparts, pd.DataFrame) and not dayparts.empty:
                try:
                    dayparts.to_excel(writer, sheet_name="Time_Dayparts", index=False)
                except Exception:
                    pass


# Example: you can run these when you want screenshots
# plot_category_revenue_gp(cat_summary_df, CONFIG)
# plot_menu_engineering(perf_df)
# plot_top_waste(perf_df, CONFIG)
# plot_booking_covers(booking_summary_df)
# plot_scenario_gp_changes(scenarios, CONFIG)

# %% [markdown]
# ## 11. ChatGPT Export Block (The Big One)
#
# Copy this block into ChatGPT with your consultant prompt to generate the narrative report.


# %%
def to_markdown_table(df: pd.DataFrame, cols: list, index: bool = False) -> str:
    if df.empty:
        return pd.DataFrame(columns=cols).to_markdown(index=index)
    return df[cols].to_markdown(index=index)


def build_action_plan(opportunities: list, risks: list, scenarios: dict, config: dict = CONFIG) -> str:
    """
    Build a concise action plan text block from the top opportunities, risks and scenarios.
    Returns a multi-line string suitable to embed in the ChatGPT export block.
    """
    currency = config["currency"]
    lines = []
    
    # Get top 5 opportunities
    top_opps = opportunities[:5] if opportunities else []
    if top_opps:
        lines.append("- Protect and enhance key profit drivers:")
        for opp in top_opps:
            lines.append(f"  • {opp}")
    
    # Get top 3 risks
    top_risks = risks[:3] if risks else []
    if top_risks:
        lines.append("- Fix known margin leaks:")
        for risk in top_risks:
            lines.append(f"  • {risk}")
    
    # Include scenario levers
    if scenarios:
        lines.append("- Scenario levers to test:")
        for key, scen in scenarios.items():
            delta = scen["delta_gp_after_waste"]
            delta_str = f"{currency}{delta:,.2f}" if delta >= 0 else f"-{currency}{abs(delta):,.2f}"
            lines.append(f"  • {key} – GP change: {delta_str}")
    
    # If no content, return fallback
    if not lines:
        return "(No specific action items detected at this time.)"
    
    return "\n".join(lines)


def build_data_quality_report(menu_df: pd.DataFrame,
                              orders_df: pd.DataFrame,
                              waste_df: pd.DataFrame,
                              staff_df: pd.DataFrame,
                              bookings_df: pd.DataFrame) -> tuple[dict, list]:
    """
    Analyse data coverage and basic quality issues.

    Returns:
        diagnostics_dict: a structured dict of metrics
        notes: a list of human-readable bullet-point strings describing key data quality points
    """
    diagnostics = {}
    notes = []

    # Basic counts
    diagnostics["orders_row_count"] = int(len(orders_df)) if orders_df is not None else 0
    diagnostics["menu_item_count"] = int(len(menu_df)) if menu_df is not None else 0
    diagnostics["waste_row_count"] = int(len(waste_df)) if waste_df is not None else 0
    diagnostics["staff_row_count"] = int(len(staff_df)) if staff_df is not None else 0
    diagnostics["bookings_row_count"] = int(len(bookings_df)) if bookings_df is not None else 0

    # Dates coverage
    date_min = None
    date_max = None
    num_days = 0
    has_gaps = False
    try:
        if (orders_df is not None) and ("order_datetime" in orders_df.columns) and not orders_df["order_datetime"].dropna().empty:
            od = pd.to_datetime(orders_df["order_datetime"], errors="coerce").dropna()
            if not od.empty:
                date_min = od.min()
                date_max = od.max()
                num_days = int(od.dt.date.nunique())
                # Build full date range and compare
                full_days = pd.date_range(start=date_min.floor("D"), end=date_max.ceil("D"), freq="D").date
                has_gaps = len(full_days) != num_days
    except Exception:
        date_min = None
        date_max = None
        num_days = 0
        has_gaps = False

    diagnostics["date_min"] = str(date_min) if date_min is not None else None
    diagnostics["date_max"] = str(date_max) if date_max is not None else None
    diagnostics["num_days_covered"] = int(num_days)
    diagnostics["has_gaps_in_days"] = bool(has_gaps)

    # Price and qty checks
    missing_price_count = 0
    zero_or_negative_price_count = 0
    suspicious_high_price_count = 0
    if (menu_df is not None) and (not menu_df.empty):
        if "sell_price" in menu_df.columns and "cost_per_unit" in menu_df.columns:
            missing_price_count = int(menu_df[menu_df[["sell_price", "cost_per_unit"]].isnull().any(axis=1)].shape[0])
            zero_or_negative_price_count = int(menu_df[menu_df["sell_price"] <= 0].shape[0])
            try:
                p99 = float(menu_df[menu_df["sell_price"] > 0]["sell_price"].quantile(0.99))
                threshold = p99 * 1.5
                suspicious_high_price_count = int(menu_df[menu_df["sell_price"] > threshold].shape[0])
            except Exception:
                suspicious_high_price_count = 0
        else:
            # If expected columns missing, mark as missing
            missing_price_count = int(len(menu_df))

    zero_or_negative_qty_count = 0
    if (orders_df is not None) and (not orders_df.empty):
        if "qty" in orders_df.columns:
            zero_or_negative_qty_count = int(orders_df[orders_df["qty"] <= 0].shape[0])
        elif "quantity" in orders_df.columns:
            zero_or_negative_qty_count = int(orders_df[orders_df["quantity"] <= 0].shape[0])

    diagnostics["missing_price_count"] = int(missing_price_count)
    diagnostics["zero_or_negative_price_count"] = int(zero_or_negative_price_count)
    diagnostics["zero_or_negative_qty_count"] = int(zero_or_negative_qty_count)
    diagnostics["suspicious_high_price_count"] = int(suspicious_high_price_count)

    # Notes (human readable)
    if diagnostics["orders_row_count"] == 0:
        notes.append("No sales rows were found; menu analysis is based only on the menu file.")
    else:
        notes.append(f"Sales rows: {diagnostics['orders_row_count']}; data covers {diagnostics['num_days_covered']} distinct days.")

    if diagnostics["has_gaps_in_days"]:
        notes.append("Sales data has gaps in the date range; trend analysis may be incomplete.")

    if diagnostics["missing_price_count"] > 0:
        notes.append("Some menu items are missing price or cost; these rows were excluded from GP calculations.")

    if diagnostics["zero_or_negative_price_count"] > 0:
        notes.append("There are items with zero or negative sell_price values; please check the menu export.")

    if diagnostics["zero_or_negative_qty_count"] > 0:
        notes.append("There are orders with zero or negative quantities; these were ignored in volume metrics.")

    if diagnostics["suspicious_high_price_count"] > 0:
        notes.append("Some items have unusually high prices compared to the rest of the menu; this may indicate data entry issues.")

    # Positive note if no issues
    if (diagnostics["orders_row_count"] > 0 and diagnostics["missing_price_count"] == 0 and
            diagnostics["zero_or_negative_price_count"] == 0 and diagnostics["zero_or_negative_qty_count"] == 0 and
            diagnostics["suspicious_high_price_count"] == 0 and not diagnostics["has_gaps_in_days"]):
        notes.append("Sales and menu data appear complete and consistent for the analysed period.")

    return diagnostics, notes


def build_time_performance(orders_df: pd.DataFrame,
                           perf_df: pd.DataFrame,
                           config: dict) -> tuple[dict, list]:
    """
    Build time-based performance summaries with safe fallbacks.

    Returns:
        time_summary: dict with keys 'has_datetime','has_date_only','monthly','dow','dayparts'
        time_notes: list of human-readable strings
    """
    time_summary = {
        "has_datetime": False,
        "has_date_only": False,
        "monthly": pd.DataFrame(),
        "dow": pd.DataFrame(),
        "dayparts": pd.DataFrame(),
    }
    notes = []

    if orders_df is None or orders_df.empty or "order_datetime" not in orders_df.columns:
        notes.append("No timestamp column found; time-based analysis skipped.")
        return time_summary, notes

    # Parse datetimes safely
    od = pd.to_datetime(orders_df["order_datetime"], errors="coerce")
    if od.dropna().empty:
        notes.append("Timestamp column exists but could not be parsed; skipping time analysis.")
        return time_summary, notes

    orders = orders_df.copy()
    orders["_order_dt"] = od
    time_summary["has_datetime"] = True

    # Determine if time-of-day information exists
    hours = orders["_order_dt"].dt.hour.fillna(0)
    minutes = orders["_order_dt"].dt.minute.fillna(0)
    if (hours.eq(0) & minutes.eq(0)).all():
        time_summary["has_date_only"] = True
        notes.append("Only date available (no time-of-day); daypart analysis skipped.")
    else:
        time_summary["has_date_only"] = False

    # Safe qty column
    qty_col = "qty" if "qty" in orders.columns else ("quantity" if "quantity" in orders.columns else None)
    if qty_col is None:
        orders["_qty"] = 1
        notes.append("No quantity column found; assuming qty=1 for each order line.")
    else:
        orders["_qty"] = pd.to_numeric(orders[qty_col], errors="coerce").fillna(0)

    # Merge pricing and per-unit GP info from perf_df
    perf_cols = [c for c in ["item_id", "sell_price", "gp_per_unit", "units_sold", "waste_cost"] if c in perf_df.columns]
    perf_small = perf_df[perf_cols].copy() if perf_cols else pd.DataFrame()
    merged = orders.merge(perf_small, on="item_id", how="left") if not perf_small.empty else orders.copy()

    # Fill missing pricing/gp with zeros
    merged["sell_price"] = pd.to_numeric(merged.get("sell_price", 0), errors="coerce").fillna(0)
    merged["gp_per_unit"] = pd.to_numeric(merged.get("gp_per_unit", 0), errors="coerce").fillna(0)
    merged["units_sold"] = pd.to_numeric(merged.get("units_sold", 0), errors="coerce").fillna(0)
    merged["waste_cost"] = pd.to_numeric(merged.get("waste_cost", 0), errors="coerce").fillna(0)

    # Compute per-unit waste cost where possible
    merged["per_unit_waste_cost"] = 0.0
    try:
        with pd.option_context("mode.use_inf_as_na", True):
            mask = merged["units_sold"] > 0
            merged.loc[mask, "per_unit_waste_cost"] = merged.loc[mask, "waste_cost"] / merged.loc[mask, "units_sold"]
    except Exception:
        merged["per_unit_waste_cost"] = 0.0

    merged["per_unit_gp_after_waste"] = merged["gp_per_unit"] - merged["per_unit_waste_cost"]

    # Line-level metrics
    merged["line_revenue"] = merged["_qty"] * merged["sell_price"]
    merged["line_gp_after_waste"] = merged["_qty"] * merged["per_unit_gp_after_waste"]

    # Transaction id fallback
    txn_col = None
    for c in ["order_line_id", "order_id"]:
        if c in merged.columns:
            txn_col = c
            break
    if txn_col is None:
        merged["_txn_id"] = range(len(merged))
        txn_col = "_txn_id"

    # Monthly summary
    try:
        merged["year_month"] = merged["_order_dt"].dt.to_period("M").astype(str)
        monthly = merged.groupby("year_month").agg(
            revenue=("line_revenue", "sum"),
            gp_after_waste=("line_gp_after_waste", "sum"),
            transactions=(txn_col, "nunique"),
        ).reset_index()
    except Exception:
        monthly = pd.DataFrame()

    # Day-of-week summary
    try:
        merged["day_of_week"] = merged["_order_dt"].dt.day_name()
        dow = merged.groupby("day_of_week").agg(
            revenue=("line_revenue", "sum"),
            gp_after_waste=("line_gp_after_waste", "sum"),
            transactions=(txn_col, "nunique"),
        ).reset_index()
        # Ensure Monday->Sunday order
        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow["day_of_week"] = pd.Categorical(dow["day_of_week"], categories=dow_order, ordered=True)
        dow = dow.sort_values("day_of_week").reset_index(drop=True)
    except Exception:
        dow = pd.DataFrame()

    # Daypart summary
    dayparts_df = pd.DataFrame()
    if not time_summary["has_date_only"]:
        def assign_daypart(h: int) -> str:
            if h < 5:
                return "Overnight (00–05)"
            if h < 11:
                return "Breakfast (05–11)"
            if h < 15:
                return "Lunch (11–15)"
            if h < 17:
                return "Afternoon (15–17)"
            if h < 22:
                return "Dinner (17–22)"
            return "Late Night (22–24)"

        try:
            merged["hour"] = merged["_order_dt"].dt.hour.fillna(0).astype(int)
            merged["daypart"] = merged["hour"].apply(assign_daypart)
            dayparts_df = merged.groupby("daypart").agg(
                revenue=("line_revenue", "sum"),
                gp_after_waste=("line_gp_after_waste", "sum"),
                transactions=(txn_col, "nunique"),
            ).reset_index()
        except Exception:
            dayparts_df = pd.DataFrame()

    # Populate summary
    time_summary["monthly"] = monthly
    time_summary["dow"] = dow
    time_summary["dayparts"] = dayparts_df

    # Notes: simple consultant-style insights
    try:
        if not dow.empty:
            top = dow.loc[dow["revenue"].idxmax()]
            bot = dow.loc[dow["revenue"].idxmin()]
            notes.append(f"Strongest day by revenue: {top['day_of_week']} ({top['revenue']:.0f}). Weakest: {bot['day_of_week']} ({bot['revenue']:.0f}).")
            fri_sat = dow[dow["day_of_week"].isin(["Friday", "Saturday"])]["revenue"].sum()
            total_rev = dow["revenue"].sum() if dow["revenue"].sum() > 0 else 1
            share = fri_sat / total_rev
            notes.append(f"Friday+Saturday revenue share: {share:.0%} of total weekly revenue.")
        if not dayparts_df.empty:
            top = dayparts_df.loc[dayparts_df["revenue"].idxmax()]
            bot = dayparts_df.loc[dayparts_df["revenue"].idxmin()]
            notes.append(f"Strongest daypart: {top['daypart']} ({top['revenue']:.0f}). Weakest: {bot['daypart']} ({bot['revenue']:.0f}).")
    except Exception:
        pass

    return time_summary, notes


# =============================================================================
# HELPER FUNCTIONS FOR TABLE FORMATTING (Phase 1 Integration)
# =============================================================================

def format_category_performance_table(cat_summary_df: pd.DataFrame, config: dict) -> str:
    """Format category performance as markdown table."""
    if cat_summary_df.empty:
        return "No category data available."
    
    cat_copy = cat_summary_df.copy()
    cat_copy["gp_pct"] = cat_copy["gp_pct"] * 100
    cat_copy["gp_pct_after_waste"] = cat_copy["gp_pct_after_waste"] * 100
    
    return to_markdown_table(
        cat_copy,
        ["category", "units_sold", "total_revenue", "total_gp",
         "total_gp_after_waste", "total_waste", "gp_pct", "gp_pct_after_waste"],
        index=False,
    )

def format_top_margin_items_table(perf_df: pd.DataFrame, config: dict) -> str:
    """Format top margin items as markdown table."""
    if perf_df.empty:
        return "No performance data available."
    
    top_items = perf_df.head(10)
    cols = ["item_name", "category", "units_sold", "revenue",
            "gross_profit", "gp_after_waste", "waste_cost",
            "gp_pct", "gp_pct_after_waste", "menu_engineering_class"]
    
    return to_markdown_table(top_items, cols, index=False)

def format_menu_class_table(perf_df: pd.DataFrame, menu_class: str, config: dict) -> str:
    """Format menu engineering class items as markdown table."""
    if perf_df.empty:
        return f"No {menu_class} items found."
    
    items = perf_df[perf_df["menu_engineering_class"] == menu_class].head(10)
    if items.empty:
        return f"No {menu_class} items found."
    
    cols = ["item_name", "category", "units_sold", "revenue",
            "gross_profit", "gp_after_waste", "waste_cost",
            "gp_pct", "gp_pct_after_waste"]
    
    return to_markdown_table(items, cols, index=False)

def format_top_waste_items_table(perf_df: pd.DataFrame, config: dict) -> str:
    """Format top waste items as markdown table."""
    if perf_df.empty:
        return "No waste data available."
    
    top_waste = perf_df.sort_values("waste_cost", ascending=False).head(10)
    cols = ["item_name", "category", "units_sold", "waste_cost",
            "gp_after_waste", "gp_pct_after_waste"]
    
    return to_markdown_table(top_waste, cols, index=False)

def format_staff_performance_table(staff_perf_df: pd.DataFrame, config: dict) -> str:
    """Format staff performance as markdown table."""
    if staff_perf_df.empty:
        return "No staff data available."
    
    return to_markdown_table(
        staff_perf_df.head(20),
        ["staff_name", "role", "orders_handled", "units_sold", "revenue", "gp", "revenue_per_hour", "gp_per_hour"],
        index=False,
    )

def format_booking_summary_table(booking_summary_df: pd.DataFrame, config: dict) -> str:
    """Format booking summary as markdown table."""
    if booking_summary_df.empty:
        return "No booking data available."
    
    return to_markdown_table(
        booking_summary_df,
        ["day_of_week", "booking_count", "total_covers", "avg_covers_per_booking", "no_show_rate_pct"],
        index=False,
    )

def format_scenario_summaries(scenarios: dict, config: dict) -> str:
    """Format scenario analysis as readable text block."""
    currency = config.get("currency", "£")
    lines = []
    
    for scenario_name, scenario_data in scenarios.items():
        lines.append(f"\n{scenario_name}:")
        lines.append(f"  {scenario_data.get('label', '')}")
        # Use correct key names from apply_price_change_scenario and apply_cost_inflation_scenario
        gp_change = scenario_data.get('delta_gp_after_waste', 0)
        revenue_change = scenario_data.get('delta_revenue', 0)
        lines.append(f"  GP Impact: {currency}{gp_change:,.0f}")
        lines.append(f"  Revenue Impact: {currency}{revenue_change:,.0f}")
    
    return "\n".join(lines) if lines else "No scenario data available."


# =============================================================================
# GPT EXPORT BLOCK V1 (Legacy - Keep for Backward Compatibility)
# =============================================================================

def build_gpt_export_block(perf_df: pd.DataFrame,
                           summary_metrics: dict,
                           cat_summary_df: pd.DataFrame,
                           scenarios: dict,
                           opportunities: list,
                           risks: list,
                           staff_perf_df: pd.DataFrame,
                           booking_summary_df: pd.DataFrame,
                           config: dict,
                           data_quality_notes: list | None = None) -> str:
    currency = config["currency"]
    if data_quality_notes is None:
        data_quality_notes = []
    currency = config["currency"]
    if data_quality_notes is None:
        data_quality_notes = []

    # Category summary
    cat_summary_for_md = cat_summary_df.copy()
    cat_summary_for_md["gp_pct"] = cat_summary_for_md["gp_pct"] * 100
    cat_summary_for_md["gp_pct_after_waste"] = cat_summary_for_md["gp_pct_after_waste"] * 100
    cat_summary_md = to_markdown_table(
        cat_summary_for_md,
        ["category", "units_sold", "total_revenue", "total_gp",
         "total_gp_after_waste", "total_waste", "gp_pct", "gp_pct_after_waste"],
        index=False,
    )

    # Item-level tables
    top_margin_items = perf_df.head(10)
    stars = perf_df[perf_df["menu_engineering_class"] == "Star"].head(10)
    puzzles = perf_df[perf_df["menu_engineering_class"] == "Puzzle"].head(10)
    dogs = perf_df[perf_df["menu_engineering_class"] == "Dog"].head(10)
    top_waste = perf_df.sort_values("waste_cost", ascending=False).head(10)

    cols_core = ["item_name", "category", "units_sold", "revenue",
                 "gross_profit", "gp_after_waste", "waste_cost",
                 "gp_pct", "gp_pct_after_waste", "menu_engineering_class"]

    top_margin_md = to_markdown_table(top_margin_items, cols_core, index=False)
    stars_md = to_markdown_table(stars, cols_core, index=False)
    puzzles_md = to_markdown_table(puzzles, cols_core, index=False)
    dogs_md = to_markdown_table(dogs, cols_core, index=False)
    waste_md = to_markdown_table(top_waste, cols_core, index=False)

    # Staff performance
    staff_cols = [
        "staff_name", "role", "orders_handled", "units_sold",
        "revenue", "gp", "est_hours_worked",
        "revenue_per_hour", "gp_per_hour"
    ]
    staff_md = to_markdown_table(staff_perf_df, staff_cols, index=False)

    # Booking summary
    booking_cols = [
        "day_of_week", "booking_count", "total_covers",
        "avg_covers_per_booking", "no_show_rate_pct"
    ]
    booking_md = to_markdown_table(booking_summary_df, booking_cols, index=False)

    # Scenario summaries
    scenario_lines = []
    for key, scen in scenarios.items():
        scenario_lines.append(
            f"- {key}: {scen['label']}\n"
            f"  - Base GP (after waste): {currency}{scen['base_gp_after_waste']:,.2f}\n"
            f"  - Scenario GP (after waste): {currency}{scen['scenario_gp_after_waste']:,.2f}\n"
            f"  - Change in GP: {currency}{scen['delta_gp_after_waste']:,.2f}"
        )
    scenario_summary_text = "\n".join(scenario_lines)

    opp_text = "\n".join([f"- {o}" for o in opportunities]) or "- (None detected)"
    risk_text = "\n".join([f"- {r}" for r in risks]) or "- (None detected)"
    action_plan_text = build_action_plan(opportunities, risks, scenarios, config)

    block = f"""
RESTAURANT_NAME: {config['restaurant_name']}
PERIOD_ANALYSED: {config['period_label']}

TOPLINE_METRICS:
- Total Revenue: {currency}{summary_metrics['total_revenue']:,.2f}
- Total Gross Profit (before waste): {currency}{summary_metrics['total_gp']:,.2f}
- Total Gross Profit (after waste): {currency}{summary_metrics['total_gp_after_waste']:,.2f}
- Average GP% (before waste): {summary_metrics['avg_gp_pct']*100:.2f}%
- Average GP% (after waste): {summary_metrics['avg_gp_pct_after_waste']*100:.2f}%
- Total Recorded Waste Cost: {currency}{summary_metrics['total_waste_cost']:,.2f}
- Number of Active Menu Items: {len(perf_df)}

DATA_QUALITY_SUMMARY:
{('\n'.join([f"- {n}" for n in data_quality_notes]) if data_quality_notes else '- No major data quality issues detected.')}

CATEGORY_PERFORMANCE_TABLE (markdown):
{cat_summary_md}

TOP_MARGIN_ITEMS_TABLE (markdown):
{top_margin_md}

MENU_STARS_TABLE (markdown):
{stars_md}

MENU_PUZZLES_TABLE (markdown):
{puzzles_md}

MENU_DOGS_TABLE (markdown):
{dogs_md}

TOP_WASTE_ITEMS_TABLE (markdown):
{waste_md}

STAFF_PERFORMANCE_TABLE (markdown):
{staff_md}

BOOKING_SUMMARY_BY_DAY_TABLE (markdown):
{booking_md}

SCENARIO_SUMMARIES:
{scenario_summary_text}

KEY_OPPORTUNITIES:
{opp_text}

KEY_RISKS:
{risk_text}

ACTION_PLAN_SUMMARY:
{action_plan_text}
"""
    return block.strip()


# NOTE: Module-level calls removed. These are now computed inside run_full_analysis_v2()
# data_quality_diagnostics, data_quality_notes = build_data_quality_report(
#     menu_df, orders_df, waste_df, staff_df, bookings_df
# )

# gpt_export_block = build_gpt_export_block(
#     perf_df=perf_df,
#     summary_metrics=summary_metrics,
#     cat_summary_df=cat_summary_df,
#     scenarios=scenarios,
#     opportunities=opportunities,
#     risks=risks,
#     staff_perf_df=staff_perf_df,
#     booking_summary_df=booking_summary_df,
#     config=CONFIG,
#     data_quality_notes=data_quality_notes,
# )

# print(gpt_export_block)  # preview first part

# %% [markdown]
# ## 12. How to Use for Your Example PDF
#
# 1. Leave `DATA_SOURCE = "synthetic"` for now.
# 2. Run all cells top to bottom.
# 3. Optionally run the chart functions and screenshot them.
# 4. Scroll to the output of `gpt_export_block` and copy the **entire** text.
# 5. In ChatGPT, paste your consultant prompt, e.g.:
#
#    > "You are a senior restaurant revenue and menu engineering consultant...
#      Turn the following structured analysis into a full report..."
#
#    Then paste the block underneath.
#
# 6. Lightly edit the generated report → export as PDF → this is your **demo report**.
#
# Later, when you have a real client:
# - Set `DATA_SOURCE = "client"`.
# - Fill in `CONFIG["client_menu_path"]`, `CONFIG["client_sales_path"]`, `CONFIG["client_waste_path"]` (if they have waste data).
# - Re-run and repeat.


def run_full_analysis_v2(config: dict = CONFIG, data_source: str = "synthetic") -> dict:
    """
    Run the full Restaurant Intelligence v2 analysis and return all data objects.

    This function re-uses the existing generators, loaders and analysis functions
    defined in this file. It does not print or display anything — it only
    returns a dictionary of results so the module can be imported and reused.

    Args:
        config: Configuration dictionary (defaults to module `CONFIG`).
        data_source: Either "synthetic" or "client" to select data loading mode.

    Returns:
        A dictionary with keys: menu_df, orders_df, waste_df, staff_df, bookings_df,
        perf_df, summary_metrics, cat_summary_df, staff_perf_df, booking_summary_df,
        scenarios, opportunities, risks, gpt_export_block
    """
    # Load or generate data
    if data_source == "synthetic":
        menu_df, orders_df, waste_df, staff_df, bookings_df = generate_synthetic_restaurant(config)
    else:
        # client mode: load menu + sales, then waste; placeholders for staff/bookings
        menu_df, orders_df = load_client_menu_and_sales(config)
        waste_df = load_client_waste(config, menu_df)
        staff_df = pd.DataFrame(columns=["staff_id", "staff_name", "role", "hourly_rate"])
        bookings_df = pd.DataFrame(columns=["booking_id", "booking_datetime", "covers", "source", "status"])
    
    # Validate data (especially important for client data)
    validation_result = validate_client_data(menu_df, orders_df, waste_df, config)
    
    # Print validation summary
    print("\n" + "="*60)
    print("DATA VALIDATION RESULTS")
    print("="*60)
    
    if not validation_result["valid"]:
        print("\n❌ CRITICAL ERRORS FOUND - Analysis cannot proceed:\n")
        for error in validation_result["errors"]:
            print(f"  • {error}")
        print("\nPlease fix these errors and try again.")
        print("="*60 + "\n")
        return {
            "validation": validation_result,
            "error": "Data validation failed - see validation results above"
        }
    
    # Show warnings even if valid
    if validation_result["warnings"]:
        print("\n⚠️  WARNINGS (analysis will proceed but quality may be affected):\n")
        for warning in validation_result["warnings"]:
            print(f"  • {warning}")
    
    # Show summary
    if validation_result["summary"]:
        print("\n📊 DATA SUMMARY:")
        for key, value in validation_result["summary"].items():
            print(f"  • {key}: {value}")
    
    print("\n✅ Validation passed - proceeding with analysis")
    print("="*60 + "\n")

    # Core analyses
    perf_df, summary_metrics, cat_summary_df = build_menu_performance(menu_df, orders_df, waste_df, config)
    staff_perf_df = build_staff_performance(orders_df, staff_df, perf_df)
    booking_summary_df = build_booking_summary(bookings_df)
    scenarios = run_scenarios(perf_df, config)
    opportunities = generate_opportunities(perf_df, cat_summary_df, summary_metrics, booking_summary_df, config)
    risks = generate_risks(perf_df, cat_summary_df, summary_metrics, booking_summary_df, config)

    # Data quality diagnostics
    data_quality_diagnostics, data_quality_notes = build_data_quality_report(
        menu_df, orders_df, waste_df, staff_df, bookings_df
    )
    
    # Update summary_metrics with days_of_data from diagnostics
    if "num_days_covered" in data_quality_diagnostics:
        summary_metrics["days_of_data"] = data_quality_diagnostics["num_days_covered"]

    # Build formatted tables for GPT export
    results_dict = {
        "perf_df": perf_df,
        "cat_summary_df": cat_summary_df,
        "summary_metrics": summary_metrics,
        "data_quality_diagnostics": data_quality_diagnostics,
        "scenarios": scenarios,  # Add raw scenarios dict with numeric data
        "config": config,
        "restaurant_name": config.get("restaurant_name"),
        "period_label": config.get("period_label"),
    }
    
    # Add formatted tables from old export block (for backward compatibility)
    results_dict.update({
        "category_performance_table": format_category_performance_table(cat_summary_df, config),
        "top_margin_items_table": format_top_margin_items_table(perf_df, config),
        "menu_stars_table": format_menu_class_table(perf_df, "Star", config),
        "menu_plowhorses_table": format_menu_class_table(perf_df, "Plowhorse", config),
        "menu_puzzles_table": format_menu_class_table(perf_df, "Puzzle", config),
        "menu_dogs_table": format_menu_class_table(perf_df, "Dog", config),
        "top_waste_items_table": format_top_waste_items_table(perf_df, config),
        "scenario_summaries": format_scenario_summaries(scenarios, config),
    })
    
    # Add optional tables if data available
    if not staff_perf_df.empty:
        results_dict["staff_performance_table"] = format_staff_performance_table(staff_perf_df, config)
    if not booking_summary_df.empty:
        results_dict["booking_summary_table"] = format_booking_summary_table(booking_summary_df, config)

    # Build Phase 1 Insight Graph
    insight_graph = insights_module.build_phase1_insight_graph(results_dict)
    
    # Build GPT export block v2 with insights
    gpt_export_block = insights_module.build_gpt_export_block_v2(
        meta=config,
        results=results_dict,
        insight_graph=insight_graph,
        charts=None,  # Will be populated after chart generation
    )
    
    # Keep old export block for backward compatibility
    gpt_export_block_v1 = build_gpt_export_block(
        perf_df=perf_df,
        summary_metrics=summary_metrics,
        cat_summary_df=cat_summary_df,
        scenarios=scenarios,
        opportunities=opportunities,
        risks=risks,
        staff_perf_df=staff_perf_df,
        booking_summary_df=booking_summary_df,
        config=config,
        data_quality_notes=data_quality_notes,
    )

    return {
        "menu_df": menu_df,
        "orders_df": orders_df,
        "waste_df": waste_df,
        "staff_df": staff_df,
        "bookings_df": bookings_df,
        "perf_df": perf_df,
        "summary_metrics": summary_metrics,
        "cat_summary_df": cat_summary_df,
        "staff_perf_df": staff_perf_df,
        "booking_summary_df": booking_summary_df,
        "scenarios": scenarios,
        "opportunities": opportunities,
        "risks": risks,
        "insight_graph": insight_graph,  # Phase 1 intelligence structure
        "gpt_export_block": gpt_export_block,  # v2 with insights
        "gpt_export_block_v1": gpt_export_block_v1,  # Legacy format
        "data_quality_diagnostics": data_quality_diagnostics,
        "data_quality_notes": data_quality_notes,
        "validation_result": validation_result,  # Include validation for reference
        "config": config,  # Include config for reference
    }


if __name__ == "__main__":
    """Run a headless one-shot analysis and write outputs to `output/`.

    Files produced:
      - output/menu_df.csv
      - output/orders_df.csv
      - output/waste_df.csv
      - output/staff_df.csv
      - output/bookings_df.csv
      - output/perf_df.csv
      - output/summary_metrics.json
      - output/gpt_export_block.txt
    """
    import json
    import os

    results = run_full_analysis_v2(data_source=DATA_SOURCE)
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    # DataFrames -> CSV
    results["menu_df"].to_csv(os.path.join(out_dir, "menu_df.csv"), index=False)
    results["orders_df"].to_csv(os.path.join(out_dir, "orders_df.csv"), index=False)
    results["waste_df"].to_csv(os.path.join(out_dir, "waste_df.csv"), index=False)
    results["staff_df"].to_csv(os.path.join(out_dir, "staff_df.csv"), index=False)
    results["bookings_df"].to_csv(os.path.join(out_dir, "bookings_df.csv"), index=False)
    results["perf_df"].to_csv(os.path.join(out_dir, "perf_df.csv"), index=False)

    # JSON/text outputs
    with open(os.path.join(out_dir, "summary_metrics.json"), "w") as f:
        json.dump(results["summary_metrics"], f, indent=2, default=str)

    with open(os.path.join(out_dir, "gpt_export_block.txt"), "w", encoding="utf-8") as f:
        f.write(results["gpt_export_block"])
    
    with open(os.path.join(out_dir, "gpt_export_block_v1.txt"), "w", encoding="utf-8") as f:
        f.write(results["gpt_export_block_v1"])
    
    # Save validation results
    with open(os.path.join(out_dir, "validation_report.json"), "w") as f:
        json.dump(results["validation_result"], f, indent=2, default=str)
    
    # Save insight graph as JSON
    from dataclasses import asdict
    insight_graph_dict = asdict(results["insight_graph"])
    with open(os.path.join(out_dir, "insight_graph.json"), "w", encoding="utf-8") as f:
        json.dump(insight_graph_dict, f, indent=2)

    print(f"Wrote analysis outputs to '{out_dir}/'")
    print(f"  - Validation report: {out_dir}/validation_report.json")
    print(f"  - Insight graph: {out_dir}/insight_graph.json")
    print(f"  - GPT export v2: {out_dir}/gpt_export_block.txt")
    print(f"  - GPT export v1 (legacy): {out_dir}/gpt_export_block_v1.txt")
