"""
Synthetic Restaurant Dataset Generator
=======================================

Generates realistic restaurant data with configurable:
- Menu size (small/medium/large)
- Sales volume (low/medium/high)
- Data quality (pristine/typical/corrupted)
- Edge cases (duplicates, missing items, negative GP)

Use for:
- Automated testing
- Stress testing
- Edge case validation
- Performance benchmarking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import random
from typing import Literal


class RestaurantDataGenerator:
    """Generate synthetic restaurant datasets for testing"""
    
    # Category definitions
    CATEGORIES = ["Starters", "Mains", "Desserts", "Sides", "Drinks"]
    
    # Realistic item names by category
    ITEM_NAMES = {
        "Starters": [
            "Caesar Salad", "Soup of the Day", "Garlic Bread", "Bruschetta",
            "Chicken Wings", "Calamari", "Spring Rolls", "Nachos",
            "Mozzarella Sticks", "Hummus Plate"
        ],
        "Mains": [
            "Beef Burger", "Chicken Burger", "Veggie Burger",
            "Margherita Pizza", "Pepperoni Pizza", "BBQ Chicken Pizza",
            "Fish and Chips", "Steak", "Salmon", "Pasta Carbonara",
            "Chicken Curry", "Vegetarian Lasagna", "Roast Chicken"
        ],
        "Desserts": [
            "Chocolate Cake", "Cheesecake", "Ice Cream", "Tiramisu",
            "Apple Pie", "Brownie", "Crème Brûlée", "Panna Cotta"
        ],
        "Sides": [
            "French Fries", "Onion Rings", "Coleslaw", "Side Salad",
            "Mashed Potatoes", "Roasted Vegetables", "Garlic Mushrooms"
        ],
        "Drinks": [
            "Coca Cola", "Sprite", "Orange Juice", "Apple Juice",
            "Coffee", "Tea", "Beer Pint", "House Wine (175ml)",
            "Sparkling Water", "Still Water"
        ]
    }
    
    # Price ranges by category (min, max)
    PRICE_RANGES = {
        "Starters": (4.0, 9.0),
        "Mains": (10.0, 25.0),
        "Desserts": (5.0, 8.0),
        "Sides": (3.0, 6.0),
        "Drinks": (2.0, 8.0)
    }
    
    # Typical GP% by category
    GP_PERCENTAGES = {
        "Starters": (0.60, 0.75),
        "Mains": (0.55, 0.70),
        "Desserts": (0.65, 0.80),
        "Sides": (0.70, 0.85),
        "Drinks": (0.75, 0.90)
    }
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_menu(
        self,
        size: Literal["small", "medium", "large"] = "medium",
        quality: Literal["pristine", "typical", "corrupted"] = "pristine"
    ) -> pd.DataFrame:
        """
        Generate menu data
        
        Args:
            size: small (15 items), medium (40 items), large (100 items)
            quality: pristine (perfect), typical (minor issues), corrupted (major issues)
        
        Returns:
            DataFrame with columns: item_name, category, sell_price, cost_per_unit
        """
        # Determine item count
        if size == "small":
            item_count = random.randint(10, 20)
        elif size == "medium":
            item_count = random.randint(35, 50)
        else:  # large
            item_count = random.randint(80, 120)
        
        # Select items from each category
        items = []
        categories = []
        
        items_per_category = item_count // len(self.CATEGORIES)
        remainder = item_count % len(self.CATEGORIES)
        
        for cat in self.CATEGORIES:
            cat_items = items_per_category + (1 if remainder > 0 else 0)
            remainder -= 1
            
            # Sample items (with replacement if needed)
            available = self.ITEM_NAMES[cat]
            if cat_items <= len(available):
                selected = random.sample(available, cat_items)
            else:
                selected = random.choices(available, k=cat_items)
            
            items.extend(selected)
            categories.extend([cat] * cat_items)
        
        # Generate prices
        prices = []
        costs = []
        
        for cat in categories:
            price_min, price_max = self.PRICE_RANGES[cat]
            gp_min, gp_max = self.GP_PERCENTAGES[cat]
            
            price = round(random.uniform(price_min, price_max), 2)
            gp_pct = random.uniform(gp_min, gp_max)
            cost = round(price * (1 - gp_pct), 2)
            
            prices.append(price)
            costs.append(cost)
        
        menu_df = pd.DataFrame({
            "item_name": items,
            "category": categories,
            "sell_price": prices,
            "cost_per_unit": costs
        })
        
        # Apply quality degradation
        if quality == "typical":
            menu_df = self._add_typical_issues(menu_df)
        elif quality == "corrupted":
            menu_df = self._add_corrupted_issues(menu_df)
        
        return menu_df
    
    def generate_sales(
        self,
        menu_df: pd.DataFrame,
        volume: Literal["low", "medium", "high"] = "medium",
        days: int = 365,
        quality: Literal["pristine", "typical", "corrupted"] = "pristine"
    ) -> pd.DataFrame:
        """
        Generate sales transaction data
        
        Args:
            menu_df: Menu DataFrame to reference
            volume: low (50/day), medium (150/day), high (500/day)
            days: Number of days of data
            quality: Data quality level
        
        Returns:
            DataFrame with columns: item_name, qty, order_datetime
        """
        # Determine daily transaction count
        if volume == "low":
            daily_txns = random.randint(30, 70)
        elif volume == "medium":
            daily_txns = random.randint(100, 200)
        else:  # high
            daily_txns = random.randint(400, 600)
        
        total_txns = daily_txns * days
        
        # Create popularity weights (Pareto principle: 20% items = 80% sales)
        item_names = menu_df["item_name"].tolist()
        categories = menu_df["category"].tolist()
        
        # Weight by category popularity
        category_weights = {
            "Mains": 0.40,
            "Drinks": 0.25,
            "Starters": 0.15,
            "Sides": 0.12,
            "Desserts": 0.08
        }
        
        weights = [category_weights.get(cat, 0.1) for cat in categories]
        weights = np.array(weights) / sum(weights)  # Normalize
        
        # Add item-level variation (some items more popular than others)
        item_variation = np.random.lognormal(0, 0.5, len(item_names))
        weights = weights * item_variation
        weights = weights / sum(weights)
        
        # Generate sales
        selected_items = np.random.choice(
            item_names,
            size=total_txns,
            p=weights,
            replace=True
        )
        
        # Quantities (most orders are 1-2 items)
        quantities = np.random.choice([1, 2, 3, 4], size=total_txns, p=[0.50, 0.30, 0.15, 0.05])
        
        # Timestamps (realistic restaurant hours: 9am-11pm)
        start_date = datetime(2024, 1, 1, 9, 0)
        timestamps = []
        
        for day in range(days):
            day_txns = daily_txns + random.randint(-20, 20)  # Daily variation
            
            for _ in range(day_txns):
                # Peak hours: lunch (12-2pm) and dinner (6-9pm)
                hour = random.choices(
                    range(9, 23),
                    weights=[0.02, 0.03, 0.05, 0.15, 0.15, 0.08, 0.05, 0.03, 0.12, 0.15, 0.12, 0.03, 0.02, 0.01]
                )[0]
                minute = random.randint(0, 59)
                
                timestamp = start_date + timedelta(days=day, hours=hour, minutes=minute)
                timestamps.append(timestamp)
        
        # Trim to exact count
        selected_items = selected_items[:len(timestamps)]
        quantities = quantities[:len(timestamps)]
        
        sales_df = pd.DataFrame({
            "item_name": selected_items,
            "qty": quantities,
            "order_datetime": timestamps
        })
        
        # Apply quality degradation
        if quality == "typical":
            sales_df = self._add_typical_sales_issues(sales_df, menu_df)
        elif quality == "corrupted":
            sales_df = self._add_corrupted_sales_issues(sales_df, menu_df)
        
        return sales_df
    
    def generate_waste(
        self,
        menu_df: pd.DataFrame,
        waste_rate: float = 0.05,
        quality: Literal["pristine", "typical", "corrupted"] = "pristine"
    ) -> pd.DataFrame:
        """
        Generate waste data
        
        Args:
            menu_df: Menu DataFrame to reference
            waste_rate: Fraction of items with waste data (0.0-1.0)
            quality: Data quality level
        
        Returns:
            DataFrame with columns: item_name, waste_qty
        """
        # Select items with waste
        n_items = len(menu_df)
        n_waste_items = int(n_items * waste_rate)
        
        if n_waste_items == 0:
            return pd.DataFrame(columns=["item_name", "waste_qty"])
        
        waste_items = menu_df.sample(n=n_waste_items)
        
        # Generate waste quantities (typically 1-5% of total)
        waste_qtys = np.random.exponential(scale=10, size=n_waste_items).astype(int)
        waste_qtys = np.clip(waste_qtys, 1, 50)  # 1-50 units
        
        waste_df = pd.DataFrame({
            "item_name": waste_items["item_name"].values,
            "waste_qty": waste_qtys
        })
        
        # Apply quality degradation
        if quality == "typical":
            waste_df = self._add_typical_waste_issues(waste_df, menu_df)
        elif quality == "corrupted":
            waste_df = self._add_corrupted_waste_issues(waste_df, menu_df)
        
        return waste_df
    
    def _add_typical_issues(self, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic minor issues to menu data"""
        df = menu_df.copy()
        
        # 1. Category case variations (10% of items)
        n_vary = int(len(df) * 0.1)
        if n_vary > 0:
            vary_idx = random.sample(range(len(df)), n_vary)
            for idx in vary_idx:
                cat = df.loc[idx, "category"]
                df.loc[idx, "category"] = random.choice([cat.lower(), cat.upper(), cat.title()])
        
        # 2. Extra whitespace (5% of items)
        n_space = int(len(df) * 0.05)
        if n_space > 0:
            space_idx = random.sample(range(len(df)), n_space)
            for idx in space_idx:
                df.loc[idx, "item_name"] = "  " + df.loc[idx, "item_name"] + "  "
        
        # 3. Currency symbols in prices (5% of items)
        n_currency = int(len(df) * 0.05)
        if n_currency > 0:
            currency_idx = random.sample(range(len(df)), n_currency)
            for idx in currency_idx:
                df.loc[idx, "sell_price"] = f"£{df.loc[idx, 'sell_price']}"
        
        return df
    
    def _add_corrupted_issues(self, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add major data quality issues"""
        df = self._add_typical_issues(menu_df)
        
        # 1. Duplicate items (5%)
        n_dup = int(len(df) * 0.05)
        if n_dup > 0:
            dup_rows = df.sample(n=n_dup)
            df = pd.concat([df, dup_rows], ignore_index=True)
        
        # 2. Negative GP items (3%)
        n_neg = int(len(df) * 0.03)
        if n_neg > 0:
            neg_idx = random.sample(range(len(df)), n_neg)
            for idx in neg_idx:
                # Swap cost and price
                df.loc[idx, "cost_per_unit"] = df.loc[idx, "sell_price"] * 1.2
        
        # 3. Zero/null prices (2%)
        n_zero = int(len(df) * 0.02)
        if n_zero > 0:
            zero_idx = random.sample(range(len(df)), n_zero)
            for idx in zero_idx:
                df.loc[idx, "sell_price"] = 0.0
        
        # 4. Missing categories (5%)
        n_missing = int(len(df) * 0.05)
        if n_missing > 0:
            missing_idx = random.sample(range(len(df)), n_missing)
            for idx in missing_idx:
                df.loc[idx, "category"] = random.choice(["", " ", None])
        
        return df
    
    def _add_typical_sales_issues(self, sales_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add minor sales data issues"""
        df = sales_df.copy()
        
        # 1. Items not in menu (2% - seasonal/discontinued)
        n_unknown = int(len(df) * 0.02)
        if n_unknown > 0:
            unknown_items = ["Seasonal Special", "Daily Special", "Old Menu Item"]
            unknown_idx = random.sample(range(len(df)), n_unknown)
            for idx in unknown_idx:
                df.loc[idx, "item_name"] = random.choice(unknown_items)
        
        # 2. Name typos (1%)
        n_typo = int(len(df) * 0.01)
        if n_typo > 0:
            typo_idx = random.sample(range(len(df)), n_typo)
            for idx in typo_idx:
                name = df.loc[idx, "item_name"]
                # Add extra character
                df.loc[idx, "item_name"] = name + "s"
        
        return df
    
    def _add_corrupted_sales_issues(self, sales_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add major sales data issues"""
        df = self._add_typical_sales_issues(sales_df, menu_df)
        
        # 1. High % of unknown items (15%)
        n_unknown = int(len(df) * 0.15)
        if n_unknown > 0:
            unknown_items = [f"Unknown Item {i}" for i in range(20)]
            unknown_idx = random.sample(range(len(df)), n_unknown)
            for idx in unknown_idx:
                df.loc[idx, "item_name"] = random.choice(unknown_items)
        
        # 2. Invalid dates (10%)
        n_bad_dates = int(len(df) * 0.10)
        if n_bad_dates > 0:
            bad_date_idx = random.sample(range(len(df)), n_bad_dates)
            for idx in bad_date_idx:
                df.loc[idx, "order_datetime"] = "invalid_date"
        
        # 3. Negative quantities (1%)
        n_neg_qty = int(len(df) * 0.01)
        if n_neg_qty > 0:
            neg_idx = random.sample(range(len(df)), n_neg_qty)
            for idx in neg_idx:
                df.loc[idx, "qty"] = -abs(df.loc[idx, "qty"])
        
        return df
    
    def _add_typical_waste_issues(self, waste_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add minor waste data issues"""
        df = waste_df.copy()
        
        # 1. Items not in menu (5%)
        n_unknown = int(len(df) * 0.05)
        if n_unknown > 0:
            unknown_idx = random.sample(range(len(df)), n_unknown)
            for idx in unknown_idx:
                df.loc[idx, "item_name"] = "Discontinued Item"
        
        return df
    
    def _add_corrupted_waste_issues(self, waste_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Add major waste data issues"""
        df = self._add_typical_waste_issues(waste_df, menu_df)
        
        # 1. High % unknown items (20%)
        n_unknown = int(len(df) * 0.20)
        if n_unknown > 0:
            unknown_idx = random.sample(range(len(df)), n_unknown)
            for idx in unknown_idx:
                df.loc[idx, "item_name"] = f"Unknown {random.randint(1, 100)}"
        
        # 2. Negative waste (5%)
        n_neg = int(len(df) * 0.05)
        if n_neg > 0:
            neg_idx = random.sample(range(len(df)), n_neg)
            for idx in neg_idx:
                df.loc[idx, "waste_qty"] = -abs(df.loc[idx, "waste_qty"])
        
        return df
    
    def generate_complete_dataset(
        self,
        output_dir: str | Path,
        menu_size: Literal["small", "medium", "large"] = "medium",
        sales_volume: Literal["low", "medium", "high"] = "medium",
        quality: Literal["pristine", "typical", "corrupted"] = "pristine",
        days: int = 365
    ) -> dict:
        """
        Generate complete restaurant dataset and save to files
        
        Args:
            output_dir: Directory to save files
            menu_size: Size of menu
            sales_volume: Volume of sales
            quality: Data quality level
            days: Days of sales data
        
        Returns:
            Dictionary with file paths and metadata
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate data
        menu_df = self.generate_menu(size=menu_size, quality=quality)
        sales_df = self.generate_sales(menu_df, volume=sales_volume, days=days, quality=quality)
        waste_df = self.generate_waste(menu_df, quality=quality)
        
        # Save files
        menu_path = output_dir / f"menu_{menu_size}_{quality}.csv"
        sales_path = output_dir / f"sales_{sales_volume}_{quality}.csv"
        waste_path = output_dir / f"waste_{quality}.csv"
        
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        waste_df.to_csv(waste_path, index=False)
        
        # Metadata - handle corrupted dates safely
        try:
            date_min = sales_df['order_datetime'].min()
            date_max = sales_df['order_datetime'].max()
            date_range = f"{date_min} to {date_max}"
        except:
            date_range = "Contains invalid dates (corrupted data)"
        
        metadata = {
            "menu_path": str(menu_path),
            "sales_path": str(sales_path),
            "waste_path": str(waste_path),
            "menu_items": len(menu_df),
            "sales_transactions": len(sales_df),
            "waste_items": len(waste_df),
            "date_range": date_range,
            "quality": quality,
            "menu_size": menu_size,
            "sales_volume": sales_volume
        }
        
        return metadata


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Generate test datasets via command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic restaurant data")
    parser.add_argument("--output", "-o", default="data/synthetic", help="Output directory")
    parser.add_argument("--menu-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--sales-volume", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--quality", choices=["pristine", "typical", "corrupted"], default="pristine")
    parser.add_argument("--days", type=int, default=365, help="Days of sales data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Generate dataset
    generator = RestaurantDataGenerator(seed=args.seed)
    
    print(f"\n🏭 Generating synthetic restaurant data...")
    print(f"   Menu Size: {args.menu_size}")
    print(f"   Sales Volume: {args.sales_volume}")
    print(f"   Quality: {args.quality}")
    print(f"   Days: {args.days}")
    print(f"   Output: {args.output}\n")
    
    metadata = generator.generate_complete_dataset(
        output_dir=args.output,
        menu_size=args.menu_size,
        sales_volume=args.sales_volume,
        quality=args.quality,
        days=args.days
    )
    
    print("✅ Dataset generated successfully!\n")
    print("📊 Summary:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    print(f"\n📁 Files created:")
    print(f"   • {metadata['menu_path']}")
    print(f"   • {metadata['sales_path']}")
    print(f"   • {metadata['waste_path']}")


if __name__ == "__main__":
    main()
