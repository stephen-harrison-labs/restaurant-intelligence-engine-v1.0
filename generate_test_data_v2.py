"""
Ultra-Realistic Restaurant Dataset Generator with Ground Truth
===============================================================

Generates production-quality synthetic restaurant data with:
- Restaurant archetypes (casual_brasserie, burger_bar, dessert_cafe, gastro_pub)
- Ground truth labels for automated testing
- Realistic staff scheduling and performance
- Bookings data tied to sales volume
- Full reproducibility with seeds

Use for:
- Automated pytest validation
- Testing engine intelligence (Stars, Dogs, Puzzles, Plowhorses)
- Integration testing with known expected outcomes
- Stress testing edge cases
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal, Dict, List, Tuple
from dataclasses import dataclass, asdict


# =============================================================================
# RESTAURANT ARCHETYPES
# =============================================================================

@dataclass
class RestaurantArchetype:
    """Definition of a restaurant type with realistic characteristics"""
    name: str
    category_distribution: Dict[str, float]  # % of menu by category
    price_ranges: Dict[str, Tuple[float, float]]  # (min, max) per category
    gp_ranges: Dict[str, Tuple[float, float]]  # (min, max) GP% per category
    volume_profile: Dict[str, float]  # Relative popularity by category
    menu_size_range: Tuple[int, int]  # (min, max) total items
    signature_items: List[str]  # Must-include items for realism


RESTAURANT_ARCHETYPES = {
    "casual_brasserie": RestaurantArchetype(
        name="casual_brasserie",
        category_distribution={
            "Starters": 0.20, "Mains": 0.35, "Sides": 0.15, 
            "Desserts": 0.15, "Drinks": 0.15
        },
        price_ranges={
            "Starters": (5.5, 12.0), 
            "Mains": (12.0, 28.0),
            "Sides": (3.5, 6.5),
            "Desserts": (5.5, 9.0),
            "Drinks": (2.5, 9.0)
        },
        gp_ranges={
            "Starters": (0.60, 0.75),
            "Mains": (0.55, 0.70),
            "Sides": (0.70, 0.85),
            "Desserts": (0.65, 0.80),
            "Drinks": (0.75, 0.90)
        },
        volume_profile={
            "Starters": 0.15, "Mains": 0.45, "Sides": 0.15, 
            "Desserts": 0.10, "Drinks": 0.15
        },
        menu_size_range=(35, 50),
        signature_items=["Steak Frites", "Moules Marinières", "French Onion Soup", "Crème Brûlée"]
    ),
    
    "burger_bar": RestaurantArchetype(
        name="burger_bar",
        category_distribution={
            "Starters": 0.15, "Mains": 0.40, "Sides": 0.25,
            "Desserts": 0.10, "Drinks": 0.10
        },
        price_ranges={
            "Starters": (4.5, 8.5),
            "Mains": (8.0, 18.0),
            "Sides": (3.0, 6.0),
            "Desserts": (4.0, 7.0),
            "Drinks": (2.0, 6.0)
        },
        gp_ranges={
            "Starters": (0.65, 0.80),
            "Mains": (0.50, 0.65),
            "Sides": (0.75, 0.90),
            "Desserts": (0.70, 0.85),
            "Drinks": (0.80, 0.95)
        },
        volume_profile={
            "Starters": 0.10, "Mains": 0.50, "Sides": 0.25,
            "Desserts": 0.05, "Drinks": 0.10
        },
        menu_size_range=(25, 35),
        signature_items=["Classic Beef Burger", "Bacon Cheeseburger", "Loaded Fries", "Onion Rings"]
    ),
    
    "dessert_cafe": RestaurantArchetype(
        name="dessert_cafe",
        category_distribution={
            "Starters": 0.10, "Mains": 0.15, "Sides": 0.05,
            "Desserts": 0.50, "Drinks": 0.20
        },
        price_ranges={
            "Starters": (4.0, 7.0),
            "Mains": (7.0, 14.0),
            "Sides": (2.5, 5.0),
            "Desserts": (5.0, 12.0),
            "Drinks": (3.0, 7.5)
        },
        gp_ranges={
            "Starters": (0.60, 0.75),
            "Mains": (0.55, 0.70),
            "Sides": (0.70, 0.85),
            "Desserts": (0.70, 0.85),
            "Drinks": (0.80, 0.95)
        },
        volume_profile={
            "Starters": 0.05, "Mains": 0.10, "Sides": 0.05,
            "Desserts": 0.60, "Drinks": 0.20
        },
        menu_size_range=(30, 45),
        signature_items=["Triple Chocolate Cake", "New York Cheesecake", "Tiramisu", "Cappuccino"]
    ),
    
    "gastro_pub": RestaurantArchetype(
        name="gastro_pub",
        category_distribution={
            "Starters": 0.20, "Mains": 0.30, "Sides": 0.15,
            "Desserts": 0.10, "Drinks": 0.25
        },
        price_ranges={
            "Starters": (6.0, 14.0),
            "Mains": (14.0, 32.0),
            "Sides": (4.0, 7.5),
            "Desserts": (6.0, 10.0),
            "Drinks": (3.5, 12.0)
        },
        gp_ranges={
            "Starters": (0.60, 0.75),
            "Mains": (0.55, 0.70),
            "Sides": (0.70, 0.85),
            "Desserts": (0.65, 0.80),
            "Drinks": (0.70, 0.90)
        },
        volume_profile={
            "Starters": 0.15, "Mains": 0.35, "Sides": 0.12,
            "Desserts": 0.08, "Drinks": 0.30
        },
        menu_size_range=(40, 60),
        signature_items=["Sunday Roast", "Fish & Chips", "Scotch Egg", "Sticky Toffee Pudding", "Craft Beer"]
    )
}


# =============================================================================
# ITEM LIBRARY BY ARCHETYPE
# =============================================================================

ITEM_LIBRARY = {
    "Starters": {
        "casual_brasserie": ["French Onion Soup", "Escargots", "Pâté", "Moules Marinières", 
                             "Caesar Salad", "Niçoise Salad", "Goat Cheese Salad"],
        "burger_bar": ["Buffalo Wings", "Nachos", "Jalapeño Poppers", "Mozzarella Sticks", 
                       "Loaded Potato Skins", "Chicken Tenders"],
        "dessert_cafe": ["Fruit Salad", "Yogurt Parfait", "Granola Bowl", "Avocado Toast"],
        "gastro_pub": ["Scotch Egg", "Pork Pie", "Prawn Cocktail", "Soup of the Day", 
                       "Chicken Liver Pâté", "Smoked Salmon"]
    },
    "Mains": {
        "casual_brasserie": ["Steak Frites", "Coq au Vin", "Bouillabaisse", "Duck Confit", 
                             "Ratatouille", "Croque Monsieur", "Quiche Lorraine"],
        "burger_bar": ["Classic Beef Burger", "Bacon Cheeseburger", "Chicken Burger", 
                       "Veggie Burger", "BBQ Pulled Pork Burger", "Double Cheeseburger", 
                       "Spicy Jalapeño Burger"],
        "dessert_cafe": ["Breakfast Sandwich", "Panini", "Club Sandwich", "Soup & Salad Combo", 
                         "Quiche", "Bagel & Cream Cheese"],
        "gastro_pub": ["Sunday Roast", "Fish & Chips", "Steak & Ale Pie", "Bangers & Mash", 
                       "Shepherd's Pie", "Lamb Shank", "Pork Belly", "Sea Bass"]
    },
    "Desserts": {
        "casual_brasserie": ["Crème Brûlée", "Tarte Tatin", "Mousse au Chocolat", "Profiteroles", 
                             "Macarons", "Crêpes Suzette"],
        "burger_bar": ["Milkshake", "Brownie Sundae", "Apple Pie", "Cheesecake Slice"],
        "dessert_cafe": ["Triple Chocolate Cake", "New York Cheesecake", "Tiramisu", "Red Velvet Cake", 
                         "Carrot Cake", "Lemon Drizzle", "Banoffee Pie", "Pavlova", 
                         "Eton Mess", "Panna Cotta"],
        "gastro_pub": ["Sticky Toffee Pudding", "Apple Crumble", "Bread & Butter Pudding", 
                       "Eton Mess", "Chocolate Fondant"]
    },
    "Sides": {
        "casual_brasserie": ["French Fries", "Haricots Verts", "Gratin Dauphinois", "Ratatouille"],
        "burger_bar": ["French Fries", "Sweet Potato Fries", "Onion Rings", "Loaded Fries", 
                       "Coleslaw", "Side Salad", "Corn on the Cob"],
        "dessert_cafe": ["Cookie", "Muffin", "Croissant"],
        "gastro_pub": ["Chips", "Mash", "Roast Potatoes", "Seasonal Veg", "Mushy Peas", "Gravy"]
    },
    "Drinks": {
        "casual_brasserie": ["House Wine (175ml)", "Champagne", "Perrier", "Café au Lait", 
                             "Espresso", "Orangina"],
        "burger_bar": ["Coca Cola", "Sprite", "Milkshake", "Beer Bottle", "Iced Tea"],
        "dessert_cafe": ["Cappuccino", "Latte", "Espresso", "Hot Chocolate", "Tea", 
                         "Iced Coffee", "Smoothie", "Fresh Juice"],
        "gastro_pub": ["Craft Beer", "Cask Ale", "House Wine (175ml)", "G&T", "Cider", 
                       "Coffee", "Sparkling Water"]
    }
}


# =============================================================================
# MAIN GENERATOR CLASS
# =============================================================================

class UltraRealisticRestaurantGenerator:
    """Generate production-quality restaurant data with ground truth labels"""
    
    def __init__(self, seed: int = 42):
        """Initialize with numpy RNG for full reproducibility"""
        self.rng = np.random.default_rng(seed)
        self.archetype = None
        self.ground_truth = {"items": [], "expected_metrics": {}}
        
    def generate_restaurant_dataset(
        self,
        restaurant_name: str,
        restaurant_type: str = "casual_brasserie",
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31",
        quality_level: str = "typical",
        random_seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generate complete ultra-realistic restaurant dataset with ground truth labels.
        
        Parameters
        ----------
        restaurant_name : str
            Name of the restaurant (used in metadata)
        restaurant_type : str, optional
            One of: "casual_brasserie", "burger_bar", "dessert_cafe", "gastro_pub"
            Default: "casual_brasserie"
        start_date : str, optional
            Start date for sales data (YYYY-MM-DD format)
            Default: "2024-01-01"
        end_date : str, optional
            End date for sales data (YYYY-MM-DD format)
            Default: "2024-12-31"
        quality_level : str, optional
            Data quality: "pristine" (perfect), "typical" (minor issues), "chaotic" (major issues)
            Default: "typical"
        random_seed : int, optional
            Seed for reproducibility
            Default: 42
            
        Returns
        -------
        menu_df : pd.DataFrame
            Menu items with columns: item_id, item_name, category, sell_price, cost_per_unit
        sales_df : pd.DataFrame
            Sales transactions with columns: order_id, item_name, qty, order_datetime, staff_name
        waste_df : pd.DataFrame
            Waste data with columns: item_name, waste_qty, waste_date
        staff_df : pd.DataFrame
            Staff performance with columns: staff_name, role, orders_handled, revenue, 
            gp, revenue_per_hour, gp_per_hour, performance_level
        bookings_df : pd.DataFrame
            Bookings with columns: booking_id, booking_datetime, covers, status, day_of_week
        ground_truth : dict
            Ground truth labels for testing with structure:
            {
                "items": pd.DataFrame with columns [item_id, item_name, tags, 
                         expected_matrix_quadrant, is_opportunity, is_risk, archetype_role],
                "expected_metrics": {
                    "total_star_items": int,
                    "total_underpriced_items": int,
                    "high_waste_items": List[str],
                    "anchor_items": List[str],
                    "overpriced_low_volume_items": List[str]
                }
            }
            
        Examples
        --------
        >>> # Generate pristine dataset for testing
        >>> menu, sales, waste, staff, bookings, truth = generate_restaurant_dataset(
        ...     restaurant_name="The Test Kitchen",
        ...     restaurant_type="burger_bar",
        ...     quality_level="pristine",
        ...     random_seed=42
        ... )
        >>> 
        >>> # Use in pytest
        >>> def test_engine_detects_stars():
        ...     engine = RestaurantIntelligenceEngine(menu, sales, waste)
        ...     results = engine.analyze()
        ...     star_items = truth['items'][truth['items']['expected_matrix_quadrant'] == 'Star']
        ...     assert all(item in results['stars'] for item in star_items['item_name'])
        
        Notes
        -----
        - All randomness is deterministic via numpy RNG with provided seed
        - Ground truth labels ensure specific items behave as tagged (Star, Dog, etc.)
        - Staff and bookings data are realistically tied to sales volume patterns
        - Quality levels inject realistic noise for robustness testing
        """
        # Reinitialize RNG with provided seed
        self.rng = np.random.default_rng(random_seed)
        self.archetype = RESTAURANT_ARCHETYPES[restaurant_type]
        self.ground_truth = {"items": [], "expected_metrics": {}}
        
        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days + 1
        
        print(f"\n🏭 Generating {restaurant_type} dataset: {restaurant_name}")
        print(f"   Period: {start_date} to {end_date} ({days} days)")
        print(f"   Quality: {quality_level}")
        
        # Step 1: Generate menu with ground truth labels
        menu_df = self._generate_menu_with_truth(restaurant_type)
        print(f"   ✅ Menu: {len(menu_df)} items across {menu_df['category'].nunique()} categories")
        
        # Step 2: Generate sales tied to ground truth
        sales_df = self._generate_sales_with_staff(menu_df, start, days)
        print(f"   ✅ Sales: {len(sales_df):,} transactions")
        
        # Step 3: Generate waste (higher for tagged high_waste_risk items)
        waste_df = self._generate_waste_with_truth(menu_df)
        print(f"   ✅ Waste: {len(waste_df)} items with waste")
        
        # Step 4: Generate staff performance summary
        staff_df = self._generate_staff_summary(sales_df, menu_df)
        print(f"   ✅ Staff: {len(staff_df)} team members")
        
        # Step 5: Generate bookings tied to sales volume
        bookings_df = self._generate_bookings(sales_df, start, days)
        print(f"   ✅ Bookings: {len(bookings_df)} reservations")
        
        # Step 6: Apply quality degradation
        if quality_level != "pristine":
            menu_df, sales_df, waste_df = self._apply_quality_issues(
                menu_df, sales_df, waste_df, quality_level
            )
            print(f"   ✅ Quality noise applied ({quality_level})")
        
        # Step 7: Finalize ground truth
        ground_truth = self._finalize_ground_truth()
        
        print(f"   ✅ Ground truth: {len(ground_truth['items'])} labeled items\n")
        
        return menu_df, sales_df, waste_df, staff_df, bookings_df, ground_truth
    
    def _generate_menu_with_truth(self, restaurant_type: str) -> pd.DataFrame:
        """Generate menu with ground truth labels for testing"""
        arch = self.archetype
        
        # Determine total items
        total_items = self.rng.integers(arch.menu_size_range[0], arch.menu_size_range[1] + 1)
        
        # Calculate items per category based on distribution
        items_by_category = {}
        for category, pct in arch.category_distribution.items():
            count = max(1, int(total_items * pct))
            items_by_category[category] = count
        
        menu_items = []
        item_id = 1
        
        for category, count in items_by_category.items():
            # Get available items for this archetype
            available_items = ITEM_LIBRARY[category][restaurant_type]
            
            # Add signature items if this category has them
            signature_in_cat = [item for item in arch.signature_items 
                                if any(sig in item for sig in available_items)]
            
            # Sample items (with replacement if needed)
            if count <= len(available_items):
                selected = self.rng.choice(available_items, size=count, replace=False).tolist()
            else:
                selected = self.rng.choice(available_items, size=count, replace=True).tolist()
            
            # Ensure signature items are included
            for sig in signature_in_cat[:count]:
                if sig not in selected and len(selected) > 0:
                    selected[0] = sig
            
            # Generate prices and costs
            price_min, price_max = arch.price_ranges[category]
            gp_min, gp_max = arch.gp_ranges[category]
            
            for item_name in selected:
                price = round(self.rng.uniform(price_min, price_max), 2)
                gp_pct = self.rng.uniform(gp_min, gp_max)
                cost = round(price * (1 - gp_pct), 2)
                
                # Assign ground truth tags
                tags = [category.lower()]
                expected_quadrant = None
                is_opportunity = False
                is_risk = False
                archetype_role = "standard"
                
                # Tag special items (deterministic based on item_id for reproducibility)
                roll = self.rng.random()
                
                if roll < 0.10:  # 10% underpriced high performers (Stars)
                    tags.append("underpriced_high_performer")
                    expected_quadrant = "Star"
                    is_opportunity = True
                    archetype_role = "opportunity"
                    # Make price lower than it should be
                    price = round(price * 0.85, 2)
                    cost = round(price * (1 - gp_pct * 1.2), 2)  # Even lower cost
                    
                elif roll < 0.18:  # 8% overpriced low volume (Dogs)
                    tags.append("overpriced_low_volume")
                    expected_quadrant = "Dog"
                    is_risk = True
                    archetype_role = "risk"
                    # Make price too high
                    price = round(price * 1.25, 2)
                    
                elif roll < 0.25:  # 7% high waste risk
                    tags.append("high_waste_risk")
                    is_risk = True
                    archetype_role = "risk"
                    
                elif roll < 0.30:  # 5% anchor items (premium reference)
                    tags.append("anchor_item")
                    expected_quadrant = "Plowhorse"  # High volume, appropriate price
                    archetype_role = "anchor"
                    # Premium price
                    price = round(price_max * 0.95, 2)
                    cost = round(price * (1 - gp_pct), 2)
                
                menu_items.append({
                    "item_id": item_id,
                    "item_name": item_name,
                    "category": category,
                    "sell_price": price,
                    "cost_per_unit": cost,
                    "tags": tags,
                    "expected_quadrant": expected_quadrant,
                    "is_opportunity": is_opportunity,
                    "is_risk": is_risk,
                    "archetype_role": archetype_role
                })
                
                item_id += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(menu_items)
        
        # Store ground truth (separate from menu CSV)
        self.ground_truth["items"] = df[["item_id", "item_name", "tags", "expected_quadrant", 
                                          "is_opportunity", "is_risk", "archetype_role"]].copy()
        
        # Return menu without ground truth columns
        return df[["item_id", "item_name", "category", "sell_price", "cost_per_unit"]]
    
    def _generate_sales_with_staff(self, menu_df: pd.DataFrame, start_date: datetime, days: int) -> pd.DataFrame:
        """Generate sales with realistic staff assignments"""
        # Create staff roster
        staff_roster = [
            {"name": "Alice", "role": "Server", "shift": "morning", "performance": "high_performer"},
            {"name": "Bob", "role": "Server", "shift": "evening", "performance": "average"},
            {"name": "Charlie", "role": "Bartender", "shift": "evening", "performance": "high_performer"},
            {"name": "Diana", "role": "Server", "shift": "evening", "performance": "average"},
            {"name": "Eve", "role": "Head Waiter", "shift": "evening", "performance": "high_performer"},
            {"name": "Frank", "role": "Server", "shift": "morning", "performance": "trainee"},
            {"name": "Grace", "role": "Server", "shift": "evening", "performance": "average"},
            {"name": "Henry", "role": "Manager", "shift": "both", "performance": "high_performer"},
        ]
        
        # Determine daily transaction rate (based on archetype)
        base_daily_txns = 120  # Medium volume
        
        # Get volume profile from archetype
        category_popularity = self.archetype.volume_profile
        
        # Create item weights based on ground truth tags and category
        item_weights = []
        for _, item in menu_df.iterrows():
            category = item["category"]
            base_weight = category_popularity.get(category, 0.1)
            
            # Check ground truth tags to adjust weights
            truth_item = self.ground_truth["items"][
                self.ground_truth["items"]["item_name"] == item["item_name"]
            ]
            
            if len(truth_item) > 0:
                tags = truth_item.iloc[0]["tags"]
                
                # Stars and anchors: high volume
                if "underpriced_high_performer" in tags or "anchor_item" in tags:
                    base_weight *= 3.0
                # Dogs: VERY low volume (overpriced items don't sell)
                elif "overpriced_low_volume" in tags:
                    base_weight *= 0.02  # 2% of normal - truly unpopular
                # High waste: medium-high volume (to generate waste)
                elif "high_waste_risk" in tags:
                    base_weight *= 1.5
            
            item_weights.append(base_weight)
        
        # Normalize weights
        item_weights = np.array(item_weights)
        item_weights = item_weights / item_weights.sum()
        
        # Generate sales transactions
        transactions = []
        order_id = 1
        
        for day in range(days):
            # Day of week effects (busier Fri/Sat)
            current_date = start_date + timedelta(days=day)
            dow = current_date.weekday()
            
            if dow in [4, 5]:  # Fri, Sat
                daily_txns = int(base_daily_txns * 1.5)
            elif dow == 6:  # Sun
                daily_txns = int(base_daily_txns * 1.2)
            else:
                daily_txns = base_daily_txns
            
            # Add random variation
            daily_txns += self.rng.integers(-20, 21)
            
            # Generate transactions for this day
            for _ in range(daily_txns):
                # Determine time of day (9am-11pm = 14 hours)
                hour_weights = np.array([0.02, 0.03, 0.05, 0.15, 0.15, 0.08, 0.05, 0.03, 
                                         0.12, 0.15, 0.12, 0.03, 0.02, 0.01])
                hour_weights = hour_weights / hour_weights.sum()  # Normalize
                hour = int(self.rng.choice(range(9, 23), p=hour_weights))
                minute = int(self.rng.integers(0, 60))
                
                timestamp = current_date + timedelta(hours=hour, minutes=minute)
                
                # Assign staff based on shift
                if hour < 16:  # Morning shift
                    available_staff = [s for s in staff_roster if s["shift"] in ["morning", "both"]]
                else:  # Evening shift
                    available_staff = [s for s in staff_roster if s["shift"] in ["evening", "both"]]
                
                staff_member = self.rng.choice(available_staff)
                staff_name = staff_member["name"]
                
                # Select item (weighted by ground truth)
                item_idx = self.rng.choice(len(menu_df), p=item_weights)
                item_name = menu_df.iloc[item_idx]["item_name"]
                
                # Quantity (performance affects avg ticket)
                if staff_member["performance"] == "high_performer":
                    qty = self.rng.choice([1, 2, 3], p=[0.40, 0.40, 0.20])
                elif staff_member["performance"] == "trainee":
                    qty = self.rng.choice([1, 2], p=[0.80, 0.20])
                else:
                    qty = self.rng.choice([1, 2, 3], p=[0.50, 0.35, 0.15])
                
                transactions.append({
                    "order_id": order_id,
                    "item_name": item_name,
                    "qty": qty,
                    "order_datetime": timestamp,
                    "staff_name": staff_name
                })
                
                order_id += 1
        
        return pd.DataFrame(transactions)
    
    def _generate_waste_with_truth(self, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Generate waste data with higher rates for high_waste_risk items"""
        waste_records = []
        
        for _, item in menu_df.iterrows():
            item_name = item["item_name"]
            
            # Check if item is tagged as high_waste_risk
            truth_item = self.ground_truth["items"][
                self.ground_truth["items"]["item_name"] == item_name
            ]
            
            if len(truth_item) > 0:
                tags = truth_item.iloc[0]["tags"]
                
                if "high_waste_risk" in tags:
                    # High waste items: definitely have waste
                    waste_qty = self.rng.integers(15, 40)
                    waste_records.append({
                        "item_name": item_name,
                        "waste_qty": waste_qty
                    })
                elif self.rng.random() < 0.10:  # 10% of other items have some waste
                    waste_qty = self.rng.integers(2, 10)
                    waste_records.append({
                        "item_name": item_name,
                        "waste_qty": waste_qty
                    })
        
        return pd.DataFrame(waste_records)
    
    def _generate_staff_summary(self, sales_df: pd.DataFrame, menu_df: pd.DataFrame) -> pd.DataFrame:
        """Generate staff performance summary from sales transactions"""
        # Merge sales with menu to get prices
        sales_with_prices = sales_df.merge(
            menu_df[["item_name", "sell_price", "cost_per_unit"]], 
            on="item_name", 
            how="left"
        )
        
        # Calculate revenue and GP per transaction
        sales_with_prices["revenue"] = sales_with_prices["qty"] * sales_with_prices["sell_price"]
        sales_with_prices["cost"] = sales_with_prices["qty"] * sales_with_prices["cost_per_unit"]
        sales_with_prices["gp"] = sales_with_prices["revenue"] - sales_with_prices["cost"]
        
        # Group by staff
        staff_summary = sales_with_prices.groupby("staff_name").agg({
            "order_id": "count",
            "qty": "sum",
            "revenue": "sum",
            "gp": "sum"
        }).reset_index()
        
        staff_summary.columns = ["staff_name", "orders_handled", "units_sold", "revenue", "gp"]
        
        # Add roles and performance (from roster)
        roles = {
            "Alice": ("Server", "high_performer"),
            "Bob": ("Server", "average"),
            "Charlie": ("Bartender", "high_performer"),
            "Diana": ("Server", "average"),
            "Eve": ("Head Waiter", "high_performer"),
            "Frank": ("Server", "trainee"),
            "Grace": ("Server", "average"),
            "Henry": ("Manager", "high_performer")
        }
        
        staff_summary["role"] = staff_summary["staff_name"].map(lambda x: roles.get(x, ("Server", "average"))[0])
        staff_summary["performance_level"] = staff_summary["staff_name"].map(lambda x: roles.get(x, ("Server", "average"))[1])
        
        # Calculate per-hour metrics (assume 8-hour shifts, ~250 days/year)
        total_days = len(sales_df["order_datetime"].dt.date.unique()) if len(sales_df) > 0 else 1
        shifts_worked = total_days * 0.6  # Assume ~60% attendance
        hours_worked = shifts_worked * 8
        
        staff_summary["revenue_per_hour"] = staff_summary["revenue"] / hours_worked
        staff_summary["gp_per_hour"] = staff_summary["gp"] / hours_worked
        
        return staff_summary
    
    def _generate_bookings(self, sales_df: pd.DataFrame, start_date: datetime, days: int) -> pd.DataFrame:
        """Generate bookings data tied to sales volume"""
        bookings = []
        booking_id = 1
        
        # Calculate avg daily covers from sales
        if len(sales_df) > 0:
            daily_covers = sales_df.groupby(sales_df["order_datetime"].dt.date)["qty"].sum()
            avg_daily_covers = daily_covers.mean()
        else:
            avg_daily_covers = 100
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            dow = current_date.weekday()
            
            # Busier on weekends
            if dow in [4, 5]:  # Fri, Sat
                daily_bookings = self.rng.integers(15, 25)
            elif dow == 6:  # Sun
                daily_bookings = self.rng.integers(10, 15)
            else:
                daily_bookings = self.rng.integers(5, 12)
            
            for _ in range(daily_bookings):
                # Booking time (mostly evening)
                hour = int(self.rng.choice([18, 19, 20, 21], p=[0.25, 0.35, 0.30, 0.10]))
                minute = int(self.rng.choice([0, 15, 30, 45]))
                
                booking_datetime = current_date + timedelta(hours=hour, minutes=minute)
                
                # Covers (party size)
                covers = self.rng.choice([2, 3, 4, 5, 6, 8], p=[0.40, 0.15, 0.25, 0.10, 0.07, 0.03])
                
                # Status (realistic no-show rate 8-12%)
                status_roll = self.rng.random()
                if status_roll < 0.85:
                    status = "show"
                elif status_roll < 0.95:
                    status = "no_show"
                else:
                    status = "cancelled"
                
                bookings.append({
                    "booking_id": booking_id,
                    "booking_datetime": booking_datetime,
                    "covers": covers,
                    "status": status,
                    "day_of_week": current_date.strftime("%A")
                })
                
                booking_id += 1
        
        return pd.DataFrame(bookings)
    
    def _apply_quality_issues(
        self, 
        menu_df: pd.DataFrame, 
        sales_df: pd.DataFrame, 
        waste_df: pd.DataFrame, 
        quality_level: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Apply realistic data quality issues based on quality level"""
        menu = menu_df.copy()
        sales = sales_df.copy()
        waste = waste_df.copy()
        
        if quality_level == "typical":
            # Minor issues (5-10% affected)
            
            # Menu: case variations
            n_case = int(len(menu) * 0.10)
            if n_case > 0:
                idx = self.rng.choice(len(menu), size=n_case, replace=False)
                for i in idx:
                    cat = menu.loc[i, "category"]
                    menu.loc[i, "category"] = self.rng.choice([cat.lower(), cat.upper()])
            
            # Menu: whitespace
            n_space = int(len(menu) * 0.05)
            if n_space > 0:
                idx = self.rng.choice(len(menu), size=n_space, replace=False)
                for i in idx:
                    menu.loc[i, "item_name"] = "  " + menu.loc[i, "item_name"] + "  "
            
            # Sales: currency symbols in numeric fields (convert to string)
            n_currency = int(len(sales) * 0.03)
            if n_currency > 0:
                # Note: This would break qty as numeric, so we'll skip for sales
                pass
            
        elif quality_level == "chaotic":
            # Major issues (10-20% affected)
            
            # Menu: missing categories
            n_missing = int(len(menu) * 0.05)
            if n_missing > 0:
                idx = self.rng.choice(len(menu), size=n_missing, replace=False)
                menu.loc[idx, "category"] = ""
            
            # Menu: duplicate items
            n_dup = int(len(menu) * 0.05)
            if n_dup > 0:
                dup_rows = menu.sample(n=n_dup, random_state=self.rng.integers(0, 10000))
                menu = pd.concat([menu, dup_rows], ignore_index=True)
            
            # Sales: invalid dates
            n_bad_dates = int(len(sales) * 0.10)
            if n_bad_dates > 0:
                idx = self.rng.choice(len(sales), size=n_bad_dates, replace=False)
                sales.loc[idx, "order_datetime"] = "invalid_date"
            
            # Sales: negative quantities
            n_neg = int(len(sales) * 0.01)
            if n_neg > 0:
                idx = self.rng.choice(len(sales), size=n_neg, replace=False)
                sales.loc[idx, "qty"] = -abs(sales.loc[idx, "qty"])
            
            # Waste: negative waste
            n_neg_waste = int(len(waste) * 0.05)
            if n_neg_waste > 0 and len(waste) > 0:
                idx = self.rng.choice(len(waste), size=n_neg_waste, replace=False)
                waste.loc[idx, "waste_qty"] = -abs(waste.loc[idx, "waste_qty"])
        
        return menu, sales, waste
    
    def _finalize_ground_truth(self) -> Dict:
        """Finalize ground truth with expected metrics"""
        items_df = self.ground_truth["items"]
        
        # Calculate expected metrics
        star_items = items_df[items_df["expected_quadrant"] == "Star"]
        underpriced_items = items_df[items_df["is_opportunity"] == True]
        high_waste_items = items_df[items_df["tags"].apply(lambda x: "high_waste_risk" in x if isinstance(x, list) else False)]
        anchor_items = items_df[items_df["tags"].apply(lambda x: "anchor_item" in x if isinstance(x, list) else False)]
        overpriced_items = items_df[items_df["tags"].apply(lambda x: "overpriced_low_volume" in x if isinstance(x, list) else False)]
        
        return {
            "items": items_df,
            "expected_metrics": {
                "total_star_items": len(star_items),
                "total_underpriced_items": len(underpriced_items),
                "high_waste_items": high_waste_items["item_name"].tolist(),
                "anchor_items": anchor_items["item_name"].tolist(),
                "overpriced_low_volume_items": overpriced_items["item_name"].tolist()
            }
        }


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def validate_ground_truth(
    dataset: Tuple,
    engine_output: Dict
) -> Dict[str, bool]:
    """
    Validate that engine output matches ground truth expectations.
    
    Parameters
    ----------
    dataset : tuple
        Output from generate_restaurant_dataset()
        (menu_df, sales_df, waste_df, staff_df, bookings_df, ground_truth)
    engine_output : dict
        Output from RestaurantIntelligenceEngine analysis
        Must contain keys for detected patterns
        
    Returns
    -------
    dict
        Assertion-friendly results:
        {
            "stars_matched": bool,
            "dogs_matched": bool,
            "high_waste_detected": bool,
            "opportunities_found": bool,
            "all_checks_passed": bool
        }
        
    Examples
    --------
    >>> dataset = generate_restaurant_dataset("Test", "burger_bar")
    >>> engine = RestaurantIntelligenceEngine(dataset[0], dataset[1], dataset[2])
    >>> results = engine.analyze()
    >>> validation = validate_ground_truth(dataset, results)
    >>> assert validation["all_checks_passed"]
    """
    _, _, _, _, _, ground_truth = dataset
    
    results = {
        "stars_matched": False,
        "dogs_matched": False,
        "high_waste_detected": False,
        "opportunities_found": False,
        "all_checks_passed": False
    }
    
    # Extract expected items
    truth_items = ground_truth["items"]
    expected_stars = truth_items[truth_items["expected_quadrant"] == "Star"]["item_name"].tolist()
    expected_dogs = truth_items[truth_items["expected_quadrant"] == "Dog"]["item_name"].tolist()
    expected_high_waste = ground_truth["expected_metrics"]["high_waste_items"]
    expected_opportunities = truth_items[truth_items["is_opportunity"] == True]["item_name"].tolist()
    
    # Validate (with flexible matching since engine may not detect all)
    if "stars" in engine_output:
        detected_stars = [item for item in expected_stars if item in engine_output["stars"]]
        results["stars_matched"] = len(detected_stars) >= len(expected_stars) * 0.7  # 70% threshold
    
    if "dogs" in engine_output:
        detected_dogs = [item for item in expected_dogs if item in engine_output["dogs"]]
        results["dogs_matched"] = len(detected_dogs) >= len(expected_dogs) * 0.5  # 50% threshold
    
    if "high_waste_items" in engine_output:
        detected_waste = [item for item in expected_high_waste if item in engine_output["high_waste_items"]]
        results["high_waste_detected"] = len(detected_waste) >= len(expected_high_waste) * 0.6
    
    if "opportunities" in engine_output:
        detected_opps = [item for item in expected_opportunities if item in engine_output["opportunities"]]
        results["opportunities_found"] = len(detected_opps) >= len(expected_opportunities) * 0.6
    
    # Overall check
    results["all_checks_passed"] = all([
        results["stars_matched"],
        results["dogs_matched"],
        results["high_waste_detected"],
        results["opportunities_found"]
    ])
    
    return results


# =============================================================================
# BACKWARD COMPATIBLE CLI INTERFACE
# =============================================================================

def main():
    """CLI interface with backward compatibility"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate ultra-realistic restaurant datasets with ground truth"
    )
    
    # New parameters
    parser.add_argument("--restaurant-name", default="Test Restaurant", 
                        help="Restaurant name")
    parser.add_argument("--restaurant-type", 
                        choices=["casual_brasserie", "burger_bar", "dessert_cafe", "gastro_pub"],
                        default="casual_brasserie",
                        help="Restaurant archetype")
    parser.add_argument("--start-date", default="2024-01-01", 
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2024-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", "-o", default="data/synthetic",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Backward compatible parameters (mapped to new system)
    parser.add_argument("--menu-size", choices=["small", "medium", "large"],
                        help="(Legacy) Menu size - overrides restaurant type sizing")
    parser.add_argument("--sales-volume", choices=["low", "medium", "high"],
                        help="(Legacy) Sales volume - ignored in new system")
    parser.add_argument("--quality", "--quality-level",
                        choices=["pristine", "typical", "chaotic", "corrupted"],
                        default="typical", dest="quality_level",
                        help="Data quality level")
    parser.add_argument("--days", type=int,
                        help="Days of data - overrides start/end dates")
    
    args = parser.parse_args()
    
    # Handle backward compatibility
    if args.days:
        args.end_date = (datetime.strptime(args.start_date, "%Y-%m-%d") + 
                         timedelta(days=args.days-1)).strftime("%Y-%m-%d")
    
    quality = args.quality_level
    if quality == "corrupted":
        quality = "chaotic"  # Map old term to new
    
    # Generate dataset
    generator = UltraRealisticRestaurantGenerator(seed=args.seed)
    
    menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
        restaurant_name=args.restaurant_name,
        restaurant_type=args.restaurant_type,
        start_date=args.start_date,
        end_date=args.end_date,
        quality_level=quality,
        random_seed=args.seed
    )
    
    # Save files
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{args.restaurant_type}_{quality}"
    
    menu.to_csv(output_dir / f"menu_{prefix}.csv", index=False)
    sales.to_csv(output_dir / f"sales_{prefix}.csv", index=False)
    waste.to_csv(output_dir / f"waste_{prefix}.csv", index=False)
    staff.to_csv(output_dir / f"staff_{prefix}.csv", index=False)
    bookings.to_csv(output_dir / f"bookings_{prefix}.csv", index=False)
    
    # Save ground truth separately
    truth["items"].to_csv(output_dir / f"ground_truth_{prefix}.csv", index=False)
    
    import json
    with open(output_dir / f"expected_metrics_{prefix}.json", "w") as f:
        json.dump(truth["expected_metrics"], f, indent=2)
    
    print(f"\n✅ Complete dataset saved to {output_dir}/")
    print(f"\n📊 Ground Truth Summary:")
    print(f"   Star items (expected): {truth['expected_metrics']['total_star_items']}")
    print(f"   Underpriced items: {truth['expected_metrics']['total_underpriced_items']}")
    print(f"   High waste items: {len(truth['expected_metrics']['high_waste_items'])}")
    print(f"   Anchor items: {len(truth['expected_metrics']['anchor_items'])}")
    print(f"   Overpriced low-volume: {len(truth['expected_metrics']['overpriced_low_volume_items'])}")


if __name__ == "__main__":
    main()
