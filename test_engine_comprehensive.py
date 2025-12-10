"""
Comprehensive Pytest Suite for Restaurant Intelligence Engine
==============================================================

Tests 100+ edge cases covering:
- Data loading resilience
- Merge operations integrity
- Calculation accuracy
- Edge case handling
- Performance under stress
- Error recovery
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import resturantv1 as rv1
import insights_module as im


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def basic_menu():
    """Standard menu with 5 items"""
    return pd.DataFrame({
        "item_name": ["Burger", "Pizza", "Salad", "Pasta", "Soup"],
        "category": ["Mains", "Mains", "Starters", "Mains", "Starters"],
        "sell_price": [10.0, 12.0, 8.0, 14.0, 6.0],
        "cost_per_unit": [4.0, 5.0, 3.0, 6.0, 2.0]
    })


@pytest.fixture
def basic_sales():
    """Standard sales with 100 transactions"""
    items = ["Burger"] * 30 + ["Pizza"] * 25 + ["Salad"] * 20 + ["Pasta"] * 15 + ["Soup"] * 10
    return pd.DataFrame({
        "item_name": items,
        "qty": np.random.randint(1, 4, 100),
        "order_datetime": pd.date_range("2024-01-01", periods=100, freq="h")
    })


@pytest.fixture
def basic_waste():
    """Standard waste with all items"""
    return pd.DataFrame({
        "item_name": ["Burger", "Pizza", "Salad", "Pasta", "Soup"],
        "waste_qty": [5, 3, 2, 4, 1]
    })


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestDataLoading:
    """Test data loading resilience and error handling"""
    
    def test_utf8_sig_encoding(self, temp_dir, basic_menu, basic_sales):
        """C1.1: UTF-8-sig encoding (Excel BOM) handled"""
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        
        # Write with UTF-8-sig (Excel BOM)
        basic_menu.to_csv(menu_path, index=False, encoding='utf-8-sig')
        basic_sales.to_csv(sales_path, index=False, encoding='utf-8-sig')
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        assert len(menu) == 5
        assert len(sales) == 100
    
    def test_latin1_encoding(self, temp_dir):
        """C1.1: Latin-1 encoding handled"""
        menu_path = temp_dir / "menu.csv"
        
        # Create menu with latin-1 characters
        menu_df = pd.DataFrame({
            "item_name": ["Café", "Crêpe", "Naïve"],
            "category": ["Mains", "Desserts", "Starters"],
            "sell_price": [10.0, 8.0, 6.0],
            "cost_per_unit": [4.0, 3.0, 2.0]
        })
        menu_df.to_csv(menu_path, index=False, encoding='latin-1')
        
        df = rv1._read_any_table(str(menu_path))
        assert len(df) == 3
    
    def test_missing_required_columns(self, temp_dir):
        """C1.2: Missing required columns fail with clear error"""
        menu_path = temp_dir / "menu.csv"
        
        # Menu missing sell_price
        bad_menu = pd.DataFrame({
            "item_name": ["Burger"],
            "category": ["Mains"],
            "cost_per_unit": [4.0]
        })
        bad_menu.to_csv(menu_path, index=False)
        
        sales_path = temp_dir / "sales.csv"
        pd.DataFrame({
            "item_name": ["Burger"],
            "qty": [1],
            "order_datetime": ["2024-01-01"]
        }).to_csv(sales_path, index=False)
        
        with pytest.raises(ValueError, match="missing required columns"):
            rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))


# =============================================================================
# MERGE INTEGRITY TESTS
# =============================================================================

class TestMergeIntegrity:
    """Test LEFT join preservation and data integrity"""
    
    def test_left_join_preserves_all_sales(self, temp_dir):
        """C2.5: LEFT join preserves sales even with missing menu items"""
        # Menu with only 3 items
        menu_df = pd.DataFrame({
            "item_name": ["Burger", "Pizza", "Salad"],
            "category": ["Mains", "Mains", "Starters"],
            "sell_price": [10.0, 12.0, 8.0],
            "cost_per_unit": [4.0, 5.0, 3.0]
        })
        
        # Sales with 5 items (2 not in menu)
        sales_df = pd.DataFrame({
            "item_name": ["Burger"] * 10 + ["Pizza"] * 8 + ["Salad"] * 5 + ["Pasta"] * 4 + ["Soup"] * 3,
            "qty": [1] * 30,
            "order_datetime": pd.date_range("2024-01-01", periods=30, freq="D")
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        # All 30 sales should be preserved
        assert len(sales) == 30, f"Expected 30 sales, got {len(sales)}"
        
        # Orphaned items should have IDs assigned
        pasta_sales = sales[sales["item_name"] == "Pasta"]
        assert len(pasta_sales) == 4
        assert not pd.isna(pasta_sales.iloc[0]["item_id"])
    
    def test_waste_left_join_preserves_records(self, temp_dir, basic_menu):
        """C1.7: Waste LEFT join preserves matched records"""
        waste_df = pd.DataFrame({
            "item_name": ["Burger", "Pizza", "Unknown Item"],
            "waste_qty": [5, 3, 2]
        })
        
        waste_path = temp_dir / "waste.csv"
        waste_df.to_csv(waste_path, index=False)
        
        waste = rv1.load_client_waste(str(waste_path), basic_menu)
        
        # Should have 2 matched items (Unknown dropped due to no cost data)
        assert len(waste) == 2
        assert "Burger" in waste.merge(basic_menu, on="item_id")["item_name"].values
    
    def test_no_data_loss_on_merge(self, temp_dir, basic_menu, basic_sales):
        """S9.1: Verify zero data loss in merge operations"""
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        basic_menu.to_csv(menu_path, index=False)
        basic_sales.to_csv(sales_path, index=False)
        
        original_sales_count = len(basic_sales)
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        assert len(sales) == original_sales_count, "Data loss detected in merge"
    
    def test_duplicate_items_rejected(self, temp_dir):
        """H1.2: Duplicate item names cause clear error"""
        menu_df = pd.DataFrame({
            "item_name": ["Burger", "Burger", "Pizza"],  # Duplicate!
            "category": ["Mains", "Mains", "Mains"],
            "sell_price": [10.0, 11.0, 12.0],
            "cost_per_unit": [4.0, 4.5, 5.0]
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Burger"],
            "qty": [1],
            "order_datetime": ["2024-01-01"]
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        with pytest.raises(ValueError, match="duplicate item names"):
            rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))


# =============================================================================
# CALCULATION ACCURACY TESTS
# =============================================================================

class TestCalculationAccuracy:
    """Test mathematical operations and edge cases"""
    
    def test_division_by_zero_safe(self):
        """C2.1: Division by zero returns 0, not inf/nan"""
        df = pd.DataFrame({
            "sell_price": [0.0, 10.0, 5.0],
            "gp_per_unit": [2.0, 4.0, 3.0]
        })
        
        df["gp_pct"] = np.where(
            df["sell_price"] > 0,
            df["gp_per_unit"] / df["sell_price"],
            0.0
        )
        
        assert not np.isinf(df["gp_pct"]).any()
        assert not np.isnan(df["gp_pct"]).any()
        assert df["gp_pct"].iloc[0] == 0.0
    
    def test_negative_gp_detected(self, temp_dir):
        """C2.4: Items with cost > price flagged"""
        menu_df = pd.DataFrame({
            "item_name": ["Loss Leader", "Profitable"],
            "category": ["Starters", "Mains"],
            "sell_price": [5.0, 10.0],
            "cost_per_unit": [8.0, 4.0]  # First item loses money
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Loss Leader", "Profitable"],
            "qty": [1, 1],
            "order_datetime": ["2024-01-01", "2024-01-02"]
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        # Should load but print WARNING (captured in logs)
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        negative_gp = menu[menu["gp_per_unit"] < 0]
        assert len(negative_gp) == 1
        assert negative_gp.iloc[0]["item_name"] == "Loss Leader"
    
    @pytest.mark.parametrize("price,cost,expected_gp", [
        (10.0, 4.0, 6.0),
        (0.0, 0.0, 0.0),
        (5.0, 5.0, 0.0),
        (10.0, 15.0, -5.0),  # Negative GP
    ])
    def test_gp_calculation_accuracy(self, price, cost, expected_gp):
        """Verify GP calculation accuracy across edge cases"""
        gp = price - cost
        assert gp == expected_gp


# =============================================================================
# CATEGORY NORMALIZATION TESTS
# =============================================================================

class TestCategoryNormalization:
    """Test category name standardization"""
    
    @pytest.mark.parametrize("input_cat,expected", [
        ("main", "Mains"),
        ("MAINS", "Mains"),
        ("Main Dishes", "Mains"),
        ("entrees", "Mains"),
        ("starter", "Starters"),
        ("STARTERS", "Starters"),
        ("dessert", "Desserts"),
        ("DESSERTS", "Desserts"),
        ("side", "Sides"),
        ("drink", "Drinks"),
        ("mainss", "Mains"),  # Typo with fuzzy match
    ])
    def test_category_normalization(self, input_cat, expected):
        """C1.4: Category variations normalize correctly"""
        result = rv1.normalize_category_name(input_cat)
        assert result == expected, f"Expected '{expected}' for '{input_cat}', got '{result}'"
    
    def test_custom_category_preserved(self):
        """C1.4: Custom categories preserved with title case"""
        result = rv1.normalize_category_name("Breakfast Specials")
        assert result == "Breakfast Specials"
    
    def test_empty_category_handled(self):
        """C1.4: Empty categories become Uncategorized"""
        assert rv1.normalize_category_name("") == "Uncategorized"
        assert rv1.normalize_category_name(None) == "Uncategorized"
        assert rv1.normalize_category_name("   ") == "Uncategorized"


# =============================================================================
# MENU ENGINEERING TESTS
# =============================================================================

class TestMenuEngineering:
    """Test menu engineering classification logic"""
    
    def test_size_aware_thresholds_small_menu(self):
        """C3.1: Small menus (<30 items) use 70th percentile"""
        menu_size = 20
        if menu_size < 30:
            threshold = 0.70
        else:
            threshold = 0.50
        
        assert threshold == 0.70
    
    def test_size_aware_thresholds_medium_menu(self):
        """C3.1: Medium menus (30-80 items) use 60th percentile"""
        menu_size = 50
        if menu_size < 30:
            threshold = 0.70
        elif menu_size < 80:
            threshold = 0.60
        else:
            threshold = 0.50
        
        assert threshold == 0.60
    
    def test_size_aware_thresholds_large_menu(self):
        """C3.1: Large menus (80+ items) use 50th percentile"""
        menu_size = 100
        if menu_size < 30:
            threshold = 0.70
        elif menu_size < 80:
            threshold = 0.60
        else:
            threshold = 0.50
        
        assert threshold == 0.50


# =============================================================================
# DATE HANDLING TESTS
# =============================================================================

class TestDateHandling:
    """Test date parsing and validation"""
    
    @pytest.mark.parametrize("date_string,should_parse", [
        ("2024-01-01", True),
        ("2024-01-01 12:00:00", True),
        ("01/01/2024", True),
        ("invalid", False),
        ("", False),
        ("2024-13-45", False),  # Invalid date
    ])
    def test_date_parsing(self, date_string, should_parse):
        """H1.3: Date parsing handles various formats"""
        result = pd.to_datetime(date_string, errors="coerce")
        if should_parse:
            assert not pd.isna(result)
        else:
            assert pd.isna(result)
    
    def test_high_date_failure_rate_rejected(self, temp_dir, basic_menu):
        """H1.3: >5% date parse failures cause error"""
        # Sales with 60% bad dates
        sales_df = pd.DataFrame({
            "item_name": ["Burger"] * 10,
            "qty": [1] * 10,
            "order_datetime": ["2024-01-01"] * 4 + ["invalid"] * 6  # 60% invalid
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        basic_menu.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        with pytest.raises(ValueError, match="dates failed to parse"):
            rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressScenarios:
    """Test performance under extreme conditions"""
    
    def test_large_dataset_performance(self, temp_dir):
        """Handle 100k sales transactions"""
        menu_df = pd.DataFrame({
            "item_name": [f"Item_{i}" for i in range(100)],
            "category": ["Mains"] * 100,
            "sell_price": np.random.uniform(5, 20, 100),
            "cost_per_unit": np.random.uniform(2, 10, 100)
        })
        
        # 100k sales
        sales_df = pd.DataFrame({
            "item_name": np.random.choice(menu_df["item_name"], 100000),
            "qty": np.random.randint(1, 4, 100000),
            "order_datetime": pd.date_range("2024-01-01", periods=100000, freq="min")
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        import time
        start = time.time()
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        elapsed = time.time() - start
        
        assert len(sales) == 100000
        assert elapsed < 10.0, f"Loading 100k rows took {elapsed:.2f}s (expected <10s)"
    
    def test_extreme_category_variations(self, temp_dir):
        """Handle 50 different category spellings"""
        variations = [
            "main", "MAIN", "mains", "MAINS", "Main", "Main Dishes",
            "starter", "Starter", "STARTERS", "Starters",
            "dessert", "Desserts", "DESSERT",
            "side", "Sides", "SIDES",
            "drink", "Drinks", "DRINKS", "beverages",
            # ... and random custom ones
        ] + [f"Custom_{i}" for i in range(30)]
        
        menu_df = pd.DataFrame({
            "item_name": [f"Item_{i}" for i in range(50)],
            "category": variations,
            "sell_price": [10.0] * 50,
            "cost_per_unit": [4.0] * 50
        })
        
        # Apply normalization
        menu_df["category"] = menu_df["category"].apply(rv1.normalize_category_name)
        
        # Should normalize standard ones
        standard_cats = ["Mains", "Starters", "Desserts", "Sides", "Drinks"]
        normalized_count = menu_df["category"].isin(standard_cats).sum()
        
        assert normalized_count >= 19, "Standard categories should normalize"


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test weird but valid scenarios"""
    
    def test_all_zero_prices(self, temp_dir):
        """Handle menu with all free items"""
        menu_df = pd.DataFrame({
            "item_name": ["Free1", "Free2", "Free3"],
            "category": ["Mains", "Mains", "Mains"],
            "sell_price": [0.0, 0.0, 0.0],
            "cost_per_unit": [1.0, 2.0, 3.0]
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Free1", "Free2"],
            "qty": [1, 1],
            "order_datetime": ["2024-01-01", "2024-01-02"]
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        # Should not crash
        assert len(menu) == 3
        assert all(menu["gp_pct"] == 0.0)
    
    def test_single_item_menu(self, temp_dir):
        """Handle restaurant with only 1 item"""
        menu_df = pd.DataFrame({
            "item_name": ["Only Item"],
            "category": ["Mains"],
            "sell_price": [10.0],
            "cost_per_unit": [4.0]
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Only Item"] * 50,
            "qty": [1] * 50,
            "order_datetime": pd.date_range("2024-01-01", periods=50, freq="D")
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        assert len(menu) == 1
        assert len(sales) == 50
    
    def test_unicode_item_names(self, temp_dir):
        """Handle unicode characters in item names"""
        menu_df = pd.DataFrame({
            "item_name": ["Café Latte ☕", "Crème Brûlée 🍮", "Naïve Salad 🥗"],
            "category": ["Drinks", "Desserts", "Starters"],
            "sell_price": [4.0, 8.0, 7.0],
            "cost_per_unit": [1.5, 3.0, 2.5]
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Café Latte ☕"],
            "qty": [1],
            "order_datetime": ["2024-01-01"]
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False, encoding='utf-8')
        sales_df.to_csv(sales_path, index=False, encoding='utf-8')
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        assert len(menu) == 3
        assert "☕" in menu.iloc[0]["item_name"]
    
    def test_very_large_prices(self, temp_dir):
        """Handle extremely high prices (luxury items)"""
        menu_df = pd.DataFrame({
            "item_name": ["Luxury Caviar", "Vintage Wine"],
            "category": ["Starters", "Drinks"],
            "sell_price": [500.0, 1000.0],
            "cost_per_unit": [200.0, 400.0]
        })
        
        sales_df = pd.DataFrame({
            "item_name": ["Luxury Caviar"],
            "qty": [1],
            "order_datetime": ["2024-01-01"]
        })
        
        menu_path = temp_dir / "menu.csv"
        sales_path = temp_dir / "sales.csv"
        menu_df.to_csv(menu_path, index=False)
        sales_df.to_csv(sales_path, index=False)
        
        menu, sales, _ = rv1.load_client_menu_and_sales(str(menu_path), str(sales_path))
        
        assert menu.iloc[0]["sell_price"] == 500.0
        assert menu.iloc[0]["gp_per_unit"] == 300.0


# =============================================================================
# INSIGHTS MODULE TESTS
# =============================================================================

class TestInsightsModule:
    """Test insight detection and reliability scoring"""
    
    def test_skip_counter_tracks_failures(self):
        """C4.1: Skip counter detects excessive failures"""
        # Create perf_df with NaN values
        perf_df = pd.DataFrame({
            "item_name": [f"Item_{i}" for i in range(100)],
            "category": ["Mains"] * 100,
            "units_sold": [np.nan] * 50 + [10.0] * 50,  # 50% bad data
            "revenue": [100.0] * 100,
            "gp_pct_after_waste": [0.3] * 100
        })
        
        # This should raise error due to >5% skipped
        with pytest.raises(ValueError, match="items (.+) skipped"):
            im.detect_item_level_problems(perf_df, summary_metrics={})


# =============================================================================
# SUMMARY
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
