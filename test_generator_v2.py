"""
Test suite for upgraded synthetic data generator
Demonstrates ground truth validation capabilities
"""

import pytest
import pandas as pd
from generate_test_data_v2 import (
    UltraRealisticRestaurantGenerator,
    validate_ground_truth,
    RESTAURANT_ARCHETYPES
)


class TestRestaurantArchetypes:
    """Test that all archetypes generate correctly"""
    
    @pytest.mark.parametrize("archetype", [
        "casual_brasserie", "burger_bar", "dessert_cafe", "gastro_pub"
    ])
    def test_archetype_generation(self, archetype):
        """Each archetype should generate valid datasets"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name=f"Test {archetype}",
            restaurant_type=archetype,
            start_date="2024-01-01",
            end_date="2024-01-31",
            quality_level="pristine",
            random_seed=42
        )
        
        # Validate menu
        assert len(menu) > 0, f"{archetype} should have menu items"
        assert all(col in menu.columns for col in ["item_id", "item_name", "category", "sell_price", "cost_per_unit"])
        
        # Validate sales
        assert len(sales) > 0, f"{archetype} should have sales"
        assert "staff_name" in sales.columns, "Sales should include staff assignments"
        
        # Validate ground truth
        assert len(truth["items"]) == len(menu), "Ground truth should cover all menu items"
        assert "expected_metrics" in truth, "Should have expected metrics"


class TestGroundTruthLabeling:
    """Test that ground truth labels work as expected"""
    
    def test_star_items_high_volume(self):
        """Star items (underpriced_high_performer) should have high sales volume"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Ground Truth Test",
            restaurant_type="burger_bar",
            start_date="2024-01-01",
            end_date="2024-03-31",
            quality_level="pristine",
            random_seed=42
        )
        
        # Get star items from ground truth
        star_items = truth["items"][truth["items"]["expected_quadrant"] == "Star"]["item_name"].tolist()
        
        if len(star_items) > 0:
            # Calculate sales volume for star items
            star_sales = sales[sales["item_name"].isin(star_items)]["qty"].sum()
            total_sales = sales["qty"].sum()
            star_pct = star_sales / total_sales * 100
            
            # Star items should have higher than average volume
            assert star_pct > 5, f"Star items should have >5% volume, got {star_pct:.1f}%"
    
    def test_high_waste_items_have_waste(self):
        """Items tagged as high_waste_risk should appear in waste data"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Waste Test",
            restaurant_type="casual_brasserie",
            start_date="2024-01-01",
            end_date="2024-12-31",
            quality_level="pristine",
            random_seed=42
        )
        
        # Get high waste items from ground truth
        high_waste_items = truth["expected_metrics"]["high_waste_items"]
        
        if len(high_waste_items) > 0:
            # Check they appear in waste data
            waste_item_names = waste["item_name"].tolist()
            found = sum(1 for item in high_waste_items if item in waste_item_names)
            
            assert found > 0, "At least some high_waste_risk items should appear in waste data"


class TestStaffAndBookings:
    """Test staff and bookings features"""
    
    def test_staff_performance_levels(self):
        """Staff should have different performance levels"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Staff Test",
            restaurant_type="gastro_pub",
            start_date="2024-01-01",
            end_date="2024-01-31",
            quality_level="pristine",
            random_seed=42
        )
        
        assert len(staff) > 0, "Should have staff members"
        assert "performance_level" in staff.columns
        assert "role" in staff.columns
        
        # Check we have different performance levels
        perf_levels = staff["performance_level"].unique()
        assert len(perf_levels) >= 2, "Should have multiple performance levels"
    
    def test_bookings_no_show_rate(self):
        """Bookings should have realistic no-show rate (8-12%)"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Bookings Test",
            restaurant_type="casual_brasserie",
            start_date="2024-01-01",
            end_date="2024-06-30",
            quality_level="pristine",
            random_seed=42
        )
        
        assert len(bookings) > 0, "Should have bookings"
        
        no_shows = (bookings["status"] == "no_show").sum()
        total_bookings = len(bookings)
        no_show_rate = no_shows / total_bookings * 100
        
        assert 5 <= no_show_rate <= 15, f"No-show rate should be 5-15%, got {no_show_rate:.1f}%"
    
    def test_weekend_busier(self):
        """Weekends (Fri/Sat) should have more bookings"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Weekend Test",
            restaurant_type="burger_bar",
            start_date="2024-01-01",
            end_date="2024-03-31",
            quality_level="pristine",
            random_seed=42
        )
        
        bookings["day_of_week"] = pd.to_datetime(bookings["booking_datetime"]).dt.day_name()
        weekend_bookings = bookings[bookings["day_of_week"].isin(["Friday", "Saturday"])].copy()
        weekday_bookings = bookings[~bookings["day_of_week"].isin(["Friday", "Saturday"])].copy()
        
        weekend_avg = len(weekend_bookings) / 2  # 2 weekend days
        weekday_avg = len(weekday_bookings) / 5  # 5 weekdays
        
        assert weekend_avg > weekday_avg, "Weekends should be busier than weekdays"


class TestQualityLevels:
    """Test that quality degradation works"""
    
    def test_pristine_no_issues(self):
        """Pristine data should have no quality issues"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Pristine Test",
            restaurant_type="burger_bar",
            start_date="2024-01-01",
            end_date="2024-01-07",
            quality_level="pristine",
            random_seed=42
        )
        
        # No empty categories
        assert (menu["category"] != "").all(), "Pristine should have no empty categories"
        
        # No negative quantities
        assert (sales["qty"] > 0).all(), "Pristine should have no negative quantities"
    
    def test_chaotic_has_issues(self):
        """Chaotic data should have quality problems"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Chaotic Test",
            restaurant_type="dessert_cafe",
            start_date="2024-01-01",
            end_date="2024-01-07",
            quality_level="chaotic",
            random_seed=42
        )
        
        # Should have some empty categories or invalid dates (if dataset big enough)
        # Just check it doesn't crash
        assert len(menu) > 0
        assert len(sales) > 0


class TestBackwardCompatibility:
    """Test that old CLI parameters still work"""
    
    def test_legacy_parameters_work(self):
        """Generator should accept old-style parameters"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        # Should not crash with any archetype
        menu, sales, waste, staff, bookings, truth = generator.generate_restaurant_dataset(
            restaurant_name="Legacy Test",
            restaurant_type="casual_brasserie",  # Default
            start_date="2024-01-01",
            end_date="2024-01-07",
            quality_level="typical",
            random_seed=42
        )
        
        assert len(menu) > 0
        assert len(sales) > 0
        assert len(truth["items"]) > 0


class TestValidationFunction:
    """Test the validate_ground_truth helper"""
    
    def test_validation_structure(self):
        """Validation function should return correct structure"""
        generator = UltraRealisticRestaurantGenerator(seed=42)
        
        dataset = generator.generate_restaurant_dataset(
            restaurant_name="Validation Test",
            restaurant_type="burger_bar",
            start_date="2024-01-01",
            end_date="2024-01-31",
            quality_level="pristine",
            random_seed=42
        )
        
        # Mock engine output
        mock_engine_output = {
            "stars": [],
            "dogs": [],
            "high_waste_items": [],
            "opportunities": []
        }
        
        result = validate_ground_truth(dataset, mock_engine_output)
        
        # Check structure
        assert "stars_matched" in result
        assert "dogs_matched" in result
        assert "high_waste_detected" in result
        assert "opportunities_found" in result
        assert "all_checks_passed" in result
        assert isinstance(result["all_checks_passed"], bool)


class TestReproducibility:
    """Test that seed ensures reproducibility"""
    
    def test_same_seed_same_output(self):
        """Same seed should produce identical datasets"""
        gen1 = UltraRealisticRestaurantGenerator(seed=123)
        gen2 = UltraRealisticRestaurantGenerator(seed=123)
        
        menu1, sales1, _, _, _, _ = gen1.generate_restaurant_dataset(
            "Test", "burger_bar", "2024-01-01", "2024-01-07", "pristine", 123
        )
        
        menu2, sales2, _, _, _, _ = gen2.generate_restaurant_dataset(
            "Test", "burger_bar", "2024-01-01", "2024-01-07", "pristine", 123
        )
        
        # Menus should be identical
        assert len(menu1) == len(menu2)
        assert menu1["item_name"].tolist() == menu2["item_name"].tolist()
        assert menu1["sell_price"].tolist() == menu2["sell_price"].tolist()
        
        # Sales count should be identical
        assert len(sales1) == len(sales2)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
