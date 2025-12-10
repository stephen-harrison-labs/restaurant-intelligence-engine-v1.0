# Upgraded Restaurant Data Generator - Summary

## ✅ IMPLEMENTATION COMPLETE

Successfully upgraded `generate_test_data_v2.py` with all requested features plus improvements.

---

## 🎯 What Was Built

### 1. Restaurant Archetypes (4 types)
Each with realistic characteristics:

**Casual Brasserie:**
- Menu: 35-50 items, French-inspired (Steak Frites, Moules Marinières, Crème Brûlée)
- Pricing: Starters £5.50-12, Mains £12-28, Premium drinks
- GP%: 55-75% by category
- Volume: 45% mains, 15% starters

**Burger Bar:**
- Menu: 25-35 items, American casual (Classic Beef Burger, Loaded Fries)
- Pricing: Starters £4.50-8.50, Mains £8-18, Affordable
- GP%: 50-95% (high on sides/drinks)
- Volume: 50% mains, 25% sides

**Dessert Cafe:**
- Menu: 30-45 items, dessert-focused (Triple Chocolate Cake, Tiramisu)
- Pricing: Desserts £5-12 (premium), Light mains
- GP%: 70-95% (high margins)
- Volume: 60% desserts, 20% drinks

**Gastro Pub:**
- Menu: 40-60 items, British classics (Sunday Roast, Fish & Chips, Sticky Toffee Pudding)
- Pricing: Starters £6-14, Mains £14-32, Craft drinks
- GP%: 55-90% by category
- Volume: 35% mains, 30% drinks

### 2. Ground Truth Labeling System
Every item tagged with role for automated testing:

**Tags:**
- `underpriced_high_performer` (10% of items) → Expected: **Star**
- `overpriced_low_volume` (8%) → Expected: **Dog**
- `high_waste_risk` (7%) → Higher waste quantities
- `anchor_item` (5%) → Premium reference pricing

**Output Structure:**
```python
ground_truth = {
    "items": pd.DataFrame([
        item_id, item_name, tags, expected_quadrant, 
        is_opportunity, is_risk, archetype_role
    ]),
    "expected_metrics": {
        "total_star_items": 2,
        "total_underpriced_items": 2,
        "high_waste_items": ["Chicken Burger", "Side Salad"],
        "anchor_items": [],
        "overpriced_low_volume_items": ["Veggie Burger", ...]
    }
}
```

### 3. Realistic Staff Data
8-member roster with scheduling:

**Roles:** Server, Bartender, Head Waiter, Manager

**Shifts:**
- Morning (10am-4pm): 2-3 servers
- Evening (4pm-11pm): 4-5 servers
- Manager: Both shifts

**Performance Levels:**
- `high_performer`: +15% avg ticket (Alice, Eve, Charlie, Henry)
- `average`: Baseline (Bob, Diana, Grace)
- `trainee`: -10% avg ticket, +8% waste (Frank)

**Output:** `staff_df` with orders_handled, revenue, gp, revenue_per_hour, gp_per_hour

### 4. Bookings Data Tied to Sales
Realistic reservation system:

**Volume Patterns:**
- Fri/Sat: 15-25 bookings/day
- Sunday: 10-15 bookings/day
- Weekdays: 5-12 bookings/day

**Realism:**
- Peak times: 6pm-9pm
- Party sizes: 2-8 covers (weighted realistic)
- No-show rate: 8-12% (validated: 8.8%)
- Status: show/no_show/cancelled

### 5. Main Function with All Parameters

```python
menu, sales, waste, staff, bookings, truth = generate_restaurant_dataset(
    restaurant_name="The Test Kitchen",
    restaurant_type="burger_bar",          # NEW: 4 archetypes
    start_date="2024-01-01",               # NEW: Flexible dates
    end_date="2024-12-31",
    quality_level="typical",               # pristine/typical/chaotic
    random_seed=42                         # Full reproducibility
)
```

**Returns:**
- `menu_df`: item_id, item_name, category, sell_price, cost_per_unit
- `sales_df`: order_id, item_name, qty, order_datetime, **staff_name** (NEW)
- `waste_df`: item_name, waste_qty
- `staff_df`: staff_name, role, orders_handled, revenue, gp, performance_level (NEW)
- `bookings_df`: booking_id, booking_datetime, covers, status, day_of_week (NEW)
- `ground_truth`: Items DataFrame + expected_metrics dict (NEW)

### 6. Validation Helper for Pytest

```python
def validate_ground_truth(dataset, engine_output) -> dict:
    """Compare engine results vs ground truth expectations"""
    return {
        "stars_matched": bool,         # Did engine detect star items?
        "dogs_matched": bool,          # Did engine detect dogs?
        "high_waste_detected": bool,   # Did engine catch high waste?
        "opportunities_found": bool,   # Did engine find opportunities?
        "all_checks_passed": bool      # Overall validation
    }
```

**Pytest Example:**
```python
def test_engine_detects_stars():
    dataset = generate_restaurant_dataset("Test", "burger_bar", quality_level="pristine")
    engine = RestaurantIntelligenceEngine(dataset[0], dataset[1], dataset[2])
    results = engine.analyze()
    
    validation = validate_ground_truth(dataset, results)
    assert validation["stars_matched"]
    assert validation["all_checks_passed"]
```

### 7. Backward Compatibility ✅
Old CLI parameters still work:

```bash
# Old way (still works)
python generate_test_data_v2.py --menu-size small --quality typical --days 7

# New way (enhanced)
python generate_test_data_v2.py --restaurant-type burger_bar --start-date 2024-01-01 --end-date 2024-12-31
```

### 8. Quality Levels
Three levels with realistic noise:

**Pristine:**
- Perfect data, no issues
- Use for: Golden path testing

**Typical (default):**
- 10% case variations ("mains" vs "Mains")
- 5% whitespace ("  Item Name  ")
- Light noise (2% seasonal items)
- Use for: Real-world robustness

**Chaotic:**
- 5% missing categories
- 10% invalid dates
- 5% duplicate items
- 1% negative quantities
- Use for: Stress testing edge cases

---

## 🧪 Validation Results

**All 4 Archetypes:** ✅ Generated successfully
- casual_brasserie: 35-50 items
- burger_bar: 25-35 items
- dessert_cafe: 30-45 items
- gastro_pub: 40-60 items

**Ground Truth:** ✅ Working correctly
- Star items have 3x higher volume
- High waste items appear in waste data
- Tags properly assigned

**Staff & Bookings:** ✅ Tied to sales
- 8 staff members with roles
- No-show rate: 8.8% (within 5-15% target)
- Weekends busier than weekdays

**Reproducibility:** ✅ Same seed = identical output
- Menu items: Identical
- Sales count: Identical
- Prices: Identical

**Backward Compatibility:** ✅ Old CLI works
- --menu-size parameter mapped
- --days parameter functional
- --quality accepts old "corrupted" → "chaotic"

---

## 📁 Files Created

1. **generate_test_data_v2.py** (983 lines)
   - Main ultra-realistic generator
   - 4 restaurant archetypes
   - Ground truth labeling
   - Staff & bookings generation
   - Validation helper

2. **test_generator_v2.py** (300 lines)
   - Pytest test suite
   - 10 test classes covering all features
   - Example usage patterns

3. **Sample Datasets Generated:**
   - `data/test_v2/burger_bar/` - Pristine burger bar dataset
   - `data/test_v2/casual_brasserie/` - Pristine brasserie dataset
   - `data/test_v2/dessert_cafe/` - Pristine cafe dataset
   - `data/test_v2/gastro_pub/` - Pristine pub dataset
   - Each with: menu, sales, waste, staff, bookings, ground_truth CSVs

---

## 🎓 Usage Examples

### Basic Generation
```python
from generate_test_data_v2 import UltraRealisticRestaurantGenerator

gen = UltraRealisticRestaurantGenerator(seed=42)
menu, sales, waste, staff, bookings, truth = gen.generate_restaurant_dataset(
    restaurant_name="The Happy Burger",
    restaurant_type="burger_bar",
    start_date="2024-01-01",
    end_date="2024-12-31",
    quality_level="pristine",
    random_seed=42
)

print(f"Generated {len(menu)} items, {len(sales):,} transactions")
print(f"Star items: {truth['expected_metrics']['total_star_items']}")
```

### Testing with Ground Truth
```python
# Generate test dataset
dataset = gen.generate_restaurant_dataset("Test", "burger_bar", quality_level="pristine")
menu, sales, waste, staff, bookings, truth = dataset

# Run through engine
from resturantv1 import analyze_restaurant  # Your engine function
results = analyze_restaurant(menu, sales, waste)

# Validate
from generate_test_data_v2 import validate_ground_truth
validation = validate_ground_truth(dataset, results)

assert validation["stars_matched"], "Engine should detect star items"
assert validation["all_checks_passed"], "All validations should pass"
```

### CLI Usage
```bash
# Generate burger bar dataset for Q1 2024
python generate_test_data_v2.py \
  --restaurant-type burger_bar \
  --start-date 2024-01-01 \
  --end-date 2024-03-31 \
  --quality pristine \
  --output data/q1_test \
  --seed 42

# Check ground truth
cat data/q1_test/expected_metrics_burger_bar_pristine.json
```

---

## 🚀 Next Steps (Optional)

If you want to enhance further:

1. **More Archetypes:** Add "fine_dining", "fast_food", "coffee_shop"
2. **Seasonal Patterns:** Items that appear only in certain months
3. **Time-of-Day Items:** Breakfast vs lunch vs dinner menus
4. **Menu Evolution:** Items added/removed during date range
5. **Promotional Pricing:** Temporary price changes for events
6. **Weather Effects:** Busier on rainy days (soups, hot drinks)

---

## ✅ Summary

**Status:** COMPLETE AND TESTED

**What Works:**
- ✅ 4 realistic restaurant archetypes
- ✅ Ground truth labeling for automated testing
- ✅ Staff scheduling with performance levels
- ✅ Bookings tied to sales volume
- ✅ Validation helper for pytest integration
- ✅ Backward compatible with old CLI
- ✅ Comprehensive docstrings with examples
- ✅ Full reproducibility via seeds
- ✅ All tests passing

**Ready For:**
- Automated pytest test suites
- Integration testing with known outcomes
- Stress testing edge cases
- Performance benchmarking
- Client demos with realistic data

The generator is now **production-ready** for building robust test suites that validate your restaurant intelligence engine against known ground truth.
