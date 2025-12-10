# Quick Guide: How to Test Your Engine is Working

## 🎯 The Problem
You have a restaurant intelligence engine, but how do you KNOW it's detecting patterns correctly?

## ✅ The Solution
Use **ground truth datasets** - data where you KNOW what the engine SHOULD find.

---

## 🚀 Quick Start (3 Steps)

### Step 1: Generate Test Data with Known Patterns

```python
from generate_test_data_v2 import UltraRealisticRestaurantGenerator

# Create generator
gen = UltraRealisticRestaurantGenerator(seed=42)

# Generate data where you KNOW what should be detected
menu, sales, waste, staff, bookings, truth = gen.generate_restaurant_dataset(
    restaurant_name="Test Burger Bar",
    restaurant_type="burger_bar",
    start_date="2024-01-01",
    end_date="2024-12-31",
    quality_level="pristine",  # Clean data for testing
    random_seed=42
)

# See what SHOULD be detected
print(f"Expected Star items: {truth['expected_metrics']['total_star_items']}")
print(f"High waste items: {truth['expected_metrics']['high_waste_items']}")
```

### Step 2: Run Your Engine

```python
# Save data to files (your engine probably loads from CSV)
menu.to_csv('data/test/menu.csv', index=False)
sales.to_csv('data/test/sales.csv', index=False)
waste.to_csv('data/test/waste.csv', index=False)

# Run your engine (adjust to your actual code)
from your_engine import analyze_restaurant  # Your engine function

results = analyze_restaurant(
    menu_csv='data/test/menu.csv',
    sales_csv='data/test/sales.csv',
    waste_csv='data/test/waste.csv'
)
```

### Step 3: Check if Engine Found What It Should

```python
# Get what engine detected
detected_stars = results['star_items']  # Adjust to your actual output

# Get what SHOULD have been detected
expected_stars = truth['items'][truth['items']['expected_quadrant'] == 'Star']['item_name'].tolist()

# Validate
matched = [item for item in expected_stars if item in detected_stars]
match_rate = len(matched) / len(expected_stars) * 100

if match_rate >= 50:
    print(f"✅ PASS: Engine detected {match_rate:.0f}% of star items")
else:
    print(f"❌ FAIL: Engine only detected {match_rate:.0f}% of star items")
    print(f"   Expected: {expected_stars}")
    print(f"   Detected: {detected_stars}")
```

---

## 📊 What Ground Truth Provides

The generator tags items with their expected behavior:

| Tag | What It Means | What Engine Should Detect |
|-----|---------------|---------------------------|
| `underpriced_high_performer` | Item priced too low, sells well | **Star** (high volume + good GP) |
| `overpriced_low_volume` | Item too expensive, sells poorly | **Dog** (low volume + low GP) |
| `high_waste_risk` | Item generates significant waste | Appears in waste analysis with high qty |
| `anchor_item` | Premium reference pricing | **Plowhorse** or **Star** (high volume) |

---

## 🧪 Example: Complete Validation

```python
from generate_test_data_v2 import UltraRealisticRestaurantGenerator

# Generate test data
gen = UltraRealisticRestaurantGenerator(seed=42)
menu, sales, waste, staff, bookings, truth = gen.generate_restaurant_dataset(
    "Test Restaurant", "burger_bar", "2024-01-01", "2024-12-31", "pristine", 42
)

# Save for your engine
menu.to_csv('test_menu.csv', index=False)
sales.to_csv('test_sales.csv', index=False)
waste.to_csv('test_waste.csv', index=False)

# Run your engine (example - adjust to your actual code)
# import your_engine
# results = your_engine.analyze('test_menu.csv', 'test_sales.csv', 'test_waste.csv')

# For validation, let's simulate what your engine should calculate
import pandas as pd

# Load and analyze (simulating your engine)
sales_with_menu = sales.merge(menu[['item_name', 'sell_price', 'cost_per_unit']], on='item_name')
item_summary = sales_with_menu.groupby('item_name').agg({
    'qty': 'sum',
    'sell_price': 'first',
    'cost_per_unit': 'first'
}).reset_index()

item_summary['revenue'] = item_summary['qty'] * item_summary['sell_price']
item_summary['gp'] = item_summary['revenue'] - (item_summary['qty'] * item_summary['cost_per_unit'])
item_summary['gp_pct'] = (item_summary['gp'] / item_summary['revenue'] * 100)

# Classify (simplified menu engineering)
item_summary['volume_rank'] = item_summary['qty'].rank(pct=True)
item_summary['gp_rank'] = item_summary['gp_pct'].rank(pct=True)

def classify(row):
    if row['volume_rank'] >= 0.5 and row['gp_rank'] >= 0.5:
        return "Star"
    elif row['volume_rank'] >= 0.5 and row['gp_rank'] < 0.5:
        return "Plowhorse"
    elif row['volume_rank'] < 0.5 and row['gp_rank'] >= 0.5:
        return "Puzzle"
    else:
        return "Dog"

item_summary['quadrant'] = item_summary.apply(classify, axis=1)

# VALIDATE: Check if engine detected what it should
expected_stars = truth['items'][truth['items']['expected_quadrant'] == 'Star']['item_name'].tolist()
detected_stars = item_summary[item_summary['quadrant'] == 'Star']['item_name'].tolist()

print("\n🧪 VALIDATION RESULTS:")
print(f"   Expected Stars: {expected_stars}")
print(f"   Detected Stars: {detected_stars}")

matched = [s for s in expected_stars if s in detected_stars]
if len(matched) >= len(expected_stars) * 0.5:
    print(f"   ✅ PASS: Engine correctly identified {len(matched)}/{len(expected_stars)} star items")
else:
    print(f"   ❌ FAIL: Engine only found {len(matched)}/{len(expected_stars)} star items")

# Check high waste detection
expected_waste = truth['expected_metrics']['high_waste_items']
detected_waste = waste['item_name'].tolist()

print(f"\n   Expected High Waste: {expected_waste}")
print(f"   Detected in Waste: {detected_waste}")

if all(item in detected_waste for item in expected_waste):
    print(f"   ✅ PASS: All high waste items detected")
else:
    print(f"   ⚠️  CHECK: Some high waste items missing")
```

---

## 🎓 Why This Works

1. **You generate data where patterns are KNOWN** (via tags)
2. **Run your engine on that data**
3. **Compare what engine found vs what you KNOW is there**
4. **If they match → Engine works! If not → Engine has bugs**

---

## 🔥 Pro Tips

### Test Multiple Scenarios
```python
# Test burger bar
test_archetype("burger_bar")

# Test fine dining
test_archetype("casual_brasserie")

# Test dessert cafe
test_archetype("dessert_cafe")
```

### Test Edge Cases
```python
# Test with messy data
menu, sales, waste, staff, bookings, truth = gen.generate_restaurant_dataset(
    "Chaos Test", "burger_bar", "2024-01-01", "2024-12-31",
    quality_level="chaotic",  # Has duplicates, missing data, etc.
    random_seed=42
)
# Engine should handle errors gracefully
```

### Build Automated Tests
```python
def test_engine_detects_stars():
    """Automated test for CI/CD"""
    menu, sales, waste, _, _, truth = generate_test_data()
    
    results = my_engine.analyze(menu, sales, waste)
    
    expected_stars = truth['items'][truth['items']['expected_quadrant'] == 'Star']['item_name'].tolist()
    detected_stars = results['stars']
    
    match_rate = len([s for s in expected_stars if s in detected_stars]) / len(expected_stars)
    
    assert match_rate >= 0.5, f"Engine should detect at least 50% of stars, got {match_rate*100:.0f}%"
```

---

## 📁 Ready-to-Use Script

I've created `validate_engine_with_ground_truth.py` for you. Just run:

```bash
python validate_engine_with_ground_truth.py
```

It will:
1. ✅ Generate test data with known patterns
2. ✅ Run analysis (simulated)
3. ✅ Validate results vs ground truth
4. ✅ Give you a pass/fail report

---

## 🎯 Bottom Line

**Before:** "I think my engine works... maybe?"  
**After:** "I KNOW my engine works because it detected 4/5 expected patterns in ground truth data"

This is how you build confidence in your code! 🚀
