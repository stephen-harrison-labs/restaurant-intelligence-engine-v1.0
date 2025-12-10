"""
Phase 1 Insights Module - Structured Intelligence Layer

This module defines the core data structures for the Insight Graph.
All problem detection, opportunity calculation, and risk analysis
flows into these structures.

Architecture:
- L0 (engine) → L1 (detectors) → L2 (InsightGraph) → L3 (AI narrative)
"""

from dataclasses import dataclass, field
from typing import Any, List
import pandas as pd
import numpy as np


@dataclass
class Insight:
    """
    A single insight/finding/recommendation from the analysis.
    
    Can represent:
    - An opportunity (e.g., underpriced high-volume item)
    - A risk (e.g., negative GP after waste)
    - A pattern (e.g., category underperforming)
    - An action item (e.g., remove low-performing dish)
    
    Attributes:
        id: Unique identifier (e.g., "waste_margherita_pizza")
        type: Category of insight ("waste", "pricing", "menu", "risk", "opportunity")
        label: Short human-readable label (e.g., "Margherita Pizza waste issue")
        summary: One-sentence description of the insight
        impact_low: Lower bound annual £ impact estimate
        impact_high: Upper bound annual £ impact estimate
        strength: How strong the pattern is (0-100)
        confidence: Data quality confidence (0-100)
        urgency: Priority level ("low", "medium", "high")
        tags: Free-form tags for filtering/grouping
    """
    id: str
    type: str
    label: str
    summary: str
    impact_low: float
    impact_high: float
    strength: int
    confidence: int
    urgency: str
    tags: List[str] = field(default_factory=list)


@dataclass
class InsightGraphPhase1:
    """
    Phase 1 Insight Graph - Minimal viable intelligence structure.
    
    Contains only essential sections for MVP:
    - meta: Restaurant metadata and data quality info
    - opportunities: List of profit improvement opportunities
    - risks: List of red flags and problem areas
    
    Future phases will add:
    - sections.categories
    - sections.menu_engineering
    - sections.waste
    - sections.staff
    - sections.bookings
    - sections.seasonality
    - sections.projections
    """
    meta: dict[str, Any] = field(default_factory=dict)
    opportunities: List[Insight] = field(default_factory=list)
    risks: List[Insight] = field(default_factory=list)


# =============================================================================
# PROBLEM DETECTION - Item Level
# =============================================================================

def detect_item_level_problems(
    perf_df: pd.DataFrame,
    cat_summary_df: pd.DataFrame,
    overall_gp_pct_after_waste: float,
) -> dict[str, list[dict]]:
    """
    Detects item-level risks and opportunities from performance data.
    
    Args:
        perf_df: Item-level performance with columns:
            - item_name, category, units_sold, revenue, gross_profit
            - gp_after_waste, waste_cost, gp_pct, gp_pct_after_waste
            - menu_engineering_class ("Star", "Plowhorse", "Puzzle", "Dog")
        cat_summary_df: Category-level summary with:
            - category, gp_pct_after_waste
        overall_gp_pct_after_waste: Restaurant-wide GP% after waste
    
    Returns:
        {
            "risks": [list of risk dicts],
            "opportunities": [list of opportunity dicts]
        }
    """
    risks = []
    opportunities = []
    
    # Safe handling for empty data
    if perf_df.empty or cat_summary_df.empty:
        return {"risks": [], "opportunities": []}
    
    # Compute volume threshold (median)
    median_volume = perf_df["units_sold"].median()
    if pd.isna(median_volume):
        median_volume = 0
    
    # Create category lookup for GP%
    cat_gp_map = {}
    for _, row in cat_summary_df.iterrows():
        cat = row.get("category", "")
        gp_pct = row.get("gp_pct_after_waste", 0)
        if pd.notna(cat) and pd.notna(gp_pct):
            cat_gp_map[cat] = gp_pct
    
    # Analyze each item
    skipped_items = 0
    for idx, item in perf_df.iterrows():
        item_name = item.get("item_name", "Unknown")
        category = item.get("category", "Unknown")
        units_sold = item.get("units_sold", 0)
        revenue = item.get("revenue", 0)
        gross_profit = item.get("gross_profit", 0)
        gp_after_waste = item.get("gp_after_waste", 0)
        waste_cost = item.get("waste_cost", 0)
        gp_pct = item.get("gp_pct", 0)
        gp_pct_after_waste = item.get("gp_pct_after_waste", 0)
        menu_class = item.get("menu_engineering_class", "")
        
        # Safe conversions
        try:
            units_sold = float(units_sold) if pd.notna(units_sold) else 0
            revenue = float(revenue) if pd.notna(revenue) else 0
            gross_profit = float(gross_profit) if pd.notna(gross_profit) else 0
            gp_after_waste = float(gp_after_waste) if pd.notna(gp_after_waste) else 0
            waste_cost = float(waste_cost) if pd.notna(waste_cost) else 0
            gp_pct = float(gp_pct) if pd.notna(gp_pct) else 0
            gp_pct_after_waste = float(gp_pct_after_waste) if pd.notna(gp_pct_after_waste) else 0
        except (ValueError, TypeError) as e:
            skipped_items += 1
            # C4.1: Fail if too many items skipped (>5%)
            if skipped_items > len(perf_df) * 0.05:
                raise ValueError(
                    f"❌ CRITICAL: {skipped_items} items ({(skipped_items/len(perf_df)*100):.1f}%) skipped due to data quality issues.\n"
                    f"This is too high to proceed. Check for NaN/inf values in perf_df.\n"
                    f"Last error: {str(e)}"
                )
            continue  # Skip items with bad data
        
        # Get category benchmark
        cat_gp_pct = cat_gp_map.get(category, overall_gp_pct_after_waste)
        
        # --- RISK DETECTION ---
        
        # Risk 1: Negative GP after waste (CRITICAL)
        if gp_after_waste < 0:
            risks.append({
                "item_name": item_name,
                "category": category,
                "issue": "negative_gp_after_waste",
                "reason": f"Item loses £{abs(gp_after_waste):.2f} after waste. Immediate removal or recipe redesign needed.",
                "impact_estimate": abs(gp_after_waste),
                "severity": "critical",
            })
        
        # Risk 2: High waste relative to profit
        elif gross_profit > 0 and waste_cost > 0:
            waste_ratio = waste_cost / gross_profit
            if waste_ratio > 0.20:  # Waste exceeds 20% of gross profit
                risks.append({
                    "item_name": item_name,
                    "category": category,
                    "issue": "high_waste_ratio",
                    "reason": f"Waste represents {waste_ratio*100:.1f}% of gross profit. Investigate portion control, prep waste, or demand forecasting.",
                    "impact_estimate": waste_cost * 0.5,  # Assume 50% of waste is avoidable
                    "severity": "high",
                })
        
        # Risk 3: Overpriced Puzzles/Dogs (low volume, not beating category GP)
        if menu_class in ["Puzzle", "Dog"]:
            if units_sold < median_volume and gp_pct_after_waste <= cat_gp_pct + 0.01:
                risks.append({
                    "item_name": item_name,
                    "category": category,
                    "issue": "overpriced_low_volume",
                    "reason": f"{menu_class} with {units_sold:.0f} units sold and GP% ({gp_pct_after_waste*100:.1f}%) not materially better than category average. Consider price reduction or removal.",
                    "impact_estimate": revenue * 0.02,  # Small negative impact estimate
                    "severity": "medium",
                })
        
        # Risk 4: Low volume + low GP dragging category down
        if units_sold < median_volume * 0.3:  # Very low volume
            if gp_pct_after_waste < cat_gp_pct - 0.05:  # 5pp below category
                if revenue > 0:  # Has some sales
                    risks.append({
                        "item_name": item_name,
                        "category": category,
                        "issue": "underperformer",
                        "reason": f"Low volume ({units_sold:.0f} units) with GP% ({gp_pct_after_waste*100:.1f}%) dragging {category} category down. Candidate for removal.",
                        "impact_estimate": abs(gp_after_waste) if gp_after_waste < 0 else revenue * 0.01,
                        "severity": "medium",
                    })
        
        # --- OPPORTUNITY DETECTION ---
        
        # Opportunity 1: High-performing Stars/Plowhorses (underpriced)
        if menu_class in ["Star", "Plowhorse"]:
            # Item GP% significantly above category
            if gp_pct_after_waste >= cat_gp_pct + 0.03:
                # Also above restaurant overall
                if gp_pct_after_waste >= overall_gp_pct_after_waste + 0.03:
                    opportunities.append({
                        "item_name": item_name,
                        "category": category,
                        "issue": "underpriced_high_performer",
                        "reason": f"{menu_class} with {units_sold:.0f} units sold and strong GP% ({gp_pct_after_waste*100:.1f}%, {((gp_pct_after_waste - overall_gp_pct_after_waste)*100):.1f}pp above average). Consider controlled price increase.",
                        "impact_estimate": revenue * 0.03,  # 3% price increase potential
                        "severity": "high",
                    })
        
        # Opportunity 2: Items with avoidable waste but still profitable
        if gp_after_waste > 0 and waste_cost > 0:
            if gross_profit > 0:
                waste_ratio = waste_cost / gross_profit
                # Moderate waste (10-30%) - likely fixable
                if 0.10 < waste_ratio <= 0.30:
                    opportunities.append({
                        "item_name": item_name,
                        "category": category,
                        "issue": "avoidable_waste",
                        "reason": f"Item is profitable (£{gp_after_waste:.2f}) but waste represents {waste_ratio*100:.1f}% of GP. Waste reduction could improve margin significantly.",
                        "impact_estimate": waste_cost * 0.4,  # 40% waste reduction achievable
                        "severity": "medium",
                    })
        
        # Opportunity 3: High-volume items beating restaurant average
        if units_sold >= median_volume * 1.5:  # 50% above median
            if gp_pct_after_waste >= overall_gp_pct_after_waste + 0.05:  # 5pp above average
                # Don't double-count if already flagged as underpriced
                already_flagged = any(
                    opp["item_name"] == item_name and opp["issue"] == "underpriced_high_performer"
                    for opp in opportunities
                )
                if not already_flagged:
                    opportunities.append({
                        "item_name": item_name,
                        "category": category,
                        "issue": "high_volume_strong_margin",
                        "reason": f"High volume ({units_sold:.0f} units) with excellent GP% ({gp_pct_after_waste*100:.1f}%). Key profit driver - protect pricing and quality.",
                        "impact_estimate": revenue * 0.02,  # 2% price opportunity
                        "severity": "medium",
                    })
    
    # C4.1: Report skipped items at end
    if skipped_items > 0:
        print(f"⚠️  WARNING: Skipped {skipped_items} items ({(skipped_items/len(perf_df)*100):.1f}%) due to data quality issues")
    
    return {
        "risks": risks,
        "opportunities": opportunities,
    }


# =============================================================================
# DATA RELIABILITY SCORING
# =============================================================================

def compute_data_reliability_score(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """
    Computes an overall data reliability score (0-100) based on data quality diagnostics.
    
    Args:
        diagnostics: Data quality diagnostics dict from build_data_quality_report()
            Accepts both old and new key naming conventions:
            - orders_row_count OR sales_row_count: int
            - num_days_covered OR days_of_data: int
            - has_gaps_in_days: bool
            - menu_item_count OR menu_row_count: int
            - waste_row_count: int (optional)
            - staff_row_count: int (optional)
            - bookings_row_count: int (optional)
    
    Returns:
        {
            "score": int (0-100),
            "level": "low" | "medium" | "high",
            "notes": [list of reliability issues]
        }
    """
    score = 100
    notes = []
    
    # Safe extraction with defaults - handle both old and new key names
    sales_rows = diagnostics.get("orders_row_count") or diagnostics.get("sales_row_count", 0)
    days_of_data = diagnostics.get("num_days_covered") or diagnostics.get("days_of_data", 0)
    has_gaps = diagnostics.get("has_gaps_in_days", False)
    menu_rows = diagnostics.get("menu_item_count") or diagnostics.get("menu_row_count", 0)
    waste_rows = diagnostics.get("waste_row_count", 0)
    staff_rows = diagnostics.get("staff_row_count", 0)
    bookings_rows = diagnostics.get("bookings_row_count", 0)
    
    # Convert to int safely
    try:
        sales_rows = int(sales_rows) if sales_rows is not None and pd.notna(sales_rows) else 0
        days_of_data = int(days_of_data) if days_of_data is not None and pd.notna(days_of_data) else 0
        menu_rows = int(menu_rows) if menu_rows is not None and pd.notna(menu_rows) else 0
        waste_rows = int(waste_rows) if waste_rows is not None and pd.notna(waste_rows) else 0
        staff_rows = int(staff_rows) if staff_rows is not None and pd.notna(staff_rows) else 0
        bookings_rows = int(bookings_rows) if bookings_rows is not None and pd.notna(bookings_rows) else 0
    except (ValueError, TypeError):
        pass  # Keep defaults
    
    # --- SALES ROW COUNT ---
    if sales_rows == 0:
        score -= 20
        notes.append("No sales data provided. Analysis is based on aggregated tables only.")
    elif sales_rows < 5000:
        score -= 20
        notes.append(f"Very low sales row count ({sales_rows:,}). Recommend 20,000+ rows for robust analysis.")
    elif sales_rows < 20000:
        score -= 10
        notes.append(f"Low sales row count ({sales_rows:,}). Results may be less reliable for low-volume items.")
    
    # --- DATE COVERAGE ---
    if days_of_data == 0:
        score -= 20
        notes.append("No date coverage information. Unable to assess seasonality.")
    elif days_of_data < 90:
        score -= 20
        notes.append(f"Short date coverage ({days_of_data} days). Recommend 180+ days for seasonal patterns.")
    elif days_of_data < 180:
        score -= 10
        notes.append(f"Medium date coverage ({days_of_data} days). Some seasonal patterns may not be visible.")
    
    # --- DATE GAPS ---
    if has_gaps:
        score -= 10
        notes.append("Sales data has gaps in the date range. Trend analysis may be incomplete.")
    
    # --- MISSING MENU DATA ---
    if menu_rows == 0:
        score -= 5
        notes.append("No menu data provided. GP calculations may be inaccurate.")
    
    # --- MISSING WASTE DATA ---
    if waste_rows == 0:
        score -= 5
        notes.append("No waste data provided. GP after waste calculations assume zero waste.")
    
    # --- MISSING STAFF DATA ---
    # Only penalize if it's expected but missing (some clients won't have staff data)
    # We'll be lenient here - only flag if diagnostics explicitly shows 0
    if "staff_row_count" in diagnostics and staff_rows == 0:
        score -= 5
        notes.append("No staff performance data available. Staff insights excluded.")
    
    # --- MISSING BOOKINGS DATA ---
    if "bookings_row_count" in diagnostics and bookings_rows == 0:
        score -= 5
        notes.append("No bookings/reservations data available. Demand analysis excluded.")
    
    # Clamp score to 0-100
    score = max(0, min(100, score))
    
    # Determine level
    if score >= 80:
        level = "high"
    elif score >= 60:
        level = "medium"
    else:
        level = "low"
    
    return {
        "score": score,
        "level": level,
        "notes": notes,
    }


def validate_metrics_vs_tables(
    summary_metrics: dict[str, Any],
    cat_summary_df: pd.DataFrame,
    data_reliability_notes: list[str]
) -> list[str]:
    """
    Validate that topline metrics are consistent with category performance tables.
    
    Args:
        summary_metrics: Dict with total_revenue, total_gp_after_waste, etc.
        cat_summary_df: Category performance dataframe
        data_reliability_notes: Existing reliability notes to append to
    
    Returns:
        Updated list of reliability notes with any warnings about mismatches
    """
    notes = data_reliability_notes.copy()
    
    if cat_summary_df.empty:
        return notes
    
    # Sum category units_sold
    if "units_sold" in cat_summary_df.columns:
        cat_total_units = cat_summary_df["units_sold"].sum()
        
        # Check if summary_metrics has units but is zero
        topline_units = summary_metrics.get("total_units_sold", None)
        
        if topline_units is not None:
            if topline_units == 0 and cat_total_units > 0:
                notes.append(f"⚠️ Mismatch: Topline shows 0 units but category table shows {cat_total_units:,.0f} units. Check data wiring.")
            elif abs(topline_units - cat_total_units) > 2:
                notes.append(f"⚠️ Mismatch: Topline units ({topline_units:,.0f}) differs from category total ({cat_total_units:,.0f}).")
    
    # Sum category revenue
    if "total_revenue" in cat_summary_df.columns:
        cat_total_revenue = cat_summary_df["total_revenue"].sum()
        topline_revenue = summary_metrics.get("total_revenue", 0)
        
        if topline_revenue == 0 and cat_total_revenue > 0:
            notes.append(f"⚠️ Mismatch: Topline shows £0 revenue but category table shows £{cat_total_revenue:,.0f}. Check data wiring.")
        elif topline_revenue > 0 and abs(topline_revenue - cat_total_revenue) > 1:
            # Allow 1 pound difference due to rounding
            diff_pct = abs(topline_revenue - cat_total_revenue) / topline_revenue * 100
            if diff_pct > 0.1:
                notes.append(f"⚠️ Mismatch: Topline revenue (£{topline_revenue:,.0f}) differs from category total (£{cat_total_revenue:,.0f}) by {diff_pct:.1f}%.")
    
    return notes


# =============================================================================
# OPPORTUNITY VALUE ESTIMATION
# =============================================================================

def estimate_opportunity_range_for_item(
    row: pd.Series,
    target_gp_pct_after_waste: float,
    confidence: float = 0.6,
) -> tuple[float, float]:
    """
    Estimates annual GP uplift range (low/high) for a single menu item.
    
    Use cases:
    - Underpriced Stars/Plowhorses (price increase opportunity)
    - Items below category GP% (margin improvement opportunity)
    - Waste reduction scenarios
    
    Args:
        row: Single row from perf_df with columns:
            - gp_after_waste: Current annual GP after waste (£)
            - units_sold: Annual unit sales volume
            - sell_price: Current menu price (£)
        target_gp_pct_after_waste: Target GP% after waste (as decimal, e.g., 0.70 for 70%)
        confidence: Confidence factor (0-1) for conservative estimation. Default 0.6.
    
    Returns:
        (low_estimate, high_estimate): Tuple of annual £ impact estimates
    
    Example:
        Item sells 2000 units at £10 with current GP% of 60%.
        Target GP% is 70%.
        Current GP/unit: £6
        Target GP/unit: £7
        Uplift: (£7 - £6) * 2000 = £2000
        With confidence 0.6:
            Low: £2000 * 0.5 * 0.6 = £600
            High: £2000 * 1.5 * 0.6 = £1800
    """
    # Safe extraction with NaN handling
    gp_after_waste = np.nan_to_num(row.get("gp_after_waste", 0), nan=0.0)
    units_sold = np.nan_to_num(row.get("units_sold", 0), nan=0.0)
    sell_price = np.nan_to_num(row.get("sell_price", 0), nan=0.0)
    
    # Avoid division by zero
    units_sold = max(float(units_sold), 1.0)
    
    # Current GP per unit
    current_gp_per_unit = float(gp_after_waste) / units_sold
    
    # Target GP per unit at desired margin
    target_gp_per_unit = float(target_gp_pct_after_waste) * float(sell_price)
    
    # Annual uplift if we hit target
    uplift = (target_gp_per_unit - current_gp_per_unit) * units_sold
    
    # If no opportunity (already at or above target), return zero
    if uplift <= 0:
        return (0.0, 0.0)
    
    # Apply confidence factor and create range
    # Low estimate: conservative (50% achievement * confidence)
    # High estimate: optimistic (150% achievement * confidence)
    low_estimate = uplift * 0.5 * confidence
    high_estimate = uplift * 1.5 * confidence
    
    return (float(low_estimate), float(high_estimate))


# =============================================================================
# INSIGHT GRAPH BUILDER - PHASE 1
# =============================================================================

def build_phase1_insight_graph(results: dict[str, Any]) -> InsightGraphPhase1:
    """
    Builds the Phase 1 Insight Graph from analysis results.
    
    This is the central intelligence object that structures all findings
    from L0 (engine) and L1 (detectors) into a queryable, AI-friendly format.
    
    Args:
        results: Full analysis results dict from run_full_analysis_v2() containing:
            - perf_df: Item-level performance
            - cat_summary_df: Category summaries
            - summary_metrics: Restaurant-wide metrics
            - data_quality_diagnostics: Data quality report
            - restaurant_name, period_label: Metadata
    
    Returns:
        InsightGraphPhase1: Structured intelligence with:
            - meta: Restaurant info + data reliability
            - opportunities: List of Insight objects (profit improvements)
            - risks: List of Insight objects (red flags)
    """
    # Safe extraction of core data
    perf_df = results.get("perf_df")
    cat_summary_df = results.get("cat_summary_df")
    summary_metrics = results.get("summary_metrics", {}).copy()  # Make a copy to modify
    diagnostics = results.get("data_quality_diagnostics", {})
    
    # Extract config for metadata
    config = results.get("config", {})
    restaurant_name = config.get("restaurant_name", "Unknown Restaurant")
    period_label = config.get("period_label", "Unknown Period")
    
    # Handle missing data gracefully
    if perf_df is None or perf_df.empty:
        return InsightGraphPhase1(
            meta={
                "restaurant_name": restaurant_name,
                "period_label": period_label,
                "data_reliability": {"score": 0, "level": "low", "notes": ["No performance data available"]},
                "overall_gp_pct_after_waste": 0,
            },
            opportunities=[],
            risks=[],
        )
    
    if cat_summary_df is None or cat_summary_df.empty:
        cat_summary_df = pd.DataFrame(columns=["category", "gp_pct_after_waste"])
    
    # Compute overall GP%
    overall_gp_pct_after_waste = summary_metrics.get("avg_gp_pct_after_waste", 0)
    if pd.isna(overall_gp_pct_after_waste):
        overall_gp_pct_after_waste = 0
    
    # CRITICAL FIX: Add missing metrics from perf_df if not in summary_metrics
    if "total_units_sold" not in summary_metrics or summary_metrics.get("total_units_sold", 0) == 0:
        if "units_sold" in perf_df.columns:
            summary_metrics["total_units_sold"] = int(perf_df["units_sold"].sum())
    
    # Add days_of_data from diagnostics if missing from summary_metrics
    days_from_diag = diagnostics.get("num_days_covered") or diagnostics.get("days_of_data", 0)
    if days_from_diag:
        summary_metrics["days_of_data"] = int(days_from_diag)
    
    # Run detectors
    problems = detect_item_level_problems(perf_df, cat_summary_df, overall_gp_pct_after_waste)
    reliability = compute_data_reliability_score(diagnostics)
    
    # Validate metrics consistency and add warnings to reliability notes
    reliability["notes"] = validate_metrics_vs_tables(
        summary_metrics, cat_summary_df, reliability["notes"]
    )
    
    # Helper: Create item ID slug
    def make_id(issue_type: str, item_name: str) -> str:
        """Create unique insight ID from issue and item name."""
        item_slug = item_name.lower().replace(" ", "_").replace("(", "").replace(")", "")[:30]
        return f"{issue_type}_{item_slug}"
    
    # Helper: Map issue type to strength score
    def get_strength(issue_type: str) -> int:
        """Heuristic strength score based on issue type."""
        strength_map = {
            "negative_gp_after_waste": 95,
            "high_waste_ratio": 85,
            "underpriced_high_performer": 80,
            "avoidable_waste": 75,
            "high_volume_strong_margin": 70,
            "overpriced_low_volume": 70,
            "underperformer": 60,
        }
        return strength_map.get(issue_type, 50)
    
    # Helper: Map issue type to urgency
    def get_urgency(issue_type: str) -> str:
        """Determine urgency level based on issue type."""
        if issue_type in ["negative_gp_after_waste", "high_waste_ratio"]:
            return "high"
        elif issue_type in ["underpriced_high_performer", "avoidable_waste", "overpriced_low_volume"]:
            return "medium"
        else:
            return "low"
    
    # Helper: Convert raw problem dict to Insight object
    def problem_to_insight(problem: dict, is_opportunity: bool) -> Insight:
        """Convert detector output to Insight object with impact estimates."""
        item_name = problem.get("item_name", "Unknown")
        category = problem.get("category", "Unknown")
        issue = problem.get("issue", "unknown")
        reason = problem.get("reason", "")
        impact_estimate = problem.get("impact_estimate", 0)
        
        # Get item row from perf_df for detailed calculations
        item_row = None
        try:
            item_match = perf_df[perf_df["item_name"] == item_name]
            if not item_match.empty:
                item_row = item_match.iloc[0]
        except Exception:
            pass
        
        # Calculate impact range
        impact_low = 0.0
        impact_high = 0.0
        
        if impact_estimate > 0:
            # Use detector's estimate as midpoint, create range
            impact_low = float(impact_estimate) * 0.5
            impact_high = float(impact_estimate) * 1.5
        elif item_row is not None and issue in ["underpriced_high_performer", "high_volume_strong_margin"]:
            # Use sophisticated estimate for margin improvement opportunities
            try:
                target_gp = overall_gp_pct_after_waste + 0.05  # Target 5pp above current
                impact_low, impact_high = estimate_opportunity_range_for_item(
                    item_row, 
                    target_gp, 
                    confidence=0.6
                )
            except Exception:
                # Fallback to simple estimate
                revenue = float(item_row.get("revenue", 0))
                impact_low = revenue * 0.02
                impact_high = revenue * 0.05
        
        # Get menu engineering class for tagging
        menu_class = ""
        if item_row is not None:
            menu_class = item_row.get("menu_engineering_class", "")
        
        # Build tags
        tags = [issue, category.lower()]
        if menu_class:
            tags.append(menu_class.lower())
        
        # Create insight
        return Insight(
            id=make_id(issue, item_name),
            type="opportunity" if is_opportunity else "risk",
            label=f"{item_name} – {issue.replace('_', ' ').title()}",
            summary=reason,
            impact_low=impact_low,
            impact_high=impact_high,
            strength=get_strength(issue),
            confidence=reliability["score"],
            urgency=get_urgency(issue),
            tags=tags,
        )
    
    # Convert all problems to Insight objects
    opportunities = []
    risks = []
    
    try:
        for problem in problems.get("opportunities", []):
            insight = problem_to_insight(problem, is_opportunity=True)
            opportunities.append(insight)
    except Exception as e:
        # Log but don't crash
        print(f"Warning: Error processing opportunities: {e}")
    
    try:
        for problem in problems.get("risks", []):
            insight = problem_to_insight(problem, is_opportunity=False)
            risks.append(insight)
    except Exception as e:
        # Log but don't crash
        print(f"Warning: Error processing risks: {e}")
    
    # Sort by impact (high estimate descending)
    opportunities.sort(key=lambda x: x.impact_high, reverse=True)
    risks.sort(key=lambda x: x.strength, reverse=True)
    
    # Build metadata
    meta = {
        "restaurant_name": restaurant_name,
        "period_label": period_label,
        "engine_version": config.get("engine_version", "unknown"),
        "price_elasticity_assumption": config.get("price_elasticity_assumption", -1.0),
        "data_reliability": reliability,
        "overall_gp_pct_after_waste": float(overall_gp_pct_after_waste),
        "total_opportunities_found": len(opportunities),
        "total_risks_found": len(risks),
        "potential_annual_uplift_low": sum(o.impact_low for o in opportunities),
        "potential_annual_uplift_high": sum(o.impact_high for o in opportunities),
        "days_of_data": summary_metrics.get("days_of_data", 0),
        "total_units_sold": summary_metrics.get("total_units_sold", 0),
    }
    
    return InsightGraphPhase1(
        meta=meta,
        opportunities=opportunities,
        risks=risks,
    )


# =============================================================================
# GPT EXPORT BUILDER V2 - EXECUTIVE ULTRA SPECIFICATION
# =============================================================================

def build_gpt_export_block_v2(
    meta: dict,
    results: dict,
    insight_graph: InsightGraphPhase1,
    charts: dict | None = None,
) -> str:
    """
    Generate complete GPT-ready export block with Executive Ultra specification.
    
    This is the Phase 1 MVP export format that provides ALL information needed
    for AI-powered report generation. Contains structured data, insights graph,
    and embedded directives for narrative generation.
    
    Args:
        meta: Config metadata (restaurant_name, period_label, currency)
        results: Full analysis results with tables and metrics
        insight_graph: Phase 1 intelligence structure
        charts: Optional dict of chart filenames
    
    Returns:
        Massive formatted text block ready for ChatGPT consumption
    """
    import json
    import html
    from dataclasses import asdict
    
    lines = []
    
    # Helper: section separator
    def section(title: str):
        lines.append("\n" + "=" * 80)
        lines.append(f"{title}")
        lines.append("=" * 80 + "\n")
    
    # Helper: subsection
    def subsection(title: str):
        lines.append(f"\n--- {title} ---\n")
    
    # C6.4: HTML escaping helper for user input
    def safe_text(text: str) -> str:
        """Escape HTML special characters to prevent XSS."""
        return html.escape(str(text)) if text else ""
    
    # =============================================================================
    # SECTION 1: META & HIGH LEVEL
    # =============================================================================
    section("SECTION 1: META & HIGH LEVEL")
    
    restaurant_name = safe_text(meta.get("restaurant_name", "Unknown Restaurant"))
    period_label = safe_text(meta.get("period_label", "Unknown Period"))
    currency = meta.get("currency", "£")
    engine_version = meta.get("engine_version", "unknown")
    
    lines.append(f"RESTAURANT_NAME: {restaurant_name}")
    lines.append(f"PERIOD_ANALYSED: {period_label}")
    lines.append(f"CURRENCY: {currency}")
    lines.append(f"ENGINE_VERSION: {engine_version}")
    
    # Price elasticity assumption (for scenario modelling transparency)
    elasticity = meta.get("price_elasticity_assumption", -1.0)
    lines.append(f"PRICE_ELASTICITY_ASSUMPTION: {elasticity:.1f}")
    lines.append(f"  (Approx. {abs(elasticity):.0f}% volume change per 1% price change)")
    
    # Data reliability
    subsection("DATA_RELIABILITY")
    reliability = insight_graph.meta.get("data_reliability", {})
    lines.append(f"Score: {reliability.get('score', 0)}/100")
    lines.append(f"Level: {reliability.get('level', 'unknown').upper()}")
    lines.append("Notes:")
    for note in reliability.get("notes", []):
        lines.append(f"  • {note}")
    
    # Topline metrics
    subsection("TOPLINE_METRICS")
    summary = results.get("summary_metrics", {})
    
    metrics = [
        ("Total Revenue", summary.get("total_revenue", 0), True),
        ("Total GP Before Waste", summary.get("total_gp_before_waste", 0), True),
        ("Total Waste Cost", summary.get("total_waste_cost", 0), True),
        ("Total GP After Waste", summary.get("total_gp_after_waste", 0), True),
        ("GP% Before Waste", summary.get("avg_gp_pct_before_waste", 0) * 100, False),
        ("GP% After Waste", summary.get("avg_gp_pct_after_waste", 0) * 100, False),
        ("Total Units Sold", summary.get("total_units_sold", 0), False),
        ("Days of Data", summary.get("days_of_data", 0), False),
    ]
    
    for label, value, is_currency in metrics:
        if is_currency:
            lines.append(f"{label}: {currency}{value:,.0f}")
        elif "%" in label:
            lines.append(f"{label}: {value:.1f}%")
        else:
            lines.append(f"{label}: {value:,.0f}")
    
    # =============================================================================
    # SECTION 2: CORE TABLES
    # =============================================================================
    section("SECTION 2: CORE TABLES")
    
    # Category performance
    if "category_performance_table" in results:
        subsection("CATEGORY_PERFORMANCE")
        lines.append(results["category_performance_table"])
    
    # Top margin items
    if "top_margin_items_table" in results:
        subsection("TOP_MARGIN_ITEMS")
        lines.append(results["top_margin_items_table"])
    
    # Menu engineering classes
    if "menu_stars_table" in results:
        subsection("MENU_STARS (High Volume, High GP%)")
        lines.append(results["menu_stars_table"])
    
    if "menu_plowhorses_table" in results:
        subsection("MENU_PLOWHORSES (High Volume, Low GP%)")
        lines.append(results["menu_plowhorses_table"])
    
    if "menu_puzzles_table" in results:
        subsection("MENU_PUZZLES (Low Volume, High GP%)")
        lines.append(results["menu_puzzles_table"])
    
    if "menu_dogs_table" in results:
        subsection("MENU_DOGS (Low Volume, Low GP%)")
        lines.append(results["menu_dogs_table"])
    
    # Waste analysis
    if "top_waste_items_table" in results:
        subsection("TOP_WASTE_ITEMS")
        lines.append(results["top_waste_items_table"])
    
    # Staff performance (if available)
    if "staff_performance_table" in results:
        subsection("STAFF_PERFORMANCE")
        lines.append(results["staff_performance_table"])
    
    # Booking summary (if available)
    if "booking_summary_table" in results:
        subsection("BOOKING_SUMMARY")
        lines.append(results["booking_summary_table"])
    
    # Scenario analysis - format from raw scenarios dict
    if "scenarios" in results and results["scenarios"]:
        subsection("SCENARIO_ANALYSIS")
        scenarios = results["scenarios"]
        # currency is already defined from meta at top of function
        
        # Add elasticity assumption note for transparency
        elasticity = meta.get("price_elasticity_assumption", -1.0)
        lines.append(f"\nPrice Elasticity Assumption: {elasticity:.1f}")
        lines.append(f"(Approx. {abs(elasticity):.0f}% volume change per 1% price change)")
        lines.append("Note: Revenue impacts may be small if price increases are offset by volume decreases.\n")
        
        for scenario_name, scenario_data in scenarios.items():
            lines.append(f"\n{scenario_name}:")
            lines.append(f"  Label: {scenario_data.get('label', 'No label')}")
            
            # Extract GP change with fallback logic
            gp_change = scenario_data.get('delta_gp_after_waste')
            if gp_change is None:
                # Fallback: compute from base and scenario values if available
                base_gp = scenario_data.get('base_gp_after_waste')
                scenario_gp = scenario_data.get('scenario_gp_after_waste')
                if base_gp is not None and scenario_gp is not None:
                    gp_change = scenario_gp - base_gp
                else:
                    gp_change = 0  # Default to 0 only if everything missing
            
            # Add sanity tag for AI narrative (positive/negative/neutral)
            if gp_change > 100:  # Threshold to avoid noise from rounding
                direction = "positive"
            elif gp_change < -100:
                direction = "negative"
            else:
                direction = "neutral"
            
            # Extract revenue change with fallback logic
            revenue_change = scenario_data.get('delta_revenue')
            if revenue_change is None:
                base_rev = scenario_data.get('base_revenue')
                scenario_rev = scenario_data.get('scenario_revenue')
                if base_rev is not None and scenario_rev is not None:
                    revenue_change = scenario_rev - base_rev
                else:
                    revenue_change = 0
            
            lines.append(f"  Direction: {direction}")
            lines.append(f"  GP Impact: {currency}{gp_change:,.0f}")
            lines.append(f"  Revenue Impact: {currency}{revenue_change:,.0f}")
    elif "scenario_summaries" in results:
        # Backward compatibility: use pre-formatted text if available
        subsection("SCENARIO_ANALYSIS")
        lines.append(results["scenario_summaries"])
    else:
        subsection("SCENARIO_ANALYSIS")
        lines.append("No scenario data available.")
    
    # =============================================================================
    # SECTION 3: INSIGHTS_PHASE1 (RAW STRUCTURED INTELLIGENCE)
    # =============================================================================
    section("SECTION 3: INSIGHTS_PHASE1 (RAW STRUCTURED INTELLIGENCE)")
    
    lines.append("This section contains the complete Phase 1 Intelligence Graph.")
    lines.append("All opportunities, risks, and metadata in machine-readable format.\n")
    
    # Convert insight graph to dict for JSON serialization
    insight_dict = {
        "meta": insight_graph.meta,
        "opportunities": [asdict(o) for o in insight_graph.opportunities],
        "risks": [asdict(r) for r in insight_graph.risks],
    }
    
    lines.append(json.dumps(insight_dict, indent=2))
    
    # =============================================================================
    # SECTION 4: EXECUTIVE SUMMARIES
    # =============================================================================
    section("SECTION 4: EXECUTIVE SUMMARIES")
    
    # Top opportunities
    subsection("TOP 10 PROFIT OPPORTUNITIES")
    
    opportunities_sorted = sorted(insight_graph.opportunities, key=lambda x: x.impact_high, reverse=True)[:10]
    
    if opportunities_sorted:
        for i, opp in enumerate(opportunities_sorted, 1):
            impact_range = f"{currency}{opp.impact_low:,.0f}-{currency}{opp.impact_high:,.0f}"
            lines.append(f"{i}. **{opp.label}** (Impact: {impact_range})")
            lines.append(f"   {opp.summary}")
            lines.append(f"   Urgency: {opp.urgency.upper()} | Confidence: {opp.confidence}/100")
            lines.append("")
    else:
        lines.append("No significant opportunities detected.")
    
    # Top risks
    subsection("TOP 10 CRITICAL RISKS")
    
    risks_sorted = sorted(insight_graph.risks, key=lambda x: x.strength, reverse=True)[:10]
    
    if risks_sorted:
        for i, risk in enumerate(risks_sorted, 1):
            impact_range = f"{currency}{risk.impact_low:,.0f}-{currency}{risk.impact_high:,.0f}" if risk.impact_high > 0 else "Impact not quantified"
            lines.append(f"{i}. **{risk.label}** (Severity: {risk.strength}/100)")
            lines.append(f"   {risk.summary}")
            lines.append(f"   Urgency: {risk.urgency.upper()}")
            lines.append("")
    else:
        lines.append("No critical risks detected.")
    
    # Total uplift potential
    subsection("ESTIMATED ANNUAL UPLIFT POTENTIAL")
    
    uplift_low = insight_graph.meta.get("potential_annual_uplift_low", 0)
    uplift_high = insight_graph.meta.get("potential_annual_uplift_high", 0)
    
    lines.append(f"Conservative Estimate: {currency}{uplift_low:,.0f}")
    lines.append(f"Optimistic Estimate: {currency}{uplift_high:,.0f}")
    lines.append(f"Midpoint: {currency}{(uplift_low + uplift_high) / 2:,.0f}")
    lines.append(f"\nBased on {len(insight_graph.opportunities)} identified opportunities.")
    lines.append(f"Confidence level: {reliability.get('level', 'medium').upper()} (score {reliability.get('score', 0)}/100)")
    
    # Waste strategy summary
    subsection("WASTE STRATEGY SUMMARY")
    
    # Count all waste-related insights (including high_waste_ratio, avoidable_waste, etc.)
    waste_risks = [r for r in insight_graph.risks if any(tag in r.tags for tag in ["waste", "high_waste_ratio", "avoidable_waste"])]
    waste_opps = [o for o in insight_graph.opportunities if any(tag in o.tags for tag in ["waste", "high_waste_ratio", "avoidable_waste"])]
    
    if waste_risks or waste_opps:
        lines.append(f"Waste-related issues found: {len(waste_risks)} risks, {len(waste_opps)} opportunities")
        lines.append("")
        
        if waste_risks:
            lines.append("Critical Waste Risks:")
            for risk in waste_risks[:5]:
                lines.append(f"  • {risk.label}: {risk.summary}")
        
        if waste_opps:
            lines.append("\nWaste Reduction Opportunities:")
            total_waste_saving = sum(o.impact_high for o in waste_opps)
            lines.append(f"  Potential annual saving: {currency}{total_waste_saving:,.0f}")
            for opp in waste_opps[:5]:
                lines.append(f"  • {opp.label}")
    else:
        lines.append("No significant waste issues detected.")
    
    # =============================================================================
    # SECTION 5: EMBEDDED AI DIRECTIVES (PHASE 2 NARRATIVE ENGINE)
    # =============================================================================
    section("SECTION 5: EMBEDDED AI DIRECTIVES (PHASE 2 NARRATIVE ENGINE)")
    
    lines.append("=" * 80)
    lines.append("PREMIUM DARK-SLATE + GOLD PDF LAYOUT - OFFICIAL GENERATION RULES")
    lines.append("=" * 80)
    lines.append("")
    lines.append("THESE ARE THE OFFICIAL GENERATION RULES FOR THE PREMIUM CONSULTING REPORT.")
    lines.append("THE AI MUST FOLLOW THESE RULES EXACTLY WHEN GENERATING THE FINAL REPORT.")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # A. GLOBAL OUTPUT RULES
    # -------------------------------------------------------------------------
    subsection("A. GLOBAL OUTPUT RULES")
    
    lines.append("1. Output MUST be pure HTML unless otherwise stated.")
    lines.append("2. NEVER output Markdown – no **, no ##, no code blocks.")
    lines.append("3. NEVER invent numbers. ALL numbers must come from:")
    lines.append("   - results dict (SECTION 2 tables)")
    lines.append("   - insight_graph (SECTION 3)")
    lines.append("   - topline metrics (SECTION 1)")
    lines.append("   - scenario summaries")
    lines.append("   - category tables")
    lines.append("4. NEVER hallucinate items, categories, staff, or charts.")
    lines.append("5. If a section has no data (e.g., no staff/bookings), render a blank")
    lines.append("   but professional-looking placeholder card with:")
    lines.append("   <p class='note'>No data provided for this section.</p>")
    lines.append("6. Do not change item names, category names, or values.")
    lines.append(f"7. All money values must use currency: {currency}")
    lines.append("8. Narrative tone must adjust based on data_reliability_score:")
    lines.append("   - 80–100 → Confident, assertive, strategic")
    lines.append("   - 60–79  → Balanced, careful, advisory")
    lines.append("   - 0–59   → Cautious, limited insights, conservative language")
    lines.append(f"   CURRENT RELIABILITY: {reliability.get('score', 0)}/100 ({reliability.get('level', 'unknown').upper()})")
    lines.append("9. All insights must be backed by either:")
    lines.append("   - insight_graph.opportunities")
    lines.append("   - insight_graph.risks")
    lines.append("   - category_performance tables")
    lines.append("   - scenario summaries")
    lines.append("   - waste summaries")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # B. PREMIUM PDF LAYOUT (DARK SLATE + GOLD)
    # -------------------------------------------------------------------------
    subsection("B. PREMIUM PDF LAYOUT (DARK SLATE + GOLD)")
    
    lines.append("PAGE COLORS:")
    lines.append("  - Main background: #181b22 (dark slate)")
    lines.append("  - Section card background: #1f232b")
    lines.append("  - Divider lines: rgba(255,255,255,0.08)")
    lines.append("  - Gold accent: #d3af37")
    lines.append("  - Text color: #e8e8e8 (body), #ffffff (headings)")
    lines.append("")
    
    lines.append("TYPOGRAPHY:")
    lines.append("  - Headings (H1–H3): Georgia, serif, color: #d3af37 or #ffffff")
    lines.append("  - Body text: Inter, Roboto, or similar clean sans-serif, color: #e8e8e8")
    lines.append("  - Block quotes / callouts: Georgia italic, color: #d3af37")
    lines.append("  - Font sizes: H1 (32px), H2 (24px), H3 (18px), Body (14px)")
    lines.append("")
    
    lines.append("SPACING SYSTEM:")
    lines.append("  - Section margin: 40px bottom")
    lines.append("  - Chart spacing: 25px top, 40px bottom")
    lines.append("  - Card padding: 28px")
    lines.append("  - Headline spacing: 10px bottom")
    lines.append("  - Paragraph spacing: 15px bottom")
    lines.append("")
    
    lines.append("CARDS & COMPONENTS:")
    lines.append("  - Section cards CSS:")
    lines.append("    .section-card {")
    lines.append("      background: #1f232b;")
    lines.append("      padding: 28px;")
    lines.append("      border-left: 6px solid #d3af37;")
    lines.append("      border-radius: 6px;")
    lines.append("      margin-bottom: 40px;")
    lines.append("      box-shadow: 0 2px 8px rgba(0,0,0,0.3);")
    lines.append("    }")
    lines.append("  - Sub-cards inside follow same padding but no gold bar.")
    lines.append("  - Charts are full-width with subtle box-shadow.")
    lines.append("")
    
    lines.append("COVER PAGE:")
    lines.append("  - Dark background (#181b22)")
    lines.append("  - Large title in serif (48px, color: #ffffff)")
    lines.append("  - Subtitle in gold (24px, color: #d3af37)")
    lines.append(f"  - Restaurant name: {restaurant_name}")
    lines.append(f"  - Period analysed: {period_label}")
    lines.append("  - Prepared by: Data Intelligence")
    lines.append("  - No charts on cover page.")
    lines.append("")
    
    lines.append("FOOTER:")
    lines.append("  At bottom of each page:")
    lines.append("  <p class='footer'>Prepared by Data Intelligence – Confidential</p>")
    lines.append("  Style: font-size: 11px; color: #666; text-align: center; margin-top: 60px;")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # C. REPORT STRUCTURE (FIXED ORDER)
    # -------------------------------------------------------------------------
    subsection("C. REPORT STRUCTURE (FIXED ORDER)")
    
    lines.append("AI MUST GENERATE THE REPORT SECTIONS IN THIS EXACT ORDER:")
    lines.append("")
    
    report_sections = [
        "1. Cover Page",
        "2. Executive Summary",
        "3. Data & Approach",
        "4. Topline Metrics",
        "5. Category Performance",
        "6. Menu Engineering Analysis",
        "7. Waste Analysis",
        "8. Staff Performance (if data available)",
        "9. Booking Behaviour (if data available)",
        "10. Scenario Modelling",
        "11. Opportunities & Risks",
        "12. 90-Day Action Plan",
        "13. 12-Month Profit Projection",
        "14. Strategic Recommendations",
        "15. Appendix (Full Tables)",
    ]
    
    for section_title in report_sections:
        lines.append(f"  {section_title}")
    
    lines.append("")
    lines.append("Each section must be wrapped in a <div class='section-card'> element.")
    lines.append("Each section must have an <h2> heading with the section title.")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # D. CHART HANDLING RULES
    # -------------------------------------------------------------------------
    subsection("D. CHART HANDLING RULES")
    
    lines.append("Charts must be embedded exactly like this:")
    lines.append("")
    lines.append("  <img src='CHART_PATH' class='chart' alt='DESCRIPTION'>")
    lines.append("")
    lines.append("With captions:")
    lines.append("")
    lines.append("  <p class='chart-caption'>Figure X: DESCRIPTION</p>")
    lines.append("")
    lines.append("CSS for charts:")
    lines.append("  .chart {")
    lines.append("    width: 100%;")
    lines.append("    max-width: 900px;")
    lines.append("    margin: 25px auto 40px;")
    lines.append("    display: block;")
    lines.append("    border-radius: 6px;")
    lines.append("    box-shadow: 0 4px 12px rgba(0,0,0,0.4);")
    lines.append("  }")
    lines.append("  .chart-caption {")
    lines.append("    text-align: center;")
    lines.append("    font-size: 13px;")
    lines.append("    color: #999;")
    lines.append("    margin-top: -30px;")
    lines.append("    margin-bottom: 40px;")
    lines.append("  }")
    lines.append("")
    
    if charts:
        lines.append("AVAILABLE CHARTS (in preferred order):")
        for i, (chart_name, filename) in enumerate(charts.items(), 1):
            lines.append(f"  {i}. {chart_name}: {filename}")
    else:
        lines.append("AVAILABLE CHARTS: None provided")
    
    lines.append("")
    lines.append("If chart missing → insert:")
    lines.append("  <p class='note'>Chart not provided.</p>")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # E. EXECUTIVE SUMMARY RULESET
    # -------------------------------------------------------------------------
    subsection("E. EXECUTIVE SUMMARY RULESET")
    
    lines.append("The Executive Summary must include:")
    lines.append("")
    lines.append("1. One paragraph describing the restaurant's overall performance")
    lines.append("   - Reference: Total Revenue, GP%, Days of Data (SECTION 1)")
    lines.append("   - Tone: Adjust based on data_reliability_score")
    lines.append("")
    lines.append("2. 3–5 high-impact opportunities from insight_graph (sorted by impact_high DESC)")
    lines.append(f"   - Available opportunities: {len(insight_graph.opportunities)}")
    lines.append("   - Format: Bullet list with impact ranges")
    lines.append("")
    lines.append("3. 3–5 critical risks from insight_graph (sorted by strength DESC)")
    lines.append(f"   - Available risks: {len(insight_graph.risks)}")
    lines.append("   - Format: Bullet list with urgency indicators")
    lines.append("")
    lines.append("4. One paragraph describing GP uplift potential")
    lines.append(f"   - Conservative: {currency}{uplift_low:,.0f}")
    lines.append(f"   - Optimistic: {currency}{uplift_high:,.0f}")
    lines.append(f"   - Midpoint: {currency}{(uplift_low + uplift_high) / 2:,.0f}")
    lines.append("")
    lines.append("5. One paragraph describing data quality/reliability")
    lines.append(f"   - Current score: {reliability.get('score', 0)}/100 ({reliability.get('level', 'unknown').upper()})")
    lines.append("   - Reference: insight_graph.meta.data_reliability.notes")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # F. SECTION-BY-SECTION NARRATIVE RULES
    # -------------------------------------------------------------------------
    subsection("F. SECTION-BY-SECTION NARRATIVE RULES")
    
    lines.append("Every section must follow this narrative template:")
    lines.append("")
    lines.append("1. Short headline sentence (what this section covers)")
    lines.append("2. Interpretation of core metrics from tables")
    lines.append("3. Insight_graph references:")
    lines.append("   - Filter opportunities/risks by tags matching the section")
    lines.append("   - Example: waste section → filter by 'waste' tag")
    lines.append("   - Example: category section → filter by category name")
    lines.append("4. Forward-looking recommendations (actionable next steps)")
    lines.append("5. If low reliability (<60), prepend:")
    lines.append("   <p class='note'>Insights limited due to data reliability.</p>")
    lines.append("")
    
    lines.append("SECTION-SPECIFIC GUIDANCE:")
    lines.append("")
    lines.append("TOPLINE METRICS:")
    lines.append("  - Display as large stat cards (revenue, GP%, waste impact)")
    lines.append("  - Use gold color (#d3af37) for key numbers")
    lines.append("  - Include period and currency")
    lines.append("")
    lines.append("CATEGORY PERFORMANCE:")
    lines.append("  - Render full category_performance_table from SECTION 2")
    lines.append("  - Highlight top/bottom performers")
    lines.append("  - Reference category-specific insights from insight_graph")
    lines.append("")
    lines.append("MENU ENGINEERING:")
    lines.append("  - Embed menu_engineering chart")
    lines.append("  - Render Stars/Plowhorses/Puzzles/Dogs tables from SECTION 2")
    lines.append("  - Reference underpriced/overpriced insights")
    lines.append("")
    lines.append("WASTE ANALYSIS:")
    lines.append("  - Embed top_waste_items chart")
    lines.append("  - Render top_waste_items_table from SECTION 2")
    lines.append("  - Reference avoidable_waste opportunities from insight_graph")
    lines.append("")
    lines.append("STAFF PERFORMANCE:")
    lines.append("  - Only render if staff_performance_table exists in SECTION 2")
    lines.append("  - Otherwise: <p class='note'>No staff data provided.</p>")
    lines.append("")
    lines.append("BOOKING BEHAVIOUR:")
    lines.append("  - Only render if booking_summary_table exists in SECTION 2")
    lines.append("  - Otherwise: <p class='note'>No booking data provided.</p>")
    lines.append("")
    lines.append("SCENARIO MODELLING:")
    lines.append("  - Render scenario_summaries from SECTION 2")
    elasticity = meta.get("price_elasticity_assumption", -1.0)
    lines.append(f"  - Note: All scenarios use price elasticity of {elasticity:.1f}")
    lines.append(f"    (meaning {abs(elasticity):.0f}% volume change per 1% price change)")
    lines.append("  - Explain impact of each scenario")
    lines.append("  - GP changes are typically more significant than revenue changes")
    lines.append("  - No invented scenarios")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # G. 90-DAY ACTION PLAN RULES
    # -------------------------------------------------------------------------
    subsection("G. 90-DAY ACTION PLAN RULES")
    
    lines.append("Must contain exactly 4 phases with specific timelines:")
    lines.append("")
    lines.append("PHASE 1 (Weeks 1–2): WASTE & LOSS CONTROL")
    lines.append("  - Tasks: Address critical waste risks from insight_graph")
    lines.append("  - Priority: HIGH urgency risks first")
    lines.append("  - Format: Bullet list with specific items to fix")
    lines.append("  - Expected impact: Sum of high_waste_ratio opportunities")
    lines.append("")
    lines.append("PHASE 2 (Weeks 3–4): PRICING ADJUSTMENTS")
    lines.append("  - Tasks: Underpriced items from insight_graph")
    lines.append("  - Reference: underpriced_high_performer opportunities")
    lines.append("  - Format: Item name + current price + suggested adjustment")
    lines.append("  - Expected impact: Sum of pricing opportunities")
    lines.append("")
    lines.append("PHASE 3 (Weeks 5–8): MENU REDESIGN & RE-ANCHORING")
    lines.append("  - Tasks: Address Dogs, promote Stars")
    lines.append("  - Reference: menu engineering tables from SECTION 2")
    lines.append("  - Format: Strategic menu changes")
    lines.append("")
    lines.append("PHASE 4 (Weeks 9–12): STAFF TRAINING + REVENUE UPLIFT")
    lines.append("  - Tasks: Operational improvements")
    lines.append("  - Reference: Staff performance insights (if available)")
    lines.append("  - Format: Training programs, process improvements")
    lines.append("")
    lines.append("Each task must reference actual insights from insight_graph.")
    lines.append("No generic tasks – everything must be data-driven.")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # H. 12-MONTH PROFIT PROJECTION RULES
    # -------------------------------------------------------------------------
    subsection("H. 12-MONTH PROFIT PROJECTION RULES")
    
    elasticity = meta.get("price_elasticity_assumption", -1.0)
    lines.append("Projection must be based ONLY on:")
    lines.append("")
    lines.append("  - Baseline GP after waste (from TOPLINE_METRICS)")
    lines.append("  - Scenario uplift estimates (from scenario_summaries)")
    lines.append("  - Waste reduction potential (from waste opportunities)")
    lines.append("  - Opportunity total_low and total_high (insight_graph.meta)")
    lines.append("")
    lines.append(f"NOTE: Scenarios calculated with price elasticity {elasticity:.1f}.")
    lines.append("Focus projections on GP improvements, not revenue growth.")
    lines.append("")
    lines.append("OUTPUT FORMAT:")
    lines.append("")
    lines.append("  Projected Annual GP After Waste:")
    lines.append(f"    - Baseline (no changes): {currency}{summary.get('total_gp_after_waste', 0):,.0f}")
    lines.append(f"    - Conservative (50% implementation): {currency}{summary.get('total_gp_after_waste', 0) + uplift_low:,.0f}")
    lines.append(f"    - Target (midpoint): {currency}{summary.get('total_gp_after_waste', 0) + (uplift_low + uplift_high) / 2:,.0f}")
    lines.append(f"    - Optimistic (full implementation): {currency}{summary.get('total_gp_after_waste', 0) + uplift_high:,.0f}")
    lines.append("")
    lines.append("NEVER invent growth rates or percentages.")
    lines.append("NEVER assume compound growth.")
    lines.append("All projections must tie directly to insight_graph opportunities.")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # I. SAFE FALLBACKS
    # -------------------------------------------------------------------------
    subsection("I. SAFE FALLBACKS")
    
    lines.append("If insight_graph empty → produce generic 'insufficient insight' messages")
    lines.append("If scenarios missing → skip scenario modelling section")
    lines.append("If staff data missing → render placeholder note card")
    lines.append("If bookings data missing → render placeholder note card")
    lines.append("If waste_df empty → skip waste card")
    lines.append("If reliability_score < 40 → soften language significantly, add warnings")
    lines.append("If reliability_score < 60 → add caveats to all sections")
    lines.append("If any table missing → skip that specific subsection gracefully")
    lines.append("")
    lines.append("NEVER crash. NEVER leave blank sections. Always render something professional.")
    lines.append("")
    
    # -------------------------------------------------------------------------
    # J. FINAL REQUIREMENT
    # -------------------------------------------------------------------------
    subsection("J. FINAL REQUIREMENT")
    
    lines.append("The AI MUST generate a COMPLETE, CLIENT-READY premium HTML report that can be")
    lines.append("converted directly to PDF without any modification.")
    lines.append("")
    lines.append("NO missing sections.")
    lines.append("NO placeholder TODO text.")
    lines.append("NO markdown.")
    lines.append("HTML only.")
    lines.append("Professional design matching dark-slate + gold theme.")
    lines.append("Every number must be traceable to SECTION 1, 2, or 3.")
    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF AI REPORTING RULES")
    lines.append("=" * 80)
    lines.append("")
    
    # =============================================================================
    # SECTION 6: HTML SECTION HOOKS
    # =============================================================================
    section("SECTION 6: HTML SECTION HOOKS")
    
    lines.append("HTML_SECTIONS:")
    lines.append("Use these IDs when generating HTML structure:\n")
    
    html_sections = {
        "EXEC_SUMMARY": "Executive Summary - 3-5 paragraphs covering headline findings, data quality, and overall verdict",
        "FINANCIAL_OVERVIEW": "Financial Performance - Revenue, GP%, waste impact, key metrics from TOPLINE_METRICS",
        "MENU_ENGINEERING": "Menu Engineering Analysis - Stars/Plowhorses/Puzzles/Dogs breakdown with recommendations",
        "WASTE_STRATEGY": "Waste Management Strategy - Top waste items, reduction opportunities, forecasting recommendations",
        "PRICING_RECOMMENDATIONS": "Pricing Strategy - Underpriced/overpriced items, category benchmarking, elasticity considerations",
        "CATEGORY_ANALYSIS": "Category Performance - Deep dive into each category's contribution, GP%, volume trends",
        "STAFF_ANALYSIS": "Staff Performance Analysis - Efficiency metrics, top performers, scheduling recommendations (IF DATA AVAILABLE)",
        "BOOKINGS_ANALYSIS": "Bookings & Demand Analysis - Cover trends, no-shows, peak periods, capacity utilization (IF DATA AVAILABLE)",
        "SCENARIO_ANALYSIS": "Financial Scenarios - Impact modeling from scenario analysis section",
        "OPPORTUNITIES": "Profit Improvement Opportunities - Detailed breakdown of TOP 10 OPPORTUNITIES with action steps",
        "RISKS": "Critical Risk Factors - Detailed breakdown of TOP 10 RISKS with mitigation strategies",
        "ACTION_PLAN_90_DAYS": "90-Day Action Plan - Prioritized timeline with quick wins (Week 1-2), medium-term fixes (Week 3-8), long-term initiatives (Week 9-12)",
        "PROFIT_PROJECTION_12_MONTH": "12-Month Profit Projection - Conservative/realistic/optimistic scenarios based on implementing recommendations",
        "STRATEGIC_RECOMMENDATIONS": "Strategic Direction - High-level guidance on menu strategy, operational focus, investment priorities",
    }
    
    for section_id, description in html_sections.items():
        lines.append(f"  {section_id}:")
        lines.append(f"    {description}")
        lines.append("")
    
    # =============================================================================
    # SECTION 7: CHART REFERENCES
    # =============================================================================
    section("SECTION 7: CHART REFERENCES")
    
    if charts:
        lines.append("CHARTS:")
        lines.append("These charts are available for embedding in the report:\n")
        
        for chart_name, filename in charts.items():
            lines.append(f"  {chart_name}: {filename}")
        
        lines.append("\nREFERENCE FORMAT:")
        lines.append("  In narrative: 'See [Chart Name] for visual breakdown'")
        lines.append("  In HTML: <img src='charts/{filename}' alt='{chart_name}'>")
    else:
        lines.append("No charts provided in this export.")
    
    # =============================================================================
    # FOOTER
    # =============================================================================
    lines.append("\n" + "=" * 80)
    lines.append("END OF EXPORT BLOCK")
    lines.append("=" * 80)
    lines.append(f"\nGenerated: {period_label}")
    lines.append(f"Total Opportunities: {len(insight_graph.opportunities)}")
    lines.append(f"Total Risks: {len(insight_graph.risks)}")
    lines.append(f"Data Reliability: {reliability.get('score', 0)}/100")
    lines.append(f"Potential Uplift: {currency}{uplift_low:,.0f} - {currency}{uplift_high:,.0f}")
    
    return "\n".join(lines)
