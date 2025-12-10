"""
Engine Self-Diagnostic Module v2
=================================

Real-time data quality analysis that runs BEFORE the main engine.
Prints warnings, detects anomalies, and prevents bad data from corrupting analysis.

Features:
- Pre-flight data validation
- Anomaly detection (test data, synthetic patterns)
- Quality scoring per data source
- Actionable recommendations
- Auto-fix suggestions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings


@dataclass
class DiagnosticResult:
    """Container for diagnostic findings"""
    severity: str  # "info", "warning", "error", "critical"
    category: str  # "menu", "sales", "waste", "general"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    auto_fixable: bool = False


class EngineDiagnostics:
    """Advanced diagnostic system for restaurant data"""
    
    def __init__(self):
        self.findings: List[DiagnosticResult] = []
        self.quality_scores: Dict[str, float] = {}
    
    def diagnose_all(
        self,
        menu_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        waste_df: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """
        Run complete diagnostic suite
        
        Returns:
            {
                "overall_quality": float (0-100),
                "findings": List[DiagnosticResult],
                "safe_to_proceed": bool,
                "quality_by_source": Dict[str, float]
            }
        """
        self.findings = []
        self.quality_scores = {}
        
        print("\n" + "="*80)
        print("🔍 ENGINE SELF-DIAGNOSTICS - Pre-Flight Check")
        print("="*80 + "\n")
        
        # Run all diagnostic checks
        self._diagnose_menu(menu_df)
        self._diagnose_sales(sales_df, menu_df)
        if waste_df is not None and not waste_df.empty:
            self._diagnose_waste(waste_df, menu_df)
        self._diagnose_relationships(menu_df, sales_df, waste_df)
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality()
        
        # Determine if safe to proceed
        critical_count = sum(1 for f in self.findings if f.severity == "critical")
        error_count = sum(1 for f in self.findings if f.severity == "error")
        safe_to_proceed = critical_count == 0 and error_count < 3
        
        # Print report
        self._print_diagnostic_report(overall_quality, safe_to_proceed)
        
        return {
            "overall_quality": overall_quality,
            "findings": self.findings,
            "safe_to_proceed": safe_to_proceed,
            "quality_by_source": self.quality_scores
        }
    
    def _diagnose_menu(self, menu_df: pd.DataFrame):
        """Diagnose menu data quality"""
        print("📋 Analyzing Menu Data...")
        score = 100.0
        
        # Check 1: Required columns
        required_cols = ["item_name", "category", "sell_price", "cost_per_unit"]
        missing_cols = [col for col in required_cols if col not in menu_df.columns]
        if missing_cols:
            self.findings.append(DiagnosticResult(
                severity="critical",
                category="menu",
                message=f"Missing required columns: {missing_cols}",
                details={"available_columns": list(menu_df.columns)},
                recommendation="Add missing columns to menu file or update column mapping",
                auto_fixable=False
            ))
            score -= 50
        
        if not missing_cols:
            # Check 2: Duplicate item names
            duplicates = menu_df["item_name"].duplicated().sum()
            if duplicates > 0:
                dup_names = menu_df[menu_df["item_name"].duplicated(keep=False)]["item_name"].unique()
                self.findings.append(DiagnosticResult(
                    severity="critical",
                    category="menu",
                    message=f"{duplicates} duplicate item names found",
                    details={"duplicate_names": list(dup_names)[:5]},
                    recommendation="Remove or rename duplicate items",
                    auto_fixable=False
                ))
                score -= 30
            
            # Check 3: Missing/empty categories
            null_cats = menu_df["category"].isna().sum()
            empty_cats = (menu_df["category"].astype(str).str.strip() == "").sum()
            if null_cats + empty_cats > 0:
                self.findings.append(DiagnosticResult(
                    severity="error",
                    category="menu",
                    message=f"{null_cats + empty_cats} items have missing/empty categories",
                    recommendation="Assign categories to all items",
                    auto_fixable=False
                ))
                score -= 20
            
            # Check 4: Negative GP items
            try:
                # Convert to numeric safely
                sell_price_numeric = pd.to_numeric(menu_df["sell_price"], errors="coerce")
                cost_numeric = pd.to_numeric(menu_df["cost_per_unit"], errors="coerce")
                
                negative_gp = ((cost_numeric > sell_price_numeric) & 
                              (sell_price_numeric > 0)).sum()
                if negative_gp > 0:
                    neg_items = menu_df[
                        (cost_numeric > sell_price_numeric) & 
                        (sell_price_numeric > 0)
                    ]["item_name"].tolist()
                    self.findings.append(DiagnosticResult(
                        severity="warning",
                        category="menu",
                        message=f"{negative_gp} items have cost > price (negative GP)",
                        details={"items": neg_items[:5]},
                        recommendation="Review pricing - these items lose money on every sale",
                        auto_fixable=False
                    ))
                    score -= 10
            except Exception as e:
                self.findings.append(DiagnosticResult(
                    severity="error",
                    category="menu",
                    message=f"Cannot compare cost vs price (mixed data types)",
                    details={"error": str(e)},
                    recommendation="Ensure sell_price and cost_per_unit are numeric",
                    auto_fixable=False
                ))
                score -= 15
            
            # Check 5: Zero/null prices
            try:
                sell_price_numeric = pd.to_numeric(menu_df["sell_price"], errors="coerce")
                zero_prices = (sell_price_numeric <= 0).sum()
                if zero_prices > 0:
                    self.findings.append(DiagnosticResult(
                        severity="warning",
                        category="menu",
                        message=f"{zero_prices} items have zero or negative prices",
                        recommendation="Verify if these are free items or data errors",
                        auto_fixable=False
                    ))
                    score -= 5
            except:
                pass  # Already handled by negative GP check
            
            # Check 6: Test data patterns
            unique_prices = menu_df["sell_price"].nunique()
            if unique_prices < len(menu_df) * 0.3 and len(menu_df) > 10:
                self.findings.append(DiagnosticResult(
                    severity="info",
                    category="menu",
                    message=f"Only {unique_prices} unique prices for {len(menu_df)} items",
                    details={"unique_prices": unique_prices, "total_items": len(menu_df)},
                    recommendation="This may indicate placeholder/test data",
                    auto_fixable=False
                ))
                score -= 5
            
            # Check 7: Extreme values
            try:
                sell_price_numeric = pd.to_numeric(menu_df["sell_price"], errors="coerce")
                max_price = sell_price_numeric.max()
                min_price = sell_price_numeric[sell_price_numeric > 0].min() if (sell_price_numeric > 0).any() else 0
                if max_price > 100 or (min_price > 0 and min_price < 1):
                    self.findings.append(DiagnosticResult(
                        severity="info",
                        category="menu",
                        message=f"Extreme price range: £{min_price:.2f} - £{max_price:.2f}",
                        recommendation="Verify prices are in correct currency/units",
                        auto_fixable=False
                    ))
            except:
                pass  # Skip if cannot parse prices
            
            # Check 8: Category variety
            n_categories = menu_df["category"].nunique()
            if n_categories < 3 and len(menu_df) > 20:
                self.findings.append(DiagnosticResult(
                    severity="info",
                    category="menu",
                    message=f"Only {n_categories} categories for {len(menu_df)} items",
                    recommendation="Consider adding more categories for better analysis",
                    auto_fixable=False
                ))
        
        self.quality_scores["menu"] = max(0, score)
        print(f"   Menu Quality Score: {self.quality_scores['menu']:.1f}/100")
    
    def _diagnose_sales(self, sales_df: pd.DataFrame, menu_df: pd.DataFrame):
        """Diagnose sales data quality"""
        print("📊 Analyzing Sales Data...")
        score = 100.0
        
        # Check 1: Required columns
        required_cols = ["item_name", "qty", "order_datetime"]
        missing_cols = [col for col in required_cols if col not in sales_df.columns]
        if missing_cols:
            self.findings.append(DiagnosticResult(
                severity="critical",
                category="sales",
                message=f"Missing required columns: {missing_cols}",
                recommendation="Add missing columns to sales file",
                auto_fixable=False
            ))
            score -= 50
        
        if not missing_cols:
            # Check 2: Date parsing
            sales_copy = sales_df.copy()
            sales_copy["order_datetime"] = pd.to_datetime(sales_copy["order_datetime"], errors="coerce")
            unparseable_dates = sales_copy["order_datetime"].isna().sum()
            
            if unparseable_dates > 0:
                pct_failed = (unparseable_dates / len(sales_df)) * 100
                severity = "critical" if pct_failed > 5 else "warning"
                self.findings.append(DiagnosticResult(
                    severity=severity,
                    category="sales",
                    message=f"{unparseable_dates} rows ({pct_failed:.1f}%) have unparseable dates",
                    recommendation="Fix date format (expected: YYYY-MM-DD or DD/MM/YYYY)",
                    auto_fixable=False
                ))
                score -= min(30, pct_failed * 2)
            
            # Check 3: Items not in menu
            if "item_name" in menu_df.columns:
                menu_items = set(menu_df["item_name"])
                sales_items = set(sales_df["item_name"])
                unmatched_items = sales_items - menu_items
                
                if unmatched_items:
                    unmatched_count = sales_df["item_name"].isin(unmatched_items).sum()
                    pct_unmatched = (unmatched_count / len(sales_df)) * 100
                    
                    severity = "error" if pct_unmatched > 20 else "warning" if pct_unmatched > 5 else "info"
                    self.findings.append(DiagnosticResult(
                        severity=severity,
                        category="sales",
                        message=f"{len(unmatched_items)} items in sales not found in menu ({pct_unmatched:.1f}% of transactions)",
                        details={"unmatched_items": list(unmatched_items)[:10], "pct_affected": pct_unmatched},
                        recommendation="Add missing items to menu or fix item name mismatches",
                        auto_fixable=False
                    ))
                    score -= min(25, pct_unmatched)
            
            # Check 4: Negative quantities
            negative_qty = (sales_df["qty"] < 0).sum()
            if negative_qty > 0:
                self.findings.append(DiagnosticResult(
                    severity="error",
                    category="sales",
                    message=f"{negative_qty} transactions have negative quantities",
                    recommendation="Remove or fix negative quantity records",
                    auto_fixable=False
                ))
                score -= 15
            
            # Check 5: Zero quantities
            zero_qty = (sales_df["qty"] == 0).sum()
            if zero_qty > 0:
                self.findings.append(DiagnosticResult(
                    severity="warning",
                    category="sales",
                    message=f"{zero_qty} transactions have zero quantity",
                    recommendation="Remove zero-quantity records",
                    auto_fixable=True
                ))
                score -= 5
            
            # Check 6: Data volume
            if len(sales_df) < 100:
                self.findings.append(DiagnosticResult(
                    severity="warning",
                    category="sales",
                    message=f"Only {len(sales_df)} sales transactions (very low volume)",
                    recommendation="Collect more data for reliable insights (recommended: 1000+ transactions)",
                    auto_fixable=False
                ))
                score -= 10
            
            # Check 7: Date range
            if unparseable_dates < len(sales_df):
                date_range = (sales_copy["order_datetime"].max() - sales_copy["order_datetime"].min()).days
                if date_range < 7:
                    self.findings.append(DiagnosticResult(
                        severity="info",
                        category="sales",
                        message=f"Only {date_range} days of data",
                        recommendation="More data (30+ days) provides more reliable insights",
                        auto_fixable=False
                    ))
                    score -= 5
            
            # Check 8: Unusual patterns
            item_counts = sales_df["item_name"].value_counts()
            if len(item_counts) > 0:
                top_item_pct = (item_counts.iloc[0] / len(sales_df)) * 100
                if top_item_pct > 50:
                    self.findings.append(DiagnosticResult(
                        severity="info",
                        category="sales",
                        message=f"One item represents {top_item_pct:.1f}% of all sales",
                        details={"item": item_counts.index[0], "pct": top_item_pct},
                        recommendation="Verify this is not test data",
                        auto_fixable=False
                    ))
        
        self.quality_scores["sales"] = max(0, score)
        print(f"   Sales Quality Score: {self.quality_scores['sales']:.1f}/100")
    
    def _diagnose_waste(self, waste_df: pd.DataFrame, menu_df: pd.DataFrame):
        """Diagnose waste data quality"""
        print("🗑️  Analyzing Waste Data...")
        score = 100.0
        
        # Check 1: Items not in menu
        if "item_name" in menu_df.columns and "item_name" in waste_df.columns:
            menu_items = set(menu_df["item_name"])
            waste_items = set(waste_df["item_name"])
            unmatched_items = waste_items - menu_items
            
            if unmatched_items:
                pct_unmatched = (len(unmatched_items) / len(waste_df)) * 100
                severity = "warning" if pct_unmatched > 10 else "info"
                self.findings.append(DiagnosticResult(
                    severity=severity,
                    category="waste",
                    message=f"{len(unmatched_items)} waste items not found in menu",
                    details={"unmatched_items": list(unmatched_items)[:5]},
                    recommendation="These items will be excluded from waste analysis",
                    auto_fixable=False
                ))
                score -= min(20, pct_unmatched)
        
        # Check 2: Negative waste
        if "waste_qty" in waste_df.columns:
            negative_waste = (waste_df["waste_qty"] < 0).sum()
            if negative_waste > 0:
                self.findings.append(DiagnosticResult(
                    severity="warning",
                    category="waste",
                    message=f"{negative_waste} items have negative waste quantities",
                    recommendation="Fix or remove negative waste records",
                    auto_fixable=False
                ))
                score -= 15
        
        # Check 3: Extremely high waste
        if "waste_qty" in waste_df.columns:
            high_waste = (waste_df["waste_qty"] > 100).sum()
            if high_waste > 0:
                self.findings.append(DiagnosticResult(
                    severity="info",
                    category="waste",
                    message=f"{high_waste} items have very high waste quantities (>100 units)",
                    recommendation="Verify waste quantities are correct",
                    auto_fixable=False
                ))
        
        self.quality_scores["waste"] = max(0, score)
        print(f"   Waste Quality Score: {self.quality_scores['waste']:.1f}/100")
    
    def _diagnose_relationships(self, menu_df: pd.DataFrame, sales_df: pd.DataFrame, waste_df: pd.DataFrame):
        """Diagnose relationships between data sources"""
        print("🔗 Analyzing Data Relationships...")
        
        # Check: Menu coverage in sales
        if "item_name" in menu_df.columns and "item_name" in sales_df.columns:
            menu_items = set(menu_df["item_name"])
            sales_items = set(sales_df["item_name"])
            
            unsold_items = menu_items - sales_items
            if len(unsold_items) > len(menu_df) * 0.3:
                self.findings.append(DiagnosticResult(
                    severity="info",
                    category="general",
                    message=f"{len(unsold_items)} menu items have zero sales",
                    details={"pct_unsold": (len(unsold_items) / len(menu_df)) * 100},
                    recommendation="These items may be new, seasonal, or should be removed",
                    auto_fixable=False
                ))
    
    def _calculate_overall_quality(self) -> float:
        """Calculate overall data quality score"""
        if not self.quality_scores:
            return 0.0
        
        # Weighted average (menu and sales most important)
        weights = {"menu": 0.4, "sales": 0.5, "waste": 0.1}
        total_weight = sum(weights.get(k, 0.1) for k in self.quality_scores.keys())
        
        weighted_sum = sum(
            score * weights.get(source, 0.1)
            for source, score in self.quality_scores.items()
        )
        
        return weighted_sum / total_weight
    
    def _print_diagnostic_report(self, overall_quality: float, safe_to_proceed: bool):
        """Print formatted diagnostic report"""
        print("\n" + "="*80)
        print("📊 DIAGNOSTIC SUMMARY")
        print("="*80 + "\n")
        
        # Overall quality
        if overall_quality >= 90:
            quality_label = "EXCELLENT ✅"
        elif overall_quality >= 75:
            quality_label = "GOOD ✓"
        elif overall_quality >= 60:
            quality_label = "ACCEPTABLE ⚠️"
        else:
            quality_label = "POOR ❌"
        
        print(f"Overall Data Quality: {overall_quality:.1f}/100 ({quality_label})")
        print(f"Safe to Proceed: {'YES ✅' if safe_to_proceed else 'NO ❌'}\n")
        
        # Quality by source
        print("Quality by Source:")
        for source, score in self.quality_scores.items():
            print(f"   {source.capitalize()}: {score:.1f}/100")
        
        # Findings by severity
        findings_by_severity = {
            "critical": [f for f in self.findings if f.severity == "critical"],
            "error": [f for f in self.findings if f.severity == "error"],
            "warning": [f for f in self.findings if f.severity == "warning"],
            "info": [f for f in self.findings if f.severity == "info"]
        }
        
        print(f"\nFindings: {len(self.findings)} total")
        if findings_by_severity["critical"]:
            print(f"   🔴 Critical: {len(findings_by_severity['critical'])}")
        if findings_by_severity["error"]:
            print(f"   🟠 Errors: {len(findings_by_severity['error'])}")
        if findings_by_severity["warning"]:
            print(f"   🟡 Warnings: {len(findings_by_severity['warning'])}")
        if findings_by_severity["info"]:
            print(f"   🔵 Info: {len(findings_by_severity['info'])}")
        
        # Print detailed findings
        if findings_by_severity["critical"]:
            print("\n🔴 CRITICAL ISSUES (must fix before proceeding):")
            for f in findings_by_severity["critical"]:
                print(f"\n   • {f.message}")
                if f.recommendation:
                    print(f"     → {f.recommendation}")
        
        if findings_by_severity["error"]:
            print("\n🟠 ERRORS (strongly recommended to fix):")
            for f in findings_by_severity["error"][:5]:  # Limit to 5
                print(f"\n   • {f.message}")
                if f.recommendation:
                    print(f"     → {f.recommendation}")
        
        if findings_by_severity["warning"]:
            print("\n🟡 WARNINGS (recommended to review):")
            for f in findings_by_severity["warning"][:3]:  # Limit to 3
                print(f"\n   • {f.message}")
                if f.recommendation:
                    print(f"     → {f.recommendation}")
        
        print("\n" + "="*80 + "\n")


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_diagnostics(
    menu_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    waste_df: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Quick diagnostic check - use before running main engine
    
    Example:
        diagnostics = run_diagnostics(menu_df, sales_df, waste_df)
        if diagnostics["safe_to_proceed"]:
            # Run main analysis
            results = run_full_analysis(...)
    """
    engine = EngineDiagnostics()
    return engine.diagnose_all(menu_df, sales_df, waste_df)


if __name__ == "__main__":
    # Example usage
    print("Engine Diagnostics Module - Use via run_diagnostics() function")
