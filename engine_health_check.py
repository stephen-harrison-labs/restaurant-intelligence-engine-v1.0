"""
Engine Health Dashboard
=======================

Pre-flight validation system that checks engine readiness before analysis.
Provides health scoring, recommendations, and go/no-go decision.

Use this at the start of every analysis run to catch issues early.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

# Import diagnostics module
try:
    from diagnostics_v2 import EngineDiagnostics, DiagnosticResult
except ImportError:
    print("⚠️  Warning: diagnostics_v2.py not found. Using simplified health check.")
    EngineDiagnostics = None


@dataclass
class HealthStatus:
    """Overall health status"""
    score: float  # 0-100
    level: str  # "excellent", "good", "acceptable", "poor", "critical"
    safe_to_proceed: bool
    issues_count: Dict[str, int]  # critical, error, warning, info
    recommendations: List[str]
    estimated_reliability: str  # "high", "medium", "low"


class EngineHealthDashboard:
    """Comprehensive health check system"""
    
    def __init__(self):
        self.health_status: Optional[HealthStatus] = None
        self.diagnostics_result: Optional[Dict] = None
    
    def check_health(
        self,
        menu_path: str | Path,
        sales_path: str | Path,
        waste_path: Optional[str | Path] = None
    ) -> HealthStatus:
        """
        Perform complete health check on data files
        
        Args:
            menu_path: Path to menu CSV/Excel file
            sales_path: Path to sales CSV/Excel file
            waste_path: Optional path to waste CSV/Excel file
        
        Returns:
            HealthStatus object with health metrics
        """
        print("\n" + "="*80)
        print("🏥 ENGINE HEALTH DASHBOARD")
        print("="*80)
        print("\n⏳ Running pre-flight checks...\n")
        
        # Step 1: File existence check
        print("📂 Checking file accessibility...")
        menu_path = Path(menu_path)
        sales_path = Path(sales_path)
        waste_path = Path(waste_path) if waste_path else None
        
        if not menu_path.exists():
            print(f"❌ CRITICAL: Menu file not found: {menu_path}")
            return self._create_failed_status("Menu file not found")
        
        if not sales_path.exists():
            print(f"❌ CRITICAL: Sales file not found: {sales_path}")
            return self._create_failed_status("Sales file not found")
        
        if waste_path and not waste_path.exists():
            print(f"⚠️  WARNING: Waste file not found: {waste_path}")
            waste_path = None
        
        print("   ✅ All files accessible\n")
        
        # Step 2: Load data
        print("📥 Loading data files...")
        try:
            menu_df = self._safe_load(menu_path)
            sales_df = self._safe_load(sales_path)
            waste_df = self._safe_load(waste_path) if waste_path else None
            print("   ✅ Data loaded successfully\n")
        except Exception as e:
            print(f"   ❌ CRITICAL: Failed to load data: {str(e)}\n")
            return self._create_failed_status(f"Data loading failed: {str(e)}")
        
        # Step 3: Basic validation
        print("🔍 Running basic validation...")
        basic_issues = self._basic_validation(menu_df, sales_df, waste_df)
        if basic_issues:
            print(f"   ⚠️  Found {len(basic_issues)} basic issues\n")
            for issue in basic_issues:
                print(f"      • {issue}")
        else:
            print("   ✅ Basic validation passed\n")
        
        # Step 4: Advanced diagnostics (if available)
        if EngineDiagnostics:
            print("🔬 Running advanced diagnostics...")
            diagnostics = EngineDiagnostics()
            self.diagnostics_result = diagnostics.diagnose_all(menu_df, sales_df, waste_df)
        else:
            print("⚠️  Skipping advanced diagnostics (module not available)\n")
            self.diagnostics_result = None
        
        # Step 5: Calculate health status
        health_status = self._calculate_health_status(
            menu_df, sales_df, waste_df, 
            basic_issues,
            self.diagnostics_result
        )
        
        self.health_status = health_status
        
        # Step 6: Print dashboard
        self._print_dashboard(health_status)
        
        return health_status
    
    def _safe_load(self, path: Path) -> pd.DataFrame:
        """Safely load CSV or Excel file"""
        if path is None:
            return pd.DataFrame()
        
        if path.suffix.lower() in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        else:
            # Try UTF-8 first, then latin-1
            try:
                return pd.read_csv(path, encoding='utf-8-sig')
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding='latin-1')
    
    def _basic_validation(
        self,
        menu_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        waste_df: Optional[pd.DataFrame]
    ) -> List[str]:
        """Run basic validation checks"""
        issues = []
        
        # Menu checks
        if menu_df.empty:
            issues.append("Menu file is empty")
        else:
            if "item_name" not in menu_df.columns:
                issues.append("Menu missing 'item_name' column")
            if "sell_price" not in menu_df.columns:
                issues.append("Menu missing 'sell_price' column")
            if "category" not in menu_df.columns:
                issues.append("Menu missing 'category' column")
        
        # Sales checks
        if sales_df.empty:
            issues.append("Sales file is empty")
        else:
            if "item_name" not in sales_df.columns:
                issues.append("Sales missing 'item_name' column")
            if len(sales_df) < 10:
                issues.append(f"Very few sales records ({len(sales_df)})")
        
        # Waste checks (if provided)
        if waste_df is not None and not waste_df.empty:
            if "item_name" not in waste_df.columns:
                issues.append("Waste missing 'item_name' column")
        
        return issues
    
    def _calculate_health_status(
        self,
        menu_df: pd.DataFrame,
        sales_df: pd.DataFrame,
        waste_df: Optional[pd.DataFrame],
        basic_issues: List[str],
        diagnostics_result: Optional[Dict]
    ) -> HealthStatus:
        """Calculate overall health status"""
        
        # Start with base score
        if diagnostics_result:
            score = diagnostics_result["overall_quality"]
            safe_to_proceed = diagnostics_result["safe_to_proceed"]
            
            # Count issues by severity
            findings = diagnostics_result["findings"]
            issues_count = {
                "critical": sum(1 for f in findings if f.severity == "critical"),
                "error": sum(1 for f in findings if f.severity == "error"),
                "warning": sum(1 for f in findings if f.severity == "warning"),
                "info": sum(1 for f in findings if f.severity == "info")
            }
            
            # Collect recommendations
            recommendations = [
                f.recommendation for f in findings 
                if f.recommendation and f.severity in ["critical", "error"]
            ][:5]  # Top 5
        else:
            # Fallback scoring based on basic validation
            score = 100.0
            score -= len(basic_issues) * 10
            score = max(0, score)
            
            safe_to_proceed = len(basic_issues) == 0
            issues_count = {
                "critical": len([i for i in basic_issues if "CRITICAL" in i]),
                "error": len([i for i in basic_issues if "missing" in i.lower()]),
                "warning": len([i for i in basic_issues if "few" in i.lower()]),
                "info": 0
            }
            recommendations = basic_issues[:5]
        
        # Determine health level
        if score >= 90:
            level = "excellent"
        elif score >= 75:
            level = "good"
        elif score >= 60:
            level = "acceptable"
        elif score >= 40:
            level = "poor"
        else:
            level = "critical"
        
        # Estimate reliability
        if score >= 85 and len(sales_df) > 1000:
            estimated_reliability = "high"
        elif score >= 70 and len(sales_df) > 100:
            estimated_reliability = "medium"
        else:
            estimated_reliability = "low"
        
        return HealthStatus(
            score=score,
            level=level,
            safe_to_proceed=safe_to_proceed,
            issues_count=issues_count,
            recommendations=recommendations,
            estimated_reliability=estimated_reliability
        )
    
    def _create_failed_status(self, reason: str) -> HealthStatus:
        """Create health status for critical failure"""
        return HealthStatus(
            score=0.0,
            level="critical",
            safe_to_proceed=False,
            issues_count={"critical": 1, "error": 0, "warning": 0, "info": 0},
            recommendations=[reason],
            estimated_reliability="low"
        )
    
    def _print_dashboard(self, status: HealthStatus):
        """Print formatted health dashboard"""
        print("\n" + "="*80)
        print("📊 HEALTH DASHBOARD SUMMARY")
        print("="*80 + "\n")
        
        # Overall score
        score_bar = self._create_progress_bar(status.score, 100)
        level_emoji = {
            "excellent": "🟢",
            "good": "🟢",
            "acceptable": "🟡",
            "poor": "🟠",
            "critical": "🔴"
        }
        
        print(f"Overall Health Score: {status.score:.1f}/100 {level_emoji.get(status.level, '❓')}")
        print(f"{score_bar}")
        print(f"Health Level: {status.level.upper()}")
        print(f"Estimated Reliability: {status.estimated_reliability.upper()}\n")
        
        # Issues breakdown
        print("Issues Found:")
        if status.issues_count["critical"] > 0:
            print(f"   🔴 Critical: {status.issues_count['critical']}")
        if status.issues_count["error"] > 0:
            print(f"   🟠 Errors: {status.issues_count['error']}")
        if status.issues_count["warning"] > 0:
            print(f"   🟡 Warnings: {status.issues_count['warning']}")
        if status.issues_count["info"] > 0:
            print(f"   🔵 Info: {status.issues_count['info']}")
        
        if sum(status.issues_count.values()) == 0:
            print("   ✅ No issues found!")
        
        print()
        
        # Go/No-Go decision
        if status.safe_to_proceed:
            print("✅ GO FOR LAUNCH")
            print("   Engine is ready to run. Data quality is sufficient for analysis.")
        else:
            print("❌ NO GO")
            print("   Critical issues must be fixed before running analysis.")
        
        print()
        
        # Top recommendations
        if status.recommendations:
            print("🎯 Top Recommendations:")
            for i, rec in enumerate(status.recommendations[:5], 1):
                print(f"   {i}. {rec}")
            print()
        
        # Reliability estimate
        print("🔮 Expected Analysis Quality:")
        if status.estimated_reliability == "high":
            print("   ✅ HIGH - Results will be reliable and actionable")
        elif status.estimated_reliability == "medium":
            print("   ⚠️  MEDIUM - Results should be validated against business knowledge")
        else:
            print("   ❌ LOW - Results may be unreliable, more/better data needed")
        
        print("\n" + "="*80 + "\n")
    
    def _create_progress_bar(self, value: float, max_value: float, width: int = 50) -> str:
        """Create ASCII progress bar"""
        filled = int((value / max_value) * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {value:.1f}%"


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_health_check(
    menu_path: str | Path,
    sales_path: str | Path,
    waste_path: Optional[str | Path] = None
) -> HealthStatus:
    """
    Quick health check - use at start of analysis
    
    Example:
        health = quick_health_check("data/menu.csv", "data/sales.csv")
        if health.safe_to_proceed:
            # Run analysis
            results = run_analysis(...)
        else:
            print(f"Fix these issues first: {health.recommendations}")
    """
    dashboard = EngineHealthDashboard()
    return dashboard.check_health(menu_path, sales_path, waste_path)


def automated_health_gate(
    menu_path: str | Path,
    sales_path: str | Path,
    waste_path: Optional[str | Path] = None,
    min_score: float = 60.0
) -> bool:
    """
    Automated go/no-go decision for CI/CD pipelines
    
    Returns:
        True if safe to proceed, False otherwise
    
    Example:
        if automated_health_gate("data/menu.csv", "data/sales.csv", min_score=70):
            run_analysis()
        else:
            sys.exit(1)  # Fail the pipeline
    """
    dashboard = EngineHealthDashboard()
    health = dashboard.check_health(menu_path, sales_path, waste_path)
    
    passes = health.safe_to_proceed and health.score >= min_score
    
    if passes:
        print(f"✅ AUTOMATED HEALTH GATE: PASS (Score: {health.score:.1f} >= {min_score})")
    else:
        print(f"❌ AUTOMATED HEALTH GATE: FAIL (Score: {health.score:.1f} < {min_score})")
    
    return passes


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Run health check from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check restaurant data health")
    parser.add_argument("menu", help="Path to menu file")
    parser.add_argument("sales", help="Path to sales file")
    parser.add_argument("--waste", help="Path to waste file (optional)")
    parser.add_argument("--min-score", type=float, default=60.0, help="Minimum acceptable score")
    parser.add_argument("--automated", action="store_true", help="Exit with status code for CI/CD")
    
    args = parser.parse_args()
    
    if args.automated:
        # Automated mode for CI/CD
        passes = automated_health_gate(args.menu, args.sales, args.waste, args.min_score)
        sys.exit(0 if passes else 1)
    else:
        # Interactive mode
        health = quick_health_check(args.menu, args.sales, args.waste)
        
        if health.safe_to_proceed:
            print("\n🎉 Data is ready for analysis!")
            sys.exit(0)
        else:
            print("\n⚠️  Please fix issues before proceeding.")
            sys.exit(1)


if __name__ == "__main__":
    main()
