import os
import sys
import json
import numpy as np
import resturantv1 as engine


def configure_client_paths():
    """
    Configure client data paths.
    
    User must edit the placeholder paths below to match their actual data files:
    - client_menu_path: CSV with menu items (columns: item_name, category, price, cogs, etc.)
    - client_sales_path: CSV with order/sales data (columns: date, item_name, quantity, etc.)
    - client_waste_path: CSV with waste data (columns: date, item_name, waste_units, waste_cost, etc.)
    """
    config = engine.CONFIG.copy()
    
    # Edit these paths to point to your actual client data files
    config["client_menu_path"] = "data/client_menu.csv"
    config["client_sales_path"] = "data/client_sales.csv"
    # If you do not have a waste file, leave as None
    config["client_waste_path"] = None
    
    print("Client configuration loaded.")
    print(f"  Menu path: {config['client_menu_path']}")
    print(f"  Sales path: {config['client_sales_path']}")
    print(f"  Waste path: {config['client_waste_path']}")
    print()
    
    return config


def main():
    """Run the restaurant intelligence engine on client data and save outputs."""
    # Client / Kaggle run entrypoint
    # Build a fresh config based on the engine defaults and point it to the
    # local client files produced by the Kaggle prep scripts.
    # We do not modify the engine module here — we only set the config used
    # for this client-mode execution.
    config = engine.CONFIG.copy()
    config["client_menu_path"] = "data/client_menu.csv"
    config["client_sales_path"] = "data/client_sales.csv"
    config["client_waste_path"] = None

    # Override restaurant metadata for this Kaggle demo run
    config["restaurant_name"] = "All Scientist Restaurant – Kaggle Dirty Data Demo"
    config["period_label"] = "Jan 2022 – Dec 2023"

    # Force the engine into client-mode for this run
    engine.DATA_SOURCE = "client"
    
    # Set output directory
    output_dir = "output_client"
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure required client data files exist (menu and sales are required; waste is optional)
    required_paths = ["client_menu_path", "client_sales_path"]
    missing = []
    for k in required_paths:
        p = config.get(k)
        if not p or not os.path.exists(p):
            missing.append(p or f"<missing {k}>")
    if missing:
        raise FileNotFoundError(
            "Client-mode data files are missing: " + ", ".join(missing) + 
            ".\nPlease provide the files at the configured paths in run_client.py and retry."
        )

    # Debug: print config before running
    print(f"DEBUG: Config restaurant_name = {config['restaurant_name']}")
    print(f"DEBUG: Config period_label = {config['period_label']}")
    print(f"DEBUG: Config client_menu_path = {config['client_menu_path']}")
    print(f"DEBUG: Config client_sales_path = {config['client_sales_path']}")
    print(f"DEBUG: About to call run_full_analysis_v2 with data_source='client'")

    # Run analysis on client data
    results = engine.run_full_analysis_v2(
        config=config,
        data_source="client",
    )
    
    # Debug: check what data was loaded
    print(f"\nDEBUG: Loaded {len(results['menu_df'])} menu items")
    print(f"DEBUG: Loaded {len(results['orders_df'])} orders")
    print(f"DEBUG: Date range in orders: {results['orders_df']['order_datetime'].min()} to {results['orders_df']['order_datetime'].max()}")
    print(f"DEBUG: Sample menu items: {results['menu_df']['item_name'].head(3).tolist()}\n")
    
    # Print summary
    print("Client summary metrics:")
    print(results["summary_metrics"])
    print()
    
    # Save DataFrames as CSV (only if present in results)
    save_map = {
        "menu_df": "menu_df.csv",
        "orders_df": "orders_df.csv",
        "perf_df": "perf_df.csv",
        "staff_df": "staff_df.csv",
        "bookings_df": "bookings_df.csv",
        "waste_df": "waste_df.csv",
    }
    for key, fname in save_map.items():
        if key in results and results.get(key) is not None:
            try:
                results[key].to_csv(os.path.join(output_dir, fname), index=False)
            except Exception:
                # ignore individual save failures
                pass
    
    # Save summary metrics as JSON (convert numpy/pandas scalars to native Python types)
    def _to_py(o):
        # recursive conversion for common numpy/pandas types
        if isinstance(o, dict):
            return {k: _to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_py(v) for v in o]
        if isinstance(o, np.ndarray):
            return _to_py(o.tolist())
        if isinstance(o, (np.generic,)):
            try:
                return o.item()
            except Exception:
                return o.tolist()
        # pandas Timestamp -> string
        try:
            import pandas as _pd

            if isinstance(o, _pd.Timestamp):
                return str(o)
        except Exception:
            pass
        return o

    serializable_summary = _to_py(results.get("summary_metrics", {}))
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(serializable_summary, f, indent=2)
    
    # Save GPT export block as text
    with open(os.path.join(output_dir, "gpt_export_block.txt"), "w", encoding="utf-8") as f:
        f.write(results["gpt_export_block"])
    
    # Save charts as PNG files
    engine.save_all_charts(results, output_dir=output_dir, config=config)

    # Save data quality diagnostics (if present)
    dq = results.get("data_quality_diagnostics")
    dq_notes = results.get("data_quality_notes")
    if dq:
        with open(os.path.join(output_dir, "data_quality_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(dq, f, indent=2)
    if dq_notes:
        with open(os.path.join(output_dir, "data_quality_notes.txt"), "w", encoding="utf-8") as f:
            for line in dq_notes:
                f.write(line.rstrip() + "\n")
    
    # Save validation report
    validation = results.get("validation_result")
    if validation:
        with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as f:
            json.dump(_to_py(validation), f, indent=2)
    
    # Optionally save time notes (if present)
    time_notes = results.get("time_notes")
    if time_notes:
        with open(os.path.join(output_dir, "time_notes.txt"), "w", encoding="utf-8") as f:
            for line in time_notes:
                f.write(line.rstrip() + "\n")

    # Export results to an Excel workbook (includes Data_Quality sheets when present)
    excel_path = os.path.join(output_dir, "report_data.xlsx")
    try:
        engine.export_results_to_excel(results, excel_path)
    except Exception:
        # keep client runner resilient; do not raise for optional excel export
        pass

    print(
        f"Wrote client outputs (CSVs, charts, export block, diagnostics JSON/TXT, Excel workbook) to ./{output_dir}"
    )


if __name__ == "__main__":
    main()
