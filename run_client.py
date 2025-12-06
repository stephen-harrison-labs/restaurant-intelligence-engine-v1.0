import os
import sys
import json
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
    config["client_waste_path"] = "data/client_waste.csv"
    
    print("Client configuration loaded.")
    print(f"  Menu path: {config['client_menu_path']}")
    print(f"  Sales path: {config['client_sales_path']}")
    print(f"  Waste path: {config['client_waste_path']}")
    print()
    
    return config


def main():
    """Run the restaurant intelligence engine on client data and save outputs."""
    
    # Configure client paths
    config = configure_client_paths()
    
    # Set output directory
    output_dir = "output_client"
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure client data files exist (strict client-mode only)
    required_keys = ["client_menu_path", "client_sales_path", "client_waste_path"]
    missing = []
    for k in required_keys:
        p = config.get(k)
        if not p or not os.path.exists(p):
            missing.append(p or f"<missing {k}>")
    if missing:
        raise FileNotFoundError(
            "Client-mode data files are missing: " + ", ".join(missing) + 
            ".\nPlease provide the files at the configured paths in run_client.py and retry."
        )

    # Run analysis on client data
    results = engine.run_full_analysis_v2(
        config=config,
        data_source="client",
    )
    
    # Print summary
    print("Client summary metrics:")
    print(results["summary_metrics"])
    print()
    
    # Save DataFrames as CSV
    results["menu_df"].to_csv(os.path.join(output_dir, "menu_df.csv"), index=False)
    results["orders_df"].to_csv(os.path.join(output_dir, "orders_df.csv"), index=False)
    results["perf_df"].to_csv(os.path.join(output_dir, "perf_df.csv"), index=False)
    results["staff_df"].to_csv(os.path.join(output_dir, "staff_df.csv"), index=False)
    results["bookings_df"].to_csv(os.path.join(output_dir, "bookings_df.csv"), index=False)
    results["waste_df"].to_csv(os.path.join(output_dir, "waste_df.csv"), index=False)
    
    # Save summary metrics as JSON
    with open(os.path.join(output_dir, "summary_metrics.json"), "w") as f:
        json.dump(results["summary_metrics"], f, indent=2)
    
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
