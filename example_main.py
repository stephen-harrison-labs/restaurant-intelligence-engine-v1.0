import os
import json
import resturantv1 as engine


def main():
    """Run the restaurant intelligence engine and save outputs."""
    
    # Run analysis
    results = engine.run_full_analysis_v2(
        config=engine.CONFIG,
        data_source="synthetic",  # Change to "client" later when you have client data
    )
    
    # Print summary
    print("Summary metrics:")
    print(results["summary_metrics"])
    
    print("\nTop 5 items by margin:")
    print(
        results["perf_df"][["item_name", "category", "units_sold", "revenue", "gp_after_waste"]]
        .head(5)
        .to_string(index=False)
    )
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    # Save DataFrames as CSV
    results["menu_df"].to_csv("output/menu_df.csv", index=False)
    results["orders_df"].to_csv("output/orders_df.csv", index=False)
    results["perf_df"].to_csv("output/perf_df.csv", index=False)
    results["staff_df"].to_csv("output/staff_df.csv", index=False)
    results["bookings_df"].to_csv("output/bookings_df.csv", index=False)
    results["waste_df"].to_csv("output/waste_df.csv", index=False)
    
    # Save summary metrics as JSON
    with open("output/summary_metrics.json", "w") as f:
        json.dump(results["summary_metrics"], f, indent=2)
    
    # Save GPT export block as text
    with open("output/gpt_export_block.txt", "w", encoding="utf-8") as f:
        f.write(results["gpt_export_block"])
    
    # Save charts as PNG files
    engine.save_all_charts(results, output_dir="output", config=engine.CONFIG)
    
    # Save combined Excel workbook
    excel_path = os.path.join("output", "report_data.xlsx")
    engine.export_results_to_excel(results, excel_path)
    
    print("\nWrote outputs to ./output")
    print("Saved chart images to ./output")
    print(f"Saved combined Excel workbook to {excel_path}")


if __name__ == "__main__":
    main()
