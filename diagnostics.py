import os
import json
import traceback
import resturantv1 as engine

# Try to import run_client if available (may raise ImportError)
try:
    import run_client
    _HAS_RUN_CLIENT = True
except Exception:
    run_client = None
    _HAS_RUN_CLIENT = False


def _fmt_currency(value, currency_symbol="£"):
    try:
        return f"{currency_symbol}{float(value):,.2f}"
    except Exception:
        return str(value)


def _fmt_percent(value):
    try:
        return f"{float(value) * 100:.1f}%" if abs(float(value)) <= 10 else f"{float(value):.2f}"
    except Exception:
        return str(value)


def run_synthetic_diagnostics():
    print("=== SYNTHETIC ANALYSIS DIAGNOSTICS ===")
    try:
        results = engine.run_full_analysis_v2(config=engine.CONFIG, data_source="synthetic")
    except Exception as e:
        print("ERROR: Synthetic run failed:", str(e))
        print(traceback.format_exc())
        return

    # Topline metrics
    print("\n-- Topline Metrics --")
    sm = results.get("summary_metrics") or {}
    currency = engine.CONFIG.get("currency", "£")

    # Heuristic lookups for common summary keys
    revenue = sm.get("total_revenue") or sm.get("revenue") or sm.get("total_sales")
    gp_before = sm.get("total_gp_before_waste") or sm.get("gp_before_waste") or sm.get("total_gp")
    gp_after = sm.get("total_gp_after_waste") or sm.get("gp_after_waste") or sm.get("gp_after")
    avg_gp_before = sm.get("avg_gp_pct") or sm.get("avg_gp_before") or sm.get("avg_gp")
    avg_gp_after = sm.get("avg_gp_pct_after_waste") or sm.get("avg_gp_after")

    print(f"Total revenue: { _fmt_currency(revenue, currency) if revenue is not None else 'N/A' }")
    print(f"Total GP before waste: { _fmt_currency(gp_before, currency) if gp_before is not None else 'N/A' }")
    print(f"Total GP after waste: { _fmt_currency(gp_after, currency) if gp_after is not None else 'N/A' }")
    if avg_gp_before is not None:
        print(f"Avg GP% before waste: { _fmt_percent(avg_gp_before) }")
    else:
        print("Avg GP% before waste: N/A")
    if avg_gp_after is not None:
        print(f"Avg GP% after waste: { _fmt_percent(avg_gp_after) }")
    else:
        print("Avg GP% after waste: N/A")

    # Data quality notes (first 3)
    print("\n-- Data Quality Notes (first 3) --")
    dq_notes = results.get("data_quality_notes") or []
    if dq_notes:
        for i, line in enumerate(dq_notes[:3], 1):
            print(f"{i}. {line}")
    else:
        print("None")

    # Time notes
    print("\n-- Time Notes (first 3) --")
    time_notes = results.get("time_notes") or []
    if time_notes:
        for i, line in enumerate(time_notes[:3], 1):
            print(f"{i}. {line}")
    else:
        print("None")

    # Scenarios
    print("\n-- Scenarios Summary --")
    scenarios = results.get("scenarios") or {}
    if isinstance(scenarios, dict):
        items = scenarios.items()
    elif isinstance(scenarios, list):
        items = enumerate(scenarios)
    else:
        items = []

    any_scen = False
    for key, sc in items:
        any_scen = True
        # sc may be a dict or string; be defensive
        if isinstance(sc, dict):
            label = sc.get("label") or sc.get("name") or sc.get("description") or "(no label)"
            delta = sc.get("delta_gp_after_waste") or sc.get("gp_change") or sc.get("delta_gp") or sc.get("change_in_gp")
        else:
            label = str(sc)
            delta = None
        if delta is not None:
            print(f"- {key}: {label} -> { _fmt_currency(delta, currency) }")
        else:
            print(f"- {key}: {label}")
    if not any_scen:
        print("No scenarios found.")

    # GPT export block preview
    print("\n-- GPT Export Block (preview) --")
    gpt_block = results.get("gpt_export_block") or ""
    if gpt_block:
        lines = gpt_block.splitlines()
        preview = "\n".join(lines[:10])
        if not preview.strip():
            preview = gpt_block[:500]
        print(preview)
        if len(gpt_block) > 500:
            print("... [truncated]")
    else:
        print("(empty)")

    # Check ./output directory
    print("\n-- Files in ./output --")
    out_dir = "output"
    if os.path.exists(out_dir) and os.path.isdir(out_dir):
        files = sorted(os.listdir(out_dir))
        if files:
            for f in files:
                print(f" - {f}")
        else:
            print("(output directory exists but is empty)")
    else:
        print("NOTE: ./output directory not found. You may need to run example_main.py to generate output files.")


def run_client_diagnostics():
    print("\n=== CLIENT MODE DIAGNOSTICS ===")
    if not _HAS_RUN_CLIENT:
        print("run_client.py not found or could not be imported; skipping client diagnostics.")
        return

    print("Attempting to run run_client.main() (client-mode). This may fail if client files are missing.")
    try:
        # Call the runner; it is expected to raise a clear error if client files are missing
        run_client.main()
        print("run_client.py completed successfully.")
        # list output_client
        out_dir = "output_client"
        if os.path.exists(out_dir) and os.path.isdir(out_dir):
            files = sorted(os.listdir(out_dir))
            print("Files in ./output_client:")
            for f in files:
                print(f" - {f}")
        else:
            print("run_client.py did not produce ./output_client or it is empty.")
    except Exception as e:
        print("run_client.py raised an exception (this may be expected if client files are missing):")
        print(str(e))
        print("--- Traceback (short) ---")
        tb = traceback.format_exc()
        # Print only the last ~10 lines of the traceback to keep output readable
        tb_lines = tb.splitlines()
        print("\n".join(tb_lines[-10:]))


def main():
    print("=== RESTAURANT ENGINE DIAGNOSTICS ===")
    try:
        run_synthetic_diagnostics()
    except Exception:
        print("Unexpected error during synthetic diagnostics:")
        print(traceback.format_exc())

    print("\n----------------------------------------\n")

    try:
        run_client_diagnostics()
    except Exception:
        print("Unexpected error during client diagnostics:")
        print(traceback.format_exc())

    print("\n=== DIAGNOSTICS COMPLETE ===")


if __name__ == "__main__":
    main()
