import pandas as pd


def select_first_conversion_per_lead(
    df_leads: pd.DataFrame,
    lead_key_col: str = "lead_key",
    date_col: str = "data_hora",
) -> pd.DataFrame:
    """Keep the earliest conversion in the selected period for each lead."""
    if df_leads is None:
        return pd.DataFrame()

    out = df_leads.copy()
    if out.empty or lead_key_col not in out.columns:
        return out

    if date_col in out.columns:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out.sort_values(date_col, kind="stable", na_position="last")

    return out.drop_duplicates(subset=[lead_key_col], keep="first").copy()


def filter_opportunities_by_origins(
    df_opportunities: pd.DataFrame,
    selected_origins,
) -> pd.DataFrame:
    """Apply the dashboard origin selection to opportunity rows."""
    if df_opportunities is None:
        return pd.DataFrame()

    out = df_opportunities.copy()
    if out.empty or "origem" not in out.columns:
        return out

    return out[out["origem"].isin(selected_origins)].copy()
