import pandas as pd


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
