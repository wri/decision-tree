import pandas as pd
from datetime import datetime, timedelta

def parse_condition(value, actual):
    """
    Evaluate simple condition like '>=1', '<2', or direct value.
    Interprets and evaluates conditional expressions (like ">=1") 
    from the rules CSV.
    """
    if isinstance(value, str) and any(op in value for op in [">=", "<=", ">", "<", "=="]):
        try:
            return eval(f"{actual} {value}")
        except Exception:
            return False
    return actual == value

def _image_timing_tier(row):
    """
    Classify a polygon's image availability into a timing tier.

    Returns:
    - 'hq_1yr'   : at least one image within 1-year baseline window
    - 'ext_18mo' : no image within 1 year, but at least one within 1.5 years
    - 'none'     : no usable imagery in either window
    """
    if row['baseline_img_count'] >= 1:
        return 'hq_1yr'
    elif row['baseline_ext_img_count'] >= 1:
        return 'ext_18mo'
    else:
        return 'none'

def apply_rules_baseline(rules_file_path, df):
    """
    Decision tree for baseline classification.

    * If a polygon was already flagged as problematic during cleaning, 
    carry that flag through as its decision by ref 'notes' column
    
    PHASE 1:
      • assign first_decision {mangrove, remote, field}  
        (intentionally ignores slope entirely)
    
    PHASE 2:
      • for first_decision=='remote', final_decision is driven by both
        img_count rules AND the timing tier of available imagery:
          - 'hq_1yr'   : image exists within 1yr  -> apply rules normally
          - 'ext_18mo' : image only within 1.5yrs  -> cap at 'weak remote'
          - 'none'     : no imagery in either window -> 'review required'
      • for first_decision=='field', look only at slope rules  
      • mangrove stays as is (final_decision = 'mangrove')
    """
    rules = pd.read_csv(rules_file_path)

    # Clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # Clean input df
    for col in ['baseline_canopy', 'target_sys', 'practice', 'slope']:
        df[col] = df[col].astype(str).str.strip().str.lower()

    # Pre-compute timing tier for each polygon
    df['img_tier'] = df.apply(_image_timing_tier, axis=1)

    # Split rules into subsets each phase needs
    base_rules = rules[['canopy', 'target_sys', 'practice', 'img_count', 'first_decision']]
    remote_rules = rules[rules['first_decision'] == 'remote'][[
        'canopy', 'target_sys', 'practice', 'img_count', 'final_decision'
    ]]
    field_rules = rules[rules['first_decision'] == 'field'][[
        'canopy', 'target_sys', 'practice', 'slope', 'final_decision'
    ]]

    decisions = []
    for _, row in df.iterrows():

        # check notes column before assigning decision
        if pd.notna(row['notes']):
            decisions.append(row['notes'])
            continue

        # PHASE 1: first_decision — assign 'mangrove', 'remote' or 'field'
        if row['target_sys'] == 'mangrove':
            base = 'mangrove'
        else:
            base = None
            for _, rule in base_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys']      == rule['target_sys'] and
                    row['practice']        == rule['practice'] and
                    parse_condition(rule['img_count'], row['baseline_img_count'])
                ):
                    base = rule['first_decision']
                    break

        # ——— PHASE 2: final_decision — refine using img_count, timing and slope
        if base == 'remote':
            tier = row['img_tier']

            if tier == 'hq_1yr':
                # Ideal case: image within 1 year — apply rules normally
                final = None
                for _, rule in remote_rules.iterrows():
                    if (
                        row['baseline_canopy'] == rule['canopy'] and
                        row['target_sys']      == rule['target_sys'] and
                        row['practice']        == rule['practice'] and
                        parse_condition(rule['img_count'], row['baseline_img_count'])
                    ):
                        final = rule['final_decision']
                        break
                final = final or 'review required'

            elif tier == 'ext_18mo':
                # Flexible: image exists within 1.5 years but not within 1 year
                # Cap at weak remote regardless of count
                final = 'weak remote'

            else:
                # No imagery in either window — remote sensing not possible,
                # fall back to field visit using slope rules
                final = None
                for _, rule in field_rules.iterrows():
                    if (
                        row['baseline_canopy'] == rule['canopy'] and
                        row['target_sys']      == rule['target_sys'] and
                        row['practice']        == rule['practice'] and
                        row['slope']           == rule['slope']
                    ):
                        final = rule['final_decision']
                        break
                final = final or 'review required'

        elif base == 'field':
            # only slope drives strong vs. weak field
            final = None
            for _, rule in field_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    row['slope'] == rule['slope']
                ):
                    final = rule['final_decision']
                    break

        else:
            final = base

        decisions.append(final or 'review required')

    df['baseline_decision'] = decisions

    # Summary
    summary = df['baseline_decision'].value_counts().reset_index()
    summary.columns = ['baseline_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df))*100)
    print("\nBaseline Decision Summary:\n")
    print(summary)

    return df

def apply_rules_ev(params, rules_file_path, df):
    """
    Decision tree for early verification (EV) classification.

    PHASE 1:
      • assign first_decision ∈ {mangrove, remote, field}
    
    PHASE 2:
      • for first_decision=='remote', look only at img_count rules  
      • for first_decision=='field', look only at slope rules  
      • mangrove stays as is (final_decision = 'mangrove')

    Additional Rule:
      • if EV year is current or future, decision is 'not available'
    """
    df = df.copy()
    rules = pd.read_csv(rules_file_path)

    # Clean rules df
    for col in rules.columns:
        if rules[col].dtype == object:
            rules[col] = rules[col].map(
                lambda x: x.strip().lower() if isinstance(x, str) else x
            )

    # Clean input df
    for col in ['ev_canopy', 'target_sys', 'practice', 'slope']:
        df.loc[:, col] = df[col].astype(str).str.strip().str.lower()


    # Filter EV-specific rule columns
    base_rules = rules[['canopy','target_sys','practice','img_count', 'first_decision']]
    remote_rules = rules[rules['first_decision']=='remote'][[
        'canopy','target_sys','practice','img_count','final_decision'
    ]]
    field_rules = rules[rules['first_decision']=='field'][[
        'canopy','target_sys','practice','slope','final_decision'
    ]]

    today = datetime.today()
    ev_days_start, ev_days_end = params['criteria']['ev_range']

    decisions = []

    for _, row in df.iterrows():
        if pd.isna(row['plantstart']):
            decisions.append('not available')
            continue

        plant_start = pd.to_datetime(row['plantstart'])
        ev_window_start = plant_start + timedelta(days=ev_days_start)
        if today < ev_window_start:
            decisions.append('not available')
            continue

        # PHASE 1 — first_decision: 'mangrove', 'remote', or 'field'
        # NOTE: EV will use the baseline canopy if enough time has passed
        # that we need to verify
        if row['target_sys'] == 'mangrove':
            base = 'mangrove'
        else:
            base = None
            for _, rule in base_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['ev_img_count'])
                ):
                    base = rule['first_decision']
                    break

        # PHASE 2 — final_decision
        if base == 'remote':
            final = None
            for _, rule in remote_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    parse_condition(rule['img_count'], row['ev_img_count'])
                ):
                    final = rule['final_decision']
                    break

        elif base == 'field':
            final = None
            for _, rule in field_rules.iterrows():
                if (
                    row['baseline_canopy'] == rule['canopy'] and
                    row['target_sys'] == rule['target_sys'] and
                    row['practice'] == rule['practice'] and
                    row['slope'] == rule['slope']
                ):
                    final = rule['final_decision']
                    break
        else:
            final = base

        decisions.append(final or 'review required')

    # Add column for ev decision
    df['ev_decision'] = decisions

    # Summary
    summary = df['ev_decision'].value_counts().reset_index()
    summary.columns = ['ev_decision', 'count']
    summary['proportion'] = round((summary['count'] / len(df)) * 100)
    print("\nEV Decision Summary:\n")
    print(summary)

    return df