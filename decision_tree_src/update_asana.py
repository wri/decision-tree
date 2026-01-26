import yaml
import asana
from asana import UsersApi, SectionsApi, TasksApi, CustomFieldsApi, ProjectsApi, StoriesApi
import pandas as pd
from typing import List, Optional, Union
from datetime import date, datetime
import difflib

def update_asana_status_by_gid(params: dict, 
                               secrets: dict, 
                               csv_path: str,
                               fuzzy: bool   = True,
                               fuzzy_threshold: float = 0.9,
                               baseline_col: str = "baseline_project_label",
                               ev_col: str = "ev_project_label",
                               ) -> None:
    """
    Bulk-update two custom fields ("Dtree Baseline", "Dtree EV") for tasks in a
    given section, using rows from a CSV. Matches tasks by project_name
    (substring, optional fuzzy). No task creation. No due dates.
    """
    project_gid      = params["asana"]["project_gid"]
    section_name     = params["asana"].get("section", "Remote")
    dry_run          = bool(params["asana"].get("dry_run", True))
    project_name_col = params["asana"].get("project_name_col", "project_name")

    ASANA_TOKEN = (secrets.get("asana", {}).get("pat"))

    # ---- Build Asana client ----
    cfg = asana.Configuration()
    cfg.access_token = ASANA_TOKEN
    cfg.return_page_iterator = False
    client = asana.ApiClient(cfg)

    users_api    = UsersApi(client)
    sections_api = SectionsApi(client)
    tasks_api    = TasksApi(client)
    custom_api   = CustomFieldsApi(client)

    # Optional: verify credentials
    _ = users_api.get_user("me", {"opt_fields": "workspaces.gid"})

    # ---- Resolve section ----
    secs = sections_api.get_sections_for_project(
        project_gid, {"opt_fields": "gid,name"}
    ).get("data", [])
    sec = next((s for s in secs if s["name"] == section_name), None)
    if not sec:
        raise ValueError(f"Section '{section_name}' not found in project {project_gid!r}.")
    section_gid = sec["gid"]

    # ---- Load tasks once ----
    tasks = tasks_api.get_tasks_for_section(
        section_gid,
        {"opt_fields": "gid,name,custom_fields.name,custom_fields.gid,custom_fields.display_value"}
    ).get("data", [])

    # ---- Helpers ----
    def _find_task(q: str):
        # substring (case-insensitive)
        matches = [t for t in tasks if q.lower() in t["name"].lower()]
        if matches:
            exact = [t for t in matches if t["name"].strip().lower() == q.strip().lower()]
            return exact[0] if len(exact) == 1 else sorted(matches, key=lambda x: x["name"])[0]
        # fuzzy fallback
        if fuzzy:
            scored = []
            for t in tasks:
                r = difflib.SequenceMatcher(None, q.lower(), t["name"].lower()).ratio()
                if r >= fuzzy_threshold:
                    scored.append((t, r))
            if scored:
                scored.sort(key=lambda x: x[1], reverse=True)
                print(f"Fuzzy matched '{q}' → '{scored[0][0]['name']}' (score={scored[0][1]:.2f})")
                return scored[0][0]
        return None

    def _enum_or_value(field_gid: str, desired: str):
        meta = custom_api.get_custom_field(
            field_gid, {"opt_fields": "resource_subtype,enum_options.name,enum_options.gid"}
        )
        field_meta = meta.get("data", meta)
        if field_meta.get("resource_subtype", "") == "enum":
            opts = field_meta.get("enum_options", [])
            name2gid = {o["name"]: o["gid"] for o in opts}
            if desired not in name2gid:
                raise ValueError(
                    f"Invalid enum value {desired!r} for field gid={field_gid}. "
                    f"Valid: {list(name2gid.keys())}"
                )
            return name2gid[desired]
        return desired  # text/number

    def _update_two_fields(task: dict, baseline_val: str, ev_val: str):
        name2cf = {cf["name"]: cf for cf in (task.get("custom_fields") or []) if "name" in cf}
        missing = [n for n in ("Dtree Baseline", "Dtree EV") if n not in name2cf]
        if missing:
            print(f" • skipping '{task['name']}' (missing fields: {missing})")
            return False, {}

        payload = {"custom_fields": {}}
        payload["custom_fields"][name2cf["Dtree Baseline"]["gid"]] = _enum_or_value(
            name2cf["Dtree Baseline"]["gid"], str(baseline_val)
        )
        payload["custom_fields"][name2cf["Dtree EV"]["gid"]] = _enum_or_value(
            name2cf["Dtree EV"]["gid"], str(ev_val)
        )

        if dry_run:
            print(f"[DRY-RUN] Would update '{task['name']}' ({task['gid']}) with: {payload}")
            return False, payload

        tasks_api.update_task({"data": payload}, task["gid"], {})
        print(f"→ Updated '{task['name']}' with {payload}")
        return True, payload

    # ---- Bulk from CSV ----
    df = pd.read_csv(csv_path)
    needed = [project_name_col, baseline_col, ev_col]
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV missing required column(s): {missing_cols}")

    n_rows = len(df)
    n_found = n_updated = 0

    for _, row in df.iterrows():
        q = str(row[project_name_col])
        t = _find_task(q)
        if not t:
            print(f"× No task found for '{q}' in section '{section_name}'.")
            continue
        n_found += 1
        updated, _ = _update_two_fields(t, str(row[baseline_col]), str(row[ev_col]))
        if updated:
            n_updated += 1

    mode = "DRY-RUN" if dry_run else "LIVE"
    print(f"[{mode}] Processed {n_rows} rows — found {n_found} tasks; updated {n_updated}.")
