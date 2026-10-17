"""Integration check (needs AWS access): tunnel opens and both results tables exist.

Reads RDS settings from params.yaml and secrets.yaml (aws: bastion_id_for_rds,
land_aws_profile) in the project root.
"""
import os

import pytest

from decision_tree.constants import PROJECT_ROOT
from decision_tree.rds_tunnel_no_tf import rds_connection
from decision_tree.tools import load_secrets, load_yaml


@pytest.mark.network
def test_rds_query():
    rds = load_yaml(os.path.join(PROJECT_ROOT, "params.yaml"))["rds"]
    aws_secrets = load_secrets(os.path.join(PROJECT_ROOT, "secrets.yaml")).get("aws") or {}
    with rds_connection(
        rds["database"],
        env=rds["env"],
        bastion_id=aws_secrets.get("bastion_id_for_rds"),
        aws_profile=aws_secrets.get("land_aws_profile"),
    ) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_name = ANY(%s)",
            (rds["schema"], [rds["poly_table"], rds["prj_table"]]),
        ).fetchall()
    assert {r[0] for r in rows} == {rds["poly_table"], rds["prj_table"]}
