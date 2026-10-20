"""RDS Data Access Object: write decision-tree outputs to, and read them back from, RDS PostgreSQL
through an SSM bastion tunnel (no Terraform).

Library module. The pipeline calls :func:`upload_results` from
``VerificationDecisionTree._upload_outputs``:

    from decision_tree.rds_dao import upload_results

    upload_results(poly_results=poly_df, run_datetime=run_dt, cohort="c2",
                   env="lr", database="indicators", schema="public",
                   poly_table="decision_tree_poly_results_log", prj_table="decision_tree_prj_results_log",
                   bastion_id="i-0123456789abcdef0", aws_profile=None)

All settings are passed in by the caller; the pipeline reads them from ``params.yaml``
(``rds:`` env, database, schema, tables) and ``secrets.yaml`` (``aws:``
bastion_id_for_rds, land_aws_profile). This module has no built-in defaults for them.
The AWS region is the one configured for ``aws_profile`` (``region =`` in ~/.aws/config).

:func:`read_latest_poly_results` reads the polygon table back (the newest row of each
polygon of the given projects), for the project step.

Use :func:`rds_connection` directly when you need an open connection for other queries,
or :func:`write_results` / :func:`read_latest_rows` when you already have one.

How it connects (same flow as ``open_tunnel.sh`` + ``db_url.sh``, minus Terraform):

1. Bastion: the EC2 instance ``bastion_id``.
2. DB URL: read ``/gri/<env>/app-db-url`` from Secrets Manager.
3. Tunnel: ``aws ssm start-session`` port-forwards a local port to the RDS
   host:port through the bastion. The tunnel is closed when the call returns.
4. Connect with psycopg to the requested database.

Tables: :func:`write_results` first creates the polygon and project tables with
``CREATE TABLE IF NOT EXISTS`` (definitions below, from local_create_sql_tables.sql), so a
new database or schema works without manual setup. The polygon table's geometry column is a
PostGIS ``geometry``, so the database needs the PostGIS extension.

How rows are written:

- The target table's columns and types are read from ``information_schema`` at upload
  time; DataFrame columns are matched to table columns by name and converted to the
  column's type. DataFrame columns with no matching table column are skipped.
- Derived columns: ``run_datetime`` (same value for every row of a run),
  ``ttc_by_year`` (JSON built from the ``ttc_<year>`` columns), ``cohort`` (filled
  from the ``cohort`` argument when the DataFrame has no cohort values) and
  ``geometry`` (shapely objects or WKT, written as PostGIS geometry with SRID 4326).
- Values are checked before anything is sent: text longer than the column allows,
  integers out of range, unparseable booleans/JSON and duplicate primary keys raise
  ``ValueError`` naming the column. Non-integer values headed for an integer column
  are rounded with a warning.
- Polygon and project rows are inserted in one transaction: either both land or neither.

Requirements on the calling machine:
    - AWS CLI v2 and the Session Manager plugin on PATH
    - boto3 and psycopg (``uv add boto3 "psycopg[binary]"``)

Profile: ``aws_profile`` is used for both Secrets Manager and the SSM tunnel. Pass ``None``
to use the default boto3 credential chain (env vars, instance role, etc.).
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import math
import os
import re
import signal
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import boto3
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from psycopg.types.json import Json

__all__ = [
    "UploadSummary",
    "upload_results",
    "write_results",
    "ensure_tables",
    "read_latest_poly_results",
    "read_latest_rows",
    "rds_connection",
]

logger = logging.getLogger(__name__)

GEOMETRY_SRID = 4326  # TerraMatch polygons are WGS 84 lon/lat

# Table definitions (from local_create_sql_tables.sql); the table names come from params.yaml.
POLY_TABLE_COLUMNS = """
    cohort   VARCHAR(100),
    project_id   VARCHAR(100),
    poly_id   VARCHAR(100),
    site_id   VARCHAR(100),
    project_name   VARCHAR(100),
    geometry   geometry,
    plantstart   DATE,
    practice   VARCHAR(100),
    target_sys   VARCHAR(100),
    area   REAL,
    ttc_by_year   JSON,
    baseline_img_count   SMALLINT,
    baseline_ext_img_count   SMALLINT,
    ev_img_count   SMALLINT,
    plantstart_dt   DATE,
    baseline_year   SMALLINT,
    baseline_canopy   VARCHAR(100),
    ev_canopy   VARCHAR(100),
    slope_area   REAL,
    slope   VARCHAR(100),
    img_tier   VARCHAR(100),
    baseline_decision   VARCHAR(100),
    notes_base   TEXT,
    ev_decision   VARCHAR(100),
    notes_ev   VARCHAR(100),
    baseline_canopy_missing   BOOLEAN,
    ev_canopy_missing   BOOLEAN,
    baseline_img_missing   BOOLEAN,
    ev_img_missing   BOOLEAN,
    slope_missing   BOOLEAN,
    baseline_remote_suitability   REAL,
    baseline_remote_suitability_wtd   REAL,
    ev_remote_suitability   REAL,
    ev_remote_suitability_wtd   REAL,
    baseline_cost   REAL,
    ev_cost   REAL,
    run_datetime   TIMESTAMP,
    PRIMARY KEY (project_id, poly_id, run_datetime)
"""

PRJ_TABLE_COLUMNS = """
    project_id   VARCHAR(100),
    project_name   VARCHAR(100),
    total_area   REAL,
    base_conf   REAL,
    base_label   VARCHAR(100),
    base_pct_area_scored   REAL,
    base_poly_frac_scored   REAL,
    base_scored_area   REAL,
    base_frac_mangrove   REAL,
    base_data_quality_flag   TEXT,
    ev_conf   REAL,
    ev_label   VARCHAR(100),
    ev_pct_area_scored   REAL,
    ev_poly_frac_scored   REAL,
    ev_scored_area   REAL,
    ev_frac_mangrove   REAL,
    ev_data_quality_flag TEXT,
    run_datetime   TIMESTAMP,
    PRIMARY KEY (project_id, run_datetime)
"""

_TTC_COL = re.compile(r"^ttc_(\d{4})$")
_INT_RANGES = {
    "smallint": (-32768, 32767),
    "integer": (-2147483648, 2147483647),
    "bigint": (-9223372036854775808, 9223372036854775807),
}


@dataclass(frozen=True)
class UploadSummary:
    poly_rows: int
    prj_rows: int
    run_datetime: dt.datetime


@dataclass(frozen=True)
class _DbTarget:
    host: str
    port: int


@dataclass(frozen=True)
class _Column:
    name: str
    data_type: str          # information_schema data_type, e.g. 'character varying'
    max_length: int | None  # for varchar/char
    udt_name: str = ""      # e.g. 'geometry' for PostGIS columns (data_type 'USER-DEFINED')


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def upload_results(
    poly_results: pd.DataFrame | None = None,
    prj_results: pd.DataFrame | None = None,
    *,
    run_datetime: dt.datetime | None = None,
    cohort: str | None = None,
    env: str,
    database: str,
    schema: str,
    poly_table: str,
    prj_table: str,
    bastion_id: str,
    aws_profile: str | None,
    local_port: int | None = None,
) -> UploadSummary:
    """Insert polygon and/or project results into their RDS tables.

    Args:
        poly_results: Output of ``compute_polygon_results``. ``None`` skips the polygon table.
        prj_results: Output of ``compute_project_results``. ``None`` skips the project table.
        run_datetime: Value for the ``run_datetime`` column. Pass the same value for the
            polygon and project uploads of one run. Defaults to now (UTC, no time zone).
        cohort: Fallback for the ``cohort`` column when ``poly_results`` has no cohort values.
        database, schema, poly_table, prj_table: Where to write.
        env, bastion_id, aws_profile, local_port:
            Connection settings, see :func:`rds_connection`.

    Returns:
        Number of rows inserted into each table and the ``run_datetime`` used.

    Raises:
        ValueError: Data does not fit the table (see module docstring).
        RuntimeError / TimeoutError: Bastion lookup or tunnel failures.
        psycopg.Error: Database errors; nothing is committed in that case.
    """
    if _is_empty(poly_results) and _is_empty(prj_results):
        logger.info("upload_results: nothing to upload")
        return UploadSummary(0, 0, run_datetime or _utc_now())

    with rds_connection(
        database,
        env=env,
        bastion_id=bastion_id,
        aws_profile=aws_profile,
        local_port=local_port,
    ) as conn:
        return write_results(
            conn,
            poly_results,
            prj_results,
            run_datetime=run_datetime,
            cohort=cohort,
            schema=schema,
            poly_table=poly_table,
            prj_table=prj_table,
        )


def ensure_tables(conn: psycopg.Connection, *, schema: str, poly_table: str, prj_table: str) -> None:
    """Create the polygon and project tables if they don't exist (existing tables are left as is)."""
    try:
        with conn.transaction():
            for table, columns in ((poly_table, POLY_TABLE_COLUMNS), (prj_table, PRJ_TABLE_COLUMNS)):
                conn.execute(sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                    sql.Identifier(schema), sql.Identifier(table), sql.SQL(columns.strip())))
    except psycopg.errors.UndefinedObject as exc:
        if 'type "geometry" does not exist' in str(exc):
            raise RuntimeError("The PostGIS extension is not enabled in this database, so the polygon "
                               "table's geometry column can't be created. A database admin can enable it "
                               "with: CREATE EXTENSION postgis;") from exc
        raise


def write_results(
    conn: psycopg.Connection,
    poly_results: pd.DataFrame | None = None,
    prj_results: pd.DataFrame | None = None,
    *,
    run_datetime: dt.datetime | None = None,
    cohort: str | None = None,
    schema: str,
    poly_table: str,
    prj_table: str,
) -> UploadSummary:
    """Same as :func:`upload_results`, using an already open connection."""
    run_dt = _naive_utc(run_datetime) if run_datetime else _utc_now()
    ensure_tables(conn, schema=schema, poly_table=poly_table, prj_table=prj_table)

    jobs: list[tuple[str, pd.DataFrame]] = []
    if not _is_empty(poly_results):
        jobs.append((poly_table, _prepare_poly(poly_results, run_dt, cohort)))
    if not _is_empty(prj_results):
        jobs.append((prj_table, _with_run_datetime(prj_results, run_dt)))

    # Validate and convert everything before sending any rows.
    planned: list[tuple[str, list[str], list[tuple], list[str]]] = []
    for table, df in jobs:
        columns = _table_columns(conn, schema, table)
        names, rows = _build_rows(df, columns, table)
        _check_primary_key(conn, schema, table, names, rows)
        udt = {c.name: c.udt_name for c in columns}
        planned.append((table, names, rows, [udt[n] for n in names]))

    counts: dict[str, int] = {}
    with conn.transaction():
        with conn.cursor() as cur:
            for table, names, rows, udts in planned:
                values = [
                    sql.SQL("ST_GeomFromEWKT(%s)") if u == "geometry" else sql.Placeholder()
                    for u in udts
                ]
                query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
                    sql.Identifier(schema),
                    sql.Identifier(table),
                    sql.SQL(", ").join(map(sql.Identifier, names)),
                    sql.SQL(", ").join(values),
                )
                cur.executemany(query, rows)
                counts[table] = len(rows)
                logger.info("Inserted %d row(s) into %s.%s", len(rows), schema, table)

    return UploadSummary(
        poly_rows=counts.get(poly_table, 0),
        prj_rows=counts.get(prj_table, 0),
        run_datetime=run_dt,
    )


def read_latest_poly_results(
    *,
    project_ids: list[str],
    env: str,
    database: str,
    schema: str,
    poly_table: str,
    bastion_id: str,
    aws_profile: str | None,
    local_port: int | None = None,
) -> pd.DataFrame:
    """Return the most recent row of every polygon of the given projects.

    Args:
        project_ids: Projects to read.
        env, database, schema, poly_table, bastion_id, aws_profile, local_port:
            Where to read and how to connect, see :func:`rds_connection`.

    Returns:
        A DataFrame shaped like the in-memory polygon results: booleans as nullable
        booleans, numbers as floats, ``ttc_by_year`` expanded back into ``ttc_<year>``
        columns. Projects with no rows in the table are simply absent.
    """
    with rds_connection(
        database,
        env=env,
        bastion_id=bastion_id,
        aws_profile=aws_profile,
        local_port=local_port,
    ) as conn:
        return read_latest_rows(conn, schema=schema, table=poly_table,
                                filters={"project_id": list(project_ids)})


def read_latest_rows(
    conn: psycopg.Connection,
    *,
    schema: str,
    table: str,
    filters: dict[str, list] | None = None,
) -> pd.DataFrame:
    """Newest row (by ``run_datetime``) per primary key, e.g. per (project_id, poly_id).

    ``filters`` maps column -> allowed values, e.g. ``{"project_id": ["p1", "p2"]}``.
    """
    columns = _table_columns(conn, schema, table)
    names = [c.name for c in columns]
    if "run_datetime" not in names:
        raise ValueError(f"{schema}.{table} has no run_datetime column.")
    key = [c for c in _primary_key(conn, schema, table) if c != "run_datetime"]
    if not key:
        raise ValueError(f"{schema}.{table} needs a primary key besides run_datetime "
                         "to pick each row's latest version.")

    where = sql.SQL("")
    params: tuple = ()
    if filters:
        unknown = [c for c in filters if c not in names]
        if unknown:
            raise ValueError(f"{schema}.{table} has no column(s) {unknown} to filter on.")
        where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(
            sql.SQL("{} = ANY(%s)").format(sql.Identifier(c)) for c in filters
        )
        params = tuple(list(v) for v in filters.values())

    select = [
        sql.SQL("ST_AsText({c}) AS {c}").format(c=sql.Identifier(c.name)) if c.udt_name == "geometry"
        else sql.Identifier(c.name)
        for c in columns
    ]
    query = sql.SQL(
        "SELECT DISTINCT ON ({key}) {cols} FROM {schema}.{table} {where} "
        "ORDER BY {key}, {run_dt} DESC"
    ).format(
        key=sql.SQL(", ").join(map(sql.Identifier, key)),
        cols=sql.SQL(", ").join(select),
        schema=sql.Identifier(schema),
        table=sql.Identifier(table),
        where=where,
        run_dt=sql.Identifier("run_datetime"),
    )
    rows = conn.execute(query, params).fetchall()
    logger.info("Read %d latest row(s) from %s.%s", len(rows), schema, table)
    return _to_frame(rows, columns)


@contextlib.contextmanager
def rds_connection(
    database: str,
    *,
    env: str,
    bastion_id: str,
    aws_profile: str | None,
    local_port: int | None = None,
    connect_timeout: int = 15,
    tunnel_timeout: float = 60.0,
) -> Iterator[psycopg.Connection]:
    """Open a tunnel and yield a psycopg connection; both are closed on exit.

    Args:
        database: Database to connect to (replaces the name in the secret's URL).
        env: Secrets Manager environment, i.e. ``/gri/<env>/app-db-url``.
        bastion_id: Bastion EC2 instance id.
        aws_profile: AWS profile for Secrets Manager and the SSM session
            (``None`` = default credentials). Its configured region is used.
        local_port: Local tunnel port. ``None`` picks a free port.
    """
    if not bastion_id:
        raise ValueError("bastion_id is required (set aws.bastion_id_for_rds in secrets.yaml).")

    region = _profile_region(aws_profile)
    db_url = _get_db_url(env, profile=aws_profile, region=region)
    target = _parse_target(db_url)
    port = local_port or _free_port()
    conn_url = _to_psycopg_url(_with_database(_to_local(db_url, port), database))

    with _ssm_tunnel(bastion_id, target, port, profile=aws_profile, region=region,
                     timeout=tunnel_timeout):
        with psycopg.connect(conn_url, connect_timeout=connect_timeout) as conn:
            logger.info("Connected to RDS database %r via localhost:%d", database, port)
            yield conn


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _profile_region(profile: str | None) -> str:
    """Region configured for the AWS profile (or the default credential chain when None).

    boto3 reads AWS_DEFAULT_REGION or the profile's ``region``; AWS_REGION (honoured by
    the AWS CLI but not boto3) is accepted as a last resort.
    """
    region = boto3.Session(profile_name=profile).region_name or os.environ.get("AWS_REGION")
    if not region:
        who = f"AWS profile {profile!r}" if profile else "the default AWS credentials"
        fix = f"aws configure set region us-east-1 --profile {profile}" if profile else \
              "aws configure set region us-east-1 (or set AWS_DEFAULT_REGION)"
        raise ValueError(f"No region is configured for {who}. Set one with: {fix}")
    logger.info("Using AWS region %s from %s", region, f"profile {profile!r}" if profile else "default credentials")
    return region


def _get_db_url(env: str, *, profile: str | None, region: str) -> str:
    session = boto3.Session(profile_name=profile, region_name=region)
    resp = session.client("secretsmanager").get_secret_value(SecretId=f"/gri/{env}/app-db-url")
    return resp["SecretString"].strip()


def _split_netloc(netloc: str) -> tuple[str, str, int | None]:
    """(userinfo, host, port); rpartition keeps '@' inside passwords intact."""
    userinfo, _, hostport = netloc.rpartition("@")
    host, sep, port = hostport.rpartition(":")
    if not sep or not port.isdigit():
        return userinfo, hostport, None
    return userinfo, host, int(port)


def _parse_target(db_url: str) -> _DbTarget:
    _, host, port = _split_netloc(urlsplit(db_url).netloc)
    return _DbTarget(host=host, port=port or 5432)


def _to_local(db_url: str, port: int) -> str:
    parts = urlsplit(db_url)
    userinfo, _, _ = _split_netloc(parts.netloc)
    netloc = f"{userinfo}@localhost:{port}" if userinfo else f"localhost:{port}"
    return urlunsplit(parts._replace(netloc=netloc))


def _with_database(db_url: str, database: str) -> str:
    return urlunsplit(urlsplit(db_url)._replace(path=f"/{database}"))


def _to_psycopg_url(db_url: str) -> str:
    """Strip SQLAlchemy driver suffixes ('postgresql+psycopg2://' -> 'postgresql://')."""
    scheme, sep, rest = db_url.partition("://")
    return f"{scheme.split('+', 1)[0]}{sep}{rest}"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _port_open(port: int) -> bool:
    with contextlib.suppress(OSError):
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    return False


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "posix":  # kill the group so session-manager-plugin exits too
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextlib.contextmanager
def _ssm_tunnel(
    bastion_id: str,
    target: _DbTarget,
    local_port: int,
    *,
    profile: str | None,
    region: str,
    timeout: float,
) -> Iterator[None]:
    if _port_open(local_port):
        raise RuntimeError(f"localhost:{local_port} is already in use; pick another local_port.")

    params = (
        f'{{"host":["{target.host}"],"portNumber":["{target.port}"],'
        f'"localPortNumber":["{local_port}"]}}'
    )
    cmd = [
        "aws", "ssm", "start-session",
        "--region", region,
        "--target", bastion_id,
        "--document-name", "AWS-StartPortForwardingSessionToRemoteHost",
        "--parameters", params,
    ]
    env = dict(os.environ)
    if profile:
        env["AWS_PROFILE"] = profile

    logger.info("Opening SSM tunnel localhost:%d -> %s:%d", local_port, target.host, target.port)
    # Log output goes to a temp file so a full pipe can never stall the session.
    with tempfile.TemporaryFile(mode="w+") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            deadline = time.monotonic() + timeout
            while not _port_open(local_port):
                if proc.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(
                        f"SSM session exited early (code {proc.returncode}):\n{log.read()}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Tunnel did not open on localhost:{local_port} "
                                       f"within {timeout:.0f}s")
                time.sleep(0.5)
            yield
        finally:
            _stop(proc)
            logger.info("SSM tunnel closed")


# --------------------------------------------------------------------------- #
# Upload internals
# --------------------------------------------------------------------------- #
def _is_empty(df: pd.DataFrame | None) -> bool:
    return df is None or df.empty


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def _naive_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return value


def _with_run_datetime(df: pd.DataFrame, run_dt: dt.datetime) -> pd.DataFrame:
    out = df.copy()
    out["run_datetime"] = run_dt
    return out


def _prepare_poly(df: pd.DataFrame, run_dt: dt.datetime, cohort: str | None) -> pd.DataFrame:
    """Add the derived polygon columns: run_datetime, ttc_by_year, cohort, WKT geometry."""
    out = _with_run_datetime(df, run_dt)

    ttc_cols = sorted((c for c in out.columns if _TTC_COL.match(str(c))), key=str)
    if "ttc_by_year" not in out.columns and ttc_cols:
        years = [_TTC_COL.match(str(c)).group(1) for c in ttc_cols]  # type: ignore[union-attr]

        def to_json(row: pd.Series) -> dict[str, float] | None:
            vals = {y: float(v) for y, v in zip(years, row) if not _missing(v)}
            return vals or None

        out["ttc_by_year"] = out[ttc_cols].apply(to_json, axis=1)

    if "cohort" not in out.columns or out["cohort"].isna().all():
        out["cohort"] = cohort
    out["cohort"] = out["cohort"].map(_normalize_cohort)

    if "geometry" in out.columns:
        out["geometry"] = out["geometry"].map(
            lambda g: g.wkt if hasattr(g, "wkt") else g
        )
    return out


def _normalize_cohort(value: Any) -> Any:
    """'["terrafund-cohort-2"]' (API path) -> 'terrafund-cohort-2'."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value.startswith("["):
        with contextlib.suppress(ValueError):
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return ",".join(str(v) for v in parsed)
    return value


def _table_columns(conn: psycopg.Connection, schema: str, table: str) -> list[_Column]:
    rows = conn.execute(
        """
        SELECT column_name, data_type, character_maximum_length, udt_name
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """,
        (schema, table),
    ).fetchall()
    if not rows:
        raise ValueError(f"Table {schema}.{table} does not exist (or is not visible to this user).")
    return [_Column(str(n), str(t), m, str(u)) for n, t, m, u in rows]


def _primary_key(conn: psycopg.Connection, schema: str, table: str) -> list[str]:
    return [
        str(r[0])
        for r in conn.execute(
            """
            SELECT a.attname
            FROM pg_index i
            JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
            WHERE i.indrelid = %s::regclass AND i.indisprimary
            ORDER BY array_position(i.indkey, a.attnum)
            """,
            (f'"{schema}"."{table}"',),
        ).fetchall()
    ]


def _to_frame(rows: list[tuple], columns: list[_Column]) -> pd.DataFrame:
    """Rows from the database -> DataFrame with pipeline-friendly dtypes."""
    df = pd.DataFrame(rows, columns=[c.name for c in columns])
    for c in columns:
        t = c.data_type
        if t == "boolean":
            df[c.name] = df[c.name].astype("boolean")
        elif t in _INT_RANGES or t in ("real", "double precision", "numeric"):
            df[c.name] = pd.to_numeric(df[c.name], errors="coerce").astype(float)
        elif t == "date" or t.startswith("timestamp"):
            df[c.name] = pd.to_datetime(df[c.name])
    if "ttc_by_year" in df.columns:
        ttc = pd.DataFrame(
            [v if isinstance(v, dict) else {} for v in df["ttc_by_year"]], index=df.index
        )
        if not ttc.empty:
            ttc = ttc.reindex(sorted(ttc.columns), axis=1).apply(pd.to_numeric, errors="coerce")
            ttc.columns = [f"ttc_{y}" for y in ttc.columns]
            df = pd.concat([df.drop(columns="ttc_by_year"), ttc], axis=1)
        else:
            df = df.drop(columns="ttc_by_year")
    return df


def _check_primary_key(
    conn: psycopg.Connection, schema: str, table: str, names: list[str], rows: list[tuple]
) -> None:
    pk = _primary_key(conn, schema, table)
    if not pk:
        return
    missing = [c for c in pk if c not in names]
    if missing:
        raise ValueError(f"{table}: primary-key column(s) {missing} are not in the data.")
    idx = [names.index(c) for c in pk]
    keys = [tuple(r[i] for i in idx) for r in rows]
    null_keys = sum(any(k is None for k in key) for key in keys)
    if null_keys:
        raise ValueError(f"{table}: {null_keys} row(s) have an empty primary-key value {pk}.")
    seen: set[tuple] = set()
    dupes = [k for k in keys if k in seen or seen.add(k)]  # type: ignore[func-returns-value]
    if dupes:
        raise ValueError(
            f"{table}: {len(dupes)} duplicate primary key(s) {pk}, e.g. {dupes[0]}."
        )


def _build_rows(
    df: pd.DataFrame, columns: list[_Column], table: str
) -> tuple[list[str], list[tuple]]:
    """Match DataFrame columns to table columns and convert values to the column types."""
    by_name = {c.name: c for c in columns}
    skipped = [c for c in df.columns if c not in by_name and not _TTC_COL.match(str(c))]
    if skipped:
        logger.info("%s: DataFrame column(s) not in table, skipped: %s", table, skipped)
    absent = [c.name for c in columns if c.name not in df.columns]
    if absent:
        logger.warning("%s: table column(s) not in DataFrame, written as NULL: %s", table, absent)

    names = [c.name for c in columns if c.name in df.columns]
    converted = [_convert_column(df[n], by_name[n], table) for n in names]
    rows = list(zip(*converted)) if converted else []
    return names, rows


def _convert_column(series: pd.Series, col: _Column, table: str) -> list[Any]:
    t = col.data_type
    where = f"{table}.{col.name}"
    if t in ("character varying", "character", "text"):
        return _to_text(series, col, where)
    if t in _INT_RANGES:
        return _to_int(series, t, where)
    if t in ("real", "double precision", "numeric"):
        return _to_float(series, where)
    if t == "boolean":
        return [_to_bool(v, where) for v in series]
    if t == "date":
        return [None if v is None else v.date() for v in _to_datetimes(series, where)]
    if t.startswith("timestamp"):
        return _to_datetimes(series, where)
    if t in ("json", "jsonb"):
        return [_to_json(v, where) for v in series]
    if col.udt_name == "geometry":
        return [_to_ewkt(v, where) for v in series]
    logger.warning("%s: unhandled column type %r; values sent as-is", where, t)
    return [None if _missing(v) else v for v in series]


def _missing(v: Any) -> bool:
    if v is None or v is pd.NA or v is pd.NaT:
        return True
    if isinstance(v, float | np.floating):
        return math.isnan(v)
    return False


def _to_text(series: pd.Series, col: _Column, where: str) -> list[str | None]:
    out: list[str | None] = []
    for v in series:
        if _missing(v):
            out.append(None)
        elif isinstance(v, float | np.floating) and float(v).is_integer():
            out.append(str(int(v)))  # 2024.0 -> '2024'
        else:
            out.append(str(v))
    if col.max_length is not None:
        too_long = [s for s in out if s is not None and len(s) > col.max_length]
        if too_long:
            longest = max(len(s) for s in too_long)
            raise ValueError(
                f"{where}: {len(too_long)} value(s) longer than {col.data_type}({col.max_length}); "
                f"longest is {longest} characters. Widen the column (e.g. TEXT) or shorten the data."
            )
    return out


def _numeric(series: pd.Series, where: str) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    bad = int((nums.isna() & ~series.map(_missing)).sum())
    if bad:
        raise ValueError(f"{where}: {bad} value(s) are not numeric.")
    return nums


def _to_float(series: pd.Series, where: str) -> list[float | None]:
    return [None if _missing(v) else float(v) for v in _numeric(series, where)]


def _to_int(series: pd.Series, data_type: str, where: str) -> list[int | None]:
    nums = _numeric(series, where)
    present = nums.dropna()
    fractional = int((present != present.round()).sum())
    if fractional:
        logger.warning(
            "%s: %d non-integer value(s) rounded for %s column (e.g. %s)",
            where, fractional, data_type, present[present != present.round()].iloc[0],
        )
    lo, hi = _INT_RANGES[data_type]
    rounded = nums.round()
    out_of_range = int(((rounded < lo) | (rounded > hi)).sum())
    if out_of_range:
        raise ValueError(f"{where}: {out_of_range} value(s) outside the {data_type} range.")
    return [None if _missing(v) else int(v) for v in rounded]


_TRUE = {"true", "t", "yes", "y", "1"}
_FALSE = {"false", "f", "no", "n", "0"}


def _to_bool(v: Any, where: str) -> bool | None:
    if _missing(v):
        return None
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | float | np.integer | np.floating) and v in (0, 1):
        return bool(v)
    s = str(v).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise ValueError(f"{where}: cannot interpret {v!r} as a boolean.")


def _to_datetimes(series: pd.Series, where: str) -> list[dt.datetime | None]:
    parsed = pd.to_datetime(series, errors="coerce")
    bad = int((parsed.isna() & ~series.map(_missing)).sum())
    if bad:
        raise ValueError(f"{where}: {bad} value(s) are not valid dates/times.")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return [None if pd.isna(v) else v.to_pydatetime() for v in parsed]


def _to_ewkt(v: Any, where: str) -> str | None:
    """shapely geometry or WKT/EWKT text -> EWKT with SRID (WGS 84 unless the value has one)."""
    if _missing(v):
        return None
    wkt = v.wkt if hasattr(v, "wkt") else str(v).strip()
    if not wkt:
        return None
    if wkt.upper().startswith("SRID="):
        return wkt
    if not re.match(r"^[A-Za-z]+\s*(Z|M|ZM)?\s*(\(|EMPTY)", wkt):
        raise ValueError(f"{where}: value is not WKT geometry: {wkt[:60]!r}")
    return f"SRID={GEOMETRY_SRID};{wkt}"


def _to_json(v: Any, where: str) -> Json | None:
    if isinstance(v, dict | list):
        return Json(v)
    if _missing(v):
        return None
    if isinstance(v, str):
        try:
            return Json(json.loads(v))
        except ValueError as exc:
            raise ValueError(f"{where}: value is not valid JSON: {v[:80]!r}") from exc
    return Json(v)
