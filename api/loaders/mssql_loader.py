"""Microsoft SQL Server loader for schema discovery and query execution."""

import contextlib
import datetime
import decimal
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import pymssql

from api.config import Config
from api.loaders.base_loader import BaseLoader
from api.loaders.graph_loader import load_to_graph
from api.loaders.introspection import run_introspection


class MSSQLQueryError(Exception):
    """Raised when SQL Server rejects or cannot complete a query."""


class MSSQLLoader(BaseLoader):
    """Load one SQL Server schema into QueryWeaver's schema graph."""

    SCHEMA_MODIFYING_OPERATIONS = {
        "CREATE", "ALTER", "DROP", "RENAME", "TRUNCATE",
    }
    SCHEMA_PATTERNS = [
        r"^\s*CREATE\s+(?:TABLE|INDEX|VIEW|SCHEMA)",
        r"^\s*ALTER\s+TABLE",
        r"^\s*DROP\s+(?:TABLE|INDEX|VIEW|SCHEMA)",
        r"^\s*TRUNCATE\s+TABLE",
        r"^\s*EXEC(?:UTE)?\s+SP_RENAME",
    ]

    @staticmethod
    def _parse_url(connection_url: str) -> Dict[str, Any]:
        parsed = urlparse(connection_url)
        if parsed.scheme.lower() not in ("mssql", "sqlserver"):
            raise ValueError("Invalid SQL Server URL. Expected mssql://user:password@host:1433/database")
        database = unquote(parsed.path.lstrip("/").split("/", 1)[0])
        if not parsed.hostname or not parsed.username or not database:
            raise ValueError("SQL Server URL must include username, host, and database")
        query = parse_qs(parsed.query)
        schema = query.get("schema", ["dbo"])[0].strip() or "dbo"
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_@$#]*", schema):
            raise ValueError("Invalid SQL Server schema name")
        return {
            "server": parsed.hostname,
            "port": parsed.port or 1433,
            "user": unquote(parsed.username),
            "password": unquote(parsed.password or ""),
            "database": database,
            "schema": schema,
        }

    @staticmethod
    def _connect(connection_url: str, timeout: int):
        params = MSSQLLoader._parse_url(connection_url)
        return pymssql.connect(  # pylint: disable=no-member
            server=params["server"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            database=params["database"],
            login_timeout=Config.DB_CONNECT_TIMEOUT,
            timeout=timeout,
            as_dict=True,
        )

    @staticmethod
    def _quote_identifier(value: str) -> str:
        return "[" + value.replace("]", "]]" ) + "]"

    @staticmethod
    def _qualified_table(table_name: str) -> str:
        return ".".join(
            MSSQLLoader._quote_identifier(part)
            for part in table_name.split(".", 1)
        )

    @staticmethod
    def _execute_sample_query(
        cursor, table_name: str, col_name: str, sample_size: int = 3
    ) -> List[Any]:
        qualified = MSSQLLoader._qualified_table(table_name)
        column = MSSQLLoader._quote_identifier(col_name)
        cursor.execute(
            f"SELECT TOP ({int(sample_size)}) {column} "
            f"FROM {qualified} WHERE {column} IS NOT NULL"
        )
        rows = cursor.fetchall()
        return [row[col_name] for row in rows if row.get(col_name) is not None]

    @staticmethod
    def _serialize_value(value):
        if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
            return value.isoformat()
        if isinstance(value, decimal.Decimal):
            return float(value)
        if isinstance(value, uuid.UUID):
            return str(value)
        if isinstance(value, bytes):
            return value.hex()
        return value

    @staticmethod
    def _introspect_schema(connection_url: str, schema: str):
        conn = None
        cursor = None
        try:
            conn = MSSQLLoader._connect(connection_url, Config.DB_SCHEMA_TIMEOUT)
            cursor = conn.cursor(as_dict=True)
            entities = MSSQLLoader.extract_tables_info(cursor, schema)
            relationships = MSSQLLoader.extract_relationships(cursor, schema)
            return entities, relationships
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    @staticmethod
    async def load(
        prefix: str, connection_url: str, db=None
    ) -> AsyncGenerator[tuple[bool, str], None]:
        try:
            params = MSSQLLoader._parse_url(connection_url)
            yield True, "Extracting table information..."
            entities, relationships = await run_introspection(
                MSSQLLoader._introspect_schema, connection_url, params["schema"]
            )
            yield True, "Loading data into graph..."
            await load_to_graph(
                f"{prefix}_{params['database']}",
                entities,
                relationships,
                db_name=params["database"],
                db_url=connection_url,
                db=db,
            )
            yield True, (
                "Microsoft SQL Server schema loaded successfully. "
                f"Found {len(entities)} tables."
            )
        except (pymssql.Error, ValueError) as exc:
            logging.error("SQL Server connection error: %s", exc)
            yield False, "Failed to connect to Microsoft SQL Server database"
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.error("Error loading SQL Server schema: %s", exc)
            yield False, "Failed to load Microsoft SQL Server schema"

    @staticmethod
    def extract_tables_info(cursor, schema: str) -> Dict[str, Any]:
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
            ORDER BY TABLE_NAME
            """,
            (schema,),
        )
        entities: Dict[str, Any] = {}
        for table_row in cursor.fetchall():
            table_name = table_row["TABLE_NAME"]
            columns = MSSQLLoader.extract_columns_info(cursor, schema, table_name)
            foreign_keys = MSSQLLoader.extract_foreign_keys(cursor, schema, table_name)
            entities[table_name] = {
                "description": f"Table: {schema}.{table_name}",
                "columns": columns,
                "foreign_keys": foreign_keys,
                "col_descriptions": [column["description"] for column in columns.values()],
            }
        return entities

    @staticmethod
    def extract_columns_info(cursor, schema: str, table_name: str) -> Dict[str, Any]:
        cursor.execute(
            """
            SELECT c.COLUMN_NAME, c.DATA_TYPE, c.IS_NULLABLE, c.COLUMN_DEFAULT,
                   CASE
                     WHEN pk.COLUMN_NAME IS NOT NULL THEN 'PRIMARY KEY'
                     WHEN fk.COLUMN_NAME IS NOT NULL THEN 'FOREIGN KEY'
                     WHEN uq.COLUMN_NAME IS NOT NULL THEN 'UNIQUE KEY'
                     ELSE 'NONE'
                   END AS KEY_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS c
            OUTER APPLY (
                SELECT TOP 1 kcu.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                  AND kcu.TABLE_SCHEMA = c.TABLE_SCHEMA
                  AND kcu.TABLE_NAME = c.TABLE_NAME
                  AND kcu.COLUMN_NAME = c.COLUMN_NAME
            ) pk
            OUTER APPLY (
                SELECT TOP 1 kcu.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
                  AND kcu.TABLE_SCHEMA = c.TABLE_SCHEMA
                  AND kcu.TABLE_NAME = c.TABLE_NAME
                  AND kcu.COLUMN_NAME = c.COLUMN_NAME
            ) fk
            OUTER APPLY (
                SELECT TOP 1 kcu.COLUMN_NAME
                FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                  ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                 AND tc.CONSTRAINT_SCHEMA = kcu.CONSTRAINT_SCHEMA
                WHERE tc.CONSTRAINT_TYPE = 'UNIQUE'
                  AND kcu.TABLE_SCHEMA = c.TABLE_SCHEMA
                  AND kcu.TABLE_NAME = c.TABLE_NAME
                  AND kcu.COLUMN_NAME = c.COLUMN_NAME
            ) uq
            WHERE c.TABLE_SCHEMA = %s AND c.TABLE_NAME = %s
            ORDER BY c.ORDINAL_POSITION
            """,
            (schema, table_name),
        )
        columns: Dict[str, Any] = {}
        for row in cursor.fetchall():
            name = row["COLUMN_NAME"]
            description = [f"Column {name} of type {row['DATA_TYPE']}"]
            if row["KEY_TYPE"] != "NONE":
                description.append(f"({row['KEY_TYPE']})")
            if row["IS_NULLABLE"] == "NO":
                description.append("(NOT NULL)")
            if row["COLUMN_DEFAULT"] is not None:
                description.append(f"(Default: {row['COLUMN_DEFAULT']})")
            columns[name] = {
                "type": row["DATA_TYPE"],
                "null": row["IS_NULLABLE"],
                "key": row["KEY_TYPE"],
                "description": " ".join(description),
                "default": row["COLUMN_DEFAULT"],
                "sample_values": [],
            }

        # One small read per table instead of one query per column. A database
        # with hundreds of tables commonly has thousands of columns; issuing a
        # separate sample query for each makes initial schema loading needlessly
        # slow and can exhaust connection/proxy limits.
        if columns:
            qualified = MSSQLLoader._qualified_table(f"{schema}.{table_name}")
            cursor.execute(f"SELECT TOP (3) * FROM {qualified}")
            sample_rows = cursor.fetchall()
            for name in columns:
                values = [row.get(name) for row in sample_rows if row.get(name) is not None]
                if values and isinstance(values[0], (str, int, float)):
                    columns[name]["sample_values"] = [str(value) for value in values]
        return columns

    @staticmethod
    def extract_foreign_keys(cursor, schema: str, table_name: str) -> List[Dict[str, str]]:
        cursor.execute(
            """
            SELECT fk.name AS CONSTRAINT_NAME,
                   pc.name AS COLUMN_NAME,
                   rt.name AS REFERENCED_TABLE_NAME,
                   rc.name AS REFERENCED_COLUMN_NAME
            FROM sys.foreign_keys fk
            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            JOIN sys.tables pt ON fkc.parent_object_id = pt.object_id
            JOIN sys.schemas ps ON pt.schema_id = ps.schema_id
            JOIN sys.columns pc
              ON pc.object_id = pt.object_id AND pc.column_id = fkc.parent_column_id
            JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id
            JOIN sys.columns rc
              ON rc.object_id = rt.object_id AND rc.column_id = fkc.referenced_column_id
            WHERE ps.name = %s AND pt.name = %s
            ORDER BY fk.name, fkc.constraint_column_id
            """,
            (schema, table_name),
        )
        return [
            {
                "constraint_name": row["CONSTRAINT_NAME"],
                "column": row["COLUMN_NAME"],
                "referenced_table": row["REFERENCED_TABLE_NAME"],
                "referenced_column": row["REFERENCED_COLUMN_NAME"],
            }
            for row in cursor.fetchall()
        ]

    @staticmethod
    def extract_relationships(cursor, schema: str) -> Dict[str, List[Dict[str, str]]]:
        cursor.execute(
            """
            SELECT fk.name AS CONSTRAINT_NAME,
                   pt.name AS TABLE_NAME,
                   pc.name AS COLUMN_NAME,
                   rt.name AS REFERENCED_TABLE_NAME,
                   rc.name AS REFERENCED_COLUMN_NAME
            FROM sys.foreign_keys fk
            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            JOIN sys.tables pt ON fkc.parent_object_id = pt.object_id
            JOIN sys.schemas ps ON pt.schema_id = ps.schema_id
            JOIN sys.columns pc
              ON pc.object_id = pt.object_id AND pc.column_id = fkc.parent_column_id
            JOIN sys.tables rt ON fkc.referenced_object_id = rt.object_id
            JOIN sys.columns rc
              ON rc.object_id = rt.object_id AND rc.column_id = fkc.referenced_column_id
            WHERE ps.name = %s
            ORDER BY fk.name, fkc.constraint_column_id
            """,
            (schema,),
        )
        relationships: Dict[str, List[Dict[str, str]]] = {}
        for row in cursor.fetchall():
            name = row["CONSTRAINT_NAME"]
            relationships.setdefault(name, []).append({
                "from": row["TABLE_NAME"],
                "to": row["REFERENCED_TABLE_NAME"],
                "source_column": row["COLUMN_NAME"],
                "target_column": row["REFERENCED_COLUMN_NAME"],
                "note": f"Foreign key constraint: {name}",
            })
        return relationships

    @staticmethod
    def is_schema_modifying_query(sql_query: str) -> Tuple[bool, str]:
        if not sql_query or not sql_query.strip():
            return False, ""
        normalized = sql_query.strip().upper()
        first_word = normalized.split()[0]
        if first_word in MSSQLLoader.SCHEMA_MODIFYING_OPERATIONS:
            return True, first_word
        if first_word in ("EXEC", "EXECUTE") and any(
            re.match(pattern, normalized, re.IGNORECASE)
            for pattern in MSSQLLoader.SCHEMA_PATTERNS
        ):
            return True, "RENAME"
        return False, ""

    @staticmethod
    async def refresh_graph_schema(graph_id: str, db_url: str, db=None) -> Tuple[bool, str]:
        try:
            from api.core.db_resolver import resolve_db  # pylint: disable=import-outside-toplevel

            await resolve_db(db).select_graph(graph_id).delete()
            database = MSSQLLoader._parse_url(db_url)["database"]
            suffix = f"_{database}"
            prefix = graph_id[:-len(suffix)] if graph_id.endswith(suffix) else graph_id
            success = False
            message = "Failed to reload schema"
            async for success, message in MSSQLLoader.load(prefix, db_url, db=db):
                pass
            return success, message
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logging.error("Error refreshing SQL Server schema: %s", exc)
            return False, "Error refreshing SQL Server schema"

    @staticmethod
    def execute_sql_query(sql_query: str, db_url: str) -> List[Dict[str, Any]]:
        conn = None
        cursor = None
        try:
            conn = MSSQLLoader._connect(db_url, Config.DB_STATEMENT_TIMEOUT)
            cursor = conn.cursor(as_dict=True)
            cursor.execute(sql_query)
            if cursor.description is not None:
                return [
                    {key: MSSQLLoader._serialize_value(value) for key, value in row.items()}
                    for row in cursor.fetchall()
                ]
            affected = cursor.rowcount
            operation = sql_query.strip().split()[0].upper()
            conn.commit()
            result = {"operation": operation, "status": "success"}
            if operation in ("INSERT", "UPDATE", "DELETE", "MERGE"):
                result["affected_rows"] = affected
            return [result]
        except pymssql.Error as exc:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.rollback()
            raise MSSQLQueryError(f"Microsoft SQL Server query execution error: {exc}") from exc
        finally:
            if cursor is not None:
                with contextlib.suppress(Exception):
                    cursor.close()
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
