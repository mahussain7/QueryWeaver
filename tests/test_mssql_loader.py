"""Microsoft SQL Server loader unit tests."""

from unittest.mock import MagicMock, patch

import pytest

from api.core.pipeline import MSSQLLoader, get_database_type_and_loader


@pytest.mark.unit
def test_detects_mssql_urls():
    db_type, loader = get_database_type_and_loader(
        "mssql://sa:secret@localhost:1433/example"
    )
    assert db_type == "mssql"
    assert loader is MSSQLLoader


@pytest.mark.unit
def test_parses_encoded_credentials_and_schema():
    params = MSSQLLoader._parse_url(
        "mssql://user:p%40ss@db.example:1444/reporting?schema=analytics"
    )
    assert params == {
        "server": "db.example",
        "port": 1444,
        "user": "user",
        "password": "p@ss",
        "database": "reporting",
        "schema": "analytics",
    }


@pytest.mark.unit
def test_rejects_invalid_schema():
    with pytest.raises(ValueError, match="schema"):
        MSSQLLoader._parse_url(
            "mssql://sa:secret@localhost/example?schema=dbo%3BDROP%20TABLE%20x"
        )


@pytest.mark.unit
def test_connect_applies_timeouts():
    with patch("api.loaders.mssql_loader.pymssql.connect") as connect:
        MSSQLLoader._connect(
            "mssql://sa:secret@localhost:1433/example", timeout=47
        )

    kwargs = connect.call_args.kwargs
    assert kwargs["login_timeout"] > 0
    assert kwargs["timeout"] == 47
    assert kwargs["as_dict"] is True


@pytest.mark.unit
def test_samples_all_columns_with_one_query_per_table():
    cursor = MagicMock()
    cursor.fetchall.side_effect = [
        [
            {
                "COLUMN_NAME": "id",
                "DATA_TYPE": "int",
                "IS_NULLABLE": "NO",
                "COLUMN_DEFAULT": None,
                "KEY_TYPE": "PRIMARY KEY",
            },
            {
                "COLUMN_NAME": "name",
                "DATA_TYPE": "nvarchar",
                "IS_NULLABLE": "YES",
                "COLUMN_DEFAULT": None,
                "KEY_TYPE": "NONE",
            },
        ],
        [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}],
    ]

    columns = MSSQLLoader.extract_columns_info(cursor, "dbo", "items")

    assert cursor.execute.call_count == 2
    assert columns["id"]["sample_values"] == ["1", "2"]
    assert columns["name"]["sample_values"] == ["first", "second"]
