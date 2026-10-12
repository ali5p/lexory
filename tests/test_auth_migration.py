import importlib.util
from pathlib import Path


def test_auth_cleanup_migration_does_not_delete_data_at_startup():
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0011_purge_anonymous_data.py"
    )
    spec = importlib.util.spec_from_file_location("cleanup_migration", migration_path)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)

    migration.upgrade()
    migration.downgrade()
