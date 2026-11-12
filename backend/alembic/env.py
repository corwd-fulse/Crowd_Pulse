from alembic import context

from app.database import engine
from app.models import Base

target_metadata = Base.metadata


def include_name(name, type_, parent_names):
    # Leave unrelated existing tables outside autogeneration's scope.
    if type_ == "table":
        return name in target_metadata.tables
    return True


options = dict(
    target_metadata=target_metadata,
    include_name=include_name,
    compare_type=True,
    compare_server_default=True,
    version_table_schema="public",
)

if context.is_offline_mode():
    context.configure(
        dialect_name="postgresql", literal_binds=True,
        dialect_opts={"paramstyle": "named"}, **options,
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    # Tests may inject a connection to keep all DDL and fixture rows in a rollback.
    connection = context.config.attributes.get("connection")
    if connection is not None:
        context.configure(connection=connection, **options)
        with context.begin_transaction():
            context.run_migrations()
    else:
        with engine.connect() as connection:
            connection.exec_driver_sql("SET search_path TO public")
            connection.commit()
            context.configure(connection=connection, **options)
            with context.begin_transaction():
                context.run_migrations()
