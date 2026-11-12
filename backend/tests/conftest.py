import pytest
from sqlalchemy.orm import Session

from app.database import engine


@pytest.fixture
def db():
    # Application commit() calls release only a savepoint. Nothing is persisted.
    with engine.connect() as connection:
        transaction = connection.begin()
        with Session(bind=connection, join_transaction_mode="create_savepoint") as session:
            try:
                yield session
            finally:
                session.close()
                transaction.rollback()
