from fastapi import Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import User, Workspace


def get_current_workspace(
    db: Session = Depends(get_db),
) -> Workspace:
    """Resolve the development workspace for the current application user."""

    if settings.environment.lower() != "development":
        raise HTTPException(
            status_code=404,
            detail="Workspace bootstrap is only available in development.",
        )

    user = db.scalar(
        select(User).where(
            User.email == settings.development_user_email
        )
    )

    if user is None:
        user = User(
            email=settings.development_user_email,
            display_name=settings.development_user_name,
        )
        db.add(user)
        db.flush()

    workspace = db.scalar(
        select(Workspace).where(
            Workspace.owner_id == user.id,
            Workspace.name == settings.development_workspace_name,
        )
    )

    if workspace is None:
        workspace = Workspace(
            owner_id=user.id,
            name=settings.development_workspace_name,
        )
        db.add(workspace)

    try:
        db.commit()
        db.refresh(workspace)
    except Exception:
        db.rollback()
        raise

    return workspace
