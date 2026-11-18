from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.models import User, Workspace
from app.schemas.workspace import WorkspaceResponse


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


@router.get("/current", response_model=WorkspaceResponse)
def get_current_workspace(db: Session = Depends(get_db)):
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

    db.commit()
    db.refresh(workspace)

    return workspace
