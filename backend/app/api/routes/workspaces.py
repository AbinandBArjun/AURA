from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_workspace
from app.models import Workspace
from app.schemas.workspace import WorkspaceResponse


router = APIRouter(prefix="/workspaces", tags=["Workspaces"])


@router.get("/current", response_model=WorkspaceResponse)
def get_workspace(
    workspace: Workspace = Depends(get_current_workspace),
):
    return workspace
