from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentResponse(BaseModel):
    id: int
    workspace_id: int
    filename: str
    size_bytes: int
    status: str
    extraction_error: str | None = None
    checksum_sha256: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
