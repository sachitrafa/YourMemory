from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from src.services.api_keys import register_agent, revoke_agent, list_agents

router = APIRouter()


class RegisterRequest(BaseModel):
    agent_id:    str
    user_id:     str
    description: Optional[str] = ""
    can_read:    Optional[list[str]] = []
    can_write:   Optional[list[str]] = ["shared", "private"]


class RevokeRequest(BaseModel):
    agent_id: str
    user_id:  str


@router.post("/agents/register")
def register(req: RegisterRequest):
    result = register_agent(
        agent_id=req.agent_id,
        user_id=req.user_id,
        description=req.description,
        can_read=req.can_read,
        can_write=req.can_write,
    )
    return result


@router.post("/agents/revoke")
def revoke(req: RevokeRequest):
    revoked = revoke_agent(req.agent_id, req.user_id)
    if not revoked:
        raise HTTPException(status_code=404, detail=f"Agent '{req.agent_id}' not found or already revoked.")
    return {"revoked": True, "agent_id": req.agent_id}


@router.get("/agents")
def get_agents(user_id: str):
    return {"agents": list_agents(user_id.strip().lower())}
