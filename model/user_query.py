from pydantic import BaseModel


class UserQuery(BaseModel):
    name: str
    query: str
