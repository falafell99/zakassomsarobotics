from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    distance: float = 5.0
    image: str
    speak: bool = True


class Detection(BaseModel):
    label: str
    raw_label: str | None = None
    category: str = "object"
    confidence: float = 0.0
    distance: float = 5.0
    position: str = "center"
    bbox: list[int] | None = None
    action: str = "CLEAR"
    source: str = "unknown"
    priority: float = 0.0
    risk: float = 0.0


class AnalyzeResponse(BaseModel):
    command: str
    message: str
    obstacle: str = "none"
    distance: float = 5.0
    detections: list[Detection] = Field(default_factory=list)
    speak: bool = False
    processing_ms: float = 0.0
    detector: str = "unknown"
    reason: str = ""
    zones: dict[str, float] = Field(default_factory=dict)
    timestamp: str | None = None
