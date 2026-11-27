"""Dataset schema v2. Unknown targets are represented by None."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

Content = Literal["chat","social_meme","map_place","shopping_product","article_reading","receipt_confirmation","app_screen","other"]
Purpose = Literal["act_on_it","try_or_buy_later","keep_proof_or_reference","share_or_fun"]
ActionType = Literal["visit_place","consider_purchase","attend_event","complete_task","read_or_review","keep_reference","share"]
NAMES = {"chat":"Messages & conversations","social_meme":"Social & entertainment","map_place":"Places & recommendations",
         "shopping_product":"Products & offers","article_reading":"Reading & information","receipt_confirmation":"Receipts & confirmations",
         "app_screen":"App screens","other":"Miscellaneous"}
TARGETS = ("is_screenshot","content_type","purpose","still_useful")

class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

class Action(Strict):
    type: ActionType
    description: str
    evidence: str
    date_time: str | None = None

class Proposal(Strict):
    is_screenshot: bool | None
    content_type: Content | None
    purpose: Purpose | None
    description: str
    tags: list[str]
    actions: list[Action]
    evidence: str

class Draft(Proposal):
    is_screenshot: bool | None = None
    content_type: Content | None = None
    purpose: Purpose | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)
    evidence: str = ""
    still_useful: bool | None = None
    corrected_transcript: str = ""
    ambiguous: bool = False

class Extensible(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

class ImageInfo(Extensible):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width: int = Field(gt=0)
    height: int = Field(gt=0)

class Labels(Extensible):
    is_screenshot: bool | None
    content_type: Content | None
    purpose: Purpose | None
    still_useful: bool | None
    target_provenance: dict

    @model_validator(mode="after")
    def provenance(self):
        if self.is_screenshot is False and any(getattr(self,k) is not None for k in ("purpose","still_useful")):
            raise ValueError("Owner screenshot targets must be unknown for photos")
        for k in TARGETS:
            origin = self.target_provenance[k]["origin"]
            if origin not in ("unknown","source","human","owner","pseudo"):
                raise ValueError("Unknown provenance")
            if k in ("purpose","still_useful") and getattr(self,k) is not None and origin != "owner":
                raise ValueError("Owner confirmation required")
        return self

class Snapshot(Extensible):
    schema_version: Literal[2]
    id: str
    source: dict
    image: ImageInfo
    labels: Labels
    status: dict
    annotations: dict
    features: dict

    @model_validator(mode="after")
    def review_contract(self):
        from pathlib import PurePosixPath
        path = PurePosixPath(self.image.path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Image path must be relative to the dataset.")
        origins = self.source.get("origins")
        if not isinstance(origins,list) or not origins:
            raise ValueError("At least one image origin is required.")
        owners = {s.get("owner") for s in origins if s.get("dataset")=="personal"}
        for k in TARGETS:
            p = self.labels.target_provenance[k]
            if getattr(self.labels,k) is not None and p["origin"]=="owner":
                if p.get("reviewer") not in owners or p.get("reviewer") not in ("A","B","C","D") or p.get("role")!="owner" or not p.get("at"):
                    raise ValueError("Owner target requires matching owner, role and timestamp.")
        r = self.annotations["review"]
        if r["state"] not in ("pending","draft","reviewed","skipped","auto_labeled") or not isinstance(r["revision"],int):
            raise ValueError("Invalid review state.")
        q = self.annotations.get("qwen")
        if q and q.get("status")=="ok":
            Proposal.model_validate(q["fields"])
        if self.annotations.get("draft") is not None:
            Draft.model_validate(self.annotations["draft"])
        if not self.status.get("duplicate_group_id"):
            raise ValueError("Missing duplicate group.")
        return self
