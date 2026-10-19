"""Page-context utilities for the PV NarrativeGuard UI.

Page context tells RAB what part of the application the user
is currently working in.

This does not perform reasoning or retrieval.
It only describes application state.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PageContext:
    """Context supplied by the Streamlit page to RAB."""

    page_name: str
    selected_case_id: Optional[str] = None
    selected_record: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a serializable context representation."""

        return {
            "page_name": self.page_name,
            "selected_case_id": self.selected_case_id,
            "selected_record": self.selected_record or {},
            "metadata": self.metadata or {},
        }


def create_page_context(
    page_name: str,
    selected_case_id: Optional[str] = None,
    selected_record: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> PageContext:
    """Create a page context object."""

    return PageContext(
        page_name=page_name,
        selected_case_id=selected_case_id,
        selected_record=selected_record,
        metadata=metadata,
    )