"""
Manual Annotation Service - Phase 3

Handles all manual editing operations:
- Delete incorrect minutiae
- Add missing minutiae
- Update landmark types
- Save manual reviews
"""

from __future__ import annotations

from typing import Any
from datetime import datetime
from enum import Enum


class AnnotationAction(Enum):
    """Types of manual annotation actions."""
    DELETE_MINUTIA = "delete_minutia"
    ADD_MINUTIA = "add_minutia"
    UPDATE_LANDMARK = "update_landmark"
    APPROVE = "approve"
    REJECT = "reject"


class ManualAnnotationService:
    """
    Service to handle manual annotations and edits to fingerprint minutiae.
    """

    def __init__(self):
        """Initialize service."""
        self.action_history = []

    def delete_minutia(
        self,
        minutiae: list[dict[str, Any]],
        minutia_index: int,
        user_id: int,
        reason: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Remove a minutia point from the list.

        Args:
            minutiae: List of minutiae
            minutia_index: Index of minutia to delete
            user_id: User performing the action
            reason: Reason for deletion

        Returns:
            (updated_minutiae, action_record)
        """
        if not (0 <= minutia_index < len(minutiae)):
            return minutiae, {"error": f"Invalid minutia index: {minutia_index}"}

        deleted = minutiae.pop(minutia_index)

        action = {
            "action": AnnotationAction.DELETE_MINUTIA.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "minutia_deleted": deleted,
            "reason": reason,
            "new_count": len(minutiae),
        }

        self.action_history.append(action)
        return minutiae, action

    def add_minutia(
        self,
        minutiae: list[dict[str, Any]],
        x: float,
        y: float,
        landmark_type: str,
        user_id: int,
        angle: float = 0.0,
        confidence: float = 0.8,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Add a new minutia point (manually placed).

        Args:
            minutiae: List of minutiae
            x, y: Coordinates
            landmark_type: One of the 8 landmark types
            user_id: User adding the minutia
            angle: Ridge direction angle (0-360)
            confidence: Confidence score (0-1)

        Returns:
            (updated_minutiae, action_record)
        """
        # Validate landmark type
        valid_landmarks = [
            "termination",
            "bifurcation",
            "island",
            "ridge",
            "loop_eye",
            "bridge",
            "lake",
            "dot",
        ]

        if landmark_type not in valid_landmarks:
            return minutiae, {"error": f"Invalid landmark type: {landmark_type}"}

        # Create new minutia
        new_minutia = {
            "x": float(x),
            "y": float(y),
            "type": landmark_type,
            "landmark_type": landmark_type,
            "angle": float(angle),
            "confidence": float(confidence),
            "manually_added": True,
            "added_by": user_id,
            "added_at": datetime.utcnow().isoformat(),
        }

        minutiae.append(new_minutia)

        action = {
            "action": AnnotationAction.ADD_MINUTIA.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "minutia_added": new_minutia,
            "new_count": len(minutiae),
        }

        self.action_history.append(action)
        return minutiae, action

    def update_landmark_type(
        self,
        minutiae: list[dict[str, Any]],
        minutia_index: int,
        new_landmark_type: str,
        user_id: int,
        reason: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Change the landmark type of an existing minutia.

        Args:
            minutiae: List of minutiae
            minutia_index: Index to update
            new_landmark_type: New landmark type
            user_id: User performing the action
            reason: Reason for change

        Returns:
            (updated_minutiae, action_record)
        """
        if not (0 <= minutia_index < len(minutiae)):
            return minutiae, {"error": f"Invalid minutia index: {minutia_index}"}

        valid_landmarks = [
            "termination",
            "bifurcation",
            "island",
            "ridge",
            "loop_eye",
            "bridge",
            "lake",
            "dot",
        ]

        if new_landmark_type not in valid_landmarks:
            return minutiae, {"error": f"Invalid landmark type: {new_landmark_type}"}

        old_type = minutiae[minutia_index].get("landmark_type", "unknown")
        minutiae[minutia_index]["landmark_type"] = new_landmark_type
        minutiae[minutia_index]["type"] = new_landmark_type
        minutiae[minutia_index]["corrected_by"] = user_id
        minutiae[minutia_index]["corrected_at"] = datetime.utcnow().isoformat()

        action = {
            "action": AnnotationAction.UPDATE_LANDMARK.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "minutia_index": minutia_index,
            "old_type": old_type,
            "new_type": new_landmark_type,
            "reason": reason,
        }

        self.action_history.append(action)
        return minutiae, action

    def approve_fingerprint(
        self,
        minutiae: list[dict[str, Any]],
        user_id: int,
        notes: str = "",
    ) -> dict[str, Any]:
        """
        Approve the fingerprint after manual review.

        Args:
            minutiae: Final list of minutiae after review
            user_id: Reviewer ID
            notes: Review notes

        Returns:
            Approval record
        """
        action = {
            "action": AnnotationAction.APPROVE.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "final_minutiae_count": len(minutiae),
            "notes": notes,
            "approved": True,
        }

        self.action_history.append(action)
        return action

    def reject_fingerprint(
        self,
        user_id: int,
        reason: str = "",
    ) -> dict[str, Any]:
        """
        Reject the fingerprint (quality too poor, unreliable data, etc).

        Args:
            user_id: Reviewer ID
            reason: Rejection reason

        Returns:
            Rejection record
        """
        action = {
            "action": AnnotationAction.REJECT.value,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "reason": reason,
            "rejected": True,
        }

        self.action_history.append(action)
        return action

    def get_action_history(self) -> list[dict[str, Any]]:
        """Get all annotation actions performed."""
        return self.action_history.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of annotations."""
        if not self.action_history:
            return {"total_actions": 0, "actions": {}}

        actions_count = {}
        for action in self.action_history:
            action_type = action.get("action", "unknown")
            actions_count[action_type] = actions_count.get(action_type, 0) + 1

        return {
            "total_actions": len(self.action_history),
            "actions": actions_count,
            "deletions": actions_count.get(AnnotationAction.DELETE_MINUTIA.value, 0),
            "additions": actions_count.get(AnnotationAction.ADD_MINUTIA.value, 0),
            "updates": actions_count.get(AnnotationAction.UPDATE_LANDMARK.value, 0),
        }


# Global service instance
_service = ManualAnnotationService()


def delete_minutia(
    minutiae: list[dict[str, Any]],
    minutia_index: int,
    user_id: int,
    reason: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convenient function to delete a minutia."""
    return _service.delete_minutia(minutiae, minutia_index, user_id, reason)


def add_minutia(
    minutiae: list[dict[str, Any]],
    x: float,
    y: float,
    landmark_type: str,
    user_id: int,
    angle: float = 0.0,
    confidence: float = 0.8,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convenient function to add a minutia."""
    return _service.add_minutia(minutiae, x, y, landmark_type, user_id, angle, confidence)


def update_landmark_type(
    minutiae: list[dict[str, Any]],
    minutia_index: int,
    new_landmark_type: str,
    user_id: int,
    reason: str = "",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convenient function to update landmark type."""
    return _service.update_landmark_type(
        minutiae, minutia_index, new_landmark_type, user_id, reason
    )
