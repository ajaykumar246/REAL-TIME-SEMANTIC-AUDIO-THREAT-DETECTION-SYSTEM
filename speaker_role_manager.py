"""
Feature 4: Speaker Role Manager (Heuristic Diarization)

Tracks speakers and classifies them as Transmitter (Spammer)
or Receiver (Victim) using keyword heuristics and turn order.

Usage (standalone test):
    python speaker_role_manager.py
"""

from dataclasses import dataclass, field
from typing import Optional

from config import SPAM_KEYWORDS


@dataclass
class SpeakerInfo:
    """Info about a detected speaker."""
    speaker_id: str
    role: str = "Unknown"         # "Transmitter" or "Receiver"
    is_transmitter: bool = False
    turn_count: int = 0
    keyword_hits: list = field(default_factory=list)


class SpeakerRoleManager:
    """
    Heuristic speaker diarization and role assignment.

    Since we don't have true diarization (would need a separate model),
    this class uses a simple turn-based approach:
      - Speaker 0 = first person to speak
      - Speaker 1 = second person to speak

    Role assignment rules (in priority order):
      1. Keyword scan: if text contains spam keywords → Transmitter
      2. Fallback: first speaker (short greeting like "hello") → Receiver,
         second speaker → Transmitter
    """

    def __init__(self, spam_keywords: list = None):
        self.spam_keywords = [kw.lower() for kw in (spam_keywords or SPAM_KEYWORDS)]
        self.speakers: dict[str, SpeakerInfo] = {}
        self.turn_count = 0
        self.role_locked = False  # once roles are assigned, lock them

        # Track which speaker is confirmed Transmitter
        self._transmitter_id: Optional[str] = None
        self._receiver_id: Optional[str] = None

    def _detect_keywords(self, text: str) -> list[str]:
        """Find spam keywords in the given text."""
        text_lower = text.lower()
        return [kw for kw in self.spam_keywords if kw in text_lower]

    def _get_or_create_speaker(self, speaker_id: str) -> SpeakerInfo:
        """Get existing speaker info or create new."""
        if speaker_id not in self.speakers:
            self.speakers[speaker_id] = SpeakerInfo(speaker_id=speaker_id)
        return self.speakers[speaker_id]

    def assign_role(self, speaker_id: str, text: str) -> dict:
        """
        Determine the role of a speaker based on their text.

        Parameters
        ----------
        speaker_id : str
            Speaker identifier (e.g. "Speaker 0", "Speaker 1").
        text : str
            The transcribed text from this speaker.

        Returns
        -------
        dict
            {"speaker_id", "role", "is_transmitter", "keyword_hits"}
        """
        speaker = self._get_or_create_speaker(speaker_id)
        speaker.turn_count += 1
        self.turn_count += 1

        # Rule 1: Keyword scan
        hits = self._detect_keywords(text)
        if hits:
            speaker.keyword_hits.extend(hits)

            if not self.role_locked or self._transmitter_id == speaker_id:
                self._transmitter_id = speaker_id
                speaker.role = "Transmitter"
                speaker.is_transmitter = True

                # Assign other speakers as Receiver
                for sid, info in self.speakers.items():
                    if sid != speaker_id:
                        info.role = "Receiver"
                        info.is_transmitter = False
                        self._receiver_id = sid

                self.role_locked = True

        # Rule 2: Fallback — turn order heuristic
        if not self.role_locked and len(self.speakers) >= 2:
            speaker_ids = list(self.speakers.keys())

            # First speaker → Receiver (usually short greeting)
            self._receiver_id = speaker_ids[0]
            self.speakers[speaker_ids[0]].role = "Receiver"
            self.speakers[speaker_ids[0]].is_transmitter = False

            # Second speaker → Transmitter
            self._transmitter_id = speaker_ids[1]
            self.speakers[speaker_ids[1]].role = "Transmitter"
            self.speakers[speaker_ids[1]].is_transmitter = True

            self.role_locked = True

        return {
            "speaker_id": speaker.speaker_id,
            "role": speaker.role,
            "is_transmitter": speaker.is_transmitter,
            "keyword_hits": hits,
        }

    def get_role(self, speaker_id: str) -> str:
        """Get the current role for a speaker."""
        if speaker_id in self.speakers:
            return self.speakers[speaker_id].role
        return "Unknown"

    def is_transmitter(self, speaker_id: str) -> bool:
        """Check if a speaker is identified as the Transmitter."""
        if speaker_id in self.speakers:
            return self.speakers[speaker_id].is_transmitter
        return False

    def get_summary(self) -> dict:
        """Return a summary of all tracked speakers and roles."""
        return {
            "total_turns": self.turn_count,
            "role_locked": self.role_locked,
            "speakers": {
                sid: {
                    "role": info.role,
                    "is_transmitter": info.is_transmitter,
                    "turn_count": info.turn_count,
                    "keyword_hits": info.keyword_hits,
                }
                for sid, info in self.speakers.items()
            },
        }

    def reset(self):
        """Reset all state for a new call."""
        self.speakers.clear()
        self.turn_count = 0
        self.role_locked = False
        self._transmitter_id = None
        self._receiver_id = None


# ──────────────────────────── standalone test
def _test():
    """Quick test with sample dialogues."""
    manager = SpeakerRoleManager()

    dialogue = [
        ("Speaker 0", "Hello?"),
        ("Speaker 1", "Good morning sir, I am calling from XYZ Bank regarding a special offer."),
        ("Speaker 0", "What offer?"),
        ("Speaker 1", "Congratulations! You have been selected for a pre-approved loan at 2% interest."),
        ("Speaker 0", "I am not interested, thank you."),
        ("Speaker 1", "Sir please, this is a limited time offer. You can win prizes too!"),
    ]

    print("=" * 70)
    print("  Speaker Role Manager — Test")
    print("=" * 70)

    for speaker_id, text in dialogue:
        result = manager.assign_role(speaker_id, text)
        kw_str = f" [keywords: {', '.join(result['keyword_hits'])}]" if result['keyword_hits'] else ""
        print(f"  {speaker_id} → {result['role']}{kw_str}")
        print(f"    Text: \"{text}\"")

    print(f"\n  Summary: {manager.get_summary()}")


if __name__ == "__main__":
    _test()
