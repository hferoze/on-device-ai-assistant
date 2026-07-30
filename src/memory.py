from typing import Dict, List


class ConversationMemory:
    """Stores a limited number of recent conversation messages."""

    def __init__(self, max_messages: int = 6) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be at least 1")

        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self._add_message("user", message)

    def add_assistant_message(self, message: str) -> None:
        self._add_message("assistant", message)

    def _add_message(self, role: str, content: str) -> None:
        content = content.strip()

        if not content:
            return

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

        self.messages = self.messages[-self.max_messages :]

    def get_messages(self) -> List[Dict[str, str]]:
        """Return a copy of the stored messages."""
        return self.messages.copy()

    def build_prompt(self, current_question: str) -> str:
        """
        Build a prompt containing recent conversation history.

        Use this when calling Ollama's /api/generate endpoint.
        """

        prompt_lines = [
            "Continue the conversation using the context below.",
            "Answer the user's latest question clearly and concisely.",
            "",
        ]

        for message in self.messages:
            role = "User" if message["role"] == "user" else "Assistant"
            prompt_lines.append(f"{role}: {message['content']}")

        prompt_lines.append(f"User: {current_question.strip()}")
        prompt_lines.append("Assistant:")

        return "\n".join(prompt_lines)

    def clear(self) -> None:
        """Remove all stored conversation history."""
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)