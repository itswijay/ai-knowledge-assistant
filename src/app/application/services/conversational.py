import re

GREETING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(hi+|hey+|hello+|howdy|hola|greetings)$", re.IGNORECASE),
    re.compile(r"^(hi+|hey+|hello+)\s+(there|assistant|bot|friend)$", re.IGNORECASE),
    re.compile(r"^good\s+(morning|afternoon|evening|day)$", re.IGNORECASE),
    re.compile(
        r"^(how\s+are\s+you(\s+doing)?|how['’]?s\s+it\s+going|what['’]?s\s+up|sup)$",
        re.IGNORECASE,
    ),
    re.compile(r"^(thanks|thank\s+you|thx)$", re.IGNORECASE),
    re.compile(r"^(bye|goodbye|see\s+you)$", re.IGNORECASE),
)


def is_conversational_greeting(text: str) -> bool:
    """Return True if the input text is a conversational greeting or pleasantry."""
    cleaned = text.strip().strip("!?.,;:'\" \t\n")
    return any(pattern.match(cleaned) for pattern in GREETING_PATTERNS)


def build_greeting_prompt(
    assistant_name: str,
    assistant_instructions: str | None = None,
) -> str:
    """Build a conversational persona system prompt for answering greetings."""
    cleaned_name = assistant_name.strip() or "AI Assistant"
    instructions = [
        f"You are {cleaned_name}, a helpful AI assistant.",
        "The user greeted you or shared a conversational pleasantry.",
        "Respond warmly, politely, and concisely (1 to 2 sentences) in your assistant persona.",
        "Introduce yourself and briefly invite the user to ask about your knowledge base, products, or services.",
        "Do not invent factual claims or answer questions not covered by the knowledge base.",
    ]
    if assistant_instructions and assistant_instructions.strip():
        instructions.append(
            f"Assistant instructions to observe:\n{assistant_instructions.strip()}"
        )
    return "\n\n".join(instructions)
