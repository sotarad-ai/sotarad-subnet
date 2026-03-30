"""
System prompt and user message builder for lung-finding extraction.

Import this module from validator.py and test scripts:

    from prompts.system_prompt import SYSTEM_PROMPT, build_user_message
"""

# ---------------------------------------------------------------------------
# Controlled vocabularies (single source of truth for validator scoring too)
# ---------------------------------------------------------------------------

TARGET_CONDITIONS = ["Pneumonia", "Tuberculosis", "Bronchitis", "Silicosis"]

ALLOWED_STATUSES = ["active", "previous"]

ALLOWED_LATERALITIES = ["bilateral", "left", "right"]

# **Left** or **right** laterality: only these (singular zone / lobe / space).
LOCATION_LEFT_OR_RIGHT_ONLY = [
    "upper zone",
    "lower zone",
    "pleural space",
    "upper lobe",
    "lower lobe",
]

# **Bilateral** laterality only: no plural variants — never use these when
# laterality is "left" or "right".
LOCATION_BILATERAL_EXCLUSIVE = [
    "diffuse",
    "hilar",
    "perihilar",
]

# For **bilateral** laterality, zone / lobe / pleural use plural forms; for
# **left** / **right**, use the dict keys (singular).
LOCATION_PLURAL_BY_SINGULAR = {
    "upper zone": "upper zones",
    "lower zone": "lower zones",
    "pleural space": "pleural spaces",
    "upper lobe": "upper lobes",
    "lower lobe": "lower lobes",
}

# Union of every token that may appear in a valid `location` string.
LOCATION_CANONICAL_SINGULAR = (
    LOCATION_LEFT_OR_RIGHT_ONLY + LOCATION_BILATERAL_EXCLUSIVE
)

ALLOWED_LOCATIONS = sorted(
    set(LOCATION_LEFT_OR_RIGHT_ONLY)
    | set(LOCATION_PLURAL_BY_SINGULAR.values())
    | set(LOCATION_BILATERAL_EXCLUSIVE)
)

ALLOWED_CERTAINTIES = ["definite", "probable"]

LOCATION_SEPARATOR = " and "

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You analyze the chest X-ray and user demographics. Output must be ONLY a JSON \
array: first character [, last character ], nothing before or after (no prose, \
labels, markdown, or wrapper objects). Use [] if none apply.

Do all reasoning silently. Never output steps, headings (e.g. "Step 1", "##"), \
explanations, or phrases like "final answer", "Answer:", or "The result is".

Report only these conditions if clearly supported by the image: Pneumonia, \
Tuberculosis, Bronchitis, Silicosis. One object per condition; do not invent \
findings or other diagnoses.

Each array element is exactly five keys: condition, status, laterality, location, certainty.
condition: "Pneumonia" | "Tuberculosis" | "Bronchitis" | "Silicosis"
status: "active" | "previous"
laterality: "bilateral" | "left" | "right"
certainty: "definite" | "probable"
location: for left/right use only singular "upper zone","lower zone","pleural space","upper lobe","lower lobe" (multiple segments joined with " and "). For bilateral use plural "upper zones","lower zones","pleural spaces","upper lobes","lower lobes" and/or "diffuse","hilar","perihilar". Never pair diffuse/hilar/perihilar with left or right.

No other keys (no descriptors, icd10, snomed, incidental). Use probable when unsure.
"""

# ---------------------------------------------------------------------------
# User message builder
# ---------------------------------------------------------------------------


def build_user_message(patient_demographics: dict) -> str:
    """
    Build the text portion of the user turn from patient demographics.
    The image is passed separately as a base64/URL attachment per the
    vision-language API being used.

    Args:
        patient_demographics: dict with keys such as
            "age_at_acquisition" (int) and "sex" ("M" | "F")

    Returns:
        Plain-text string to accompany the image in the user turn.
    """
    age = patient_demographics.get("age_at_acquisition", "unknown")
    sex_raw = patient_demographics.get("sex", "unknown")
    sex = {"M": "male", "F": "female"}.get(sex_raw, sex_raw)

    return f"Patient: {age}-year-old {sex}."
