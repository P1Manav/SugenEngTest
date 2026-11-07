"""Text processing utilities for clinical notes and ICD-11 descriptions.
Includes abbreviation expansion, text normalization, and tokenization.
"""

import re
from typing import List

# Stemming / separator pattern for postcoordination stripping
STEM_PATTERN = re.compile(r"[&/]")


def normalize_code(code: str) -> str:
    """Normalize ICD-11 code: strip postcoordination (& or /), strip whitespace, uppercase."""
    if not isinstance(code, str):
        return ""
    stem = STEM_PATTERN.split(code.strip())[0].strip().upper()
    return stem


# Common East African and outpatient clinical abbreviations
CLINICAL_ABBREVIATIONS = {
    # Symptoms / History
    r"\bc/o\b": "complains of",
    r"\bh/o\b": "history of",
    r"\bfhx\b": "family history",
    r"\bshx\b": "social history",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\bs/p\b": "status post",
    r"\byo\b": "year old",
    r"\by/o\b": "year old",
    r"\bm/o\b": "month old",

    # Common diseases / conditions
    r"\buncomplicated\b": "uncomplicated without complication",
    r"\bchickenpox\b": "varicella chickenpox",
    r"\bthrush\b": "candidiasis candidosis thrush",
    r"\bvaginal\s+thrush\b": "vulvovaginal candidosis thrush",
    r"\bdysentery\b": "amoebic dysentery amoebiasis",
    r"\bnickel\b": "nickel metal metals",
    r"\bleg\b": "leg lower extremity lower limb",
    r"\bcalf\b": "calf lower extremity lower limb",
    r"\btinea\s+capitis\b": "tinea capitis dermatophytosis of scalp ringworm",
    r"\bgestational\s+diabetes\b": "gestational diabetes mellitus in pregnancy developing during pregnancy",
    r"\bdiabetes\s+(?:that\s+has\s+come\s+with|in)\s+(?:the\s+)?pregnancy\b": "gestational diabetes mellitus in pregnancy developing during pregnancy",
    r"\burinary\s+bilharzia\b": "urinary schistosomiasis bilharzia haematobium",
    r"\bhives\b": "urticaria hives wheals",
    r"\bperiapical\s+abscess\b": "periapical abscess without sinus",
    r"\bfrozen\s+shoulder\b": "adhesive capsulitis of shoulder frozen shoulder",
    r"\bplantar\s+fasciitis\b": "plantar fascial fibromatosis fasciitis",
    r"\b(?:cervical\s+)?lymph\s+node\s+infection\b": "acute lymphadenitis cervical lymph node",
    r"\basthma\s+attack\b": "asthma with acute exacerbation",
    r"\bexertional\s+angina\b": "stable angina pectoris",
    r"\bstable\s+pattern\b": "stable angina",
    r"\braised\s+fasting\s+sugar\b": "impaired fasting glucose",
    r"\bhigh\s+fasting\s+sugar\b": "impaired fasting glucose",
    r"\bifg\b": "impaired fasting glucose",
    r"\bpeanut\b": "peanut food allergen",
    r"\bgroundnut\b": "groundnut peanut food allergen",
    r"\boa\b": "osteoarthritis",
    r"\baf\b": "atrial fibrillation",
    r"\bgord\b": "gastro-oesophageal reflux disease GERD",
    r"\bpet\b": "preeclampsia pre-eclamptic toxaemia",
    r"\bsam\b": "severe acute malnutrition",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bPE\s*:(?!\s*chest\s*pain)": "Physical Examination:",
    r"\b(?:suspected\s+|r/o\s+)?pulmonary\s+embol(?:ism)?\b": "pulmonary embolism",
    r"\burti\b": "upper respiratory tract infection",
    r"\blrti\b": "lower respiratory tract infection",
    r"\brti\b": "respiratory tract infection",
    r"\buti\b": "urinary tract infection",
    r"\bge\b": "gastroenteritis",
    r"\bcap\b": "community acquired pneumonia",
    r"\bpud\b": "peptic ulcer disease",
    r"\bgerd\b": "gastroesophageal reflux disease",
    r"\bhtn\b": "hypertension",
    r"\bdm\b": "diabetes mellitus",
    r"\bt2dm\b": "type 2 diabetes mellitus",
    r"\bt1dm\b": "type 1 diabetes mellitus",
    r"\bptb\b": "pulmonary tuberculosis",
    r"\btb\b": "tuberculosis",
    r"\bckd\b": "chronic kidney disease",
    r"\baki\b": "acute kidney injury",
    r"\bcva\b": "cerebrovascular accident stroke",
    r"\btia\b": "transient ischaemic attack",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\bpid\b": "pelvic inflammatory disease",
    r"\bsti\b": "sexually transmitted infection",
    r"\bstd\b": "sexually transmitted disease",
    r"\brvd\b": "retroviral disease HIV",
    r"\bhiv\b": "human immunodeficiency virus",
    r"\botitis\s+m\b": "otitis media",
    r"\bom\b": "otitis media",
    r"\basom\b": "acute suppurative otitis media",
    r"\bcsom\b": "chronic suppurative otitis media",

    # Labs & Investigations
    r"\bbs\s+for\s+mps\b": "blood slide for malaria parasites",
    r"\bbs\s+no\s+mps\b": "blood slide no malaria parasites negative",
    r"\bmps\b": "malaria parasites",
    r"\bmrdt\b": "malaria rapid diagnostic test",
    r"\brdt\b": "rapid diagnostic test",
    r"\bcxr\b": "chest x-ray",
    r"\baxr\b": "abdominal x-ray",
    r"\buss\b": "ultrasound scan",
    r"\bu/s\b": "ultrasound",
    r"\bfbc\b": "full blood count",
    r"\bfhg\b": "full haemogram",
    r"\bcbc\b": "complete blood count",
    r"\brbs\b": "random blood sugar",
    r"\bfbs\b": "fasting blood sugar",
    r"\bupt\b": "urine pregnancy test",
    r"\brpr\b": "rapid plasma reagin syphilis test",
    r"\bvdrl\b": "venereal disease research laboratory test",
    r"\bcrp\b": "c-reactive protein",
    r"\besr\b": "erythrocyte sedimentation rate",
    r"\bvl\b": "viral load",

    # Anatomy / Examination
    r"\bo/e\b": "on examination",
    r"\blmp\b": "last menstrual period",
    r"\bwt\b": "weight",
    r"\bht\b": "height",
    r"\bhr\b": "heart rate",
    r"\bpr\b": "pulse rate",
    r"\bbp\b": "blood pressure",
    r"\brr\b": "respiratory rate",
    r"\bspo2\b": "oxygen saturation",
    r"\bon\s+ra\b": "on room air",
    r"\bb/l\b": "bilateral",
    r"\bl\s+lower\s+zone\b": "left lower zone",
    r"\br\s+lower\s+zone\b": "right lower zone",
    r"\bl\s+upper\s+zone\b": "left upper zone",
    r"\br\s+upper\s+zone\b": "right upper zone",
    r"\blif\b": "left iliac fossa",
    r"\brif\b": "right iliac fossa",
    r"\bruq\b": "right upper quadrant",
    r"\bluq\b": "left upper quadrant",

    # Medications / Local context
    r"\bcoartem\b": "artemether lumefantrine Coartem",
    r"\btld\b": "tenofovir lamivudine dolutegravir",
    r"\brhze\b": "rifampicin isoniazid pyrazinamide ethambutol anti-TB",
    r"\borange\s+peel\b": "peau d'orange",
    r"\bjiko\b": "charcoal stove",
    r"\bboda\s+rider\b": "motorcycle taxi rider",
    r"\bjiggers\b": "tungiasis sand flea jiggers",
    r"\bbilharzia\b": "schistosomiasis bilharzia",
}

# Compiled regexes for efficiency
COMPILED_ABBREVIATIONS = [
    (re.compile(pat, re.IGNORECASE), repl)
    for pat, repl in CLINICAL_ABBREVIATIONS.items()
]

# Clinical duration patterns e.g. x3/7 -> for 3 days, 2/52 -> 2 weeks, 3/12 -> 3 months
RE_DAYS = re.compile(r"\bx?(\d+)/7\b", re.IGNORECASE)
RE_WEEKS = re.compile(r"\bx?(\d+)/52\b", re.IGNORECASE)
RE_MONTHS = re.compile(r"\bx?(\d+)/12\b", re.IGNORECASE)


def expand_durations(text: str) -> str:
    """Expand duration notation like x3/7 into 'for 3 days'."""
    text = RE_DAYS.sub(r"for \1 days", text)
    text = RE_WEEKS.sub(r"for \1 weeks", text)
    text = RE_MONTHS.sub(r"for \1 months", text)
    return text


def expand_abbreviations(text: str) -> str:
    """Expand known clinical abbreviations in text."""
    text = expand_durations(text)
    for pattern, repl in COMPILED_ABBREVIATIONS:
        text = pattern.sub(repl, text)
    return text


# Tokenization regex for BM25
RE_WORD = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z]+)?")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words/numbers."""
    if not text:
        return []
    return [w.lower() for w in RE_WORD.findall(text)]


def clean_text(text: str) -> str:
    """Basic clean up of whitespace and control characters."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()
