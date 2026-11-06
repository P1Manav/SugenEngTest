"""Multilingual support: fast deterministic language identification and clinical term translation.
Handles outpatient consults in Swahili (East African clinic context), Spanish, and French.
No external network calls, 100% offline and deterministic.
"""

import re
from typing import Tuple, Dict, List

# Stopwords and characteristic function words
SWAHILI_MARKERS = {
    "mgonjwa", "daktari", "homa", "maumivu", "tumbo", "kichwa", "asante", "kuhara",
    "kikohozi", "damu", "mzee", "karibu", "yake", "kwa", "katika", "sana", "dawa",
    "kifua", "mkojo", "kuwashwa", "vipele", "siku", "miezi", "mtoto", "mama",
    "shida", "meno", "macho", "sikio", "mgongo", "mguu", "mkono", "hali"
}

SPANISH_MARKERS = {
    "paciente", "dolor", "fiebre", "agudo", "cronico", "cefalea", "anos", "presenta",
    "refiere", "desde", "hace", "dias", "semanas", "meses", "tratamiento", "examen",
    "presion", "arterial", "frecuencia", "cardiaca", "auscultacion", "abdomen", "mujer"
}

FRENCH_MARKERS = {
    "patient", "patiente", "douleur", "fievre", "aigu", "chronique", "cephalee", "ans",
    "presente", "depuis", "jours", "semaines", "mois", "traitement", "examen", "tension",
    "arterielle", "frequence", "cardiaque", "auscultation", "abdomen", "homme", "avec"
}

# Clinical translations dictionary mapping (lang -> (pattern, english_term))
SWAHILI_CLINICAL_MAP = [
    (r"\b(falciparum)\b", "Plasmodium falciparum"),
    (r"\b(mrdt\s+chanya|malaria\s+mrdt\s+chanya)\b", "malaria falciparum rapid diagnostic test positive confirmed"),
    (r"\b(chanya)\b", "positive confirmed"),
    (r"\b(hasi)\b", "negative excluded"),
    (r"\b(homa\s+ya\s+matumbo|homa\s+ya\s+typhoid)\b", "typhoid fever"),
    (r"\b(homa\s+ya\s+mapafu)\b", "pneumonia"),
    (r"\b(homa\s+ya\s+manjano)\b", "jaundice hepatitis"),
    (r"\b(kifua\s+kikuu|tb)\b", "tuberculosis"),
    (r"\b(tetekuwanga)\b", "varicella chickenpox"),
    (r"\b(shurua)\b", "measles"),
    (r"\b(machapwe)\b", "mumps parotitis"),
    (r"\b(malaria)\b", "malaria"),
    (r"\b(shinikizo\s+la\s+damu|presha)\b", "hypertension high blood pressure"),
    (r"\b(ugonjwa\s+wa\s+sukari|sukari)\b", "diabetes mellitus"),
    (r"\b(sukari\s+ya\s+ujauzito)\b", "gestational diabetes mellitus"),
    (r"\b(kupumua\s+kwa\s+shida|shida\s+ya\s+kupumua)\b", "dyspnea shortness of breath asthma"),
    (r"\b(maumivu\s+ya\s+kifua)\b", "chest pain"),
    (r"\b(maumivu\s+ya\s+kichwa)\b", "headache cephalea"),
    (r"\b(maumivu\s+ya\s+tumbo)\b", "abdominal pain"),
    (r"\b(kuhara\s+damu|kuhara\s+kwa\s+damu)\b", "amoebic dysentery bloody diarrhea"),
    (r"\b(kuhara)\b", "diarrhea gastroenteritis"),
    (r"\b(kutapika)\b", "vomiting"),
    (r"\b(kikohozi)\b", "cough bronchitis"),
    (r"\b(homa)\b", "fever pyrexia"),
    (r"\b(kuwashwa)\b", "pruritus itching"),
    (r"\b(vipele)\b", "rash skin eruption"),
    (r"\b(mkojo\s+kuchoma|maumivu\s+ya\s+mkojo)\b", "dysuria urinary tract infection UTI"),
    (r"\b(maumivu\s+ya\s+macho)\b", "eye pain conjunctivitis"),
    (r"\b(maumivu\s+ya\s+sikio)\b", "ear pain otitis media"),
    (r"\b(maumivu\s+ya\s+meno)\b", "toothache dental periapical abscess"),
    (r"\b(maumivu\s+ya\s+magoti|maumivu\s+ya\s+viungo)\b", "osteoarthritis knee pain arthritis"),
    (r"\b(vidonda\s+vya\s+tumbo)\b", "peptic ulcer disease"),
    (r"\b(asidi|kiungulia)\b", "gastro-oesophageal reflux disease GERD"),
]

SPANISH_CLINICAL_MAP = [
    (r"\b(malaria|paludismo)\b", "malaria"),
    (r"\b(infeccion\s+(?:de\s+)?tracto\s+urinario|itu|infeccion\s+urinaria)\b", "urinary tract infection"),
    (r"\b(candidiasis\s+vulvovaginal|candidosis\s+vaginal)\b", "vulvovaginal candidosis"),
    (r"\b(faringoamigdalitis|amigdalitis\s+aguda)\b", "acute tonsillitis"),
    (r"\b(neumonia\s+adquirida\s+en\s+la\s+comunidad|neumonia\s+aguda|neumonia)\b", "acute pneumonia"),
    (r"\b(bronquitis\s+aguda)\b", "acute bronchitis"),
    (r"\b(gastroenteritis\s+aguda)\b", "infectious gastroenteritis"),
    (r"\b(varicela)\b", "varicella chickenpox without complication"),
    (r"\b(parotiditis|paperas)\b", "mumps parotitis without complication"),
    (r"\b(sarampion)\b", "measles without complication"),
    (r"\b(enfermedad\s+por\s+reflujo\s+gastroesofagico|erge|reflujo)\b", "gastro-oesophageal reflux disease GERD"),
    (r"\b(hipertension\s+arterial|hta)\b", "essential hypertension"),
    (r"\b(diabetes\s+mellitus\s+tipo\s+2|dm2)\b", "type 2 diabetes mellitus"),
    (r"\b(diabetes\s+gestacional)\b", "gestational diabetes mellitus"),
    (r"\b(artrosis\s+de\s+rodilla|gonartrosis)\b", "osteoarthritis of knee"),
    (r"\b(asma\s+bronquial|crisis\s+asmatica|asma)\b", "asthma with exacerbation"),
    (r"\b(insuficiencia\s+cardiaca\s+congestiva|icc)\b", "congestive heart failure"),
    (r"\b(dermatitis\s+alergica\s+de\s+contacto)\b", "allergic contact dermatitis"),
    (r"\b(otitis\s+media\s+aguda)\b", "acute suppurative otitis media"),
    (r"\b(disenteria\s+amebiana)\b", "acute amoebic dysentery amoebiasis"),
    (r"\b(fiebre\s+tifoidea)\b", "typhoid fever"),
    (r"\b(ulcus\s+peptico|ulcera\s+peptica)\b", "peptic ulcer disease"),
    (r"\b(conjuntivitis\s+aguda)\b", "acute conjunctivitis"),
    (r"\b(urticaria\s+aguda|urticaria)\b", "acute urticaria hives"),
    (r"\b(tuberculosis\s+pulmonar|tbc)\b", "pulmonary tuberculosis"),
]

FRENCH_CLINICAL_MAP = [
    (r"\b(paludisme)\b", "malaria falciparum"),
    (r"\b(infection\s+urinaire)\b", "urinary tract infection"),
    (r"\b(candidose\s+vulvovaginale|mycose\s+vaginale)\b", "vulvovaginal candidosis"),
    (r"\b(angine\s+aigue|amygdalite\s+aigue)\b", "acute tonsillitis"),
    (r"\b(pneumonie\s+aigue|pneumonie)\b", "acute pneumonia"),
    (r"\b(bronchite\s+aigue)\b", "acute bronchitis"),
    (r"\b(gastroenterite\s+aigue)\b", "infectious gastroenteritis"),
    (r"\b(varicelle)\b", "varicella chickenpox without complication"),
    (r"\b(oreillons)\b", "mumps parotitis without complication"),
    (r"\b(rougeole)\b", "measles without complication"),
    (r"\b(reflux\s+gastro-oesophagien|rgo)\b", "gastro-oesophageal reflux disease GERD"),
    (r"\b(hypertension\s+arterielle|hta)\b", "essential hypertension"),
    (r"\b(diabete\s+gestationnel)\b", "gestational diabetes mellitus"),
    (r"\b(arthrose\s+du\s+genou|gonarthrose)\b", "osteoarthritis of knee"),
    (r"\b(crise\s+d\'asthme|asthme)\b", "asthma with exacerbation"),
    (r"\b(insuffisance\s+cardiaque\s+congestive)\b", "congestive heart failure"),
    (r"\b(dermatite\s+allergique\s+de\s+contact)\b", "allergic contact dermatitis"),
    (r"\b(otite\s+moyenne\s+aigue)\b", "acute suppurative otitis media"),
    (r"\b(dysenterie\s+amibienne)\b", "acute amoebic dysentery amoebiasis"),
    (r"\b(fievre\s+typhoide)\b", "typhoid fever"),
    (r"\b(ulcere\s+peptique|ulcere\s+gastrique)\b", "peptic ulcer disease"),
    (r"\b(conjonctivite\s+aigue)\b", "acute conjunctivitis"),
    (r"\b(urticaire)\b", "acute urticaria hives"),
    (r"\b(tuberculose\s+pulmonaire)\b", "pulmonary tuberculosis"),
]


def detect_language(text: str) -> str:
    """Detect text language using marker word frequency.
    Returns: 'en', 'sw', 'es', 'fr'. Defaults to 'en'.
    """
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    if not words:
        return "en"

    sw_hits = len(words.intersection(SWAHILI_MARKERS))
    es_hits = len(words.intersection(SPANISH_MARKERS))
    fr_hits = len(words.intersection(FRENCH_MARKERS))

    scores = [("sw", sw_hits), ("es", es_hits), ("fr", fr_hits)]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_lang, best_hits = scores[0]
    if best_hits >= 2:
        return best_lang
    return "en"


def translate_clinical_text(text: str, source_lang: str) -> str:
    """Translate clinical terminology from source_lang into English.
    Returns expanded English translation preserving unmapped tokens.
    """
    if source_lang == "en":
        return text

    translated = text
    term_map: List[Tuple[str, str]] = []

    if source_lang == "sw":
        term_map = SWAHILI_CLINICAL_MAP
    elif source_lang == "es":
        term_map = SPANISH_CLINICAL_MAP
    elif source_lang == "fr":
        term_map = FRENCH_CLINICAL_MAP

    for pattern, replacement in term_map:
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

    return translated
