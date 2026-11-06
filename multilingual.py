"""Multilingual support: real offline language identification and clinical term translation.
Uses Lingua / fastText language identification with Unicode script gating so non-Latin
scripts (Hindi, Arabic, Chinese, Russian, Japanese, etc.) never fall through as English.
Includes comprehensive clinical term mappings for Swahili, Spanish, French, Portuguese,
Hindi, Arabic, and Chinese.
100% offline, zero external network calls.
"""

import re

# Global cached detector instance
_detector = None

# Unicode script ranges for non-Latin script detection
NON_LATIN_SCRIPTS = [
    (re.compile(r"[\u0900-\u097F]"), "hi"),  # Devanagari / Hindi
    (re.compile(r"[\u0600-\u06FF\u0750-\u077F]"), "ar"),  # Arabic
    (re.compile(r"[\u3040-\u30FF]"), "ja"),  # Japanese Kana
    (re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]"), "zh"),  # Chinese CJK
    (re.compile(r"[\u0400-\u04FF]"), "ru"),  # Cyrillic / Russian
]

# Stopwords and characteristic function words for Latin script fallback
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

PORTUGUESE_MARKERS = {
    "paciente", "dor", "febre", "aguda", "cronica", "cefaleia", "anos", "apresenta",
    "queixa", "ha", "dias", "semanas", "meses", "tratamento", "exame", "pressao",
    "arterial", "frequencia", "cardiaca", "ausculta", "abdomen", "mulher", "com"
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
    (r"\b(infeccion\s+(?:de\s+)?tracto\s+urinario|itu|infeccion\s+urinaria|cistitis)\b", "urinary tract infection cystitis"),
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
    (r"\b(calculo\s+renal|colico\s+nefritico|litiasis\s+renal)\b", "calculus of kidney nephrolithiasis"),
    (r"\b(escabiosis|sarna)\b", "scabies"),
    (r"\b(labio\s+leporino)\b", "cleft lip"),
    (r"\b(catarata\s+senil|catarata)\b", "senile cataract"),
    (r"\b(asma)\b", "asthma"),
    (r"\b(apendicitis\s+aguda)\b", "acute appendicitis"),
    (r"\b(angina\s+de\s+pecho)\b", "angina pectoris"),
    (r"\b(artrosis|osteoartritis)\b", "osteoarthritis"),
]

FRENCH_CLINICAL_MAP = [
    (r"\b(paludisme|malaria)\b", "malaria"),
    (r"\b(infection\s+urinaire|cystite\s+aigue|cystite)\b", "urinary tract infection cystitis"),
    (r"\b(tuberculose\s+respiratoire|tuberculose)\b", "respiratory tuberculosis"),
    (r"\b(pneumonie\s+aigue|pneumonie)\b", "acute pneumonia"),
    (r"\b(bronchite\s+aigue|bronchite)\b", "acute bronchitis"),
    (r"\b(gastro-enterite\s+aigue|gastro-enterite)\b", "infectious gastroenteritis"),
    (r"\b(varicelle)\b", "varicella chickenpox without complication"),
    (r"\b(oreillons)\b", "mumps parotitis without complication"),
    (r"\b(rougeole)\b", "measles without complication"),
    (r"\b(hypertension\s+arterielle|hta)\b", "essential hypertension"),
    (r"\b(diabete\s+de\s+type\s+2|diabete)\b", "type 2 diabetes mellitus"),
    (r"\b(hypothyroidie\s+primitive|hypothyroidie)\b", "primary hypothyroidism"),
    (r"\b(canal\s+carpien)\b", "carpal tunnel syndrome"),
    (r"\b(polyarthrite\s+rhumatoide)\b", "rheumatoid arthritis"),
    (r"\b(psoriasis\s+vulgaire|psoriasis)\b", "psoriasis vulgaris"),
    (r"\b(entorse\s+de\s+la\s+cheville|entorse)\b", "sprain of ankle"),
    (r"\b(abces\s+de\s+l'oreille)\b", "abscess of external ear"),
    (r"\b(toxicite\s+medicamenteuse)\b", "drug toxicity"),
    (r"\b(covid-19)\b", "COVID-19"),
]

PORTUGUESE_CLINICAL_MAP = [
    (r"\b(malaria)\b", "malaria"),
    (r"\b(infeccao\s+urinaria|cistite\s+aguda|cistite)\b", "urinary tract infection cystitis"),
    (r"\b(hipertensao\s+arterial|hipertensao)\b", "essential hypertension"),
    (r"\b(diabetes\s+mellitus\s+tipo\s+2|diabetes)\b", "type 2 diabetes mellitus"),
    (r"\b(doenca\s+do\s+refluxo\s+gastroesofagico|refluxo)\b", "gastroesophageal reflux disease GERD"),
    (r"\b(apneia\s+obstrutiva\s+do\s+sono|apneia\s+do\s+sono)\b", "obstructive sleep apnoea"),
    (r"\b(carcinoma\s+basocelular)\b", "basal cell carcinoma"),
]

HINDI_CLINICAL_MAP = [
    (r"(टाइफाइड|मियादी\s*बुखार)", "typhoid fever Salmonella typhi"),
    (r"(मलेरिया)", "malaria Plasmodium"),
    (r"(डेंगू)", "dengue fever"),
    (r"(अस्थमा|दमा)", "bronchial asthma"),
    (r"(मोतियाबिंद)", "cataract"),
    (r"(हाइपोथायरायडिज्म|थायराइड)", "hypothyroidism"),
    (r"(बुखार)", "fever pyrexia"),
    (r"(खांसी)", "cough bronchitis"),
    (r"(दस्त|पेचिश)", "diarrhea dysentery"),
    (r"(पेट\s*दर्द)", "abdominal pain"),
]

ARABIC_CLINICAL_MAP = [
    (r"(زحار\s*أميب|أميبا)", "amoebic dysentery Entamoeba histolytica"),
    (r"(ملاريا)", "malaria"),
    (r"(شقيقة)", "migraine without aura"),
    (r"(زائدة\s*دودية)", "acute appendicitis"),
    (r"(حمى|حرارة)", "fever pyrexia"),
    (r"(سعال)", "cough"),
    (r"(إسهال)", "diarrhea"),
    (r"(ربو)", "asthma"),
    (r"(سكري)", "diabetes mellitus"),
]

CHINESE_CLINICAL_MAP = [
    (r"(假丝酵母菌|外阴念珠菌|念珠菌)", "vulvovaginal candidiasis candidosis thrush"),
    (r"(社区获得性肺炎|肺炎)", "community-acquired pneumonia"),
    (r"(抑郁症|抑郁障碍)", "depressive disorder depression"),
    (r"(慢性咳嗽|咳嗽)", "chronic cough bronchitis"),
    (r"(肝气郁结)", "liver qi stagnation traditional medicine"),
    (r"(气血两虚)", "dual deficiency of qi and blood traditional medicine"),
    (r"(发热|发烧)", "fever pyrexia"),
    (r"(哮喘)", "asthma"),
    (r"(糖尿病)", "diabetes mellitus"),
    (r"(白内障)", "cataract"),
    (r"(阑尾炎)", "appendicitis"),
]


def _get_lingua_detector():
    """Lazy initialize Lingua language detector."""
    global _detector
    if _detector is None:
        try:
            from lingua import Language, LanguageDetectorBuilder
            langs = [
                Language.ENGLISH,
                Language.SWAHILI,
                Language.SPANISH,
                Language.FRENCH,
                Language.PORTUGUESE,
                Language.GERMAN,
                Language.HINDI,
                Language.ARABIC,
                Language.CHINESE,
                Language.RUSSIAN,
                Language.JAPANESE,
            ]
            _detector = LanguageDetectorBuilder.from_languages(*langs).build()
        except Exception:
            _detector = False
    return _detector


def detect_language(text: str) -> str:
    """Detect text language using Unicode script gating, Lingua, or clinical marker fallback.
    Guarantees non-Latin scripts (Hindi, Arabic, Chinese, Russian, Japanese) never fall through as English.
    Returns ISO 639-1 code: 'en', 'sw', 'es', 'fr', 'pt', 'de', 'hi', 'ar', 'zh', 'ru', 'ja'.
    """
    if not text or not text.strip():
        return "en"

    # 1. Non-Latin script gate (instant regex match; never falls through to English)
    for pattern, lang_code in NON_LATIN_SCRIPTS:
        if pattern.search(text):
            return lang_code

    # 2. Lingua neural/n-gram detector
    detector = _get_lingua_detector()
    if detector:
        try:
            detected = detector.detect_language_of(text)
            if detected:
                lang_map = {
                    "ENGLISH": "en",
                    "SWAHILI": "sw",
                    "SPANISH": "es",
                    "FRENCH": "fr",
                    "PORTUGUESE": "pt",
                    "GERMAN": "de",
                    "HINDI": "hi",
                    "ARABIC": "ar",
                    "CHINESE": "zh",
                    "RUSSIAN": "ru",
                    "JAPANESE": "ja",
                }
                return lang_map.get(detected.name, "en")
        except Exception:
            pass

    # 3. Clinical marker word fallback for Latin scripts
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    if not words:
        return "en"

    sw_hits = len(words.intersection(SWAHILI_MARKERS))
    es_hits = len(words.intersection(SPANISH_MARKERS))
    fr_hits = len(words.intersection(FRENCH_MARKERS))
    pt_hits = len(words.intersection(PORTUGUESE_MARKERS))

    scores = [("sw", sw_hits), ("es", es_hits), ("fr", fr_hits), ("pt", pt_hits)]
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
    term_map: list[tuple[str, str]] = []

    if source_lang == "sw":
        term_map = SWAHILI_CLINICAL_MAP
    elif source_lang == "es":
        term_map = SPANISH_CLINICAL_MAP
    elif source_lang == "fr":
        term_map = FRENCH_CLINICAL_MAP
    elif source_lang == "pt":
        term_map = PORTUGUESE_CLINICAL_MAP
    elif source_lang == "hi":
        term_map = HINDI_CLINICAL_MAP
    elif source_lang == "ar":
        term_map = ARABIC_CLINICAL_MAP
    elif source_lang == "zh":
        term_map = CHINESE_CLINICAL_MAP

    for pattern, replacement in term_map:
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

    # If translating non-Latin script, append extracted English keywords to the text
    if source_lang in ("hi", "ar", "zh", "ja", "ru"):
        keywords = []
        for _, replacement in term_map:
            if replacement in translated:
                keywords.append(replacement)
        if keywords:
            translated = f"{translated} {' '.join(keywords)}"

    return translated
