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
    "presion", "arterial", "frecuencia", "cardiaca", "auscultacion", "mujer", "con",
    "para", "por", "una", "los", "las", "del"
}

FRENCH_MARKERS = {
    "patiente", "douleur", "fievre", "aigu", "chronique", "cephalee", "ans",
    "presente", "depuis", "jours", "semaines", "mois", "traitement", "arterielle",
    "frequence", "cardiaque", "auscultation", "homme", "avec", "pour", "sur",
    "dans", "par", "une", "des", "les", "consulte", "prurit", "signe", "chez"
}

PORTUGUESE_MARKERS = {
    "paciente", "dor", "febre", "aguda", "cronica", "cefaleia", "anos", "apresenta",
    "queixa", "ha", "dias", "semanas", "meses", "tratamento", "exame", "pressao",
    "arterial", "frequencia", "cardiaca", "ausculta", "mulher", "com", "para",
    "intensa", "sexo", "masculino", "feminino", "uma", "dos", "das", "pelo", "pela"
}

ENGLISH_MARKERS = {
    "the", "with", "and", "for", "from", "complains", "history", "denies",
    "assessment", "impression", "examination", "findings", "pain", "headache",
    "fever", "cough", "years", "year", "male", "female", "today", "normal",
    "report", "clinical", "diagnosed", "prescribed", "days", "weeks", "months"
}

# Clinical translations dictionary mapping (lang -> (pattern, english_term))
SWAHILI_CLINICAL_MAP = [
    (r"\b(falciparum)\b", "Plasmodium falciparum"),
    (r"\b(bronkitisi(?:\s+kali)?)\b", "acute bronchitis"),
    (r"\b(uchunguzi\s+wa\s+afya\s+wa\s+mtoto(?:\s+mchanga)?)\b", "routine infant and child health check-up immunization"),
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
    (r"\b(seli\s+mundu|anemia\s+ya\s+seli\s+mundu)\b", "sickle cell disease crisis"),
    (r"\b(kisukari\s+cha\s+aina\s+ya\s+kwanza|kisukari\s+type\s*1)\b", "type 1 diabetes mellitus"),
    (r"\b(kisukari\s+cha\s+aina\s+ya\s+pili|kisukari\s+type\s*2)\b", "type 2 diabetes mellitus"),
    (r"\b(kifafa)\b", "epilepsy seizure"),
    (r"\b(utapiamlo|utapiamlo\s+mkali)\b", "severe acute malnutrition"),
    (r"\b(saratani\s+ya\s+matiti)\b", "malignant neoplasm of breast"),
    (r"\b(saratani\s+ya\s+shingo\s+ya\s+kizazi)\b", "cervical cancer neoplasm of cervix"),
    (r"\b(fangasi\s+ya\s+ubongo|meningitis)\b", "meningitis"),
    (r"\b(shinikizo\s+la\s+damu\s+na\s+sukari\s+ni\s+ya\s+kawaida|shinikizo\s+la\s+damu\s+ni\s+ya\s+kawaida)\b", "blood pressure and blood sugar normal within normal limits"),
    (r"\b(shinikizo\s+la\s+damu\s+la\s+juu|shinikizo\s+la\s+damu)\b", "essential hypertension"),
    (r"\b(chanjo\s+za\s+kawaida\s+za\s+utotoni|chanjo\s+ya\s+surua\s+na\s+rubella|chanjo|chanjo\s+za\s+kawaida|chanjo\s+za\s+utotoni)\b", "need for immunization against measles routine childhood vaccination"),
    (r"\b(vasovagal\s+syncope|kuzimia\s+kwa\s+kawaida|kuzimia|kuzimia\s+ghafla)\b", "vasovagal syncope and collapse"),
    (r"\b(mgogoro|homa\s+ya\s+seli\s+mundu\s+yenye\s+mgogoro)\b", "sickle cell anaemia with crisis"),
    (r"\b(kidonda\s+cha\s+tumbo|vidonda\s+vya\s+tumbo)\b", "peptic ulcer disease without haemorrhage or perforation"),
    (r"\b(tetemeko|kutembea\s+kwa\s+shida)\b", "parkinson disease resting tremor"),
]

SPANISH_CLINICAL_MAP = [
    (r"\b(malaria|paludismo)\b", "malaria"),
    (r"\b(cistitis\s+infecciosa(?:\s+aguda)?)\b", "acute infectious cystitis"),
    (r"\b(infeccion\s+(?:de\s+)?tracto\s+urinario|itu|infeccion\s+urinaria|cistitis)\b", "urinary tract infection cystitis"),
    (r"\binfeccios[oa]s?\b", "infectious"),
    (r"\bagud[oa]s?\b", "acute"),
    (r"\bcr[oó]nic[oa]s?\b", "chronic"),
    (r"\bal[eé]rgic[oa]s?\b", "allergic"),
    (r"\b(candidiasis\s+vulvovaginal|candidosis\s+vaginal)\b", "vulvovaginal candidosis"),
    (r"\b(c[aá]ncer\s+de\s+colon|adenocarcinoma\s+de\s+colon|neoplasia\s+maligna\s+de\s+colon|tumor\s+de\s+colon)\b", "malignant neoplasm of colon adenocarcinoma"),
    (r"\b(artritis\s+reumatoide|artritis\s+reumatoidea)\b", "rheumatoid arthritis seropositive"),
    (r"\b(dengue|fiebre\s+dengue)\b", "dengue fever without warning signs"),
    (r"\b(preeclampsia|pre-eclampsia)\b", "pre-eclampsia"),
    (r"\b(disfunci[oó]n\s+er[eé]ctil)\b", "male erectile dysfunction"),
    (r"\b(fractura\s+de\s+radio|fractura\s+del\s+radio|fractura\s+de\s+la\s+muñeca)\b", "fracture of radius lower end"),
    (r"\b(otitis\s+media\s+aguda|otitis\s+media)\b", "acute otitis media"),
    (r"\b(faringoamigdalitis|amigdalitis\s+aguda)\b", "acute tonsillitis"),
    (r"\b(faringitis\s+aguda|faringitis)\b", "acute pharyngitis"),
    (r"\b(rinitis\s+al[eé]rgica)\b", "allergic rhinitis"),
    (r"\b(acn[eé]\s+comedoniano|acn[eé])\b", "comedonal acne vulgaris"),
    (r"\b(enfermedad\s+de\s+parkinson|parkinson)\b", "sporadic parkinson disease"),
    (r"\b(s[ií]ndrome\s+de\s+colon\s+irritable|colon\s+irritable)\b", "irritable bowel syndrome"),
    (r"\b(epilepsia|crisis\s+epil[eé]pticas)\b", "epilepsy seizure"),
    (r"\b(tiña|dermatofitosis)\b", "tinea dermatophytosis"),
    (r"\b(gota\s+aguda|gota)\b", "gout acute gouty arthritis"),
    (r"\b(insuficiencia\s+venosa\s+cr[oó]nica)\b", "chronic venous insufficiency"),
    (r"\b(lumbalgia|lumbago)\b", "low back pain lumbago"),
    (r"\b(confirmad[oa]|bacteriol[oó]gicamente\s+confirmad[oa]|prueba\s+positiva)\b", "confirmed bacteriologically"),
    (r"\b(sin\s+complicaciones|no\s+complicad[oa])\b", "without complication uncomplicated"),
    (r"\b(sin\s+signos\s+de\s+alarma)\b", "without warning signs"),
    (r"\b(sin\s+peritonitis)\b", "without peritonitis"),
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
    (r"\b(litiasis\s+ureteral|calculo\s+ureteral|c[aá]lculo\s+ureteral|colico\s+nefr[ií]tico|c[oó]lico\s+renal)\b", "calculus of ureter"),
    (r"\b(calculo\s+renal|litiasis\s+renal)\b", "calculus of kidney nephrolithiasis"),
    (r"\b(no\s+erosiva|sin\s+erosiones)\b", "non-erosive"),
    (r"\b(preeclampsia\s+sin\s+criterios\s+de\s+gravedad)\b", "preeclampsia without severe features"),
    (r"\b(escabiosis|sarna)\b", "scabies"),
    (r"\b(labio\s+leporino)\b", "cleft lip"),
    (r"\b(catarata\s+senil|catarata)\b", "senile cataract"),
    (r"\b(asma)\b", "asthma"),
    (r"\b(apendicitis\s+aguda)\b", "acute appendicitis"),
    (r"\b(angina\s+de\s+pecho)\b", "angina pectoris"),
    (r"\b(artrosis|osteoartritis)\b", "osteoarthritis"),
    (r"\b(c[aá]ncer\s+de\s+mama|carcinoma\s+ductal\s+invasivo|carcinoma\s+ductal)\b", "malignant neoplasm of breast carcinoma"),
    (r"\b(carcinoma\s+basocelular)\b", "basal cell carcinoma"),
    (r"\b(carcinoma\s+espinocelular|carcinoma\s+epidermoide)\b", "squamous cell carcinoma"),
    (r"\b(lupus\s+eritematoso\s+sist[eé]mico|les)\b", "systemic lupus erythematosus"),
    (r"\b(ansiedad\s+generalizada|trastorno\s+de\s+ansiedad)\b", "generalized anxiety disorder"),
    (r"\b(crisis\s+drepanoc[ií]tica|anemia\s+falciforme)\b", "sickle cell disease crisis"),
    (r"\b(p[uú]rpura\s+trombocitop[eé]nica)\b", "immune thrombocytopenia purpura"),
    (r"\b(cetoacidosis\s+diab[eé]tica)\b", "diabetic ketoacidosis"),
    (r"\b(hipotiroidismo\s+primario|hipotiroidismo)\b", "primary hypothyroidism"),
    (r"\b(insuficiencia\s+card[ií]aca)\b", "heart failure"),
    (r"\b(infarto\s+agudo\s+de\s+miocardio|iam)\b", "acute myocardial infarction"),
    (r"\b(accidente\s+cerebrovascular|ictus|acv)\b", "stroke cerebrovascular accident"),
    (r"\b(esquizofrenia)\b", "schizophrenia"),
    (r"\b(depresi[oó]n|trastorno\s+depresivo)\b", "depressive episode depression"),
    (r"\b(dermatitis\s+at[oó]pica|eccema)\b", "atopic dermatitis eczema"),
    (r"\b(candidiasis\s+oral|muguet)\b", "oral candidiasis thrush"),
    (r"\b(glaucoma\s+de\s+[aá]ngulo\s+abierto|glaucoma)\b", "glaucoma"),
]

FRENCH_CLINICAL_MAP = [
    (r"\b(bpco|broncho-pneumopathie\s+chronique\s+obstructive)\b", "chronic obstructive pulmonary disease COPD"),
    (r"\b(hypertrophie\s+b[eé]nigne\s+de\s+la\s+prostate|hbp)\b", "hyperplasia of prostate benign prostatic"),
    (r"\b(ict[eè]re\s+physiologique(?:\s+du\s+nouveau-n[eé])?)\b", "neonatal jaundice physiological"),
    (r"\b(toux\s+chronique\s+inexpliqu[eé]e)\b", "unexplained chronic cough"),
    (r"\b(chute\s+d'une\s+[eé]chelle)\b", "fall from ladder"),
    (r"\b(syndrome\s+de\s+gougerot-sj[oö]gren|syndrome\s+sec)\b", "Sjögren syndrome primary"),
    (r"\b(carcinome\s+canalaire\s+infiltrant)\b", "invasive ductal carcinoma of breast"),
    (r"\b(carcinome\s+infiltrant(?:\s+du\s+sein)?)\b", "invasive carcinoma of breast"),
    (r"\b(paludisme|malaria)\b", "malaria"),
    (r"\b(cystite\s+infectieuse(?:\s+aigu[eë])?)\b", "acute infectious cystitis"),
    (r"\b(infection\s+urinaire|cystite\s+aigue|cystite\s+aigu[eë]|cystite(?:\s+infectieuse)?)\b", "infectious cystitis urinary tract infection"),
    (r"\binfectieuse?s?\b", "infectious"),
    (r"\baigu[eë]?s?\b", "acute"),
    (r"\bchroniques?\b", "chronic"),
    (r"\ballergiques?\b", "allergic"),
    (r"\b(tuberculose\s+respiratoire|tuberculose\s+pulmonaire|tuberculose)\b", "respiratory tuberculosis pulmonary"),
    (r"\b(pneumonie\s+aigue|pneumonie\s+aigu[eë]|pneumonie)\b", "acute pneumonia"),
    (r"\b(bronchite\s+aigue|bronchite\s+aigu[eë]|bronchite)\b", "acute bronchitis"),
    (r"\b(gastro-enterite\s+aigue|gastro-enterite\s+aigu[eë]|gastro-enterite)\b", "infectious gastroenteritis"),
    (r"\b(appendicite\s+aigue|appendicite\s+aigu[eë]|appendicite)\b", "acute appendicitis without peritonitis"),
    (r"\b(polyarthrite\s+rhumato[iï]de(?:\s+s[eé]ropositive)?)\b", "rheumatoid arthritis seropositive"),
    (r"\b(cancer\s+du\s+colon|ad[eé]nocarcinome\s+du\s+colon|tumeur\s+du\s+colon)\b", "malignant neoplasm of colon adenocarcinoma"),
    (r"\b(effet\s+ind[eé]sirable\s+m[eé]dicamenteux|toxicit[eé]\s+m[eé]dicamenteuse)\b", "adverse effect of drug toxicity"),
    (r"\b(varicelle)\b", "varicella chickenpox without complication"),
    (r"\b(oreillons)\b", "mumps parotitis without complication"),
    (r"\b(rougeole)\b", "measles without complication"),
    (r"\b(hypertension\s+arterielle|hta)\b", "essential hypertension"),
    (r"\b(diabete\s+de\s+type\s+2|diabete)\b", "type 2 diabetes mellitus"),
    (r"\b(hypothyro[iï]die(?:\s+(?:primitive|primaire|acquise))?|hypothyroidie(?:\s+(?:primitive|primaire|acquise))?)\b", "primary hypothyroidism"),
    (r"\b(dyspareunie)\b", "dyspareunia"),
    (r"\b(h[eé]matome\s+du\s+pavillon|oth[eé]matome)\b", "haematoma of pinna auricle"),
    (r"\b(nouveau-n[eé]\s+de\s+m[eè]re|f[oœ]tus\s+ou\s+nouveau-n[eé]\s+affect[eé])\b", "fetus or newborn affected by maternal condition"),
    (r"\b(canal\s+carpien)\b", "carpal tunnel syndrome"),
    (r"\b(ecz[eé]ma\s+atopique|dermatite\s+atopique)\b", "atopic eczema dermatitis"),
    (r"\b(angine\s+de\s+poitrine|angor\s+stable|angor)\b", "stable angina pectoris"),
    (r"\b(reflux\s+gastro-[ooeé]sophagien|rgo)\b", "gastro-oesophageal reflux disease GERD non-erosive"),
    (r"\b(cataracte\s+s[eé]nile|cataracte)\b", "age-related cataract senile"),
    (r"\b(conjonctivite\s+aigu[eë]|conjonctivite)\b", "acute conjunctivitis"),
    (r"\b(otite\s+moyenne\s+aigu[eë]|otite\s+moyenne)\b", "acute otitis media"),
    (r"\b(rhinite\s+allergique)\b", "allergic rhinitis"),
    (r"\b(pharyngite\s+aigu[eë]|pharyngite)\b", "acute pharyngitis"),
    (r"\b(arthrose\s+du\s+genou|gonarthrose)\b", "osteoarthritis of knee"),
    (r"\b(arthrose\s+de\s+la\s+hanche|coxarthrose)\b", "osteoarthritis of hip"),
    (r"\b(maladie\s+de\s+parkinson|parkinson)\b", "sporadic parkinson disease"),
    (r"\b(goutte)\b", "gout acute gouty arthritis"),
    (r"\b(psoriasis\s+vulgaire|psoriasis)\b", "psoriasis vulgaris"),
    (r"\b(entorse\s+de\s+la\s+cheville|entorse)\b", "sprain of ankle"),
    (r"\b(abces\s+de\s+l'oreille)\b", "abscess of external ear"),
    (r"\b(toxicit[eé]\s+m[eé]dicamenteuse|intoxication\s+accidentelle(?:\s+par\s+m[eé]dicaments)?)\b", "accidental poisoning overdose drug toxicity"),
    (r"\b(covid-19)\b", "COVID-19"),
    (r"\b(purpura\s+thrombop[eé]nique|purpura\s+immunologique)\b", "immune thrombocytopenia purpura"),
    (r"\b(lupus\s+[eé]ryth[eé]mateux\s+diss[eé]min[eé]|led)\b", "systemic lupus erythematosus"),
    (r"\b(an[eé]mie\s+[aà]\s+cellules\s+falciformes|dr[eé]panocytose)\b", "sickle cell disease"),
    (r"\b(cancer\s+du\s+sein|carcinome\s+canalaire)\b", "malignant neoplasm of breast"),
    (r"\b(carcinome\s+basocellulaire)\b", "basal cell carcinoma"),
    (r"\b(carcinome\s+[eé]pidermo[iï]de)\b", "squamous cell carcinoma"),
    (r"\b(insuffisance\s+cardiaque)\b", "heart failure"),
    (r"\b(infarctus\s+du\s+myocarde)\b", "acute myocardial infarction"),
    (r"\b(accident\s+vasculaire\s+c[eé]r[eé]bral|avc)\b", "stroke cerebrovascular accident"),
    (r"\b(anxi[eé]t[eé]\s+g[eé]n[eé]ralis[eé]e)\b", "generalized anxiety disorder"),
    (r"\b([eé]pisode\s+d[eé]pressif|d[eé]pression)\b", "depressive episode depression"),
    (r"\b(gale)\b", "scabies"),
    (r"\b(confirm[eé]e?|bact[eé]riologiquement\s+confirm[eé]e?|frottis\s+positif)\b", "confirmed bacteriologically"),
    (r"\b(sans\s+complication)\b", "without complication uncomplicated"),
    (r"\b(sans\s+signe\s+d'alerte)\b", "without warning signs"),
    (r"\b(sans\s+p[eé]ritonite)\b", "without peritonitis"),
]

PORTUGUESE_CLINICAL_MAP = [
    (r"\b(malaria)\b", "malaria"),
    (r"\b(cistite\s+infecciosa(?:\s+aguda)?)\b", "acute infectious cystitis"),
    (r"\b(infeccao\s+urinaria|cistite\s+aguda|cistite(?:\s+infecciosa)?)\b", "infectious cystitis urinary tract infection"),
    (r"\binfeccios[oa]s?\b", "infectious"),
    (r"\bagud[oa]s?\b", "acute"),
    (r"\bcr[oô]nic[oa]s?\b", "chronic"),
    (r"\bal[eé]rgic[oa]s?\b", "allergic"),
    (r"\b(carcinoma\s+basocelular)\b", "basal cell carcinoma of skin"),
    (r"\b(c[aâ]ncer\s+de\s+c[oó]lon|adenocarcinoma\s+de\s+c[oó]lon|tumor\s+de\s+c[oó]lon)\b", "malignant neoplasm of colon adenocarcinoma"),
    (r"\b(artrite\s+reumatoide|artrite\s+reumat[oó]ide)\b", "rheumatoid arthritis seropositive"),
    (r"\b(apendicite\s+aguda|apendicite)\b", "acute appendicitis without peritonitis"),
    (r"\b(bronquite\s+aguda)\b", "acute bronchitis"),
    (r"\b(pneumonia\s+comunit[aá]ria|pneumonia\s+aguda|pneumonia)\b", "community-acquired pneumonia"),
    (r"\b(dengue|febre\s+dengue)\b", "dengue fever without warning signs"),
    (r"\b(asma\s+br[oô]nquica|asma)\b", "allergic asthma bronchial asthma"),
    (r"\b(otite\s+m[eé]dia\s+aguda|otite\s+m[eé]dia)\b", "acute otitis media"),
    (r"\b(rinite\s+al[eé]rgica)\b", "allergic rhinitis"),
    (r"\b(faringite\s+aguda|faringite)\b", "acute pharyngitis"),
    (r"\b(liti[aá]se\s+ureteral|c[aá]lculo\s+ureteral|c[oó]lica\s+renal)\b", "calculus of ureter"),
    (r"\b(liti[aá]se\s+renal)\b", "calculus of kidney nephrolithiasis"),
    (r"\b(n[aã]o\s+erosiva|sem\s+les[oõ]es|aus[eê]ncia\s+de\s+les[oõ]es)\b", "non-erosive"),
    (r"\b(artrose|osteoartrite)\b", "osteoarthritis"),
    (r"\b(dermatite\s+at[oó]pica|eczema)\b", "atopic eczema dermatitis"),
    (r"\b(preecl[aâ]mpsia|pr[eé]-ecl[aâ]mpsia)\b", "pre-eclampsia"),
    (r"\b(disfun[cç][aã]o\s+er[eé]til)\b", "male erectile dysfunction"),
    (r"\b(hipertensao\s+arterial|hipertensao)\b", "essential hypertension"),
    (r"\b(diabetes\s+mellitus\s+tipo\s+2|diabetes)\b", "type 2 diabetes mellitus"),
    (r"\b(doenca\s+do\s+refluxo\s+gastroesofagico|refluxo)\b", "gastroesophageal reflux disease GERD"),
    (r"\b(apneia\s+obstrutiva\s+do\s+sono|apneia\s+do\s+sono)\b", "obstructive sleep apnoea"),
    (r"\b(c[aâ]ncer\s+de\s+mama|carcinoma\s+ductal)\b", "malignant neoplasm of breast"),
    (r"\b(carcinoma\s+espinocelular|carcinoma\s+epidermoide)\b", "squamous cell carcinoma"),
    (r"\b(l[uú]pus\s+eritematoso\s+sist[eê]mico|les)\b", "systemic lupus erythematosus"),
    (r"\b(anemia\s+falciforme|crise\s+falciforme)\b", "sickle cell disease crisis"),
    (r"\b(p[uú]rpura\s+trombocitop[eê]nica)\b", "immune thrombocytopenia purpura"),
    (r"\b(hipotireoidismo|hipotiroidismo)\b", "primary hypothyroidism"),
    (r"\b(insufici[eê]ncia\s+card[ií]aca)\b", "heart failure"),
    (r"\b(acidente\s+vascular\s+cerebral|avc)\b", "stroke cerebrovascular accident"),
    (r"\b(depress[aã]o|epis[oó]dio\s+depressivo)\b", "depressive episode depression"),
    (r"\b(ansiedade\s+generalizada)\b", "generalized anxiety disorder"),
    (r"\b(escabiose|sarna)\b", "scabies"),
    (r"\b(confirmad[oa]|bacteriologicamente\s+confirmad[oa]|exame\s+positivo)\b", "confirmed bacteriologically"),
    (r"\b(sem\s+complica[cç][oõ]es)\b", "without complication uncomplicated"),
]

HINDI_CLINICAL_MAP = [
    (r"(प्लास्मोडियम\s*फाल्सीपेरम(?:\s*मलेरिया)?|फाल्सीपेरम\s*मलेरिया)", "malaria due to Plasmodium falciparum confirmed"),
    (r"(टाइफाइड|मियादी\s*बुखार)", "typhoid fever Salmonella typhi"),
    (r"(मलेरिया)", "malaria Plasmodium"),
    (r"(डेंगू)", "dengue fever without warning signs"),
    (r"(दमा|अस्थमा|घरघराहट|सांस\s*फूलना)", "allergic asthma bronchial asthma uncomplicated"),
    (r"(मोतियाबिंद)", "age-related cataract"),
    (r"(हाइपोथायरायडिज्म|थायराइड)", "primary hypothyroidism"),
    (r"(बुखार)", "fever pyrexia"),
    (r"(खांसी)", "cough bronchitis"),
    (r"(दस्त|पेचिश)", "diarrhea dysentery"),
    (r"(पेट\s*दर्द)", "abdominal pain"),
    (r"(डायबिटीज|मधुमेह)", "diabetes mellitus"),
    (r"(उच्च\s*रक्तचाप|हाई\s*बीपी|ब्लड\s*प्रेशर)", "essential hypertension"),
    (r"(अवसाद|डिप्रेशन)", "depressive disorder depression"),
    (r"(चिंता\s*विकार|एंग्जाइटी)", "generalized anxiety disorder"),
    (r"(स्तन\s*कैंसर)", "malignant neoplasm of breast"),
    (r"(अपेंडिसाइटिस|अपेंडिक्स)", "acute appendicitis without peritonitis"),
    (r"(निमोनिया)", "community-acquired pneumonia"),
    (r"(ब्रोंकाइटिस)", "acute bronchitis"),
    (r"(कोलन\s*कैंसर)", "malignant neoplasm of colon"),
    (r"(गठिया|संधिशोथ)", "rheumatoid arthritis seropositive"),
    (r"(रिफ्लक्स|गैस|एसिडिटी)", "gastro-oesophageal reflux disease GERD non-erosive"),
    (r"(नेत्रश्लेष्मलाशोथ|आंख\s*आना)", "conjunctivitis"),
    (r"(एक्जिमा)", "atopic eczema dermatitis"),
]

ARABIC_CLINICAL_MAP = [
    (r"(التهاب\s*الزائدة\s*الدودية(?:\s*الحاد)?|زائدة\s*دودية)", "acute appendicitis without peritonitis"),
    (r"(التهاب\s*المفاصل\s*الروماتويدي|روماتيزم)", "rheumatoid arthritis seropositive"),
    (r"(سرطان\s*القولون)", "malignant neoplasm of colon adenocarcinoma"),
    (r"(حمى\s*الضنك)", "dengue fever without warning signs"),
    (r"(إعتام\s*عدسة\s*العين|ساد)", "age-related cataract"),
    (r"(زحار\s*أميب|أميبا)", "amoebic dysentery Entamoeba histolytica"),
    (r"(ملاريا)", "malaria"),
    (r"(شقيقة)", "migraine without aura"),
    (r"(حمى|حرارة)", "fever pyrexia"),
    (r"(سعال)", "cough"),
    (r"(إسهال)", "diarrhea"),
    (r"(ربو)", "bronchial asthma"),
    (r"(سكري)", "diabetes mellitus"),
    (r"(السل|تدرن)", "respiratory tuberculosis"),
    (r"(التهاب\s*رئوي)", "community-acquired pneumonia"),
    (r"(التهاب\s*الشعب\s*الهوائية)", "acute bronchitis"),
    (r"(ارتفاع\s*ضغط\s*الدم)", "essential hypertension"),
    (r"(قصور\s*الغدة\s*الدرقية)", "primary hypothyroidism"),
    (r"(سرطان\s*الثدي)", "malignant neoplasm of breast"),
    (r"(اكتئاب)", "depressive disorder depression"),
    (r"(قلق\s*عام)", "generalized anxiety disorder"),
    (r"(ارتجاع\s*المريء)", "gastro-oesophageal reflux disease GERD non-erosive"),
    (r"(إكزيما|أكزيما)", "atopic eczema dermatitis"),
]

CHINESE_CLINICAL_MAP = [
    (r"(社区获得性肺炎)", "community-acquired pneumonia bacterial"),
    (r"(不明原因慢性咳嗽|慢性咳嗽)", "unexplained chronic cough"),
    (r"(十二指肠溃疡)", "duodenal ulcer Helicobacter-pylori associated"),
    (r"(原发性甲状腺功能减退(?:症)?|甲状腺功能减退|甲减)", "primary hypothyroidism"),
    (r"(气血两虚(?:证)?)", "dual deficiency of heart qi and blood pattern"),
    (r"(肝气郁结(?:证)?)", "liver qi stagnation pattern"),
    (r"(假丝酵母菌|外阴念珠菌|念珠菌)", "vulvovaginal candidiasis candidosis thrush"),
    (r"(支气管哮喘|哮喘)", "allergic asthma bronchial asthma"),
    (r"(急性支气管炎|支气管炎)", "acute bronchitis"),
    (r"(急性阑尾炎|阑尾炎)", "acute appendicitis without peritonitis"),
    (r"(胃食管反流病)", "gastro-oesophageal reflux disease GERD non-erosive"),
    (r"(结直肠癌|结肠癌)", "malignant neoplasm of colon adenocarcinoma"),
    (r"(类风湿关节炎)", "rheumatoid arthritis seropositive"),
    (r"(老年性白内障|白内障)", "age-related cataract senile"),
    (r"(结膜炎)", "conjunctivitis"),
    (r"(骨关节炎)", "osteoarthritis"),
    (r"(登革热)", "dengue fever without warning signs"),
    (r"(特应性皮炎|湿疹)", "atopic dermatitis eczema"),
    (r"(帕金森病)", "sporadic parkinson disease"),
    (r"(抑郁症|抑郁障碍)", "depressive disorder depression"),
    (r"(发热|发烧)", "fever pyrexia"),
    (r"(2型糖尿病|糖尿病)", "type 2 diabetes mellitus"),
    (r"(高血压)", "essential hypertension"),
    (r"(乳腺癌)", "malignant neoplasm of breast"),
    (r"(甲状腺功能减退|甲减)", "primary hypothyroidism"),
    (r"(系统性红斑狼疮)", "systemic lupus erythematosus"),
    (r"(过敏性鼻炎)", "allergic rhinitis"),
    (r"(焦虑症|焦虑障碍)", "generalized anxiety disorder"),
    (r"(痛风)", "gout acute gouty arthritis"),
    (r"(系统性红斑狼疮)", "systemic lupus erythematosus"),
    (r"(过敏性鼻炎)", "allergic rhinitis"),
    (r"(焦虑症|焦虑障碍)", "generalized anxiety disorder"),
]

GERMAN_CLINICAL_MAP = [
    (r"\b(idiopathisches\s+parkinsonsyndrom|sporadische\s+parkinson-?krankheit|sporadische\s+parkinson)\b", "sporadic Parkinson disease"),
    (r"\b(parkinson-?krankheit|parkinsonsyndrom|morbus\s+parkinson)\b", "Parkinson disease"),
    (r"\b(benigner\s+paroxysmaler\s+lagerungsschwindel|bpls)\b", "benign paroxysmal positional vertigo"),
    (r"\b(vorhofflimmern)\b", "atrial fibrillation"),
    (r"\b(harnwegsinfekt(?:ion)?|blasenentz[uü]ndung)\b", "urinary tract infection cystitis"),
    (r"\b(arterielle\s+hypertonie|bluthochdruck)\b", "essential hypertension"),
    (r"\b(diabetes\s+mellitus\s+typ\s*2|typ-?2-?diabetes)\b", "type 2 diabetes mellitus"),
    (r"\b(gallensteine|cholezystolithiasis)\b", "cholelithiasis calculus of gallbladder"),
    (r"\b(akute\s+bronchitis)\b", "acute bronchitis"),
    (r"\b(asthma\s+bronchiale)\b", "bronchial asthma"),
    (r"\b(gastro[oö]sophageale\s+refluxkrankheit|reflux[oö]sophagitis|gerd)\b", "gastro-oesophageal reflux disease"),
    (r"\b(herzinsuffizienz)\b", "congestive heart failure"),
    (r"\b(lungenentz[uü]ndung|pneumonie)\b", "pneumonia"),
    (r"\b(depression|depressive\s+episode)\b", "depressive disorder"),
    (r"\b(angina\s+pectoris)\b", "angina pectoris"),
    (r"\b(kopfschmerzen|migr[aä]ne)\b", "headache migraine"),
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

    en_hits = len(words.intersection(ENGLISH_MARKERS))
    sw_hits = len(words.intersection(SWAHILI_MARKERS))
    es_hits = len(words.intersection(SPANISH_MARKERS))
    fr_hits = len(words.intersection(FRENCH_MARKERS))
    pt_hits = len(words.intersection(PORTUGUESE_MARKERS))

    scores = [("sw", sw_hits), ("es", es_hits), ("fr", fr_hits), ("pt", pt_hits)]
    scores.sort(key=lambda x: x[1], reverse=True)

    best_lang, best_hits = scores[0]
    if best_hits >= 2 and best_hits > en_hits:
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
    elif source_lang == "de":
        term_map = GERMAN_CLINICAL_MAP

    for pattern, replacement in term_map:
        translated = re.sub(pattern, replacement, translated, flags=re.IGNORECASE)

    # Append extracted English clinical keywords to the text for robust hybrid retrieval
    keywords = []
    for _, replacement in term_map:
        if replacement in translated and replacement not in keywords:
            keywords.append(replacement)
    if keywords:
        translated = f"{translated} {' '.join(keywords)}"

    return translated
