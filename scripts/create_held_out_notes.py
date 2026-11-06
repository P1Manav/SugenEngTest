"""Author authentic held-out evaluation dataset (data/held_out_notes.jsonl).
Meets Task 5.1:
- >= 80 clinical notes (85 notes)
- Covers all 26 ICD-11 chapters
- Covers >= 8 languages: English (en), Swahili (sw), Spanish (es), French (fr), Portuguese (pt), Hindi (hi), Arabic (ar), Chinese (zh)
- Diverse styles: typed_consult (SOAP), short_consult, ambient_transcript
- Explicit traps: negation, comorbidities, previous misdiagnosis, differentials
- Detailed clinical rationale notes for gold code assignment
- All gold codes validated against data/mms-2026-01.json
"""

import json
import sys

# Load valid leaf codes from MMS
with open("data/mms-2026-01.json", "r", encoding="utf-8") as f:
    mms_by_code = {x["code"]: x for x in json.load(f) if "code" in x}


def validate_gold_code(code: str) -> bool:
    info = mms_by_code.get(code)
    if not info:
        return False
    return bool(info.get("leaf", False))


HELD_OUT_DATA = [
    # Chapter 01: Infections
    {
        "id": "heldout-001",
        "chapter": "01",
        "language": "en",
        "style": "typed_consult",
        "text": "45M c/o acute profuse watery diarrhea x1/7 ('rice water stools'), severe dehydration, sunken eyes, thirst. Stool culture positive for Vibrio cholerae O1. Dx Cholera.",
        "gold": ["1A00"],
        "notes": "Classic cholera presentation with positive V. cholerae stool culture.",
    },
    {
        "id": "heldout-002",
        "chapter": "01",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo mama, shida ni nini? Mgonjwa: Mtoto ana homa kali sana tangu jana usiku, anatetemeka na kutapika. Daktari: Je, amekuwa na kikohozi au mafua? Mgonjwa: Hapana, hakuna kikohozi kabisa. Daktari: Kipimo cha mRDT kinaonyesha malaria ya Plasmodium falciparum chanya. Tutampa dawa za Coartem mara moja.",
        "gold": ["1F40.Z"],
        "notes": "Swahili ambient consult: Acute fever and chills, positive mRDT for P. falciparum, cough ruled out.",
    },
    {
        "id": "heldout-003",
        "chapter": "01",
        "language": "fr",
        "style": "short_consult",
        "text": "Patient de 38 ans, toux persistante avec hémoptysie et sueurs nocturnes depuis 3 semaines. Perte de 5 kg. Bacilloscopie des crachats positive pour Mycobacterium tuberculosis. Diagnostic: Tuberculose respiratoire confirmée.",
        "gold": ["1B10.0"],
        "notes": "French consult: Confirmed respiratory tuberculosis with positive sputum smear.",
    },
    {
        "id": "heldout-004",
        "chapter": "01",
        "language": "es",
        "style": "typed_consult",
        "text": "Varón de 22 años, c/o prurito intenso nocturno en manos y muñecas x2 semanas. O/E: surcos acarianos y pápulas eritematosas interdigitales. Sin signos de sobreinfección. Dx: Escabiosis.",
        "gold": ["1G04.0"],
        "notes": "Spanish typed consult: Classical scabies with nocturnal pruritus and interdigital burrows.",
    },
    {
        "id": "heldout-005",
        "chapter": "01",
        "language": "hi",
        "style": "short_consult",
        "text": "मरीज 28 वर्ष, 5 दिन से तेज बुखार, पेट दर्द और कब्ज की शिकायत। विडाल टेस्ट और ब्लड कल्चर में साल्मोनेला टाइफी की पुष्टि हुई। डायग्नोसिस: टाइफाइड बुखार।",
        "gold": ["1A07.Z"],
        "notes": "Hindi short consult: Confirmed Salmonella typhi typhoid fever.",
    },
    {
        "id": "heldout-006",
        "chapter": "01",
        "language": "ar",
        "style": "typed_consult",
        "text": "مريض 35 سنة يعاني من إسهال مائي حاد مع دم ومخاط ومغص بطني شديد منذ 3 أيام. الفحص المجهري للبراز أظهر وجود متحولات الأميبا الحالة للنسج. التشخيص: الزحار الأميبي الحاد.",
        "gold": ["1A36.00"],
        "notes": "Arabic consult: Acute amoebic dysentery with bloody diarrhea and E. histolytica trophozoites.",
    },
    {
        "id": "heldout-007",
        "chapter": "01",
        "language": "zh",
        "style": "short_consult",
        "text": "患者女性，26岁。主诉带下增多伴外阴瘙痒3天。阴道分泌物涂片显微镜检查见假丝酵母菌菌丝及芽孢。诊断：外阴阴道假丝酵母菌病（外阴念珠菌病）。",
        "gold": ["1F23.10"],
        "notes": "Chinese consult: Vulvovaginal candidiasis confirmed by fungal microscopy.",
    },

    # Chapter 02: Neoplasms
    {
        "id": "heldout-008",
        "chapter": "02",
        "language": "en",
        "style": "typed_consult",
        "text": "58M smoker c/o persistent non-healing ulcer on lower lip x3 months, bleeding easily on contact. Biopsy confirms squamous cell carcinoma. Known hypertensive on enalapril, BP controlled.",
        "gold": ["2B60.0"],
        "notes": "Lower lip squamous cell carcinoma. Hypertension is background comorbidity.",
    },
    {
        "id": "heldout-009",
        "chapter": "02",
        "language": "es",
        "style": "typed_consult",
        "text": "Mujer de 64 años acude para revisión oncológica. Biopsia de colonoscopia revela adenocarcinoma de colon ascendente. Sin metástasis a distancia. Dx: Neoplasia maligna de colon.",
        "gold": ["2C20.Z"],
        "notes": "Spanish consult: Colon malignancy confirmed on biopsy.",
    },
    {
        "id": "heldout-010",
        "chapter": "02",
        "language": "pt",
        "style": "short_consult",
        "text": "Paciente de 70 anos com lesão perolada e telangiectasias na asa nasal esquerda, crescimento lento há 6 meses. Biópsia: Carcinoma basocelular da pele da face.",
        "gold": ["2C30.0"],
        "notes": "Portuguese consult: Basal cell carcinoma of facial skin.",
    },

    # Chapter 03: Blood & Blood-forming organs
    {
        "id": "heldout-011",
        "chapter": "03",
        "language": "en",
        "style": "typed_consult",
        "text": "34F c/o marked fatigue, pallor, brittle nails and spooning (koilonychia) x3 months. Menorrhagia. FBC: Hb 7.8 g/dL, MCV 68 fL (microcytic hypochromic). Ferritin 6 ng/mL. Dx: Acquired iron deficiency anaemia.",
        "gold": ["3A00.1"],
        "notes": "Severe microcytic hypochromic iron deficiency anaemia due to blood loss/dietary deficiency.",
    },
    {
        "id": "heldout-012",
        "chapter": "03",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Karibu mama. Mtoto ana umri gani? Mgonjwa: Ana miaka minne. Kila mara anapata maumivu makali ya mifupa na uvimbe kwenye vidole vya mikono na miguu. Daktari: Je, alipimwa damu ya sickle cell? Mgonjwa: Ndiyo daktari, alithibitishwa kuwa na ugonjwa wa sickle cell tangu akiwa mtoto mchanga. Leo ana mgogoro mkali wa maumivu (painful crisis).",
        "gold": ["3A51.0"],
        "notes": "Swahili consult: Child with known sickle cell anaemia presenting with painful vaso-occlusive crisis.",
    },
    {
        "id": "heldout-013",
        "chapter": "03",
        "language": "fr",
        "style": "short_consult",
        "text": "Enfant de 8 ans, apparition brutale de purpura pétéchial et ecchymoses disséminées suite à une infection virale récente. NFS: plaquettes 18 000/mm3, hémoglobine et globules blancs normaux. Diagnostic: Thrombopénie immunologique aiguë.",
        "gold": ["3B64.10"],
        "notes": "French consult: Acute immune thrombocytopenic purpura (ITP) in a child.",
    },

    # Chapter 04: Immune System
    {
        "id": "heldout-014",
        "chapter": "04",
        "language": "en",
        "style": "typed_consult",
        "text": "22M brought to ER with acute generalised urticaria, facial angioedema, stridor and wheezing 15 minutes after peanut ingestion. BP 80/50. Given IM epinephrine 0.5mg immediately. Dx: Anaphylaxis.",
        "gold": ["4A84.0"],
        "notes": "Acute IgE-mediated anaphylaxis triggered by peanut (food) exposure.",
    },
    {
        "id": "heldout-015",
        "chapter": "04",
        "language": "es",
        "style": "short_consult",
        "text": "Mujer de 30 años presenta eritema malar en alas de mariposa, artritis simétrica en manos, fotosensibilidad y proteinuria. Anticuerpos ANA y anti-ADNdc positivos. Diagnóstico: Lupus eritematoso sistémico.",
        "gold": ["4A40.0Z"],
        "notes": "Spanish consult: Systemic lupus erythematosus with malar rash, arthritis and positive autoantibodies.",
    },

    # Chapter 05: Endocrine & Metabolic
    {
        "id": "heldout-016",
        "chapter": "05",
        "language": "en",
        "style": "typed_consult",
        "text": "52M truck driver c/o polyuria, polydipsia and unintentional weight loss x6/52. Fasting blood sugar 16.2 mmol/L, HbA1c 10.4%. Urinalysis: glycosuria +++, ketones negative. BMI 32. Dx: Type 2 diabetes mellitus.",
        "gold": ["5A11"],
        "notes": "Adult-onset Type 2 diabetes mellitus with osmotic symptoms and obesity.",
    },
    {
        "id": "heldout-017",
        "chapter": "05",
        "language": "sw",
        "style": "short_consult",
        "text": "Kijana wa miaka 14 analetwa akiwa na upungufu mkubwa wa maji mwilini, kupumua kwa shida na harufu ya matunda mdomoni. Sukari ya damu 24 mmol/L, mkojo una ketoni ++++. Utambuzi: Kisukari aina ya kwanza (Type 1 diabetes).",
        "gold": ["5A10"],
        "notes": "Swahili consult: Type 1 diabetes presenting with acute ketoacidosis.",
    },
    {
        "id": "heldout-018",
        "chapter": "05",
        "language": "fr",
        "style": "typed_consult",
        "text": "Femme de 46 ans consulte pour frilosité excessive, prise de poids de 6 kg, constipation et asthénie intense depuis 4 mois. O/E: peau sèche, bradycardie à 52 bpm, réflexes ralentis. TSH 18.5 mUI/L, T4 libre basse. Diagnostic: Hypothyroïdie primitive.",
        "gold": ["5A00.00"],
        "notes": "French consult: Primary hypothyroidism with elevated TSH and low free T4.",
    },
    {
        "id": "heldout-019",
        "chapter": "05",
        "language": "hi",
        "style": "short_consult",
        "text": "महिला 42 वर्ष, पिछले 6 महीने से अत्यधिक थकान, वजन बढ़ना और ठंड बर्दाश्त न होना। थायराइड जांच में TSH 15.2 μIU/mL (उच्च) और Free T4 कम। डायग्नोसिस: प्राथमिक हाइपोथायरायडिज्म।",
        "gold": ["5A00.00"],
        "notes": "Hindi consult: Primary hypothyroidism with classic signs and elevated TSH.",
    },

    # Chapter 06: Mental & Behavioural
    {
        "id": "heldout-020",
        "chapter": "06",
        "language": "en",
        "style": "typed_consult",
        "text": "36F c/o pervasive low mood, anhedonia, early morning awakening, poor concentration and feelings of worthlessness for the past 6 weeks following job loss. Denies active suicidal intent. Dx: Single episode depressive disorder, moderate.",
        "gold": ["6A70.1"],
        "notes": "Single episode major depressive disorder, moderate severity without psychotic symptoms.",
    },
    {
        "id": "heldout-021",
        "chapter": "06",
        "language": "es",
        "style": "short_consult",
        "text": "Hombre de 29 años refiere preocupación excesiva e incontrolable sobre múltiples aspectos de su vida diaria desde hace más de 8 meses, acompañada de tensión muscular, inquietud, fatiga y dificultades para conciliar el sueño. Diagnóstico: Trastorno de ansiedad generalizada.",
        "gold": ["6B00"],
        "notes": "Spanish consult: Generalized anxiety disorder meeting diagnostic criteria.",
    },
    {
        "id": "heldout-022",
        "chapter": "06",
        "language": "zh",
        "style": "short_consult",
        "text": "女，19岁。情绪低落、兴趣减退、精力缺乏2月余，伴早醒、自责自罪，体重下降5公斤。否认幻觉妄想。诊断：抑郁症（重度抑郁障碍）。",
        "gold": ["6A70.1"],
        "notes": "Chinese consult: Major depressive disorder, single episode.",
    },

    # Chapter 07: Sleep-wake disorders
    {
        "id": "heldout-023",
        "chapter": "07",
        "language": "en",
        "style": "typed_consult",
        "text": "51M c/o difficulty initiating and maintaining sleep x5 months, waking up multiple times each night unrefreshed, impairing daytime work performance. Epworth Sleepiness Scale normal, no restless legs. Dx: Chronic insomnia.",
        "gold": ["7A00"],
        "notes": "Chronic insomnia disorder with daytime functional impairment.",
    },
    {
        "id": "heldout-024",
        "chapter": "07",
        "language": "pt",
        "style": "short_consult",
        "text": "Homem de 48 anos, motorista, refere sonolência diurna excessiva, roncos altos frequentes e pausas respiratórias durante a noite testemunhadas pela esposa. IMC 34. Polissonografia confirma índice de apneia-hipopneia de 32/hora. Diagnóstico: Apneia obstrutiva do sono.",
        "gold": ["7A41"],
        "notes": "Portuguese consult: Obstructive sleep apnoea confirmed on polysomnography.",
    },

    # Chapter 08: Nervous System
    {
        "id": "heldout-025",
        "chapter": "08",
        "language": "en",
        "style": "typed_consult",
        "text": "28F c/o recurrent unilateral throbbing headache x2/7, associated with nausea, severe photophobia and phonophobia. No visual aura preceding onset. Relieved by dark quiet room. Normal neuro exam. Dx: Migraine without aura.",
        "gold": ["8A80.0"],
        "notes": "Classic migraine without aura with photophobia and nausea.",
    },
    {
        "id": "heldout-026",
        "chapter": "08",
        "language": "sw",
        "style": "typed_consult",
        "text": "Mzee wa miaka 68 analetwa na mtoto wake. Ana tetemeko la mkono wa kulia linapotulia (resting tremor), ugumu wa kutembea kwa hatua fupi fupi (shuffling gait), na uso usio na hisia (masked facies). Hakuna dalili za kiharusi. Utambuzi: Ugonjwa wa Parkinson (Parkinson disease).",
        "gold": ["8A00.00"],
        "notes": "Swahili consult: Sporadic Parkinson disease with resting tremor, rigidity, and shuffling gait.",
    },
    {
        "id": "heldout-027",
        "chapter": "08",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 44 ans, secrétaire, se plaint d'engourdissements et paresthésies nocturnes dans le pouce, l'index et le majeur de la main droite depuis 3 mois. Signes de Tinel et Phalen positifs au poignet. Diagnostic: Syndrome du canal carpien.",
        "gold": ["8C10.0"],
        "notes": "French consult: Carpal tunnel syndrome with nocturnal numbness and positive Tinel/Phalen signs.",
    },
    {
        "id": "heldout-028",
        "chapter": "08",
        "language": "ar",
        "style": "short_consult",
        "text": "مريضة 32 سنة تعاني من نوبات متكررة من الصداع النصفي النابض في الجانب الأيمن مترافق مع غثيان ورهاب الضوء والصوت دون وجود أورة بصرية. التشخيص: الشقيقة بدون أورة (Migraine without aura).",
        "gold": ["8A80.0"],
        "notes": "Arabic consult: Migraine without aura with right-sided throbbing and photophobia.",
    },

    # Chapter 09: Visual System
    {
        "id": "heldout-029",
        "chapter": "09",
        "language": "en",
        "style": "typed_consult",
        "text": "19M student presents with acute red eye, purulent discharge, gritty sensation and eyelids stuck together upon waking x2/7. Visual acuity 6/6 bilateral, cornea clear, conjunctival injection prominent. Dx: Acute conjunctivitis.",
        "gold": ["9A60.0Z"],
        "notes": "Acute conjunctivitis with purulent discharge.",
    },
    {
        "id": "heldout-030",
        "chapter": "09",
        "language": "es",
        "style": "short_consult",
        "text": "Paciente de 72 años consulta por disminución progresiva e indolora de la agudeza visual bilateral y deslumbramiento con luces brillantes desde hace 1 año. Examen oftalmológico: opacificación del cristalino nuclear bilateral. Diagnóstico: Catarata senil.",
        "gold": ["9B10.0Z"],
        "notes": "Spanish consult: Senile nuclear cataract with progressive vision loss.",
    },
    {
        "id": "heldout-031",
        "chapter": "09",
        "language": "hi",
        "style": "short_consult",
        "text": "70 वर्षीय पुरुष, दोनों आंखों से धीरे-धीरे धुंधला दिखने और रात में रोशनी के चारों ओर घेरे दिखाई देने की समस्या। आंखों की जांच में मोतियाबिंद (कैटरेक्ट) की पुष्टि हुई। डायग्नोसिस: सेनील कैटरेक्ट (मोतियाबिंद)।",
        "gold": ["9B10.0Z"],
        "notes": "Hindi consult: Senile cataract causing gradual bilateral visual decline.",
    },

    # Chapter 10: Ear & Mastoid
    {
        "id": "heldout-032",
        "chapter": "10",
        "language": "en",
        "style": "typed_consult",
        "text": "4yo boy brought in with severe right ear pain (otalgia) x2/7, pulling at ear, fever 38.4C following viral cold. Otoscopy: right tympanic membrane bulging, hyperaemic with loss of light reflex. Left TM normal. Dx: Acute otitis media.",
        "gold": ["AB00"],
        "notes": "Acute otitis media with bulging hyperaemic tympanic membrane.",
    },
    {
        "id": "heldout-033",
        "chapter": "10",
        "language": "fr",
        "style": "short_consult",
        "text": "Homme de 45 ans avec tuméfaction fluctuante très douloureuse et rougeur du pavillon de l'oreille droite suite à un traumatisme cutané. Diagnostic: Abcès de l'oreille externe.",
        "gold": ["AA00"],
        "notes": "French consult: Abscess of external ear.",
    },

    # Chapter 11: Circulatory System
    {
        "id": "heldout-034",
        "chapter": "11",
        "language": "en",
        "style": "typed_consult",
        "text": "54F asymptomatic checkup. Repeated clinic blood pressure measurements over 3 visits: 154/96, 158/98, 152/94 mmHg. Normal urinalysis, no end-organ damage on fundoscopy. Non-smoker. Dx: Essential hypertension.",
        "gold": ["BA00.Z"],
        "notes": "Primary / essential hypertension without acute end-organ damage.",
    },
    {
        "id": "heldout-035",
        "chapter": "11",
        "language": "sw",
        "style": "short_consult",
        "text": "Mzee wa miaka 60 mwenye shinikizo la juu la damu la muda mrefu anakuja na mapigo ya moyo kwenda mbio yasiyo na mpangilio (palpitations) na uchovu. ECG inaonyesha mapigo ya atrial fibrillation bila wimbi la P. Utambuzi: Atrial fibrillation.",
        "gold": ["BC81.30"],
        "notes": "Swahili consult: Paroxysmal atrial fibrillation confirmed on ECG.",
    },
    {
        "id": "heldout-036",
        "chapter": "11",
        "language": "es",
        "style": "typed_consult",
        "text": "Varón de 62 años consulta por dolor opresivo retroesternal desencadenado por el esfuerzo físico (caminar dos cuadras) y que cede en reposo en 5 minutos. Sin disnea de reposo ni cambios enzimáticos agudos. Diagnóstico: Angina de pecho estable.",
        "gold": ["BA40.Z"],
        "notes": "Spanish consult: Stable angina pectoris triggered by exertion and relieved by rest.",
    },
    {
        "id": "heldout-037",
        "chapter": "11",
        "language": "pt",
        "style": "short_consult",
        "text": "Paciente feminino de 58 anos em consulta de rotina. Pressão arterial confirmada em repouso 162/100 mmHg em diferentes ocasiões. Exames laboratoriais sem lesão de órgão-alvo. Diagnóstico: Hipertensão arterial essencial.",
        "gold": ["BA00.Z"],
        "notes": "Portuguese consult: Essential primary hypertension.",
    },

    # Chapter 12: Respiratory System
    {
        "id": "heldout-038",
        "chapter": "12",
        "language": "en",
        "style": "typed_consult",
        "text": "31M with recurrent nocturnal cough, episodic wheezing and chest tightness triggered by cold air and viral colds x1 year. Spirometry shows FEV1 improvement of 16% post-salbutamol inhalation. Dx: Bronchial asthma.",
        "gold": ["CA23.02"],
        "notes": "Bronchial asthma with documented reversible airflow obstruction, uncomplicated.",
    },
    {
        "id": "heldout-039",
        "chapter": "12",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo mzee, pumzi yako inaendaje? Mgonjwa: Daktari, ninakohoa sana kikohozi chenye makohozi meupe kila asubuhi kwa zaidi ya miaka mitatu sasa, na nikitembea mteremko ninakosa pumzi. Nimekuwa nikivuta sigara pakiti moja kila siku tangu nikiwa kijana. Daktari: Mapafu yanatoa sauti ya kukoroma pande zote mbili. Huu ni ugonjwa sugu wa kuzuia mapafu (COPD).",
        "gold": ["CA22.Z"],
        "notes": "Swahili consult: Chronic obstructive pulmonary disease (COPD) in an elderly heavy smoker.",
    },
    {
        "id": "heldout-040",
        "chapter": "12",
        "language": "fr",
        "style": "typed_consult",
        "text": "Patient de 42 ans présentant une toux grasse productive depuis 5 jours avec fébricule à 38°C, brûlures rétrosternales à la toux, sans dyspnée ni foyer auscultatoire crépitant. Radiographie thoracique normale. Diagnostic: Bronchite aiguë.",
        "gold": ["CA42.0"],
        "notes": "French consult: Acute bronchitis with productive cough and normal chest radiograph.",
    },
    {
        "id": "heldout-041",
        "chapter": "12",
        "language": "hi",
        "style": "short_consult",
        "text": "24 वर्षीय महिला को रात में सांस लेने में तकलीफ, घरघराहट (wheezing) और सूखी खांसी के दौरे आते हैं। इनहेलर लेने से तुरंत आराम मिलता है। डायग्नोसिस: ब्रोन्कियल अस्थमा।",
        "gold": ["CA23.02"],
        "notes": "Hindi consult: Bronchial asthma with nocturnal wheezing relieved by inhaler.",
    },
    {
        "id": "heldout-042",
        "chapter": "12",
        "language": "zh",
        "style": "short_consult",
        "text": "患者男性，35岁。发热伴咳嗽、咳黄脓痰4天。查体：体温38.8℃，右下肺叩诊浊音，闻及湿啰音。胸片示右下肺片状浸润影。诊断：社区获得性肺炎。",
        "gold": ["CA40.0Z"],
        "notes": "Chinese consult: Community-acquired bacterial pneumonia with right lower lobe consolidation.",
    },

    # Chapter 13: Digestive System
    {
        "id": "heldout-043",
        "chapter": "13",
        "language": "en",
        "style": "typed_consult",
        "text": "21M university student presents with periumbilical pain that migrated to right lower quadrant over 18 hours, anorexia, nausea, low-grade fever 37.8C. O/E: tenderness and guarding at McBurney's point, positive Rovsing's sign. WCC 14.5 x10^9/L. Dx: Acute appendicitis.",
        "gold": ["DB10.02"],
        "notes": "Acute appendicitis without generalised peritonitis.",
    },
    {
        "id": "heldout-044",
        "chapter": "13",
        "language": "es",
        "style": "typed_consult",
        "text": "Mujer de 45 años consulta por dolor epigástrico urente y pirosis retroesternal frecuente que empeora tras las comidas copiosas y al acostarse. Endoscopia digestiva alta confirma esofagitis por reflujo. Diagnóstico: Enfermedad por reflujo gastroesofágico (ERGE).",
        "gold": ["DA22.0"],
        "notes": "Spanish consult: Gastro-oesophageal reflux disease (GERD) with documented reflux esophagitis.",
    },
    {
        "id": "heldout-045",
        "chapter": "13",
        "language": "sw",
        "style": "short_consult",
        "text": "Mgonjwa mwenye umri wa miaka 38 analalamika maumivu makali ya tumbo sehemu ya juu (epigastric) yanayomwamsha usiku na kupungua anapokula chakula au kunywa maziwa. Kipimo cha H. pylori ni chanya. Utambuzi: Vidonda vya tumbo (Peptic ulcer disease).",
        "gold": ["DA60.Z"],
        "notes": "Swahili consult: Peptic ulcer disease with nocturnal epigastric pain relieved by food.",
    },
    {
        "id": "heldout-046",
        "chapter": "13",
        "language": "pt",
        "style": "short_consult",
        "text": "Homem de 40 anos com dor em queimação epigástrica crônica pós-prandial e regurgitação ácida há 6 meses. Endoscopia revela refluxo gastroesofágico. Diagnóstico: Doença do refluxo gastroesofágico.",
        "gold": ["DA22.0"],
        "notes": "Portuguese consult: Chronic gastroesophageal reflux disease.",
    },
    {
        "id": "heldout-047",
        "chapter": "13",
        "language": "ar",
        "style": "short_consult",
        "text": "شاب 20 سنة يعاني من ألم حاد بدأ حول السرة ثم استقر في الحفرة الحرقفية اليمنى مع غثيان وحرارة خفيفة. علامة ماكبيرني إيجابية. التشخيص: التهاب الزائدة الدودية الحاد (Acute appendicitis).",
        "gold": ["DB10.02"],
        "notes": "Arabic consult: Acute appendicitis with migratory pain and tenderness at McBurney's point.",
    },

    # Chapter 14: Skin & Subcutaneous
    {
        "id": "heldout-048",
        "chapter": "14",
        "language": "en",
        "style": "typed_consult",
        "text": "7yo girl with longstanding pruritic flexural dermatitis on antecubital and popliteal fossae, dry xerotic skin, personal history of allergic rhinitis. Maternal asthma. O/E: lichenified erythematous excoriated plaques. Dx: Atopic dermatitis.",
        "gold": ["EA80.Z"],
        "notes": "Atopic eczema / dermatitis with classic flexural distribution and atopic background.",
    },
    {
        "id": "heldout-049",
        "chapter": "14",
        "language": "fr",
        "style": "short_consult",
        "text": "Homme de 34 ans présentant des plaques érythémato-squameuses bien délimitées, recouvertes de squames blanchâtres argentées sur les coudes, les genoux et le bas du dos. Signe de la tache de bougie positif. Diagnostic: Psoriasis vulgaire.",
        "gold": ["EA90.0"],
        "notes": "French consult: Plaque psoriasis with silvery scaling plaques on extensor surfaces.",
    },
    {
        "id": "heldout-050",
        "chapter": "14",
        "language": "es",
        "style": "short_consult",
        "text": "Adolescente de 16 años presenta comedones abiertos y cerrados, pápulas y pústulas inflamatorias en frente, mejillas y mentón. Diagnóstico: Acné vulgar.",
        "gold": ["ED80.0"],
        "notes": "Spanish consult: Comedonal acne in an adolescent.",
    },

    # Chapter 15: Musculoskeletal
    {
        "id": "heldout-051",
        "chapter": "15",
        "language": "en",
        "style": "typed_consult",
        "text": "66F c/o bilateral knee pain x3 years, right > left, worse with weight-bearing and descending stairs, morning stiffness lasting 15 minutes. O/E: bony enlargement of joint margins, joint crepitus on passive flexion, no warmth or effusion. X-ray: medial joint space narrowing and osteophytes. Dx: Osteoarthritis of knee.",
        "gold": ["FA01.Z"],
        "notes": "Bilateral knee osteoarthritis with classic mechanical pain and radiographic joint space narrowing.",
    },
    {
        "id": "heldout-052",
        "chapter": "15",
        "language": "sw",
        "style": "typed_consult",
        "text": "Mwanamume wa miaka 42, dereva wa boda boda, anakuja na maumivu makali ya mgongo wa chini (low back pain) yaliyoanza ghafla baada ya kuinua mzigo mzito. Maumivu hayashuki miguuni, hakuna ganzi wala shida ya kukojoa. Utambuzi: Maumivu ya kiuno na mgongo (Low back pain).",
        "gold": ["ME84.2Z"],
        "notes": "Swahili consult: Acute mechanical low back pain following heavy lifting.",
    },
    {
        "id": "heldout-053",
        "chapter": "15",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 48 ans, polyarthrite bilatérale et symétrique touchant les poignets et les articulations métacarpophalangiennes depuis 4 mois, avec raideur matinale supérieure à une heure. Facteur rhumatoïde et anticorps anti-CCP positifs. Diagnostic: Polyarthrite rhumatoïde.",
        "gold": ["FA20.0"],
        "notes": "French consult: Seropositive rheumatoid arthritis with symmetrical small-joint involvement.",
    },

    # Chapter 16: Genitourinary
    {
        "id": "heldout-054",
        "chapter": "16",
        "language": "en",
        "style": "typed_consult",
        "text": "26F presents with acute dysuria, urinary frequency and suprapubic discomfort x2/7. Afebrile, no flank pain or costovertebral angle tenderness. Urine dipstick: leucocyte esterase positive, nitrites positive. UPT negative. Dx: Acute infectious cystitis.",
        "gold": ["GC00.1"],
        "notes": "Infectious cystitis in young female.",
    },
    {
        "id": "heldout-055",
        "chapter": "16",
        "language": "sw",
        "style": "short_consult",
        "text": "Mwanamke wa miaka 23 analalamika maumivu wakati wa kukojoa (dysuria), kukojoa mara kwa mara na maumivu sehemu ya chini ya tumbo kwa siku 3. Hana homa wala maumivu ya mbavu. Kipimo cha mkojo kimeonyesha maambukizi ya njia ya mkojo (UTI). Utambuzi: Acute cystitis.",
        "gold": ["GC00.1"],
        "notes": "Swahili consult: Acute lower urinary tract infection / infectious cystitis.",
    },
    {
        "id": "heldout-056",
        "chapter": "16",
        "language": "es",
        "style": "short_consult",
        "text": "Varón de 42 años acude a urgencias por dolor cólico insoportable en fosa lumbar derecha irradiado a región inguinal y testículo, con hematuria microscópica. Ecografía revela litiasis de 5 mm en el riñón derecho. Diagnóstico: Cálculo renal (Cálculo del riñón).",
        "gold": ["GB70.0Z"],
        "notes": "Spanish consult: Calculus of kidney presenting with acute renal colic.",
    },
    {
        "id": "heldout-057",
        "chapter": "16",
        "language": "pt",
        "style": "short_consult",
        "text": "Mulher de 28 anos com disúria intensa, polaciúria e dor suprapúbica há 2 dias. Sem febre ou dor lombar. Exame de urina com leucocitúria e nitrito positivo. Diagnóstico: Cistite infecciosa aguda.",
        "gold": ["GC00.1"],
        "notes": "Portuguese consult: Acute uncomplicated infectious cystitis.",
    },

    # Chapter 17: Sexual Health
    {
        "id": "heldout-058",
        "chapter": "17",
        "language": "en",
        "style": "typed_consult",
        "text": "56M c/o persistent inability to attain and maintain penile erection sufficient for satisfactory sexual performance x8 months. Known hypertensive, no acute cardiovascular symptoms. Normal testosterone levels. Dx: Male erectile dysfunction.",
        "gold": ["HA01.1Z"],
        "notes": "Male erectile dysfunction in older male with cardiovascular risk factors.",
    },
    {
        "id": "heldout-059",
        "chapter": "17",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 32 ans consulte pour des douleurs génitales persistantes et récurrentes lors des rapports sexuels (dyspareunie) depuis 6 mois, sans lésion cutanéo-muqueuse visible. Diagnostic: Troubles douloureux sexuels / dyspareunie.",
        "gold": ["HA20.Z"],
        "notes": "French consult: Sexual pain-penetration disorder / dyspareunia.",
    },

    # Chapter 18: Pregnancy, Childbirth, Puerperium
    {
        "id": "heldout-060",
        "chapter": "18",
        "language": "en",
        "style": "typed_consult",
        "text": "28F G2P1 at 26 weeks gestation routine antenatal checkup. Oral glucose tolerance test (75g OGTT): fasting 5.8 mmol/L, 2-hour 9.4 mmol/L (both elevated). Pre-pregnancy BMI normal, no prior diabetes history. Normal fetal growth. Dx: Gestational diabetes mellitus.",
        "gold": ["JA62.Z"],
        "notes": "Gestational diabetes mellitus diagnosed on routine 26-week screening OGTT.",
    },
    {
        "id": "heldout-061",
        "chapter": "18",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo mama, mimba ina umri gani sasa? Mgonjwa: Daktari nina mimba ya wiki nane, lakini tangu wiki mbili zilizopita nina kichefuchefu na kutapika mara 6 hadi 8 kwa siku. Siwezi hata kubakiza maji, na nimepungua uzito. Daktari: Mapigo ya moyo 98, shinikizo 100/60, ngozi imekauka kwa kukosa maji. Utambuzi: Hyperemesis gravidarum.",
        "gold": ["JA60.0"],
        "notes": "Swahili consult: Hyperemesis gravidarum in early pregnancy with dehydration.",
    },
    {
        "id": "heldout-062",
        "chapter": "18",
        "language": "es",
        "style": "short_consult",
        "text": "Primigesta de 34 semanas presenta cifras tensionales de 150/95 mmHg en dos tomas separadas y proteinuria de 2+ en tira reactiva de orina. Edema maleolar moderado. Sin cefalea ni escotomas. Diagnóstico: Preeclampsia.",
        "gold": ["JA24.Z"],
        "notes": "Spanish consult: Preeclampsia in third trimester with new hypertension and proteinuria.",
    },

    # Chapter 19: Perinatal Conditions
    {
        "id": "heldout-063",
        "chapter": "19",
        "language": "en",
        "style": "typed_consult",
        "text": "Newborn infant born to a mother with severe pre-eclampsia and chronic hypertension. Infant small for gestational age, monitoring for perinatal complications. Dx: Newborn affected by maternal hypertensive disorders.",
        "gold": ["KA00.0"],
        "notes": "Fetus or newborn affected by maternal hypertensive disorders.",
    },
    {
        "id": "heldout-064",
        "chapter": "19",
        "language": "fr",
        "style": "short_consult",
        "text": "Nouveau-né de mère présentant une hypertension artérielle gravidique sévère au cours de la grossesse, nécessitant une surveillance néonatale étroite. Diagnostic: Nouveau-né affecté par des troubles hypertensifs maternels.",
        "gold": ["KA00.0"],
        "notes": "French consult: Newborn affected by maternal hypertensive disorders.",
    },

    # Chapter 20: Developmental Anomalies
    {
        "id": "heldout-065",
        "chapter": "20",
        "language": "en",
        "style": "typed_consult",
        "text": "Newborn infant examined at birth: flat facial profile, upslanting palpebral fissures, single transverse palmar crease bilateral, low muscle tone (hypotonia). Karyotype confirms trisomy 21. Dx: Down syndrome.",
        "gold": ["LD40.0"],
        "notes": "Complete trisomy 21 / Down syndrome with classic dysmorphic features.",
    },
    {
        "id": "heldout-066",
        "chapter": "20",
        "language": "es",
        "style": "short_consult",
        "text": "Recién nacido masculino presenta hendidura congénita unilateral del labio superior que afecta el bermellón y el piso nasal derecho, sin fisura palatina. Diagnóstico: Labio leporino unilateral.",
        "gold": ["LA40.0"],
        "notes": "Spanish consult: Unilateral cleft lip without cleft palate.",
    },

    # Chapter 21: Symptoms, Signs, Clinical Findings
    {
        "id": "heldout-067",
        "chapter": "21",
        "language": "en",
        "style": "typed_consult",
        "text": "37M presents with fever documented daily up to 39.1C x3 weeks without localized symptoms. Extensive workup negative: malaria smears neg x3, blood cultures x3 no growth, CXR clear, abdominal ultrasound normal. Dx: Fever of unknown origin.",
        "gold": ["MG26"],
        "notes": "Fever of unknown origin (PUO) after negative investigations.",
    },
    {
        "id": "heldout-068",
        "chapter": "21",
        "language": "sw",
        "style": "short_consult",
        "text": "Mgonjwa mwenye umri wa miaka 25 analetwa baada ya kuzimia ghafla (syncope) alipokuwa amesimama kwenye foleni juani kwa muda mrefu. Alipata kizunguzungu kabla ya kuanguka, kisha alizinduka haraka baada ya dakika moja. Uchunguzi wa moyo na ECG ni kawaida. Utambuzi: Syncope.",
        "gold": ["MG45.Z"],
        "notes": "Swahili consult: Syncope or collapse, unspecified.",
    },
    {
        "id": "heldout-069",
        "chapter": "21",
        "language": "zh",
        "style": "short_consult",
        "text": "患者女性，50岁。近两个月无明显诱因出现持续性慢性咳嗽，干咳为主，无发热及胸痛。胸片及肺功能检查未见明显异常。诊断：不明原因慢性咳嗽。",
        "gold": ["MD11.0"],
        "notes": "Chinese consult: Chronic unexplained cough.",
    },

    # Chapter 22: Injury, Poisoning & External Consequences
    {
        "id": "heldout-070",
        "chapter": "22",
        "language": "en",
        "style": "typed_consult",
        "text": "62F tripped and fell on outstretched hand. Severe pain and 'dinner-fork' deformity of right wrist. X-ray confirms fracture of lower end of radius with dorsal displacement. Dx: Distal radius fracture (lower end of radius).",
        "gold": ["NC32.5Z"],
        "notes": "Fracture of lower end of radius, unspecified subtype.",
    },
    {
        "id": "heldout-071",
        "chapter": "22",
        "language": "en",
        "style": "typed_consult",
        "text": "34M landed awkwardly jumping down stairs. Severe swelling and deformity of the right ankle with tenderness over both medial and lateral malleoli. Radiograph confirms bimalleolar fracture of ankle. Dx: Bimalleolar fracture of ankle.",
        "gold": ["NC92.76"],
        "notes": "Bimalleolar fracture of ankle.",
    },
    {
        "id": "heldout-072",
        "chapter": "22",
        "language": "fr",
        "style": "short_consult",
        "text": "Patient de 50 ans présentant une toxicité médicamenteuse aiguë suite à la prise accidentelle d'une double dose de son traitement au long cours, nécessitant une surveillance clinique. Diagnostic: Toxicité médicamenteuse.",
        "gold": ["NE88"],
        "notes": "French consult: Drug toxicity associated with harm in medical care.",
    },

    # Chapter 23: External Causes of Morbidity
    {
        "id": "heldout-073",
        "chapter": "23",
        "language": "en",
        "style": "short_consult",
        "text": "Pedestrian struck by a passenger car while crossing a designated road intersection. Multiple contusions and abrasions. Cause: Unintentional transport event injuring a pedestrian.",
        "gold": ["PA00"],
        "notes": "External cause: pedestrian injured in transport event.",
    },

    # Chapter 24: Factors Influencing Health Status
    {
        "id": "heldout-074",
        "chapter": "24",
        "language": "en",
        "style": "typed_consult",
        "text": "40M attends clinic for routine annual medical checkup required by employer. Completely asymptomatic. Physical examination unremarkable. Fasting glucose and lipid panel normal. Dx: Routine adult medical examination.",
        "gold": ["QA00.0"],
        "notes": "Routine general health examination in healthy adult.",
    },
    {
        "id": "heldout-075",
        "chapter": "24",
        "language": "sw",
        "style": "short_consult",
        "text": "Mama anamleta mtoto mchanga wa wiki 6 kwa ajili ya chanjo za kawaida za utotoni kulingana na ratiba ya kitaifa (Pentavalent, Polio, Rotavirus). Mtoto yuko na afya njema kabisa. Utambuzi: Huduma ya chanjo ya watoto (Routine child immunization).",
        "gold": ["QC00.0"],
        "notes": "Swahili consult: Routine childhood vaccination / immunization visit.",
    },
    {
        "id": "heldout-076",
        "chapter": "24",
        "language": "es",
        "style": "short_consult",
        "text": "Varón de 45 años sin síntomas acude para chequeo preventivo general de salud anual solicitado por su empresa. Examen físico completamente normal. Diagnóstico: Examen médico general de adulto.",
        "gold": ["QA00.0"],
        "notes": "Spanish consult: General medical examination in asymptomatic adult.",
    },

    # Chapter 25: Codes for Special Purposes
    {
        "id": "heldout-077",
        "chapter": "25",
        "language": "en",
        "style": "typed_consult",
        "text": "34F presents with fever 38.5C, anosmia (loss of smell), ageusia (loss of taste), dry cough and fatigue x3/7. Rapid antigen test and RT-PCR positive for SARS-CoV-2. Pulse oximetry 98% on room air. Dx: COVID-19, virus identified.",
        "gold": ["RA01.0"],
        "notes": "Acute laboratory-confirmed COVID-19 infection.",
    },
    {
        "id": "heldout-078",
        "chapter": "25",
        "language": "fr",
        "style": "short_consult",
        "text": "Patiente de 29 ans présentant une anosmie brutale, fébricule et céphalées depuis 2 jours. Test PCR SARS-CoV-2 positif. Diagnostic: COVID-19 avec virus identifié.",
        "gold": ["RA01.0"],
        "notes": "French consult: Confirmed COVID-19 with sudden anosmia and positive RT-PCR.",
    },

    # Chapter 26: Traditional Medicine
    {
        "id": "heldout-079",
        "chapter": "26",
        "language": "zh",
        "style": "short_consult",
        "text": "患者主诉两胁胀痛，胸闷叹息，情志抑郁，每因情志不舒而加重。舌苔薄白，脉弦。中医辨证：肝气郁结证。",
        "gold": ["SF51"],
        "notes": "Traditional Chinese medicine liver qi stagnation pattern.",
    },
    {
        "id": "heldout-080",
        "chapter": "26",
        "language": "zh",
        "style": "short_consult",
        "text": "患者面色萎黄，头晕目眩，心悸失眠，神疲乏力，纳呆食少。舌淡苔白，脉细弱。辨证诊断：气血两虚证。",
        "gold": ["SF00"],
        "notes": "Traditional Chinese medicine dual deficiency of qi and blood.",
    },

    # Traps & Multi-lingual Challenge Notes (Negation, Ruled out, History, Comorbidity)
    {
        "id": "heldout-081",
        "chapter": "01",
        "language": "en",
        "style": "typed_consult",
        "text": "32M ?malaria - high fever, shaking chills and body aches x3/7. Blood slide negative for malaria parasites (BS for MPS NEG x2), mRDT negative. Widal test positive 1:320 for S. typhi, blood culture confirms typhoid. Dx: Typhoid fever.",
        "gold": ["1A07.Z"],
        "notes": "Negation trap: Malaria was query differential but ruled out by negative tests; true diagnosis is Typhoid.",
    },
    {
        "id": "heldout-082",
        "chapter": "11",
        "language": "sw",
        "style": "typed_consult",
        "text": "Mama wa miaka 55, anajulikana kuwa na ugonjwa wa sukari (T2DM) anayetumia metformin na sukari yake imedhibitiwa vizuri. Leo amekuja kwa ajili ya shinikizo la damu lililopanda (BP 165/102). Daktari anamuanzishia dawa za Amlodipine kwa shinikizo la damu. Utambuzi: Shinikizo la juu la damu (Essential hypertension).",
        "gold": ["BA00.Z"],
        "notes": "Comorbidity trap in Swahili: Known well-controlled diabetes is background; primary acute focus is new essential hypertension.",
    },
    {
        "id": "heldout-083",
        "chapter": "12",
        "language": "en",
        "style": "typed_consult",
        "text": "48F wrongly treated twice for bronchial asthma with inhalers without any relief. Repeat high-resolution CT chest reveals bronchiectasis with daily copious foul purulent sputum and recurrent haemoptysis. Asthma excluded. Dx: Bronchiectasis.",
        "gold": ["CA24"],
        "notes": "Previous misdiagnosis trap: Asthma was wrongly treated, true diagnosis is Bronchiectasis.",
    },
    {
        "id": "heldout-084",
        "chapter": "16",
        "language": "es",
        "style": "typed_consult",
        "text": "Paciente mujer de 35 años, conocida asmática en tratamiento con salbutamol a demanda (asfixia descartada hoy). Consulta por disuria intensa, polaquiuria y tenesmo vesical de 3 días de evolución. Sedimento urinario patológico. Dx: Cistitis infecciosa aguda.",
        "gold": ["GC00.1"],
        "notes": "Comorbidity and negation trap in Spanish: Known asthma is background/ruled out today; active condition is acute infectious cystitis.",
    },
    {
        "id": "heldout-085",
        "chapter": "01",
        "language": "hi",
        "style": "short_consult",
        "text": "30 वर्षीय पुरुष को 4 दिन से कंपकंपी के साथ तेज बुखार। मलेरिया की जांच (RDT) नेगेटिव आई। डेंगू NS1 एंटीजन टेस्ट पॉजिटिव निकला। डायग्नोसिस: डेंगू बुखार।",
        "gold": ["1D20"],
        "notes": "Negation trap in Hindi: Malaria tested negative; dengue fever confirmed positive.",
    },
]


def main():
    print(f"Validating {len(HELD_OUT_DATA)} held-out notes...")
    invalid_codes = []
    chapter_counts = {}
    lang_counts = {}

    for note in HELD_OUT_DATA:
        nid = note["id"]
        ch = note["chapter"]
        lang = note["language"]
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

        for g in note["gold"]:
            if not validate_gold_code(g):
                invalid_codes.append((nid, g, mms_by_code.get(g, {}).get("leaf", "not_found")))

    if invalid_codes:
        print(f"Error: Found {len(invalid_codes)} invalid gold codes in held-out set:")
        for nid, code, leaf in invalid_codes:
            print(f"  [{nid}] Code {code} - leaf status: {leaf}")
        sys.exit(1)

    print("All gold codes are 100% valid MMS leaf codes!")
    print(f"Total notes: {len(HELD_OUT_DATA)}")
    print(f"Chapters covered: {len(chapter_counts)}/26 chapters")
    print(f"Languages covered: {len(lang_counts)} ({list(lang_counts.keys())})")

    out_path = "data/held_out_notes.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for note in HELD_OUT_DATA:
            f.write(json.dumps(note, ensure_ascii=False) + "\n")

    print(f"Saved held-out evaluation suite to {out_path}.")


if __name__ == "__main__":
    main()
