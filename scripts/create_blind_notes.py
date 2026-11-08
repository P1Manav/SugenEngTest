#!/usr/bin/env python3
"""Author independent BLIND evaluation dataset (data/blind_notes.jsonl).

Criteria:
- >= 60 clinical notes (65 notes)
- Multi-chapter coverage across all 26 ICD-11 chapters
- Multilingual: English (en), Swahili (sw), Spanish (es), French (fr), Portuguese (pt),
  Arabic (ar), Chinese (zh), Hindi (hi), German (de)
- All 3 clinical styles: typed_consult (SOAP), short_consult, ambient_transcript
- Realistic distractors: negation, family history, ruled out differentials, comorbidities
- Strictly validated against data/mms-2026-01.json (must be valid leaf codes)
- Completely isolated from dev rule building!
"""

import json
import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "data", "mms-2026-01.json"), "r", encoding="utf-8") as f:
    mms_by_code = {x["code"]: x for x in json.load(f) if "code" in x}


def is_valid_leaf(code: str) -> bool:
    info = mms_by_code.get(code)
    if not info:
        return False
    return bool(info.get("leaf", False))


BLIND_DATA = [
    # Chapter 01: Infections
    {
        "id": "blind-001",
        "chapter": "01",
        "language": "en",
        "style": "typed_consult",
        "text": "34M presents with low-grade fever, malaise, and non-painful indurated genital ulcer (chancre) x2 weeks. VDRL titer 1:64, FTA-ABS strongly positive. HIV test non-reactive. Impression: Primary syphilis.",
        "gold": ["1A60.0"],
        "notes": "Primary syphilis with positive serology and classic chancre; HIV ruled out.",
    },
    {
        "id": "blind-002",
        "chapter": "01",
        "language": "es",
        "style": "short_consult",
        "text": "Niña de 4 años, fiebre de 38.8C y erupción vesicular pruriginosa en tronco y cara con aspecto de 'gotas de rocío'. No antecedentes de vacunación. Dx: Varicela sin complicaciones.",
        "gold": ["1E90.Z"],
        "notes": "Spanish pediatric consult: Uncomplicated varicela (chickenpox).",
    },
    {
        "id": "blind-003",
        "chapter": "01",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo mzee, una shida gani leo? Mgonjwa: Daktari ninaharisha sana tangu juzi, kinyesi kina kamasi na matone ya damu, na tumbo linanikata sana chini. Daktari: Je umepima choo? Mgonjwa: Ndiyo, maabara wamesema ni Entamoeba histolytica. Daktari: Hiyo ni ugonjwa wa kichocho cha amiba (amoebic dysentery). Tutakupa metronidazole mara moja.",
        "gold": ["1A36.00"],
        "notes": "Swahili ambient consult: Acute amoebic dysentery confirmed on stool microscopy.",
    },

    # Chapter 02: Neoplasms
    {
        "id": "blind-004",
        "chapter": "02",
        "language": "fr",
        "style": "typed_consult",
        "text": "Femme de 58 ans. Mammographie de dépistage montrant une masse spiculée ACR5 du quadrant supéro-externe gauche de 22 mm. Biopsie au trocart: carcinome canalaire infiltrant de grade II, RE+, RP+, HER2-. Absence d'atteinte ganglionnaire clinique. Diagnostic: Cancer infiltrant du sein gauche.",
        "gold": ["2C61.0"],
        "notes": "French typed consult: Invasive breast carcinoma of left breast.",
    },
    {
        "id": "blind-005",
        "chapter": "02",
        "language": "en",
        "style": "short_consult",
        "text": "67M smoker (40 pack-years) presents with persistent hemoptysis, weight loss, and chronic cough. CT chest shows a 4 cm centrally located mass in the right main bronchus. Bronchoscopic biopsy confirms squamous cell carcinoma. Assessment: Malignant neoplasm of right bronchus.",
        "gold": ["2C25.Z"],
        "notes": "Squamous cell carcinoma of bronchus and lung.",
    },
    {
        "id": "blind-006",
        "chapter": "02",
        "language": "de",
        "style": "typed_consult",
        "text": "52-jähriger Patient mit neu aufgetretenem, asymmetrischem, dunkel pigmentiertem Hautfleck am Rücken (Durchmesser 8 mm, unregelmäßige Ränder). Exzisionsbiopsie: Malignes Melanom, Breslow-Tiefe 1,2 mm, Clark-Level III. Diagnose: Malignes Melanom der Haut.",
        "gold": ["2C30.Z"],
        "notes": "German typed consult: Malignant melanoma of skin, unspecified site.",
    },

    # Chapter 03: Diseases of the blood
    {
        "id": "blind-007",
        "chapter": "03",
        "language": "en",
        "style": "typed_consult",
        "text": "22F c/o severe fatigue, pallor, and pica x3 months. Menorrhagia noted. Labs: Hb 7.2 g/dL, MCV 68 fL, serum ferritin 4 ng/mL. Stool occult blood negative. Diagnosis: Iron deficiency anaemia due to blood loss.",
        "gold": ["3A00.0Z"],
        "notes": "Iron deficiency anemia due to chronic menstrual blood loss.",
    },
    {
        "id": "blind-008",
        "chapter": "03",
        "language": "sw",
        "style": "short_consult",
        "text": "Kijana wa miaka 16 mwenye historia ya ugonjwa wa seli mundu (HbSS) anakuja na maumivu makali ya mifupa na kifua tangu masaa 6. O2 sat 94%, joto 38.2C. Dx: Sickle cell anaemia with acute painful crisis.",
        "gold": ["3A51.0"],
        "notes": "Swahili consult: Sickle cell anaemia with painful vaso-occlusive crisis.",
    },

    # Chapter 04: Diseases of the immune system
    {
        "id": "blind-009",
        "chapter": "04",
        "language": "es",
        "style": "typed_consult",
        "text": "Mujer de 32 años. Presenta eritema malar en alas de mariposa, poliartritis no erosiva en manos y fotosensibilidad. ANA 1:640 moteado, anti-dsDNA positivo alto, C3 y C4 disminuidos. Sin proteinuria. Impresión: Lupus eritematoso sistémico.",
        "gold": ["4A40.0Z"],
        "notes": "Spanish typed consult: Systemic lupus erythematosus without nephritis.",
    },
    {
        "id": "blind-010",
        "chapter": "04",
        "language": "fr",
        "style": "short_consult",
        "text": "Homme de 45 ans avec syndrome sec oculaire et buccal sévère depuis 1 an. Test de Schirmer positif (<5 mm), anti-SSA/Ro et anti-SSB/La fortement positifs. Biopsie salivaire: sialadénite lymphocytaire focale (score de Chisholm 4). Diagnostic: Syndrome de Sjögren primaire.",
        "gold": ["4A43.20"],
        "notes": "French consult: Primary Sjogren syndrome.",
    },

    # Chapter 05: Endocrine, nutritional or metabolic
    {
        "id": "blind-011",
        "chapter": "05",
        "language": "en",
        "style": "typed_consult",
        "text": "48M with obesity (BMI 34) presents for routine review. Fasting plasma glucose 8.4 mmol/L, HbA1c 8.2%. Urine microalbumin/creatinine ratio normal. Assessment: Type 2 diabetes mellitus without complications.",
        "gold": ["5A11"],
        "notes": "Uncomplicated type 2 diabetes mellitus.",
    },
    {
        "id": "blind-012",
        "chapter": "05",
        "language": "zh",
        "style": "short_consult",
        "text": "患者女性，42岁。畏寒、乏力、脱发、体重增加4个月。查体甲状腺无肿大。实验室检查：血清TSH 18.5 mIU/L升高，游离T4 6.2 pmol/L降低，抗TPO抗体阳性。诊断：原发性甲状腺功能减退症。",
        "gold": ["5A00.2Z"],
        "notes": "Chinese consult: Primary hypothyroidism (autoimmune/Hashimoto).",
    },
    {
        "id": "blind-013",
        "chapter": "05",
        "language": "en",
        "style": "ambient_transcript",
        "text": "Doctor: Hello, what brings you to the clinic? Patient: Doctor, I feel so shaky, sweaty, and confused whenever I delay my meals. My home fingerstick read 2.4 mmol/L this morning after my morning insulin injection, and it went away when I drank orange juice. Doctor: That is an episode of hypoglycaemia in the context of your treated diabetes mellitus without coma.",
        "gold": ["5A21.0"],
        "notes": "Ambient consult: Documented hypoglycaemia in diabetes mellitus without coma.",
    },

    # Chapter 06: Mental, behavioural or neurodevelopmental
    {
        "id": "blind-014",
        "chapter": "06",
        "language": "en",
        "style": "typed_consult",
        "text": "36F reports depressed mood, anhedonia, early morning awakening, and psychomotor slowing x6 weeks following job loss. No suicidal ideation, no psychotic features, no past manic episodes. PHQ-9 score 16 (moderate). Assessment: Single episode depressive disorder, moderate.",
        "gold": ["6A70.1"],
        "notes": "Single episode depressive disorder, moderate severity without psychotic symptoms.",
    },
    {
        "id": "blind-015",
        "chapter": "06",
        "language": "pt",
        "style": "short_consult",
        "text": "Homem de 29 anos com episódios súbitos e recorrentes de palpitações, sensação de sufocamento, tremores e medo intenso de morrer sem causa aparente, com duração de 20 minutos. ECG normal. Dx: Transtorno de pânico.",
        "gold": ["6B01"],
        "notes": "Portuguese consult: Panic disorder with recurrent panic attacks.",
    },
    {
        "id": "blind-016",
        "chapter": "06",
        "language": "es",
        "style": "typed_consult",
        "text": "Varón de 40 años refiere preocupación excesiva y persistente sobre salud, trabajo y finanzas durante más de 8 meses, acompañada de tensión muscular, irritabilidad e insomnio de conciliación. Impresión: Trastorno de ansiedad generalizada.",
        "gold": ["6B00"],
        "notes": "Spanish consult: Generalized anxiety disorder.",
    },

    # Chapter 07: Sleep-wake disorders
    {
        "id": "blind-017",
        "chapter": "07",
        "language": "en",
        "style": "typed_consult",
        "text": "55M obese male (BMI 36, neck circumference 44 cm) referred for loud disruptive snoring, witnessed apneas, and daytime somnolence (Epworth Sleepiness Scale 15). Overnight polysomnography shows Apnea-Hypopnea Index (AHI) of 28 events/hr. Diagnosis: Obstructive sleep apnoea.",
        "gold": ["7A40.0"],
        "notes": "Obstructive sleep apnoea confirmed on polysomnography.",
    },
    {
        "id": "blind-018",
        "chapter": "07",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 48 ans se plaignant de difficultés d'endormissement persistantes (latence >60 min) et de réveils nocturnes fréquents 4 nuits par semaine depuis 6 mois, avec retentissement diurne. Diagnostic: Insomnie chronique.",
        "gold": ["7A00"],
        "notes": "French consult: Chronic insomnia disorder.",
    },

    # Chapter 08: Diseases of the nervous system
    {
        "id": "blind-019",
        "chapter": "08",
        "language": "en",
        "style": "typed_consult",
        "text": "30F presents with recurrent episodic unilateral throbbing headaches associated with photophobia, nausea, and vomiting lasting 12-24 hours. No preceding visual scintillations or neurological aura. Dx: Migraine without aura.",
        "gold": ["8A80.0"],
        "notes": "Migraine without aura.",
    },
    {
        "id": "blind-020",
        "chapter": "08",
        "language": "de",
        "style": "short_consult",
        "text": "71-jähriger Mann mit seit 1 Jahr progredientem asymmetrischem Ruhetremor der rechten Hand, Zahnrad-Rigor und Bradykinese. Gutes Ansprechen auf L-Dopa. Diagnose: Idiopathisches Parkinsonsyndrom (Sporadische Parkinson-Krankheit).",
        "gold": ["8A00.00"],
        "notes": "German consult: Sporadic Parkinson disease.",
    },
    {
        "id": "blind-021",
        "chapter": "08",
        "language": "es",
        "style": "short_consult",
        "text": "Paciente de 44 años con parestesias nocturnas y dolor en los tres primeros dedos de la mano derecha. Signo de Tinel y maniobra de Phalen positivos en muñeca derecha. EMG confirma atrapamiento del nervio mediano. Dx: Síndrome del túnel carpiano.",
        "gold": ["8C10.0"],
        "notes": "Spanish consult: Carpal tunnel syndrome.",
    },

    # Chapter 09: Diseases of the visual system
    {
        "id": "blind-022",
        "chapter": "09",
        "language": "en",
        "style": "short_consult",
        "text": "74F reports gradual bilateral painless reduction in visual acuity and glare while driving at night. Slit-lamp biomicroscopy reveals dense nuclear opacification of both lenses. Fundus exam clear. Dx: Age-related nuclear cataract.",
        "gold": ["9B10.00"],
        "notes": "Age-related nuclear cataract of lens.",
    },
    {
        "id": "blind-023",
        "chapter": "09",
        "language": "es",
        "style": "typed_consult",
        "text": "Varón de 62 años en revisión oftalmológica. Presión intraocular de 26 mmHg en ojo derecho y 25 mmHg en izquierdo. Gonioscopia: ángulo abierto grado IV bilateral. Campimetría muestra escotoma arcuato superior. Excavación papilar 0.7. Dx: Glaucoma primario de ángulo abierto.",
        "gold": ["9C61.0Z"],
        "notes": "Spanish consult: Primary open-angle glaucoma, unspecified.",
    },

    # Chapter 10: Diseases of the ear
    {
        "id": "blind-024",
        "chapter": "10",
        "language": "fr",
        "style": "short_consult",
        "text": "Enfant de 5 ans avec otalgie aiguë droite intense, fièvre à 38.5C et rhinopharyngite récente. Otoscopie: membrane tympanique droite bombante, hyperhémiée avec disparition des reliefs osseux. Diagnostic: Otite moyenne aiguë suppurée.",
        "gold": ["AA90"],
        "notes": "French consult: Acute suppurative otitis media.",
    },
    {
        "id": "blind-025",
        "chapter": "10",
        "language": "en",
        "style": "typed_consult",
        "text": "45F c/o brief episodes of intense rotational vertigo lasting 30-60 seconds provoked by rolling over in bed or looking up. Dix-Hallpike maneuver produces upward and torsional nystagmus with 5 second latency in right ear. Impression: Benign paroxysmal positional vertigo.",
        "gold": ["AB31.0"],
        "notes": "Benign paroxysmal positional vertigo (BPPV).",
    },

    # Chapter 11: Diseases of the circulatory system
    {
        "id": "blind-026",
        "chapter": "11",
        "language": "en",
        "style": "typed_consult",
        "text": "56M asymptomatic presentation for insurance physical. Blood pressure today 158/98 mmHg, repeat sitting 154/96 mmHg. 24-hr ambulatory BP monitoring confirms mean daytime BP 148/92 mmHg. Secondary workup negative. Dx: Essential hypertension.",
        "gold": ["BA00.Z"],
        "notes": "Uncomplicated essential hypertension.",
    },
    {
        "id": "blind-027",
        "chapter": "11",
        "language": "es",
        "style": "short_consult",
        "text": "Varón de 60 años con dolor torácico retroesternal opresivo irradiado a mandíbula de 45 minutos de duración, sudoración fría y disnea. ECG: elevación del segmento ST de 3 mm en derivaciones V1-V4. Troponina I marcadamente elevada (8.5 ng/mL). Dx: Infarto agudo de miocardio con elevación del ST de pared anterior.",
        "gold": ["BA41.0"],
        "notes": "Spanish consult: Acute ST-elevation myocardial infarction of anterior wall.",
    },
    {
        "id": "blind-028",
        "chapter": "11",
        "language": "pt",
        "style": "typed_consult",
        "text": "Mulher de 68 anos com palpitações irregulares e cansaço aos médios esforços. ECG demonstra ritmo irregularmente irregular com ausência de ondas P e frequência ventricular de 115 bpm. Ecocardiograma: átrio esquerdo dilatado, sem trombos. Dx: Fibrilação atrial.",
        "gold": ["BC81.3Z"],
        "notes": "Portuguese consult: Atrial fibrillation, unspecified.",
    },

    # Chapter 12: Diseases of the respiratory system
    {
        "id": "blind-029",
        "chapter": "12",
        "language": "en",
        "style": "typed_consult",
        "text": "19F college student with history of eczema presents with episodic expiratory wheezing, dry nocturnal cough, and chest tightness triggered by cold air and exercise. Spirometry shows FEV1/FVC 68% with 16% post-bronchodilator reversibility. Assessment: Allergic asthma.",
        "gold": ["CA23.00"],
        "notes": "Allergic asthma with demonstrated reversibility.",
    },
    {
        "id": "blind-030",
        "chapter": "12",
        "language": "fr",
        "style": "short_consult",
        "text": "Homme de 64 ans, tabagique sevré (45 PA), dyspnée d'effort progressive stade mMRC 3 et toux productive chronique matinale depuis 3 ans. EFR: VEMS/CVF = 58% post-bronchodilatateur, non réversible. Radio pulmonaire: emphysème centro-lobulaire. Diagnostic: Bronchopneumopathie chronique obstructive (BPCO).",
        "gold": ["CA22.0"],
        "notes": "French consult: Chronic obstructive pulmonary disease (COPD).",
    },
    {
        "id": "blind-031",
        "chapter": "12",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo kaka, kikohozi kimeanza lini? Mgonjwa: Daktari ninakohoa sana tangu siku tano zilizopita, nikitoa makohozi ya njano na homa kidogo. Kifua hakina maumivu ya ndani wala kubana. Daktari: Mapafu yako yanatoa sauti ya rales ndogo, lakini hakuna dalili za nimonia. Huu ni ugonjwa wa mkamba mkali (acute bronchitis).",
        "gold": ["CA42.0"],
        "notes": "Swahili ambient consult: Acute bronchitis without pneumonia.",
    },

    # Chapter 13: Diseases of the digestive system
    {
        "id": "blind-032",
        "chapter": "13",
        "language": "es",
        "style": "typed_consult",
        "text": "Mujer de 40 años consulta por pirosis retroesternal y regurgitación ácida postprandial de 6 meses de evolución. Endoscopia digestiva alta muestra mucosa esofágica normal sin erosiones (Grado 0 de Los Ángeles). pH-metría de 24 horas positiva para reflujo patológico. Dx: Enfermedad por reflujo gastroesofágico no erosiva.",
        "gold": ["DA22.0"],
        "notes": "Spanish typed consult: Non-erosive gastroesophageal reflux disease (NERD).",
    },
    {
        "id": "blind-033",
        "chapter": "13",
        "language": "en",
        "style": "short_consult",
        "text": "24M presents to ED with severe periumbilical pain migrating to the right lower quadrant x18 hours, nausea, anorexia, and low-grade fever (38.0C). Physical exam reveals tenderness at McBurney point with positive Rovsing sign. Ultrasound confirms non-compressible appendix of 9 mm with mural thickening. Impression: Acute appendicitis without perforation or abscess.",
        "gold": ["DB10.02"],
        "notes": "Acute appendicitis without perforation, abscess or peritoneal sepsis.",
    },
    {
        "id": "blind-034",
        "chapter": "13",
        "language": "zh",
        "style": "short_consult",
        "text": "患者男性，45岁。反复上腹部饥饿痛、夜间痛3年，进食后可缓解。胃镜检查：十二指肠球部前壁见一0.8cm圆形溃疡，表面覆白苔，周边充血水肿，无活动性出血。快速尿素酶试验幽门螺杆菌阳性。诊断：十二指肠溃疡，无出血或穿孔。",
        "gold": ["DA61"],
        "notes": "Chinese consult: Duodenal ulcer without haemorrhage or perforation.",
    },

    # Chapter 14: Diseases of the skin
    {
        "id": "blind-035",
        "chapter": "14",
        "language": "en",
        "style": "typed_consult",
        "text": "7-year-old boy presents with highly pruritic, erythematous, lichenified plaques in both antecubital and popliteal fossae. Mother notes personal history of allergic rhinitis. Examination shows xerosis and excoriations. Impression: Atopic eczema.",
        "gold": ["EA80.Z"],
        "notes": "Atopic eczema (atopic dermatitis), unspecified.",
    },
    {
        "id": "blind-036",
        "chapter": "14",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 35 ans présentant des plaques érythémato-squameuses bien délimitées recouvertes de squames blanchâtres nacrées sur les coudes, les genoux et le cuir chevelu. Signe de la tache de bougie positif. Absence d'atteinte articulaire. Diagnostic: Psoriasis en plaques (vulgaire).",
        "gold": ["EA90.0"],
        "notes": "French consult: Psoriasis vulgaris in plaques.",
    },

    # Chapter 15: Diseases of the musculoskeletal system
    {
        "id": "blind-037",
        "chapter": "15",
        "language": "en",
        "style": "typed_consult",
        "text": "65F with gradual onset of mechanical right knee pain exacerbated by walking and descending stairs, relieved by rest. Morning stiffness lasts 15 minutes. Knee X-ray shows medial joint space narrowing, subchondral sclerosis, and marginal osteophytes. Dx: Primary osteoarthritis of right knee.",
        "gold": ["FA01.0"],
        "notes": "Primary osteoarthritis of knee.",
    },
    {
        "id": "blind-038",
        "chapter": "15",
        "language": "es",
        "style": "short_consult",
        "text": "Mujer de 48 años con dolor e inflamación articular simétrica en articulaciones interfalángicas proximales y metacarpofalángicas de ambas manos x4 meses. Rigidez matutina de 90 minutos. Factor reumatoide positivo (120 UI/mL) y anticuerpos anti-CCP fuertemente positivos (>250 U/mL). Dx: Artritis reumatoide seropositiva.",
        "gold": ["FA20.0"],
        "notes": "Spanish consult: Seropositive rheumatoid arthritis.",
    },
    {
        "id": "blind-039",
        "chapter": "15",
        "language": "pt",
        "style": "short_consult",
        "text": "Homem de 54 anos acorda com dor excruciante, edema e calor ruborizado na primeira articulação metatarsofalângica esquerda (podagra). Ácido úrico sérico 9.2 mg/dL. Dx: Gota aguda (artrite gotosa aguda).",
        "gold": ["FA25.0"],
        "notes": "Portuguese consult: Acute gouty arthritis.",
    },

    # Chapter 16: Diseases of the genitourinary system
    {
        "id": "blind-040",
        "chapter": "16",
        "language": "en",
        "style": "typed_consult",
        "text": "28F presents with 3-day history of burning on micturition (dysuria), urinary frequency, and suprapubic discomfort. Denies flank pain, vaginal discharge, or fever. Urinalysis shows 3+ leukocyte esterase and positive nitrites. Urine culture yields Escherichia coli >10^5 CFU/mL. Assessment: Acute cystitis.",
        "gold": ["GC00.1"],
        "notes": "Acute cystitis (urinary tract infection).",
    },
    {
        "id": "blind-041",
        "chapter": "16",
        "language": "es",
        "style": "short_consult",
        "text": "Varón de 42 años con dolor tipo cólico lumbar derecho súbito e intenso que irradia a fosa ilíaca y testículo ipsilateral, acompañado de náuseas y hematuria microscópica. TC abdominopélvica sin contraste evidencia cálculo de 5 mm en el tercio distal del uréter derecho. Dx: Litiasis ureteral (cálculo de uréter).",
        "gold": ["GB70.0Z"],
        "notes": "Spanish consult: Calculus of ureter, unspecified.",
    },
    {
        "id": "blind-042",
        "chapter": "16",
        "language": "fr",
        "style": "short_consult",
        "text": "Homme de 68 ans présentant une pollakiurie nocturne (3-4 levers par nuit), une dysurie avec jet urinaire faible et une sensation de vidange incomplète depuis 1 an. Toucher rectal: prostate augmentée de volume, lisse, indolore et élastique. PSA normal à 2.1 ng/mL. Diagnostic: Hypertrophie bénigne de la prostate.",
        "gold": ["GA90"],
        "notes": "French consult: Benign prostatic hyperplasia.",
    },

    # Chapter 17: Conditions related to sexual health
    {
        "id": "blind-043",
        "chapter": "17",
        "language": "en",
        "style": "typed_consult",
        "text": "58M with hypertension and dyslipidemia presents with consistent inability to attain and maintain penile erection sufficient for satisfactory sexual performance over the past 9 months. Morning erections diminished. Impression: Male erectile dysfunction.",
        "gold": ["HA01.1Z"],
        "notes": "Male erectile dysfunction, unspecified.",
    },
    {
        "id": "blind-044",
        "chapter": "17",
        "language": "fr",
        "style": "short_consult",
        "text": "Femme de 27 ans consultant pour des douleurs génitales superficielles vives lors des tentatives de pénétration vaginale empêchant les rapports sexuels depuis 8 mois. Examen gynécologique sans anomalie organique ni infection. Diagnostic: Dyspareunie.",
        "gold": ["HA20.Z"],
        "notes": "French consult: Dyspareunia, unspecified.",
    },

    # Chapter 18: Pregnancy, childbirth or the puerperium
    {
        "id": "blind-045",
        "chapter": "18",
        "language": "en",
        "style": "typed_consult",
        "text": "26F primigravida at 34 weeks gestation presents for routine antenatal visit. Blood pressure is 146/94 mmHg (baseline was 110/70 mmHg). Repeat sitting BP 148/92 mmHg 4 hours later. Urine dipstick shows 2+ proteinuria (confirmed with protein/creatinine ratio 0.42 mg/mg). Platelets 210,000, ALT/AST normal, no headache or visual disturbance. Assessment: Pre-eclampsia without severe features.",
        "gold": ["JA24.Z"],
        "notes": "Pre-eclampsia, unspecified (mild / without severe features).",
    },
    {
        "id": "blind-046",
        "chapter": "18",
        "language": "es",
        "style": "short_consult",
        "text": "Gestante de 26 semanas, G2P1, acude para control tras test de O'Sullivan patológico (glucemia 165 mg/dL). Sobrecarga oral con 100 g de glucosa confirma diabetes gestacional con 2 valores alterados (basal 102 mg/dL, 1h 195 mg/dL). Dx: Diabetes mellitus gestacional.",
        "gold": ["JA63.2"],
        "notes": "Spanish consult: Diabetes mellitus arising in pregnancy (gestational diabetes).",
    },

    # Chapter 19: Perinatal period
    {
        "id": "blind-047",
        "chapter": "19",
        "language": "en",
        "style": "typed_consult",
        "text": "Full-term male infant born at 39 weeks via vaginal delivery to a mother with chronic essential hypertension managed on labetalol during pregnancy. Baby examination normal, birth weight 3.1 kg, Apgars 9 and 10. Pediatric admission diagnosis: Newborn affected by maternal hypertension.",
        "gold": ["KA00.0"],
        "notes": "Fetus or newborn affected by maternal hypertension.",
    },
    {
        "id": "blind-048",
        "chapter": "19",
        "language": "fr",
        "style": "short_consult",
        "text": "Nouveau-né à terme (38 SA) présentant un ictère cutanéo-muqueux débuté au 3ème jour de vie. Bilirubine totale à 190 micromol/L, bilirubine conjuguée normale. Absence d'incompatibilité ABO ou Rhésus, examen clinique par ailleurs normal. Diagnostic: Ictère physiologique du nouveau-né.",
        "gold": ["KA61.0"],
        "notes": "French consult: Physiological jaundice of newborn.",
    },

    # Chapter 20: Developmental anomalies
    {
        "id": "blind-049",
        "chapter": "20",
        "language": "en",
        "style": "typed_consult",
        "text": "Newborn female infant examined in nursery. Physical findings include upslanting palpebral fissures, epicanthic folds, flat nasal bridge, single transverse palmar crease on left hand, and generalized hypotonia. Echocardiogram reveals an endocardial cushion defect (AVSD). Chromosomal karyotype confirms Trisomy 21. Diagnosis: Down syndrome.",
        "gold": ["LD40.0"],
        "notes": "Down syndrome (Trisomy 21).",
    },
    {
        "id": "blind-050",
        "chapter": "20",
        "language": "es",
        "style": "short_consult",
        "text": "Recién nacido varón en quien la exploración física inmediata al parto revela una fisura que afecta al labio superior izquierdo y se extiende a través de todo el paladar duro y blando. Dx: Fisura labiopalatina bilateral o labio leporino con hendidura palatina.",
        "gold": ["LA40.Z"],
        "notes": "Spanish consult: Cleft lip with cleft palate, unspecified.",
    },

    # Chapter 21: Symptoms and clinical findings
    {
        "id": "blind-051",
        "chapter": "21",
        "language": "en",
        "style": "short_consult",
        "text": "72M with known prostate carcinoma presents with involuntary urinary leakage occurring precipitously following sudden uncontrollable urge to void. Post-void residual 30 mL. Urinalysis negative for infection. Assessment: Urge urinary incontinence.",
        "gold": ["MF50.21"],
        "notes": "Urge urinary incontinence.",
    },
    {
        "id": "blind-052",
        "chapter": "21",
        "language": "sw",
        "style": "ambient_transcript",
        "text": "Daktari: Hujambo dada, nini kilitokea jana? Mgonjwa: Daktari, nilikuwa nimesimama kwenye foleni ya benki kwa muda mrefu kwenye joto kali, ghafla nikasikia kizunguzungu kikali na kuona giza, kisha nikazimia chini kwa sekunde kama thelathini. Nilipozinduka nilikuwa mzima bila mshtuko wala kukojoa. Daktari: Shinikizo la damu na sukari ni ya kawaida, hii ni kuzimia kwa kawaida (vasovagal syncope and collapse).",
        "gold": ["MG45.Z"],
        "notes": "Swahili ambient consult: Syncope and collapse (vasovagal syncope).",
    },
    {
        "id": "blind-053",
        "chapter": "21",
        "language": "fr",
        "style": "short_consult",
        "text": "Patient de 50 ans consultant pour une toux sèche persistante depuis plus de 10 semaines, sans expectoration, sans dyspnée ni fièvre. Radiographie thoracique et spirométrie strictement normales. Bilan ORL et digestif négatif. Diagnostic: Toux chronique inexpliquée.",
        "gold": ["MD11.0"],
        "notes": "French consult: Unexplained chronic cough.",
    },

    # Chapter 22: Injury, poisoning or external causes
    {
        "id": "blind-054",
        "chapter": "22",
        "language": "en",
        "style": "typed_consult",
        "text": "3-year-old child brought to emergency department 2 hours after accidentally ingesting approximately 25 paracetamol (acetaminophen) 500 mg tablets from an unlatched cabinet. Serum acetaminophen concentration at 4 hours plotted above the treatment line on Rumack-Matthew nomogram. IV N-acetylcysteine started immediately. Assessment: Accidental poisoning by paracetamol.",
        "gold": ["NE88"],
        "notes": "Drug toxicity / poisoning by paracetamol.",
    },
    {
        "id": "blind-055",
        "chapter": "22",
        "language": "es",
        "style": "short_consult",
        "text": "Varón de 25 años sufre torsión del tobillo derecho durante partido de fútbol. Tumefacción dolorosa e impotencia funcional en cara lateral. Radiografías descartan fractura ósea. Cajón anterior positivo leve. Dx: Esguince del ligamento lateral externo del tobillo derecho.",
        "gold": ["NC32.5Z"],
        "notes": "Spanish consult: Sprain of lateral collateral ligament of ankle, unspecified side.",
    },

    # Chapter 23: External causes of morbidity
    {
        "id": "blind-056",
        "chapter": "23",
        "language": "en",
        "style": "short_consult",
        "text": "30M driver involved in a two-car traffic collision where his passenger car was struck by a pickup truck on a rural highway. Restrained driver with deployed airbag. Cause code: Car occupant injured in collision with pick-up truck.",
        "gold": ["PA02"],
        "notes": "Car occupant injured in collision with pick-up truck, pick-up truck occupant.",
    },
    {
        "id": "blind-057",
        "chapter": "23",
        "language": "fr",
        "style": "typed_consult",
        "text": "Chute accidentelle d'une échelle d'une hauteur de 2 mètres lors de travaux de peinture au domicile. Cause externe: Chute d'une échelle.",
        "gold": ["PA60"],
        "notes": "French external cause consult: Fall involving ladder.",
    },

    # Chapter 24: Factors influencing health status
    {
        "id": "blind-058",
        "chapter": "24",
        "language": "en",
        "style": "typed_consult",
        "text": "50F attends outpatient clinic for routine periodic health maintenance examination. She has no acute symptoms or medical complaints. Screening mammography and colonoscopy discussed and ordered. Assessment: Routine general medical examination of adult.",
        "gold": ["QA00.0"],
        "notes": "General adult medical examination.",
    },
    {
        "id": "blind-059",
        "chapter": "24",
        "language": "sw",
        "style": "short_consult",
        "text": "Mama anamleta mtoto wa miezi 9 kwenye kliniki ya watoto kwa ajili ya kupata chanjo za kawaida za utotoni kulingana na ratiba ya kitaifa ya chanjo (chanjo ya surua na rubella). Mtoto hana ugonjwa wowote. Dx: Mahitaji ya chanjo ya kawaida ya utotoni.",
        "gold": ["QC00.0"],
        "notes": "Swahili consult: Need for routine childhood immunization.",
    },

    # Chapter 25: Codes for special purposes (COVID-19)
    {
        "id": "blind-060",
        "chapter": "25",
        "language": "en",
        "style": "typed_consult",
        "text": "45M presents with acute fever (38.7C), dry cough, anosmia, and ageusia x3 days. Nasopharyngeal RT-PCR swab is positive for SARS-CoV-2. Chest X-ray unremarkable. Diagnosis: COVID-19, virus identified.",
        "gold": ["RA01.0"],
        "notes": "COVID-19, virus identified by laboratory testing.",
    },
    {
        "id": "blind-061",
        "chapter": "25",
        "language": "es",
        "style": "short_consult",
        "text": "Paciente de 38 años con cuadro agudo de fiebre, tos seca y pérdida del gusto y del olfato. Test de antígeno rápido para SARS-CoV-2 en exudado nasofaríngeo positivo en la consulta. Dx: Infección por COVID-19 confirmada.",
        "gold": ["RA01.0"],
        "notes": "Spanish consult: Confirmed COVID-19 infection.",
    },

    # Chapter 26: Traditional Medicine
    {
        "id": "blind-062",
        "chapter": "26",
        "language": "zh",
        "style": "typed_consult",
        "text": "患者面色苍白无华，少气懒言，神疲乏力，心悸失眠，头晕目眩，舌淡苔薄白，脉细弱。辨证分析：脾胃虚弱，气血生化无源，导致气血两虚。中医辨证诊断：气血两虚证。",
        "gold": ["SD80"],
        "notes": "Chinese Traditional Medicine: Dual deficiency of qi and blood pattern.",
    },
    {
        "id": "blind-063",
        "chapter": "26",
        "language": "zh",
        "style": "short_consult",
        "text": "女性，34岁。近半年来因情志不舒，出现胸胁胀痛，善太息，情绪抑郁，烦躁易怒，乳房胀痛，舌质淡红，脉弦。中医辨证诊断：肝气郁结证。",
        "gold": ["SD80"],
        "notes": "Chinese Traditional Medicine: Liver qi stagnation pattern.",
    },

    # Additional Hindi & Arabic Multi-Chapter Cases
    {
        "id": "blind-064",
        "chapter": "01",
        "language": "hi",
        "style": "short_consult",
        "text": "मरीज 24 वर्ष, पिछले 6 दिनों से तेज बुखार, कंपकंपी, तेज सिरदर्द और उल्टी की शिकायत। खून की जांच (रक्त स्लाइड) में प्लास्मोडियम फाल्सीपेरम मलेरिया की पुष्टि हुई। डायग्नोसिस: फाल्सीपेरम मलेरिया।",
        "gold": ["1F40.Z"],
        "notes": "Hindi consult: Plasmodium falciparum malaria.",
    },
    {
        "id": "blind-065",
        "chapter": "11",
        "language": "ar",
        "style": "typed_consult",
        "text": "مريض يبلغ من العمر 52 عاماً، راجع العيادة لإجراء فحص دوري. ضغط الدم المسجل اليوم 155/95 ملم زئبق، وتم تأكيده بقراءة ثانية 152/94 ملم زئبق. لا توجد أعراض صداع أو دوار، وتخطيط القلب الكهربائي طبيعي. التشخيص: ارتفاع ضغط الدم الأساسي.",
        "gold": ["BA00.Z"],
        "notes": "Arabic consult: Essential hypertension.",
    },
]


def main():
    print(f"Validating {len(BLIND_DATA)} blind notes against MMS 2026-01...")
    invalid_codes = []
    chapter_counts = {}
    lang_counts = {}
    style_counts = {}

    for row in BLIND_DATA:
        ch = row["chapter"]
        lang = row["language"]
        style = row["style"]
        chapter_counts[ch] = chapter_counts.get(ch, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        style_counts[style] = style_counts.get(style, 0) + 1

        for g in row["gold"]:
            if not is_valid_leaf(g):
                invalid_codes.append((row["id"], g))

    if invalid_codes:
        print("ERROR: Found invalid non-leaf gold codes:")
        for nid, c in invalid_codes:
            print(f"  {nid}: {c}")
        sys.exit(1)

    print(f"Validation PASSED! All {len(BLIND_DATA)} notes have 100% valid MMS leaf codes.")
    print(f"Chapters covered: {len(chapter_counts)}/26 chapters")
    print(f"Languages: {dict(sorted(lang_counts.items()))}")
    print(f"Styles: {dict(sorted(style_counts.items()))}")

    out_path = os.path.join(ROOT, "data", "blind_notes.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in BLIND_DATA:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Successfully saved {len(BLIND_DATA)} blind notes to {out_path}.")


if __name__ == "__main__":
    main()
