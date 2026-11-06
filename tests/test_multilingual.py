"""Multilingual validation test suite.
Evaluates pipeline on hand-crafted multi-chapter clinical notes in Swahili, Spanish, and French.
"""

import sys
sys.path.insert(0, ".")
import json
from pipeline import ICD11Pipeline

def test_multilingual_generalisation():
    p = ICD11Pipeline()

    test_cases = [
        {
            "id": "multi-sw-01",
            "text": "Mgonjwa ana homa kali na maumivu ya tumbo x4/7. Daktari anasema ni homa ya matumbo.",
            "gold": ["1A07.Z"],
            "lang": "sw",
            "desc": "Swahili Typhoid fever"
        },
        {
            "id": "multi-sw-02",
            "text": "Daktari: Hali yako vipi? Mgonjwa: Nina homa kali na kutetemeka, maumivu ya viungo. Kipimo cha malaria mRDT chanya kwa falciparum.",
            "gold": ["1F40.Z"],
            "lang": "sw",
            "desc": "Swahili Falciparum Malaria"
        },
        {
            "id": "multi-es-01",
            "text": "Paciente femenina de 24 años con dolor de garganta intenso, fiebre de 38.5C y odinofagia de 3 dias. Diagnostico: Amigdalitis aguda.",
            "gold": ["CA03.Z"],
            "lang": "es",
            "desc": "Spanish Acute Tonsillitis"
        },
        {
            "id": "multi-es-02",
            "text": "Hombre de 65 anos consulta por dolor en rodilla derecha al caminar, rigidez matutina menor de 20 minutos. Impresion: Artrosis de rodilla.",
            "gold": ["FA01.Z"],
            "lang": "es",
            "desc": "Spanish Osteoarthritis of Knee"
        },
        {
            "id": "multi-fr-01",
            "text": "Patient de 45 ans consulte pour une toux productive depuis 5 jours avec febricule. Auscultation: rales bronchiques. Diagnostic: Bronchite aigue.",
            "gold": ["CA42.Z"],
            "lang": "fr",
            "desc": "French Acute Bronchitis"
        },
        {
            "id": "multi-fr-02",
            "text": "Patiente de 28 ans avec prurit vulvaire intense et pertes vaginales blanches grumeleuses. Diagnostic: Candidose vulvovaginale.",
            "gold": ["1F23.10"],
            "lang": "fr",
            "desc": "French Vulvovaginal Candidiasis"
        },
    ]

    print("=" * 60)
    print("MULTILINGUAL GENERALISATION TEST")
    print("=" * 60)

    top1 = 0
    top3 = 0
    for tc in test_cases:
        res = p.predict_note(tc)
        preds = res["codes"]
        hit1 = preds[0] in tc["gold"]
        hit3 = any(c in tc["gold"] for c in preds[:3])
        if hit1:
            top1 += 1
        if hit3:
            top3 += 1
        status = "PASS (Top-1)" if hit1 else ("PASS (Top-3)" if hit3 else "FAIL")
        print(f"[{status}] {tc['desc']} ({tc['lang']}): Gold={tc['gold']} Pred={preds}")

    n = len(test_cases)
    print("-" * 60)
    print(f"Multilingual Summary: Top-1: {top1}/{n} ({top1/n*100:.1f}%), Top-3: {top3}/{n} ({top3/n*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    test_multilingual_generalisation()
