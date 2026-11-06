"""Unit tests for offline language identification and non-Latin script gating."""

import sys

sys.path.insert(0, ".")
from multilingual import detect_language


def test_language_identification():
    cases = [
        # Non-Latin scripts - must NEVER fall through to English
        ("मरीज को बुखार और सिरदर्द है।", "hi", "Hindi"),
        ("المريض يعاني من ألم في الصدر وضيق في التنفس.", "ar", "Arabic"),
        ("患者有发热和剧烈腹痛三天。", "zh", "Chinese"),
        ("Пациент жалуется на острую боль в груди и кашель.", "ru", "Russian"),
        ("患者は発熱と激しい腹痛を訴えている。", "ja", "Japanese"),

        # Latin languages
        ("Patient complains of mild sore throat and cough x3/7.", "en", "English"),
        ("Mgonjwa ana homa kali na maumivu ya tumbo x3/7.", "sw", "Swahili"),
        ("Paciente femenina refiere dolor abdominal agudo y fiebre de 3 dias.", "es", "Spanish"),
        ("Patiente consulte pour prurit vulvaire et leucorrhees blanchatres.", "fr", "French"),
        ("Paciente do sexo masculino refere dor lombar intensa ha 5 dias.", "pt", "Portuguese"),
    ]

    print("=" * 60)
    print("LANGUAGE IDENTIFICATION VERIFICATION")
    print("=" * 60)
    passed = 0
    for text, expected_lang, lang_name in cases:
        detected = detect_language(text)
        assert detected != "en" if expected_lang != "en" else True, f"{lang_name} must NEVER fall through to English!"
        is_exact = (detected == expected_lang)
        status = "PASS" if is_exact else "WARN"
        if is_exact:
            passed += 1
        print(f"[{status}] {lang_name:<12} Expected: {expected_lang} | Detected: {detected} | Snippet: {text[:35]}")

    print("-" * 60)
    print(f"Result: {passed}/{len(cases)} exact matches ({passed/len(cases)*100:.1f}%)")
    assert passed == len(cases), f"All {len(cases)} test cases must pass exact language identification!"
    print("All language identification tests PASSED successfully!")

if __name__ == "__main__":
    test_language_identification()

