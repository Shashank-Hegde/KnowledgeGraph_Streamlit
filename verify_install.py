#!/usr/bin/env python3
"""
FILE VERIFICATION SCRIPT
========================
Run this FIRST after copying files to verify you have the correct versions.

Usage: python3 verify_install.py
"""

import sys
import os

print("=" * 60)
print("  TRIAGE ENGINE — FILE VERIFICATION")
print("=" * 60)

errors = []

# 1. Check all required files exist
required = [
    "knowledge_base.py", "graph_engine.py", "question_selector.py",
    "response_parser.py", "enhanced_parser.py", "llm_extractor.py",
    "entity_mapper.py", "smart_resolver.py", "question_dedup.py",
    "clinician_feedback.py", "triage_engine_v2.py",
]
print("\n  File existence:")
for f in required:
    exists = os.path.exists(f)
    print("    %s %s" % ("✅" if exists else "❌", f))
    if not exists:
        errors.append("MISSING: %s" % f)

# 2. Check critical code markers
print("\n  Code version checks:")
checks = [
    ("enhanced_parser.py", "not there", "expanded yes/no detection with 'not there'"),
    ("enhanced_parser.py", "SmartSymptomResolver", "smart resolver integration"),
    ("enhanced_parser.py", "weeks ago", "temporal confirmation patterns"),
    ("question_selector.py", "asked_elicits", "dedup-aware question selection"),
    ("question_dedup.py", "_block", "dedup gate with blocking registration"),
    ("triage_engine_v2.py", "SmartSymptomResolver", "smart resolver in engine"),
    ("triage_engine_v2.py", "dedup_gate", "dedup gate integration"),
    ("triage_engine_v2.py", "method_str", "parse debug output"),
    ("llm_extractor.py", "1024", "increased LLM token limit"),
    ("smart_resolver.py", "BODY_PART_SYMPTOM_MAP", "body-part-aware resolution"),
]

for filename, marker, description in checks:
    if not os.path.exists(filename):
        print("    ❌ %s — FILE MISSING" % description)
        errors.append("MISSING: %s" % filename)
        continue
    with open(filename) as f:
        content = f.read()
    found = marker in content
    print("    %s %s (%s)" % ("✅" if found else "❌", description, filename))
    if not found:
        errors.append("OLD VERSION: %s missing '%s'" % (filename, marker))

# 3. Functional test — does the parser actually work?
print("\n  Functional tests:")
try:
    sys.path.insert(0, ".")
    from enhanced_parser import EnhancedResponseParser
    from graph_engine import KnowledgeGraph
    from llm_extractor import LLMExtractor
    from smart_resolver import SmartSymptomResolver

    graph = KnowledgeGraph()
    parser = EnhancedResponseParser(graph, llm_extractor=LLMExtractor(base_url="http://localhost:99999"))

    test_cases = [
        ("it is there for a while", True, False, []),
        ("not there", False, True, []),
        ("there is a pain in the leg", True, False, ["body_pain"]),
        ("went to somalia few weeks ago", True, False, []),
        ("seems to be all ok", False, True, []),
    ]

    func_ok = True
    for text, exp_yes, exp_no, exp_syms in test_cases:
        r = parser.parse(text)
        ok = r["is_yes"] == exp_yes and r["is_no"] == exp_no
        for s in exp_syms:
            if s not in r["detected_symptoms"]:
                ok = False
        print("    %s \"%s\" → yes=%s no=%s syms=%s" % (
            "✅" if ok else "❌", text, r["is_yes"], r["is_no"], r["detected_symptoms"]))
        if not ok:
            func_ok = False
            errors.append("FUNCTIONAL: '%s' failed" % text)

    # Test smart resolver
    resolver = SmartSymptomResolver(graph)
    r = resolver.resolve("pain in the upper part of the body")
    syms = [s for s, _ in r]
    ok = "chest_pain" in syms
    print("    %s smart resolver: 'pain in upper body' → %s" % ("✅" if ok else "❌", syms))
    if not ok:
        errors.append("FUNCTIONAL: smart resolver failed")

except Exception as e:
    print("    ❌ Import/test error: %s" % str(e))
    errors.append("IMPORT ERROR: %s" % str(e))

# 4. Summary
print("\n" + "=" * 60)
if errors:
    print("  ❌ %d ERRORS FOUND:" % len(errors))
    for e in errors:
        print("    • %s" % e)
    print("\n  FIX: Download ALL .py files from Claude and replace every file.")
    print("       rm -f *.py && # copy new files here")
else:
    print("  ✅ ALL CHECKS PASSED — you have the correct files!")
    print("  Run: python3 triage_engine_v2.py")
print("=" * 60)
