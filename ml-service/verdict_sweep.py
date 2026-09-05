"""Print the verdict this service gives for every shape of claim it handles.

    python verdict_sweep.py

WHAT THIS IS FOR:
The unit tests assert one property each. This prints the whole behaviour
matrix in one table, which is the fastest way to (a) sanity-check a change to
the verdict logic and (b) show someone what the system actually does without
waiting on live search.

HOW IT WORKS:
Search and NLI are replaced with deterministic doubles: the search double
returns a realistic pool of on-topic-but-unrelated coverage, and the NLI
double entails or contradicts only passages containing a per-case keyword.
So what you are reading is the verdict logic in isolation — not the quality of
any live provider, and not the accuracy of the real NLI model.

Two cases exercise the failure modes worth understanding: "search failed"
must never produce a finding about the claim, and "NLI down" must never
produce one either.
"""
import sys
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient
import evidence_pipeline, main
from providers import ProviderDiagnostic, SearchResult

BACKGROUND = [
    ("reuters.com", "Markets steady as investors await data",
     "Traders positioned ahead of the week's economic releases."),
    ("apnews.com", "Regulators review platform competition",
     "Officials discussed oversight of large technology platforms."),
    ("bbc.com", "Weather disrupts travel in several regions",
     "Services were delayed across a number of routes."),
    ("theguardian.com", "Company reports quarterly earnings",
     "Revenue rose modestly against analyst expectations."),
    ("npr.org", "Lawmakers debate budget priorities",
     "The session covered spending across several departments."),
    ("wsj.com", "Sector outlook revised for the year",
     "Analysts adjusted forecasts following the latest figures."),
]


def make_results(specs):
    return [SearchResult(url=f"https://{d}/story-{i}", title=t, snippet=b,
                         text=(b + " ") * 30, provider="gnews", source=d)
            for i, (d, t, b) in enumerate(specs)]


class NLI:
    def __init__(self, entail=None, contradict=None, ok=True):
        self.entail, self.contradict, self._ok = entail, contradict, ok
        self.status = {"status": "ready" if ok else "failed", "enabled": True,
                       "model": "stub", "error": None}
        self.is_ready = ok

    @property
    def is_available(self):
        return self._ok

    def score_many(self, claim, passages):
        out = []
        for p in passages:
            low = p.lower()
            if not self._ok:
                out.append({"entailment": 0., "contradiction": 0., "neutral": 1., "available": False})
            elif self.entail and self.entail in low:
                out.append({"entailment": .93, "contradiction": .02, "neutral": .05, "available": True})
            elif self.contradict and self.contradict in low:
                out.append({"entailment": .02, "contradiction": .91, "neutral": .07, "available": True})
            else:
                out.append({"entailment": .04, "contradiction": .03, "neutral": .93, "available": True})
        return out


# (label, statement, extra search hits, nli, search_status)
CASES = [
    ("fabricated headline", "Elon Musk bought the Eiffel Tower for 3 trillion dollars", [], NLI(), "success"),
    ("fabricated + absolute", "The United States banned Google across all its cities", [], NLI(), "success"),
    ("future prediction", "The United States is Going to ban Google across all its cities", [], NLI(), "success"),
    ("future, reported plan", "Apple is planning to move all manufacturing to India",
     [("reuters.com", "Apple plans India manufacturing shift",
       "Apple is planning to move all manufacturing to India, sources said.")],
     NLI(entail="planning to move all manufacturing"), "success"),
    ("announced future", "Apple announced it will move manufacturing to India by 2027", [], NLI(), "success"),
    ("supported news", "The prime minister of India resigned this morning",
     [("reuters.com", "India's prime minister resigns", "The prime minister resigned on Tuesday."),
      ("apnews.com", "Delhi in turmoil as PM steps down", "The prime minister resigned, ending speculation."),
      ("bbc.com", "PM quits after coalition collapse", "The prime minister resigned amid infighting.")],
     NLI(entail="prime minister resigned"), "success"),
    ("contradicted claim", "The Great Wall of China is visible from the Moon",
     [("politifact.com", "No, the Great Wall is not visible from the Moon",
       "Astronauts confirm the wall is not visible from the moon with the naked eye.")],
     NLI(contradict="not visible from the moon"), "success"),
    ("mixed evidence", "A four-day workweek improves productivity",
     [("bbc.com", "Trial finds four-day week improves productivity",
       "The pilot found a four-day workweek improves productivity."),
      ("wsj.com", "Study casts doubt on four-day week gains",
       "Researchers said output fell under the shorter schedule.")],
     NLI(entail="four-day workweek improves productivity", contradict="output fell"), "success"),
    ("negated claim", "The United States did not ban Google", [], NLI(), "success"),
    ("timeless fact", "Water freezes at 0°C at sea level", [], NLI(), "success"),
    ("known falsehood", "A triangle has four sides", [], NLI(), "success"),
    ("arithmetic", "2 + 2 = 5", [], NLI(), "success"),
    ("subjective", "Pizza is the best food in the world", [], NLI(), "success"),
    ("factual superlative", "The best-selling car in 2024 was the Tesla Model Y", [], NLI(), "success"),
    ("question", "Is the earth actually flat?", [], NLI(), "success"),
    ("gibberish", "asdkjh asdkjh asdkjh qwe", [], NLI(), "success"),
    ("bare link", "https://example.com/some-article", [], NLI(), "success"),
    ("noun phrase", "unemployment rate figures", [], NLI(), "success"),
    ("irregular verb", "The greatest earthquake in history hit Chile in 1960", [], NLI(), "success"),
    ("low-salience claim", "A regional bakery updated its supplier contracts", [], NLI(), "success"),
    ("search failed", "A cyclone made landfall in Odisha overnight", [], NLI(), "failed"),
    ("NLI down", "Elon Musk bought the Eiffel Tower for 3 trillion dollars", [], NLI(ok=False), "success"),
    ("ALL CAPS headline", "GOOGLE BANNED ACROSS ALL UNITED STATES CITIES", [], NLI(), "success"),
    ("multi-claim", "The prime minister resigned this morning. The finance minister was arrested.",
     [], NLI(), "success"),
    ("very long claim", "According to several officials who spoke on condition of anonymity, the "
     "government is understood to have quietly approved a sweeping new framework that would "
     "fundamentally restructure how digital platforms operate nationwide", [], NLI(), "success"),
]

cm = TestClient(main.app)
client = cm.__enter__()
print(f"{'case':<24} {'status':<26} {'conf':<7} verdict")
print("-" * 110)
for label, statement, extra, nli, status in CASES:
    results = make_results(BACKGROUND + extra) if status == "success" else []
    diags = [ProviderDiagnostic(provider="gnews", query="q", enabled=True, status=status,
                                raw_result_count=len(results), new_result_count=len(results))]
    with patch.object(evidence_pipeline, "search_all_providers", lambda q, **k: (list(results), diags)), \
         patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
         patch.object(main, "get_nli_service", lambda: nli):
        r = client.post("/api/check", json={"statement": statement}).json()
    print(f"{label:<24} {r['verification']['status']:<26} {r['confidence']:<7} {r['verdict']}")
cm.__exit__(None, None, None)
