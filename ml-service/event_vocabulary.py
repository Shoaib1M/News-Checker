"""
FILE PURPOSE:
The shared vocabulary of *events* a claim can assert — "ban", "resign",
"acquire" — with the surface forms each one is written in, and which events
are opposites of each other.

WHY THIS EXISTS:
Two separate stages need to know what action a claim is about, and they must
agree:

  - `query_generator.py` builds search queries. The single most useful query
    for "the prime minister of India resigned this morning" is one containing
    *resigned*. Without a notion of which word is the event, the generator
    picked whichever token its regex matched first — "morning" — and spent two
    of its four query slots on `"India" "morning"`.

  - `relevance_filter.py` decides whether a retrieved document is about the
    claim. A document mentioning the claim's subjects but not its event is
    background, not evidence.

Grouping surface forms by event is what lets a document written in different
words than the claim still match: a claim saying "resigned" and a headline
saying "steps down" are about the same event.

The vocabulary is curated rather than derived. A word missing here weakens
both stages but cannot make either of them wrong — an unrecognised event
makes relevance scoring neutral and query generation fall back to its
previous behaviour.
"""

from __future__ import annotations

import re


# ── Events and how they are written ──────────────────────────────────
EVENT_VERBS: dict[str, tuple[str, ...]] = {
    "ban": ("ban", "bans", "banned", "banning", "outlaw", "outlaws", "outlawed",
            "prohibit", "prohibits", "prohibited", "prohibition", "forbid",
            "forbids", "forbidden", "block", "blocks", "blocked", "restrict",
            "restricts", "restricted", "crackdown"),
    # Newsrooms have many ways of saying "left the job". Multi-word forms are
    # preferred over single ambiguous words: "vacates office" is unmistakable
    # where a bare "vacates" is not, and "bar"/"barred" were left out of the
    # ban family for the same reason.
    # "sacked"/"dismissed"/"axed" are one-sided removals rather than
    # resignations, but for retrieval they are the same event: the person is
    # out of the job, and coverage of one is coverage of the other. A bare
    # "fired" is deliberately absent — police fire tear gas, and a rifle is
    # fired — so only the passive forms qualify.
    "resign": ("resign", "resigns", "resigned", "resignation", "quit", "quits",
               "sacked", "dismissed", "axed", "was fired", "were fired",
               "step down", "steps down", "stepped down", "stepping down",
               "steps aside", "stepped aside", "stepping aside",
               "stands down", "stood down", "standing down",
               "vacates office", "vacated office", "hands over power",
               "handed over power", "relieved of duties", "leaves office",
               "left office", "ousted", "removed from office"),
    "arrest": ("arrest", "arrests", "arrested", "detained", "custody",
               "taken into custody", "apprehended", "charged", "indicted",
               "indictment"),
    "acquire": ("acquire", "acquires", "acquired", "acquisition", "buy",
                "buys", "bought", "purchase", "purchases", "purchased",
                "takeover", "merger", "merges", "merged"),
    "launch": ("launch", "launches", "launched", "unveil", "unveils",
               "unveiled", "release", "releases", "released", "introduce",
               "introduces", "introduced", "rollout", "rolls out"),
    "die": ("die", "dies", "died", "death", "deaths", "dead", "killed",
            "fatal", "fatality", "fatalities", "perished", "passed away",
            "lost their lives"),
    "invade": ("invade", "invades", "invaded", "invasion", "attack",
               "attacks", "attacked", "strike", "strikes", "war"),
    # "raise"/"raised" also appears in "raised concerns", which is not an
    # increase. It is included anyway: in news copy the quantitative sense
    # dominates, and a missing word makes the relevance gate fail outright
    # where a loose one only makes it slightly less selective.
    #
    # "jump" and "climb" were tried and removed. They matched "the children
    # jumped into the lake" and "he climbed the stairs", and soar/surge/rise
    # already cover the sense — the same reasoning that keeps bare "up" and
    # "down" out below.
    "increase": ("raise", "raises", "raised", "hike", "hikes", "hiked",
                 "soar", "soars", "soared", "spike", "spiked",
                 "increase", "increases", "increased", "rise", "rises",
                 "rose", "rising", "surge", "surges", "surged", "grew",
                 "growth", "higher", "improve", "improves", "improved",
                 "improvement", "boost", "boosts", "boosted", "gain", "gains"),
    "decrease": ("lower", "lowers", "lowered", "slash", "slashes", "slashed",
                 "plunge", "plunges", "plunged", "tumble", "tumbles",
                 "tumbled", "sank", "shrank", "shrink", "shrinks",
                 "reduce", "reduces", "reduced", "halve", "halved",
                 "decrease", "decreases", "decreased", "fall", "falls",
                 "fell", "falling", "drop", "drops", "dropped", "decline",
                 "declines", "declined", "lower", "cut", "cuts",
                 "worsen", "worsens", "worsened", "harm", "harms", "harmed",
                 "casts doubt", "no effect"),
    "approve": ("uphold", "upholds", "upheld", "ratify", "ratifies",
                "ratified", "pardon", "pardons", "pardoned", "authorise",
                "authorised", "authorize", "authorized", "greenlight",
                "greenlit", "approve", "approves", "approved", "approval",
                "pass",
                "passes", "passed", "enact", "enacts", "enacted", "signed"),
    "reject": ("repeal", "repeals", "repealed", "overturn", "overturns",
               "overturned", "withdraw", "withdraws", "withdrew",
               "withdrawal", "scrap", "scraps", "scrapped", "quash",
               "quashed", "revoke", "revokes", "revoked",
               "reject", "rejects", "rejected", "deny", "denies", "denied",
               "refuse", "refuses", "refused", "veto", "vetoed", "struck down"),
    "rename": ("rename", "renames", "renamed", "renaming", "name change",
               "rebrand", "rebrands", "rebranded", "new name"),
    "fine": ("fine", "fines", "fined", "penalty", "penalties", "sued",
             "lawsuit", "settlement", "antitrust"),
    "close": ("recall", "recalls", "recalled", "halt", "halts", "halted",
              "suspend", "suspends", "suspended", "cease", "ceases", "ceased",
              "close", "closes", "closed", "shut", "shuts", "shutdown",
              "shut down", "collapse", "collapses", "collapsed", "bankrupt",
              "bankruptcy", "dissolve", "dissolved"),
    "elect": ("clinch", "clinches", "clinched",
              "elect", "elects", "elected", "election", "wins", "won",
              "victory", "sworn in", "inaugurated"),
    "legalize": ("legalize", "legalizes", "legalized", "legalise",
                 "legalised", "decriminalize", "decriminalized"),
    # Disasters are a large share of breaking news and had no event of their
    # own, so a claim about an earthquake matched coverage on entities alone.
    "disaster": ("earthquake", "quake", "flood", "floods", "flooding",
                 "wildfire", "wildfires", "hurricane", "typhoon", "cyclone",
                 "tsunami", "landslide", "eruption", "erupted", "derailed",
                 "devastate", "devastates", "devastated", "evacuate",
                 "evacuates", "evacuated", "evacuation"),
    "announce": ("announce", "announces", "announced", "announcement",
                 "confirm", "confirms", "confirmed", "declare", "declares",
                 "declared", "plans", "proposal", "proposes", "proposed"),
}

# Bare "up" and "down" are deliberately absent. They appear in an enormous
# number of unrelated sentences ("shares were down", "down the road"), so as
# event words they made the relevance gate match almost any financial story —
# and inside "steps down" they made a resignation register as a *decrease* as
# well, giving one claim two contradictory events.

# Reverse index: surface form -> canonical event.
SURFACE_TO_EVENT: dict[str, str] = {
    form: event
    for event, forms in EVENT_VERBS.items()
    for form in forms
}

# Evidence *against* a claim usually states the opposite action, not the
# claim's own: the study refuting "a four-day workweek improves productivity"
# is headlined "output fell under the shorter schedule", with no word from the
# "increase" family in it. Matching only the claim's own action therefore
# filtered out exactly the contradicting sources the system exists to find.
ANTONYMS: dict[str, str] = {}
for _a, _b in (
    ("increase", "decrease"),
    ("approve", "reject"),
    ("ban", "legalize"),
    ("launch", "close"),
):
    ANTONYMS[_a] = _b
    ANTONYMS[_b] = _a


def _contains(surface: str, haystack: str) -> bool:
    """Whole-word match for one surface form inside already-lowercased text."""
    return bool(re.search(rf"(?<![a-z]){re.escape(surface)}(?![a-z])", haystack))


def events_in(text: str) -> set[str]:
    """Canonical events mentioned in ``text``.

    Empty when the text uses no recognised event word, which callers treat as
    "no opinion" rather than "no event".
    """
    haystack = f" {(text or '').lower()} "
    return {
        event for surface, event in SURFACE_TO_EVENT.items()
        if _contains(surface, haystack)
    }


def surface_forms_in(text: str) -> list[str]:
    """The event words actually written in ``text``, in the order they appear.

    Query generation needs the claim's own wording, not the canonical event
    name: searching for the word the claim used finds the coverage that used
    it too.
    """
    lowered = (text or "").lower()
    found: list[tuple[int, str]] = []
    for surface in SURFACE_TO_EVENT:
        match = re.search(rf"(?<![a-z]){re.escape(surface)}(?![a-z])", lowered)
        if match:
            found.append((match.start(), surface))
    found.sort()
    # Prefer the longest form at each position ("steps down" over "down").
    chosen: list[str] = []
    for _position, surface in found:
        if not any(surface in existing and surface != existing for existing in chosen):
            chosen.append(surface)
    return chosen
