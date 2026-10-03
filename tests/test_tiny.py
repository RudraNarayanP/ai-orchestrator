"""The external answer is tiny: one sentence (two at most) and at most one short caveat."""

from __future__ import annotations

from backend.evidence.sources import SourceTier
from backend.research.style import split_sentences, tiny
from tests.test_early_stop import LAW_Q, OPENED_PRIMARY, URL9, providers
from tests.test_architecture import run

LONG = (
    "Under the Data Protection Act 2018, the minimum age at which a child can independently give consent to the processing of "
    "their personal data in relation to information society services in the UK is 13 years old. Under Section 9 of the Data "
    "Protection Act 2018, the age is set at 13. Section 9 modifies the default EU GDPR Article 8 threshold of 16 down to 13."
)


def test_tiny_keeps_one_or_two_sentences_and_never_cuts_mid_sentence():
    out = tiny(LONG)
    assert out.endswith("13 years old.") and len(split_sentences(out)) == 1, "the first sentence is already long: no second one"
    short = "Yes \u2014 it is 13. The law is s. 9 of the Act. A third sentence nobody asked for."
    assert tiny(short) == "Yes \u2014 it is 13. The law is s. 9 of the Act."
    assert tiny("") == "" and tiny("4") == "4"


def test_abbreviations_do_not_split_sentences():
    assert split_sentences("See s. 9 of the Act. It says 13.") == ["See s. 9 of the Act.", "It says 13."]
    assert split_sentences("Article 8 applies. No. 5 is repealed.")[0] == "Article 8 applies."


async def test_the_final_answer_is_tiny_with_at_most_one_caveat_and_the_emoji_only_at_the_end(net):
    net.confirm(URL9, tier=SourceTier.PRIMARY_OFFICIAL)
    job, _, _ = await run(providers("chatgpt"), {"chatgpt": OPENED_PRIMARY}, LAW_Q)
    ans = job.final.answer.replace("\ufe0f", "")
    assert len(split_sentences(ans)) <= 2 and len(ans) <= 330
    assert len(job.final.caveats) <= 1 and all(len(c) <= 200 for c in job.final.caveats)
    import re

    emojis = re.findall(r"[\u2600-\u27bf\U0001F300-\U0001FAFF]", ans)
    assert len(emojis) <= 1 and (not emojis or ans.rstrip().endswith(emojis[0]))