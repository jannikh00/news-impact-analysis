import re

import pytest

from utils import TOKEN_PATTERN, clean_text


@pytest.mark.parametrize("raw, symbol, expected", [
    ("Raises Price Target to $88", None, "raises price target to moneytoken"),
    ("Shares Up 5.2% After Earnings", None, "shares up pcttoken after earnings"),
    ("Lilly Will Engage 40K People Over Next 2 Years", None,
     "lilly will engage numtoken people over next numtoken years"),
    ("Agilent (A) Gears Up for Q2 Earnings", "A", "agilent tickertoken gears up for quarter earnings"),
    ("American Express Company (NYSE: AXP) Update", None, "american express company tickertoken update"),
    ("The Vetr community has downgraded $A to 4-Stars", None,
     "the vetr community has downgraded tickertoken to numtoken stars"),
    ("Kroger Doesn&#39;t Expect Better-Than-Expected Sales", None,
     "kroger does not expect better-than-expected sales"),
    ("<b>FedEx</b> can't &amp; won't", None, "fedex can not will not"),
])
def test_clean_text(raw, symbol, expected):
    assert clean_text(raw, symbol) == expected


def test_token_pattern_keeps_hyphenated_terms():
    tokens = re.findall(TOKEN_PATTERN, clean_text("Better-than-expected Q3, not a loss"))
    assert tokens == ["better-than-expected", "quarter", "not", "loss"]
