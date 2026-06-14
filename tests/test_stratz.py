from dota2tuned.clients.stratz import MATCH_QUERY


def test_match_query_uses_current_pickban_fields():
    pickban_block = MATCH_QUERY.split("pickBans {", 1)[1].split("}", 1)[0]

    assert "isRadiant" in pickban_block
    assert "team" not in pickban_block
