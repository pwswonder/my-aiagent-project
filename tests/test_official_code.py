from paper_agent_v2.official_code import GitHubSourceResolver


def test_official_repository_can_match_paper_title_acronym() -> None:
    assert GitHubSourceResolver.matches_paper(
        "whai362/PVT",
        "Official implementation of PVT series",
        "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions",
    )


def test_unrelated_repository_does_not_match_paper() -> None:
    assert not GitHubSourceResolver.matches_paper(
        "someone/unrelated",
        "A collection of image utilities",
        "Pyramid Vision Transformer",
    )
