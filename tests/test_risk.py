from backend import config


def test_prototype_display_bands_use_configured_thresholds(monkeypatch):
    fixture = {
        name: {"low_max": .25, "high_min": .75, "clinical_cutoff": False}
        for name in ("cad", "lad", "lcx", "rca")
    }
    monkeypatch.setattr(config, "RISK_THRESHOLDS", fixture)
    for target, thresholds in fixture.items():
        assert thresholds["clinical_cutoff"] is False
        assert thresholds["low_max"] < thresholds["high_min"]
        assert config.classify_risk(target, .1) == "LOW"
        assert config.classify_risk(target, .5) == "MODERATE"
        assert config.classify_risk(target, .75) == "HIGH"
