from adsrt.validation import run_validation_suite


def test_full_validation_suite_passes():
    report = run_validation_suite()
    assert report["summary"]["passed"], report
