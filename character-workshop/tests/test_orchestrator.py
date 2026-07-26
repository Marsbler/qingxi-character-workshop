# character-workshop/tests/test_orchestrator.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.orchestrator import JobState, generate_card_job, revise_job, animate_job
from src.paths import OUTPUTS


def test_generate_card_job_mock():
    result = generate_card_job(
        user_text="青衣空系旅人，背着灯笼",
        affinity_pref="空",
        ref_image=None,
        mock=True,
    )
    assert result.state == JobState.READY
    assert result.job_dir.exists()
    assert (result.job_dir / "character.json").exists()
    assert (result.job_dir / "portrait.png").exists()
    assert (result.job_dir / "card.png").exists()
    assert result.card is not None
    assert result.error is None


def test_revise_and_animate_mock():
    base = generate_card_job("黑发念系少年", affinity_pref="念", mock=True)
    rev = revise_job(base.job_id, "改成金色短发", mock=True)
    assert rev.state == JobState.READY
    assert "金" in rev.card.appearance or "金" in rev.card.image_prompt
    anim = animate_job(rev.job_id, mock=True)
    assert anim.state in {JobState.DONE, JobState.READY}
    # video path or soft fail both ok in mock if file exists
    assert anim.error is None or anim.video_path is None
