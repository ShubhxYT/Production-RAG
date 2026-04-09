"""Background scheduler for continuous evaluation."""

from observability.logging import get_logger

from evaluation.continuous import ContinuousEvaluator

logger = get_logger(__name__)


class BackgroundEvaluationScheduler:
    """Manages an APScheduler background job for periodic evaluation runs."""

    def __init__(self) -> None:
        self._scheduler = None
        self._evaluator = ContinuousEvaluator()

    def start(self, interval_hours: int) -> None:
        """Start the background scheduler.

        Args:
            interval_hours: How often to run evaluation (in hours).
        """
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler = BackgroundScheduler()
        self._scheduler.add_job(
            self._evaluator.run,
            trigger=IntervalTrigger(hours=interval_hours),
            id="continuous_eval",
            replace_existing=True,
        )
        self._scheduler.start()

        logger.info(
            "Continuous evaluation scheduler started",
            extra={"interval_hours": interval_hours},
        )

    def stop(self) -> None:
        """Stop the background scheduler if running."""
        if self._scheduler is not None and self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Continuous evaluation scheduler stopped")
