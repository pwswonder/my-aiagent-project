from __future__ import annotations

import time

from paper_agent_v2.checkpointing import postgres_checkpointer
from paper_agent_v2.config import get_settings
from paper_agent_v2.db import SessionLocal
from paper_agent_v2.jobs import claim_next_run, process_run, recover_stale_runs


def main() -> None:
    settings = get_settings()
    with postgres_checkpointer(settings.database_url) as checkpointer:
        if checkpointer is not None:
            checkpointer.setup()
        with SessionLocal() as session:
            recover_stale_runs(session)
        while True:
            with SessionLocal() as session:
                run = claim_next_run(session)
                if run:
                    process_run(session, run, settings)
            if run is None:
                time.sleep(settings.worker_poll_seconds)


if __name__ == "__main__":
    main()
