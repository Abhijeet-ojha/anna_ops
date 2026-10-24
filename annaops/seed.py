from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from annaops.agents.matching import FICTIONAL_NGOS
from annaops.config import settings
from annaops.db import Database
from annaops.workflow import DonationWorkflow


def seed(with_scenarios: bool = False) -> None:
    database = Database(settings)
    database.initialize()
    count = database.seed_ngos(FICTIONAL_NGOS)
    print(f"Seeded {count} fictional NGO profiles into {database.path}")
    if not with_scenarios:
        return
    workflow = DonationWorkflow(database, settings)
    now = datetime.now(settings.timezone).replace(microsecond=0)
    prepared = (now - timedelta(minutes=20)).isoformat()
    deadline = (now + timedelta(hours=3)).isoformat()
    messages = [
        f"40 portions vegetarian biryani prepared at {prepared}, kept hot, pickup by {deadline} from Demo Campus Jaipur, coordinates 26.9124, 75.7873",
        "We have some food left after an event.",
        f"25 portions rice prepared at {(now + timedelta(hours=2)).isoformat()}, kept hot, pickup by {deadline} from Demo Hall Jaipur, coordinates 26.90, 75.78",
    ]
    for message in messages:
        donation = database.create_donation(message, "INTAKE_PROCESSING", synthetic=True)
        result = workflow.process_new(donation["id"], message)
        print(f"Seeded scenario {result['id']} -> {result['status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed अन्न ops synthetic demo data")
    parser.add_argument("--with-scenarios", action="store_true", help="Also seed three clearly synthetic offers")
    arguments = parser.parse_args()
    seed(arguments.with_scenarios)

