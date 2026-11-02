"""Run one battery on one model: generate, score, and log every item.

Each item's prompt digest, response, label and outcome go to an item log, a hash-chained
``ActionLog`` of its own (one file per run), so a reader can re-derive every outcome and a
labeller can read the responses. The run itself keeps only response digests.

Item logs are sensitive: they hold full responses to harmful prompts. Keep them outside the
repository tree (the gate writes them under ``output.dir``/items/), never commit them, and
never bundle them with a signed record; a record carries response digests only.
"""

from __future__ import annotations

from changegate.battery import Battery, BatteryRun, Decoding, items_digest
from changegate.battery.generate import Generator
from changegate.digests import sha256_json
from changegate.errors import ChangegateError
from changegate.log import ActionLog

ITEM_EVENT = "battery.item"


def run_battery(
    battery: Battery,
    generator: Generator,
    decoding: Decoding | None = None,
    item_log: ActionLog | None = None,
) -> BatteryRun:
    """Generate a response to every item, score them, and log each item.

    Args:
        battery: the battery.
        generator: the loaded model under test.
        decoding: overrides the battery's decoding (for example a smaller token budget).
        item_log: where to append one ``battery.item`` row per item, response included.

    Returns:
        The run: one outcome per item, with the model, scorer, decoding and template digest.

    Raises:
        ChangegateError: if the generator returns the wrong number of responses.

    Does not establish: that the model saw the prompts exactly as a deployment would; the
    prompts are rendered by the model's own chat template with no system prompt unless the
    item carries one.
    """
    items = list(battery.items())
    dec = decoding or battery.decoding
    responses = generator.generate([item.messages for item in items], dec)
    if len(responses) != len(items):
        raise ChangegateError(f"{len(items)} items but {len(responses)} responses")
    outcomes = battery.score(items, responses)
    run = BatteryRun(
        battery=battery.name,
        battery_version=battery.version,
        items_sha256=items_digest(items),
        model=generator.model,
        scorer=battery.scorer,
        outcomes=tuple(outcomes),
        decoding=dec,
        template_sha256=generator.template_sha256,
        controls=tuple(sorted(battery.control_categories)),
    )
    if item_log is not None:
        for item, response, outcome in zip(items, responses, outcomes, strict=True):
            item_log.append(
                ITEM_EVENT,
                battery=battery.name,
                model_sha256=generator.model.sha256,
                item_id=item.id,
                category=item.category,
                messages_sha256=sha256_json([m.model_dump() for m in item.messages]),
                response=response,
                response_sha256=outcome.response_sha256,
                label=outcome.label,
                passed=outcome.passed,
            )
    return run
