# ---------- main.py ----------
import os

from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration import metadata
from tfx.orchestration import pipeline as tfx_pipeline
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

import configs
import components

PUSHGATEWAY_ADDR = os.getenv("PUSHGATEWAY_ADDR", "localhost:9091")
JOB_NAME = configs.PIPELINE_NAME


def push_metric(name: str, value: float, labels: dict = None):
    registry = CollectorRegistry()
    g = Gauge(
        name,
        "TFX pipeline metric",
        labelnames=labels.keys() if labels else [],
        registry=registry,
    )
    if labels:
        g.labels(**labels).set(value)
    else:
        g.set(value)
    push_to_gateway(PUSHGATEWAY_ADDR, job=JOB_NAME, registry=registry)


def create_pipeline():
    # 1. ExampleGen
    example_gen = components.create_example_gen()

    # 2. StatisticsGen
    stats_gen = components.create_statistics_gen(example_gen.outputs["examples"])

    # 3. SchemaGen
    schema_gen = components.create_schema_gen(stats_gen.outputs["statistics"])

    # 4. ExampleValidator
    example_validator = components.create_example_validator(
        stats_gen.outputs["statistics"], schema_gen.outputs["schema"]
    )

    # 5. Transform
    transform = components.create_transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join(os.getcwd(), "modules", "transform.py"),
    )

    # 6. Trainer
    trainer = components.create_trainer(
        module_file=os.path.join(os.getcwd(), "modules", "trainer.py"),
        examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        hyperparameters=transform.outputs["transform_graph"],
    )

    # 7. Resolver
    model_resolver = components.create_model_resolver()

    # 8. Evaluator
    evaluator = components.create_evaluator(
        examples=example_gen.outputs["examples"], model=trainer.outputs["model"]
    )

    # 9. Pusher
    pusher = components.create_pusher(
        model=trainer.outputs["model"], model_blessing=evaluator.outputs["blessing"]
    )

    p = tfx_pipeline.Pipeline(
        pipeline_name=configs.PIPELINE_NAME,
        pipeline_root=configs.PIPELINE_ROOT,
        components=[
            example_gen,
            stats_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            configs.METADATA_PATH
        ),
        beam_pipeline_args=[
            f"--runner={configs.RUNNER}",
            *[f"--experiments={exp}" for exp in configs.BEAM_EXPERIMENTS],
        ],
    )

    # Hook: setelah pipeline selesai, push informasi sukses/gagal
    def on_success(*args, **kwargs):
        push_metric('pipeline_runs_total', 1, {'status':'success'})
    def on_failure(*args, **kwargs):
        push_metric('pipeline_runs_total', 1, {'status':'failure'})

    p.on_success = on_success
    p.on_failure = on_failure

    return p


if __name__ == "__main__":
    BeamDagRunner().run(create_pipeline())
