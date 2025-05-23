# ---------- components.py ----------
# Semua fungsi untuk membuat komponen TFX dalam satu file.

from tfx.components import (
    CsvExampleGen, StatisticsGen, SchemaGen, ExampleValidator,
    Transform, Trainer, Evaluator, Pusher
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.experimental.latest_blessed_model_resolver import LatestBlessedModelResolver
from tfx.proto import trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing
from tensorflow_model_analysis.proto import config_pb2 as tfma_config

import configs  # untuk akses hyperparameter


def create_example_gen():
    return CsvExampleGen(input_base=configs.DATA_ROOT)


def create_statistics_gen(examples):
    return StatisticsGen(examples=examples)


def create_schema_gen(statistics):
    return SchemaGen(statistics=statistics)


def create_example_validator(statistics, schema):
    return ExampleValidator(statistics=statistics, schema=schema)


def create_transform(examples, schema, module_file: str):
    return Transform(
        examples=examples,
        schema=schema,
        module_file=module_file
    )


def create_trainer(module_file: str, examples, schema, transform_graph, hyperparameters):
    return Trainer(
        module_file=module_file,
        examples=examples,
        schema=schema,
        transform_graph=transform_graph,
        train_args=trainer_pb2.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
        eval_args=trainer_pb2.EvalArgs(num_steps=configs.EVAL_NUM_STEPS)
    )


def create_model_resolver():
    return Resolver(
        strategy_class=LatestBlessedModelResolver,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id('latest_blessed_model_resolver')


def create_evaluator(examples, model):
    eval_config = tfma_config.EvalConfig(
        model_specs=[tfma_config.ModelSpec(label_key='label')],
        metrics_specs=[
            tfma_config.MetricsSpec(
                metrics=[
                    tfma_config.MetricConfig(
                        class_name='BinaryAccuracy',
                        threshold=tfma_config.MetricThreshold(
                            value_threshold={'lower_bound': {'value': configs.BINARY_ACCURACY_THRESHOLD}}
                        )
                    )
                ]
            )
        ],
        slicing_specs=[tfma_config.SlicingSpec()]
    )
    return Evaluator(examples=examples, model=model, eval_config=eval_config)


def create_pusher(model, model_blessing):
    return Pusher(
        model=model,
        model_blessing=model_blessing,
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=configs.SERVING_MODEL_DIR
            )
        )
    )
