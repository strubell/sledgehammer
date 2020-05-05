"""
The ``evaluate`` subcommand can be used to
evaluate a trained model against a dataset
and report any metrics calculated by the model.
This version supports reading confidence thresholds.

.. code-block:: bash

    $ allennlp evaluate --help
    usage: allennlp evaluate [-h] [--output-file OUTPUT_FILE]
                             [--weights-file WEIGHTS_FILE]
                             [--cuda-device CUDA_DEVICE] [-o OVERRIDES]
                             [--batch-weight-key BATCH_WEIGHT_KEY]
                             [--extend-vocab]
                             [--embedding-sources-mapping EMBEDDING_SOURCES_MAPPING]
                             [--include-package INCLUDE_PACKAGE]
                             archive_file input_file

    Evaluate the specified model + dataset

    positional arguments:
    archive_file            path to an archived trained model
    input_file              path to the file containing the evaluation data

    optional arguments:
    -h, --help              show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file to save metrics
    --weights-file WEIGHTS_FILE
                            a path that overrides which weights file to use
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a JSON structure used to override the experiment
                            configuration
    --batch-weight-key BATCH_WEIGHT_KEY
                            If non-empty, name of metric used to weight the loss
                            on a per-batch basis.
    --extend-vocab          if specified, we will use the instances in your new
                            dataset to extend your vocabulary. If pretrained-file
                            was used to initialize embedding layers, you may also
                            need to pass --embedding-sources-mapping.
    --embedding-sources-mapping EMBEDDING_SOURCES_MAPPING
                            a JSON dict defining mapping from embedding module
                            path to embeddingpretrained-file used during training.
                            If not passed, and embedding needs to be extended, we
                            will try to use the original file paths used during
                            training. If they are not available we will use random
                            vectors for embedding extension.
    --include-package INCLUDE_PACKAGE
                            additional packages to include
"""
from typing import Dict, Any
import argparse
import logging
import json
from timeit import default_timer as timer

from allennlp.commands.subcommand import Subcommand
from allennlp.common.util import prepare_environment

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.training.util import evaluate
from allennlp.common import Params
from allennlp.common.util import import_submodules

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Evaluate(Subcommand):
    def add_parser(self, parser: argparse):
        # pylint: disable=protected-access
        parser.add_argument('archive_file', type=str, help='path to an archived trained model')

        parser.add_argument('input_file', type=str, help='path to the file containing the evaluation data')

        parser.add_argument('--output-file', type=str, help='path to output file')

        parser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        cuda_device = parser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device',
                                 type=int,
                                 default=-1,
                                 help='id of GPU to use (if any)')

        parser.add_argument('-t', '--thresholds',
                               type=str,
                               default="0",
                               help='A comma (,) or underscore (_) separated list of temperature thresholds to consider')

        parser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a JSON structure used to override the experiment configuration')

        parser.add_argument('--batch-weight-key',
                               type=str,
                               default="",
                               help='If non-empty, name of metric used to weight the loss on a per-batch basis.')

        parser.add_argument('--extend-vocab',
                               action='store_true',
                               default=False,
                               help='if specified, we will use the instances in your new dataset to '
                                    'extend your vocabulary. If pretrained-file was used to initialize '
                                    'embedding layers, you may also need to pass --embedding-sources-mapping.')

        parser.add_argument('--embedding-sources-mapping',
                               type=str,
                               default="",
                               help='a JSON dict defining mapping from embedding module path to embedding'
                               'pretrained-file used during training. If not passed, and embedding needs to be '
                               'extended, we will try to use the original file paths used during training. If '
                               'they are not available we will use random vectors for embedding extension.')
        parser.add_argument('--include-package',
                               type=str,
                               action='append',
                               default=[],
                               help='additional packages to include')
        parser.set_defaults(func=evaluate_from_args)


def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    logging.getLogger('allennlp.common.params').disabled = True
    logging.getLogger('allennlp.nn.initializers').disabled = True
    logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    print("config: ", config.as_dict())

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    logger.info("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    embedding_sources: Dict[str, str] = (json.loads(args.embedding_sources_mapping)
                                         if args.embedding_sources_mapping else {})
    if args.extend_vocab:
        logger.info("Vocabulary is being extended with test instances.")
        model.vocab.extend_from_instances(Params({}), instances=instances)
        model.extend_embedder_vocab(embedding_sources)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    thrs = args.thresholds.replace("_", ",").split(",")

    for thr in thrs:
        model._temperature_threshold = float(thr)
        start_time = timer()
        metrics = evaluate(model, instances, iterator, args.cuda_device, args.batch_weight_key)
        elapsed_time = timer() - start_time
        metrics['time'] = elapsed_time

        logger.info("Finished evaluating.")
        logger.info("Metrics:")
        for key, metric in metrics.items():
            logger.info("%s: %s: %s", thr, key, metric)

        output_file = args.output_file
        if output_file:
            with open(output_file + "_" + thr, "w") as file:
                json.dump(metrics, file, indent=4)
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AllenNLP for the MultilossBertForClassification model with multiple thresholds")
    e = Evaluate()
    e.add_parser(parser)
    args = parser.parse_args()
    for package_name in getattr(args, 'include_package', ()):
        import_submodules(package_name)

    args.func(args)
