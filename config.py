import argparse
import logging


def get_arguments() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="""Compute interpolation of 3D images onto a high-definition grid
        based on the accuracy of images alignments (i.e. the distance between the
        projected voxel center of the grid and the voxel center of the image).""",
    )

    # Input/Output arguments and options
    parser.add_argument("data_dir", help="path to data directory")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="specify an output directory",
    )
    parser.add_argument(
        "--study-name",
        default="",
        help="name of the study",
    )

    # Script specific options
    parser.add_argument(
        "-r",
        "--resolution",
        default=1,
        help="resolution of the MNI template grid",
    )
    parser.add_argument(
        "--transform-dir",
        default="ANTs_iteration_0",
        help="name of the transforms to apply (should be a directory in `data_dir`)",
    )
    parser.add_argument(
        "-w",
        "--no-weight",
        default=True,
        action="store_false",
        help="specify to NOT weight the interpolation using the projected distances",
    )
    parser.add_argument(
        "-b",
        "--n-batches",
        default=100,
        help="number of coordinates batches to compute in parallel",
    )

    parser.add_argument(
        "-j",
        "--n-jobs",
        default=1,
        help="number of jobs to send using joblib",
    )

    parser.add_argument(
        "-v",
        "--verbosity",
        action="count",
        default=1,
        help="""increase output verbosity (-v: standard logging infos; -vv: logging
        infos and NiLearn verbose; -vvv: debug)""",
    )

    args = parser.parse_args()

    return args


def logger_config(verbosity_level: int = 1):
    logging_level_map = {
        0: logging.WARN,
        1: logging.INFO,
        2: logging.DEBUG,
    }

    logging.basicConfig(
        # filename='example.log',
        # format='%(asctime)s %(levelname)s:%(message)s',
        format="%(levelname)s: %(message)s",
        level=logging_level_map[min([verbosity_level, 2])],
    )

    logging.captureWarnings(True)
