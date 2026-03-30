import argparse
from qscore.q_score import calculate_per_residue_q_scores

def add_args(parser):
    parser.add_argument(
        "--structure-path",
        "--s",
        "-s",
        help="Path to structure mmCIF/PDB file",
        required=True,
    )
    parser.add_argument(
        "--volume-path",
        "--v",
        "-v",
        help="Path to cryo-EM map MRC file",
        required=True,
    )
    parser.add_argument(
        "--output-path",
        "--o",
        "-o",
        help="Where to write output files",
        default="output.csv",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=8,
        help="Number of radial samples per radius. Use smaller values to trade fidelity for speed.",
    )
    return parser

def main(parsed_args):
    calculate_per_residue_q_scores(
        parsed_args.structure_path,
        parsed_args.volume_path,
        parsed_args.output_path,
        num_points=parsed_args.num_points,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)
