"""
General utility functions not covered elsewhere.
"""
from pathlib import Path
from argparse import Namespace
import json


def save_new_ingest_json(file_name: Path, 
                         args: Namespace):
    """
    Will save a new ingest json file to the indicated file_name. The file will
    be populated with default values. Any existing file will be overwritten.

    !file_name! the file to save

    !arg! the args Namespace which will contain any requested values or defaults

    !target! the target name or will default to file_name.stem if omitted
    """
    new_args = vars(args)
    
    # Remove stuff that should not be seen in the JSON file
    for k in ["file", "new_file", "mission", "author"]:
        if k in new_args:
            new_args.pop(k)

    new_args["target"] = new_args.get("sys_name") or file_name.stem

    # Add some dummy values to demonstrate how various settings are written
    if new_args["clips"] is None or len(new_args["clips"]) == 0:
        new_args["clips"] = [[45000.0, 45001.0]]
    new_args["fitting_params"]["dummy_token"] = "value"

    # Set up a default auto-poly known to work well on TESS light-curves
    new_args["polies"] = [
        { "term": "sf", "degree": 1, "gap_threshold": 0.5 }
    ]

    with open(file_name, "w") as f:
        json.dump(new_args, f, ensure_ascii=False, indent=2)
        print(f"New ingest target JSON file saved to '{f.name}'")
    return