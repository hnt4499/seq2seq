"""
Utilities functions.
"""


def compare_config(cfg_1, cfg_2):
    """Compare two config dictionaries. Useful for checking when resuming from
    previous session.

    Parameters
    ----------
    cfg_1 : dict
    cfg_2 : dict

    Returns
    -------
    Returns True when the two configs match (with some exclusions), False
    otherwise.
    """
    # This might need to be modified every time an API is changed
    to_compare = [
        "data.bitext_files", "data.src_vocab_path", "data.tgt_vocab_path",
        "model", "training.work_dir", "training.num_epochs", "training.testing"
    ]

    for component in to_compare:
        curr_scfg_1, curr_scfg_2 = cfg_1, cfg_2  # sub configs
        for key in component.split("."):
            if key not in curr_scfg_1 or key not in curr_scfg_2:
                return False
            curr_scfg_1 = curr_scfg_1[key]
            curr_scfg_2 = curr_scfg_2[key]
        if curr_scfg_1 != curr_scfg_2:
            return False
    return True
