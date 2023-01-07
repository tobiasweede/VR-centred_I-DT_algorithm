#!/usr/bin/python
# -*- coding: utf-8 -*-

# I-DT multiprocessing
from typing import Type
from IDT_alg_VR_centred_SMI import IDTVR
from pathlib import Path
import multiprocessing
import pandas as pd


def get_file_names(source_dir="./gazelogs"):
    p = sorted(Path(source_dir).glob("*gazelog*"))
    return p


def create_fixations(gazelog: Path, save_dir="./fixations"):
    df_et = pd.read_csv(
        gazelog,
        skiprows=15,
        sep=";",
        encoding="utf-8",
        low_memory=False,
    )

    # head pose
    df_et[["head_pos_x", "head_pos_y", "head_pos_z"]] = df_et[
        "head_pos"
    ].str.split(",", expand=True)

    # gaze vector
    df_et[["et_x", "et_y", "et_z"]] = df_et["bino_hitObject_pos"].str.split(
        ",", expand=True
    )

    # POR
    df_et[["POR X", "POR Y"]] = df_et["smi_bino_por"].str.split(
        ",", expand=True
    )

    # fix timestamp
    firstTimestamp = df_et.loc[0, "#timestamp_unity"]
    if isinstance(firstTimestamp, float):
        df_et["elapsedTime"] = df_et["#timestamp_unity"] - firstTimestamp
    else:
        raise(TypeError("Wrong firstTimestamp type"))

    # keep relevant cols
    df_et = df_et[
        [
            "head_pos_x",
            "head_pos_y",
            "head_pos_z",
            "et_x",
            "et_y",
            "et_z",
            "POR X",
            "POR Y",
            "elapsedTime",
            "bino_hitObject",
            "bino_hitObject_Feature",
        ]
    ]

    # cast columns to float (except the last 3)
    df_et[df_et.columns[:-3]] = df_et[df_et.columns[:-3]].astype(float)

    # calculate time for seach sample
    df_et["time_delta"] = df_et["elapsedTime"] - df_et["elapsedTime"].shift(1)

    # create fixations
    idt_vr = IDTVR(numba_allow=True, time_th=0.1)
    _, df_et_fixations = idt_vr.fit_compute(
        df_et, time="elapsedTime", debug=False
    )

    # write fixation csv
    subject = gazelog.name.split("_")[0]
    write_path = Path(save_dir + "/" + subject + "_idt_fixations.csv")
    df_et_fixations.to_csv(write_path)


def main(debug=False):

    gaze_files = get_file_names()

    if debug:
        for gazelog in gaze_files:
            create_fixations(gazelog)
        exit

    max_jobs = 24
    pool = multiprocessing.Pool(max_jobs)
    results = []

    for gazelog in gaze_files:
        r = pool.apply_async(create_fixations, args=(gazelog,))
        results.append(r)

    for r in results:
        r.wait()
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
