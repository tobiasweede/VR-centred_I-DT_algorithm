# I-DT multiprocessing
from IDT_alg_VR_centred import IDTVR
from itertools import product
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
    )

    # head pose
    df_et[["head_pose_x", "head_pose_y", "head_pose_z"]] = df_et[
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
    df_et["elapsedTime"] = df_et["#timestamp_unity"] - firstTimestamp

    # create fixations
    idt_vr = IDTVR(numba_allow=True, time_th=0.1)
    _, df_et_fixations = idt_vr.fit_compute(
        df_et, time="elapsedTime", debug=True
    )

    # write fixation csv
    subject = gazelog.name[:4]
    write_path = Path(save_dir + "/" + subject + "_idt_fixations.csv")
    df_et_fixations.to_csv(write_path)


def main():
    gaze_files = get_file_names()

    max_jobs = 10
    pool = multiprocessing.Pool(max_jobs)
    results = []

    for gazelog in gaze_files:
        r = pool.apply_async(create_fixations, args=(gazelog,))
        results.append(r)
        break

    for r in results:
        r.wait()
    
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
