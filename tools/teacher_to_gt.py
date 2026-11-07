"""Turn the RGB teacher's unlabelled detections (tools/rgb_teacher.py report: work/curation/teacher/unlabelled_<seq>.txt,
MOT gt lines with class 13 and conf 0 on every second source frame) into gt/gt_teacher.txt files next to the gt.txt of
every sequence folder the monitor passes play: the source-frame-rate folders (MOT17 train, PersonPath22 windows) and
their 60 fps renderings (tools/make60.py). The labeller (tools/record_dataset.py) reads gt_teacher.txt as regions to
ignore. A teacher frame's boxes are given to every frame whose nearest teacher frame it is.
Usage: teacher_to_gt.py [TEACHER_DIR work/curation/teacher]
"""
import configparser
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
W = ROOT / "work"
TREES = [  # (source folder, 60 fps folder or None)
    (W / "datasets/mot17/unzipped/MOT17/train", W / "datasets/mot17_60/train"),
    (W / "datasets/pp22", W / "datasets/pp22_60"),
]


def seq_len_fps(seq_dir):
    c = configparser.ConfigParser()
    c.read(Path(seq_dir) / "seqinfo.ini")
    return int(c["Sequence"]["seqLength"]), float(c["Sequence"]["frameRate"])


def read_unlabelled(path):
    """{1-based source frame: [(x, y, w, h, conf)]}"""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split(",")
        if len(f) >= 9:
            out.setdefault(int(float(f[0])), []).append(tuple(float(v) for v in (f[2], f[3], f[4], f[5], f[8])))
    return out


def write_gt(seq_dir, mapping, teacher):
    """mapping: 1-based frame of seq_dir -> 1-based teacher frame"""
    lines = []
    for frame, tf in mapping.items():
        for x, y, w, h, conf in teacher.get(tf, []):
            lines.append(f"{frame},-1,{x:.1f},{y:.1f},{w:.1f},{h:.1f},0,13,{conf:.2f}")
    (Path(seq_dir) / "gt" / "gt_teacher.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main(teacher_dir):
    teacher_dir = Path(teacher_dir)
    n_seq = n_lines = 0
    for src_root, root60 in TREES:
        for seq in sorted(p for p in src_root.glob("**/seqinfo.ini")):
            seq_dir = seq.parent
            unl = teacher_dir / f"unlabelled_{seq_dir.name}.txt"
            if not unl.exists():
                continue
            teacher = read_unlabelled(unl)
            tframes = sorted(teacher)
            if not tframes:
                continue
            n, fps = seq_len_fps(seq_dir)

            def nearest(f):  # nearest teacher frame (1-based) to the 1-based source frame f
                return min(tframes, key=lambda t: abs(t - f))

            n_lines += write_gt(seq_dir, {f: nearest(f) for f in range(1, n + 1)}, teacher)
            n_seq += 1
            if root60 is not None:
                for d60 in root60.glob(f"**/{seq_dir.name}"):
                    if (d60 / "seqinfo.ini").exists():
                        n60, _ = seq_len_fps(d60)
                        # 60 fps frame k (0-based) shows source position k * fps / 60 (make60.py)
                        n_lines += write_gt(d60, {k + 1: nearest(int(round(min(k * fps / 60, n - 1))) + 1) for k in range(n60)},
                                            teacher)
    print(f"{n_seq} sequences, {n_lines} ignore-region lines written as gt/gt_teacher.txt")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else W / "curation/teacher")
