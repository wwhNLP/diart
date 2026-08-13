def read_rttm(path):
    segments = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            _, file_id, channel, start, dur, spk = parts[:6]
            start, dur = float(start), float(dur)
            segments.append(
                {
                    "file": file_id,
                    "channel": channel,
                    "start": start,
                    "end": round(start + dur, 3),
                    "duration": dur,
                    "speaker": spk,
                }
            )
    return segments


segs = read_rttm("output.rttm")
for s in segs:
    print(f"{s['speaker']}: {s['start']:.2f}s - {s['end']:.2f}s")
