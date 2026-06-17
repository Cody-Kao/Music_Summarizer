# pipeline/candidate_generation.py

def build_bar_segments(downbeats):
    """
    Convert downbeats into bar segments.
    """
    bars = []
    downbeat_times = [d["time"] for d in downbeats if d["beat"] == 1]

    for i in range(len(downbeat_times) - 1):
        bars.append({
            "bar_index": i,
            "start": downbeat_times[i],
            "end": downbeat_times[i + 1]
        })

    return bars


def assign_bars_to_sections(bars, sections):
    """
    Match bars into SongFormer sections with strict boundary tracking.
    """
    results = []

    for bar in bars:
        center = (bar["start"] + bar["end"]) / 2
        matched_label = None
        sec_idx = -1
        sec_start = 0.0
        sec_end = 0.0

        for idx, sec in enumerate(sections):
            if sec["start"] <= center < sec["end"]:
                matched_label = sec["label"]
                sec_idx = idx            # 👈 Tracks the exact JSON block
                sec_start = sec["start"] # 👈 Saves the true semantic start
                sec_end = sec["end"]     # 👈 Saves the true semantic end
                break

        results.append({
            **bar,
            "section": matched_label,
            "section_index": sec_idx,
            "section_start": sec_start,
            "section_end": sec_end
        })

    return results


def generate_candidates(
    bars,
    min_duration=8,
    max_duration=20,
    bleed_tolerance=0.3 # 👈 NEW: Allow up to 300ms of bleed for slight AI model misalignment
):
    """
    Generate candidate excerpts strictly contained within semantic boundaries.
    """
    candidates = []
    n = len(bars)

    for i in range(n):
        current_section = bars[i]["section"]
        sec_idx = bars[i].get("section_index", -1)
        sec_start = bars[i].get("section_start", 0.0)
        sec_end = bars[i].get("section_end", 0.0)

        # Skip bars that couldn't be matched to a section
        if current_section is None:
            continue

        # ==========================================
        # 🛑 STRICT START FILTER
        # ==========================================
        # If this bar starts significantly BEFORE the official section begins, 
        # it bleeds backward into the previous section. Do not allow it to start a candidate.
        if bars[i]["start"] < (sec_start - bleed_tolerance):
            continue

        for j in range(i + 1, n):
            
            # Ensure it's the exact same section block (prevents merging adjacent same-label sections)
            if bars[j]["section"] != current_section or bars[j].get("section_index", -1) != sec_idx:
                break

            # ==========================================
            # 🛑 STRICT END FILTER
            # ==========================================
            # If this bar ends significantly AFTER the official section ends,
            # it bleeds forward into the next section. Stop expanding this candidate!
            if bars[j]["end"] > (sec_end + bleed_tolerance):
                break

            start = bars[i]["start"]
            end = bars[j]["end"]
            duration = end - start

            if duration < min_duration:
                continue

            if duration > max_duration:
                break

            candidates.append({
                "start_bar": bars[i]["bar_index"],
                "end_bar": bars[j]["bar_index"],
                "start_time": start,
                "end_time": end,
                "duration": duration,
                "section": current_section
            })

    return candidates