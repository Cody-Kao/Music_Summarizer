# pipeline/scoring/optimizer.py

from pipeline.scoring.transition import transition_score
from pipeline.scoring.overlap import overlaps
from pipeline.features.embedding import cosine_sim

SECTION_REPEAT_PENALTY = 0.05
SIMILARITY_THRESHOLD = 0.90
SIMILARITY_PENALTY = 0.25
PEAK_MULTIPLIER = 1.5
CHORUS_MULTIPLIER = 1.3  
PEAK_CHORUS_SYNERGY = 2.0  
ESCALATION_MULTIPLIER = 2.0
SCORE_EXPONENT = 1.8
BEAM_WIDTH = 150


def optimize_summary(
    candidates,
    max_duration=50,
    beam_width=BEAM_WIDTH,
    # ==========================================
    # 🎛️ NEW: TUNABLE BEAM PARAMETERS
    # ==========================================
    patient_ratio=0.45,        # % of beam reserved for paths saving their budget
    patience_threshold=0.40,   # Max % of budget a path can use and still be "patient"
    structural_ratio=0.45      # % of beam reserved for Peak/Chorus combos
):
    candidates = sorted(candidates, key=lambda x: x["start_time"])
    beam = [(0.0, 0.0, [], {}, False, False)]

    for candidate in candidates:
        new_beam = []

        for (score, duration, selected, section_counts, has_peak, long_enough) in beam:
            # OPTION 1 — skip
            new_beam.append((score, duration, selected, section_counts, has_peak, long_enough))

            # OPTION 2 — take
            candidate_duration = candidate["duration"]
            if duration + candidate_duration > max_duration:
                continue

            if any(overlaps(prev, candidate) for prev in selected):
                continue

            # Base score
            candidate_score = candidate["score"] ** SCORE_EXPONENT

            is_peak = candidate.get("is_peak", False)
            section_name = candidate.get("section", "")
            is_chorus = section_name.lower() == "chorus"

            # Synergy Scoring
            if is_peak and is_chorus: 
                candidate_score *= PEAK_CHORUS_SYNERGY
            elif is_peak: 
                candidate_score *= PEAK_MULTIPLIER
            if is_chorus: candidate_score *= CHORUS_MULTIPLIER
            
            # ==========================================
            # 🎬 NEW: "Opening Act" Bonus
            # ==========================================
            # Force the algorithm to start the musical journey with a Verse/Intro
            if len(selected) == 0:
                if section_name.lower() in ["verse", "intro"]:
                    candidate_score *= 1.8  # Make starting with Verse irresistible
                elif is_chorus:
                    candidate_score *= 0.8  # Penalize jumping straight into a Chorus

            # Tension Escalation Curve
            rel_pos = candidate["features"]["relative_position"]
            escalation_boost = 1.0 + ((ESCALATION_MULTIPLIER - 1.0) * rel_pos)
            candidate_score *= escalation_boost

            # The "True Climax" Finish Strong Mechanic
            is_closing_segment = (duration + candidate_duration) > (max_duration * 0.75)
            is_late_in_song = rel_pos > 0.65

            if is_closing_segment:
                if (is_peak or is_chorus) and is_late_in_song:
                    candidate_score *= 1.8 
                elif (is_peak or is_chorus) and not is_late_in_song:
                    candidate_score *= 0.8
                elif not has_peak:
                    candidate_score *= 0.1 

            # ==========================================
            # OPTIONAL: Budget Utilization Bonus
            # ==========================================
            # Gently reward paths that get closer to the max duration.
            # E.g., using 45s/60s (75%) gives a 15% bonus to the segment score.
            utilization_ratio = (duration + candidate_duration) / max_duration
            if not long_enough and utilization_ratio > 0.75: # Only reward it if it's actually filling the budget
                candidate_score *= (1.0 + (0.15 * utilization_ratio))
                long_enough = True

            new_score = score + candidate_score

            # Similarity suppression (Acoustic Twin Exemption)
            similarity_penalty = 0.0
            for prev in selected:
                sim = cosine_sim(prev["embedding"], candidate["embedding"])
                if sim > SIMILARITY_THRESHOLD:
                    if is_chorus and prev.get("section", "").lower() == "chorus":
                        similarity_penalty += (SIMILARITY_PENALTY * 0.2) 
                    else:
                        similarity_penalty += SIMILARITY_PENALTY
            new_score -= similarity_penalty

            # Transition bonus
            if len(selected) > 0:
                trans = transition_score(selected[-1], candidate)
                new_score += (0.2 * trans)

            # ==========================================
            # Diversity Penalty & "Chorus Fatigue" Progressive Penalty
            # ==========================================
            repeat_count = section_counts.get(section_name, 0)
            
            if is_chorus:
                if repeat_count <= 4:
                    effective_repeat_count = 0  # 1st and 2nd choruses are penalty-free
                else:
                    effective_repeat_count = repeat_count+2 # 5th+ chorus triggers a MASSIVE penalty to force diversity
            else:
                effective_repeat_count = repeat_count

            new_score -= (effective_repeat_count * SECTION_REPEAT_PENALTY)

            # Update state
            new_counts = dict(section_counts)
            new_counts[section_name] = repeat_count + 1
            new_selected = selected + [candidate]
            new_has_peak = has_peak or is_peak

            new_beam.append(
                (new_score, duration + candidate_duration, new_selected, new_counts, new_has_peak, long_enough)
            )

        # ==========================================
        # 🎛️ IMPLEMENTING THE TUNABLE PRUNING
        # ==========================================
        new_beam = sorted(new_beam, key=lambda x: x[0], reverse=True)
        
        patient = []
        peak_and_chorus = []
        just_chorus = []
        
        # Determine the cutoff for a path to be considered "Patient"
        budget_limit = max_duration * patience_threshold 
        
        for i, state in enumerate(new_beam):
            if state[1] <= budget_limit:
                patient.append((i, state))
                
            has_p = state[4]
            has_c = state[3].get("Chorus", 0) > 0 or state[3].get("chorus", 0) > 0
            
            if has_p and has_c: peak_and_chorus.append((i, state))
            elif has_c: just_chorus.append((i, state))

        protected = []
        seen = set()
        
        def pull_into_beam(pool, quota):
            added = 0
            for idx, state in pool:
                if idx not in seen:
                    protected.append(state)
                    seen.add(idx)
                    added += 1
                if added >= quota:
                    break
                    
        # Calculate exactly how many slots each category gets based on your ratios
        patient_slots = int(beam_width * patient_ratio)
        peak_chorus_slots = int(beam_width * (structural_ratio * (3/4)))
        just_chorus_slots = int(beam_width * (structural_ratio * (1/4)))
        
        # Fill the reserved slots safely
        pull_into_beam(patient, patient_slots)
        pull_into_beam(peak_and_chorus, peak_chorus_slots)
        pull_into_beam(just_chorus, just_chorus_slots)
        
        # Fill whatever slots are remaining with the best "General" paths
        for i, state in enumerate(new_beam):
            if i not in seen:
                protected.append(state)
                seen.add(i)
            if len(protected) >= beam_width:
                break
                
        beam = protected 

    # -----------------------------------
    # Multi-Tier Selection 
    # -----------------------------------
    def count_choruses(state):
        return state[3].get("Chorus", 0) + state[3].get("chorus", 0)

    tier_1 = [s for s in beam if s[4] and count_choruses(s) >= 2]
    tier_2 = [s for s in beam if s[4] and count_choruses(s) >= 1]
    tier_3 = [s for s in beam if s[4]]
    
    if tier_1: best_tier = tier_1
    elif tier_2: best_tier = tier_2
    elif tier_3: best_tier = tier_3
    else: best_tier = beam 
        
    best = max(best_tier, key=lambda x: x[0])
    final_selection = best[2]

    # ==========================================
    # PRINT SUMMARY WITH CHORUS TRACKING
    # ==========================================
    print("\n" + "="*50)
    print("🏆 FINAL SUMMARY SELECTION")
    print("="*50)
    
    total_duration = 0
    chorus_count = 0
    
    for i, cand in enumerate(final_selection):
        start = cand["start_time"]
        end = cand["end_time"]
        dur = cand["duration"]
        sec = cand.get("section", "N/A")
        
        is_peak = cand.get("is_peak", False)
        peak_str = "⭐ PEAK" if is_peak else "      "
        if sec.lower() == "chorus":
            chorus_count += 1
        
        print(f"[{i+1}] {start:05.1f}s -> {end:05.1f}s | Dur: {dur:04.1f}s | {peak_str} | Sec: {sec}")
        total_duration += dur

    print("-" * 50)
    print(f"Total Duration : {total_duration:.1f}s")
    print(f"Total Choruses : {chorus_count}")
    print("=" * 50 + "\n")

    return final_selection