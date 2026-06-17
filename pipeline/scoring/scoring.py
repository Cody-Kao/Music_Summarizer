# pipeline/scoring/scoring.py

""" WEIGHTS = {
    # high-level salience
    "climax": 0.30,
    # musical support signals
    "repetition": 0.10,
    "representativeness": 0.15,
    "vocal_density": 0.10,
    "rms": 0.08,
    "novelty": 0.20,
    "structural_importance": 0.07
} """

def attach_candidate_scores(candidates, WEIGHTS):
    # -----------------------------------
    # Assign score
    # -----------------------------------

    for i, candidate in enumerate(candidates):

        f = candidate["features"]

        score = (
            WEIGHTS["representativeness"]
            * f["representativeness"]

            + WEIGHTS["novelty"]
            * f["novelty"]

            + WEIGHTS["vocal_density"]
            * f["vocal_density"]

            + WEIGHTS["structural_importance"]
            * f["structural_importance"]

            + WEIGHTS["rms"]
            * f["rms"] 

            + WEIGHTS["repetition"]
            * f["repetition"] 
        )
        # heuristic: late segment bonus
        position_bonus = (
            1.0
            + 0.08
            * f["relative_position"]
        )

        score *= position_bonus

        candidate["score"] = float(score)