"""Reporting statistics shared by the paper builders.

Kept outside ``src/`` deliberately: every frozen study package pins the whole
``src`` tree by checksum, so a helper added there would invalidate the frozen
packages that ``verify_package`` re-checks against the working source.
"""
from __future__ import annotations


def sign_flip_resolution_floor(nonzero_blocks):
    """Smallest two-sided sign-flip p attainable with this many non-zero blocks.

    An exact sign-flip test enumerates $2^m$ equally likely sign assignments of
    the $m$ non-zero paired differences. The most extreme outcome keeps two of
    them (all-positive and all-negative), so no configuration can report below
    $2/2^{m}$. A contrast whose floor exceeds the chosen level cannot reach
    significance whatever its data show, which is a property of the design
    rather than of the result.
    """
    if type(nonzero_blocks) is not int or nonzero_blocks < 0:
        raise ValueError("Non-zero block count must be a non-negative integer")
    return 1.0 if nonzero_blocks == 0 else min(1.0, 2.0 / 2 ** nonzero_blocks)


def wins_losses_ties(differences, tolerance=1e-12):
    """Split paired per-question differences, treating near-zero as a tie.

    Ties stay in the estimand and in the interval; they only drop out of the
    exact null, where a zero difference carries no sign to flip.
    """
    wins = sum(1 for d in differences if d > tolerance)
    losses = sum(1 for d in differences if d < -tolerance)
    return {"wins": wins, "losses": losses, "ties": len(differences) - wins - losses,
            "questions": len(differences), "nonzero": wins + losses,
            "resolution_floor": sign_flip_resolution_floor(wins + losses)}
