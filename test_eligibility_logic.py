from quiz_data import check_eligibility, requires_physics

# Reported case: CS BSc does not use physics, but fails average and math.
eligible, reason = check_eligibility(
    "CS", "tawjihi", "bachelor", 40, None, 62.5
)
assert eligible is False
assert "Overall average is 62.5%" in reason
assert "Mathematics is 40.0%" in reason
assert "Physics is not a separate minimum requirement" in reason

# CS BSc passes at 80 average and 80 mathematics.
eligible, reason = check_eligibility(
    "CS", "tawjihi", "bachelor", 80, None, 80
)
assert eligible is True, reason

# Engineering BSc requires average, math, and physics.
eligible, reason = check_eligibility(
    "Electrical Engineering", "tawjihi", "bachelor", 80, 79, 80
)
assert eligible is False
assert "Physics is 79.0%" in reason

# Technical CS uses 70/70.
eligible, reason = check_eligibility(
    "Computer Science", "tawjihi", "tech", 70, None, 70
)
assert eligible is True, reason

assert requires_physics("Electrical Engineering") is True
assert requires_physics("CS") is False

print("PASS: eligibility logic checks")
