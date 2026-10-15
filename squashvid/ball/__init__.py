"""Ball tracking stream: zero-shot heatmap trackers, a labelling app and per-region evaluation.

The squash ball in these clips is 1.6-4.5 px wide, so every detector here runs on
short frame stacks (temporal heatmap networks) rather than single frames, and
every number is scored per image region because the front wall, the floor and
the door strip behave like different problems.
"""
