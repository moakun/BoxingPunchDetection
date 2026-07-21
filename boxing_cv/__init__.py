"""Boxing Punch Detection — real-time upper-body pose analysis.

Modules
-------
constants        : landmark indices, skeleton connections, tunable config
pose             : MediaPipe Pose wrapper -> (33, 4) landmark array
features         : normalization, joint angles, kinematics, ring buffer
stance           : orthodox / southpaw detection
spotter          : per-arm punch state machine (temporal segmentation)
classify_rules   : v1 rule-based (type, zone) classifier
classify_nn      : v2 temporal neural net (definition + ONNX inference)
naming           : (lead/rear, trajectory, stance) -> boxing term
overlay          : skeleton / label / counter rendering
"""

__version__ = "0.1.0"
