# Hand connections as index pairs (same topology as the old HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),            # thumb
    (0,5), (5,6), (6,7), (7,8),            # index
    (0,9), (9,10),(10,11), (11,12),       # middle
    (0,13), (13,14), (14,15), (15,16),     # ring
    (0,17), (17,18), (18,19), (19,20),     # pinky
    (5,9), (9,13), (13,17)             # palm
]

# IDs of fingertips
FINGERTIP_IDS = {4, 8, 12, 16, 20}
