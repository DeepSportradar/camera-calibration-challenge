import numpy as np
from typing import List, Tuple

# Intersections FIBA court
#                                                       KEY AREA
#                                                        width
#                                             <--------------------------->
#    +----------+----------------------------+-----------------------------+----------------------------+------------+
#    |          |                        ^   |     ^ BOARD            ^    |                            |            |
#    |          |                        |   |     | offset           | RIM                             |            |
#    |          |                        |   |     v   ------------   | offset                          |            |
#    |          |                        |   |            ( X )       v    |                            |            |
#    | 3-POINTS |                        |   |             ˘-\             |                            |            |
#    |  limit   |               KEY AREA |   |                \            |                            |            |
#    |<-------->|                lenght  |   |                 \           |                            |            |
#    |          |                        |   |                  \          |                            |            |
#    |           |                       |   |                   \         |                           |             |
#    |           |                       |   |                    \        |                           |             |
#    |            \                      |   |                     \       |                          /              |
#    |             |                     |   |                      \      |                         |               |
#    |              \                    v   |                       \     |                        /                |
#    |               \                       +-----+-----------------+\----+                       /                 |
#    |                 \                           |<--------------->| \                         /                   |
#    |                   '.                         \    CIRCLE d   /   \ 3-POINTS            .'                     |
#    |                     '-.                        '.         .'      \  dist           .-'                       |
#    |                        '-.                        `-----´          \             .-'                          |
#    |                           '--.                                      \        .--'                             |
#    |                               '--.                                   \   .--'                                 |
#    |                                   '---___                         ___-ˇ-'                                     |
#    |                                          ''-------_______-------''                                            |
#    |                                                                                                               |

# Lines color list:
# 0. BACKGROUND
# 1. court.edges[0] = top left - top right
# 2. court.edges[2] = top right - bottom right
# 3. court.edges[3] = bottom right - bottom left
# 4. court.edges[4] = bottom left - top left
# 5. midcourt-line = top mid - bottom mid -> Point3D(self.w/2,0,0), Point3D(self.w/2,self.h,0)
# 6. central circle
# LEFT PAINT
# 7. left, key-area top - key-area lenght, key-area top -> Point3D(0, self.h/2-w/2, 0), Point3D(l, self.h/2-w/2, 0)
# 8. key-area lenght, key-area top - key-area lenght, key-area bottom
# 9. left, key-area bottom - key-area lenght, key-area bottom
# 10. left key area circle
# RIGHT PAINT
# 11. right, key-area top - key-area lenght, key-area top
# 12. key-area lenght, key-area top - key-area lenght, key-area bottom
# 13. right, key-area bottom - key-area lenght, key-area bottom
# 14. right key area circle
# LEFT 3-points-line
# 15. left 3 points arc
# 16. left top line 3 points arc
# 17. left bottom line 3 points arc
# RIGHT 3-points-line
# 18. right 3 points arc
# 19. right top line 3 points arc
# 20. right bottom line 3 points arc

# Intersections list
# 1 - 4 = (0, 0)
# 1 - 2 = (2800, 0)
# 1 - 5 = (1400, 0)
# 1 - 8 = (575, 0)
# 1 - 12 = (2225, 0)
# 2 - 19 = (2800, 90)
# 2 - 13 = (2800, 995) # 1500/2 + 490/2
# 2 - 11 = (2800, 505) # 1500/2 - 490/2
# 2 - 20 = (2800, 1410) # 1550 - 90
# 3 - 4 = (0, 1500)
# 3 - 2 = (2800, 1500)
# 3 - 5 = (1400, 1500)
# 3 - 8 = (575, 1500)
# 3 - 12 = (2225, 1500)
# 4 - 16 = (0, 90)
# 4 - 7 = (0, 995) # 1500/2 + 490/2
# 4 - 9 = (0, 505) # 1500/2 - 490/2
# 4 - 17 = (0, 1410) # 1550 - 90
# 7 - 8 = (575, 505) # 1500/2 - 490/2
# 9 - 8 = (575, 995) # 1500/2 + 490/2
# 11 - 12 = (2225, 505) # 1500/2 - 490/2
# 13 - 12 = (2225, 995) # 1500/2 + 490/2
# 8 - 10 top = (575, 570) # 1500/2 - 360/2
# 8 - 10 bottom = (575, 930) # 1500/2 + 360/2
# 8 - 10 top = (575, 570) # 1500/2 - 360/2
# 8 - 10 bottom = (575, 930) # 1500/2 + 360/2
# 16 - 5 = (1400, 90)
# 17 - 5 = (1400, 1410)
# 19 - 5 = (1400, 90)
# 20 - 5 = (1400, 1410)

INTERSECTIONS = {
    "1-4": (0.0, 0.0, 0.0),
    "1-2": (2800.0, 0.0, 0.0),
    "1-5": (1400.0, 0.0, 0.0),
    "1-8": (575.0, 0.0, 0.0),
    "1-12": (2225.0, 0.0, 0.0),
    "2-19": (2800.0, 90.0, 0.0),
    "2-13": (2800.0, 995.0, 0.0),  # 1500/2 + 490/2
    "2-11": (2800.0, 505.0, 0.0),  # 1500/2-490/2
    "2-20": (2800.0, 1410.0, 0.0),  # 1550-90
    "3-4": (0.0, 1500.0, 0.0),
    "3-2": (2800.0, 1500.0, 0.0),
    "3-5": (1400.0, 1500.0, 0.0),
    "3-8": (575.0, 1500.0, 0.0),
    "3-12": (2225.0, 1500.0, 0.0),
    "4-16": (0.0, 90.0, 0.0),
    "4-7": (0.0, 505.0, 0.0),  # 0, 1500/2 + 490/2
    "4-9": (0.0, 995.0, 0.0),  # 1500/2 - 490/2
    "4-17": (0.0, 1410.0, 0.0),  # 1550-90
    "7-8": (575.0, 505.0, 0.0),  # 1500/2-490/2
    "9-8": (575.0, 995.0, 0.0),  # 1500/2 + 490/2
    "11-12": (2225.0, 505.0, 0.0),  # 1500/2-490/2
    "13-12": (2225.0, 995.0, 0.0),  # 1500/2 + 490/2
    "16-5": (1400.0, 90.0, 0.0),
    "17-5": (1400.0, 1410.0, 0.0),
    "19-5": (1400.0, 90.0, 0.0),
    "20-5": (1400.0, 1410.0, 0.0),
}


def return_3D_points_mean(str_: str):
    sp = str_.split("_")
    firstp = np.array(INTERSECTIONS[sp[0]])
    secondp = np.array(INTERSECTIONS[sp[1]])
    return tuple(np.mean((firstp, secondp), 0))


MIDPOINTS = {
    "1-4_1-8": return_3D_points_mean("1-4_1-8"),
    "1-8_1-5": return_3D_points_mean("1-8_1-5"),
    "1-5_1-12": return_3D_points_mean("1-5_1-12"),
    "1-12_1-2": return_3D_points_mean("1-12_1-2"),
    "2-11_2-13": return_3D_points_mean("2-11_2-13"),
    "3-12_3-2": return_3D_points_mean("3-12_3-2"),
    "3-5_3-12": return_3D_points_mean("3-5_3-12"),
    "3-8_3-5": return_3D_points_mean("3-8_3-5"),
    "3-4_3-8": return_3D_points_mean("3-4_3-8"),
    "4-7_4-9": return_3D_points_mean("4-7_4-9"),
    "4-7_7-8": return_3D_points_mean("4-7_7-8"),
    "7-8_9-8": return_3D_points_mean("7-8_9-8"),
    "4-9_9-8": return_3D_points_mean("4-9_9-8"),
    "16-5_17-5": return_3D_points_mean("16-5_17-5"),
    "19-5_20-5": return_3D_points_mean("19-5_20-5"),
    "11-12_13-12": return_3D_points_mean("11-12_13-12"),
    "11-12_2-11": return_3D_points_mean("11-12_2-11"),
    "13-12_2-13": return_3D_points_mean("13-12_2-13"),
}


def list_segments(img, num_classes=21):
    points = [0]
    for i in range(1, num_classes):
        points.append(np.argwhere(img == i))
    return points


def mid_points(points):
    ml = (
        (points[0][0] + points[1][0]) * 0.5,
        (points[0][1] + points[1][1]) * 0.5,
    )
    return ml


def find_intersections(img: np.ndarray) -> Tuple[List, List]:
    """Find lines and intersections between lines.
    The intersections will be used as 2D to 3D correspondences.

    Args:
        img (np.ndarray): _description_

    Returns:
        _type_: _description_
    """

    points = list_segments(img)

    points3d = []
    points2d = []
    found_2d = {}

    for key, value in INTERSECTIONS.items():
        sp = key.split("-")
        idx1, idx2 = (int(sp[0]), int(sp[1]))
        if points[idx1].size > 0 and points[idx2].size > 0:
            try:
                m1, b1 = np.polyfit(points[idx1][:, 1], points[idx1][:, 0], 1)
                m2, b2 = np.polyfit(points[idx2][:, 1], points[idx2][:, 0], 1)
            except:
                continue
            xi = (b1 - b2) / (m2 - m1)
            yi = m1 * xi + b1
            points3d.append(value)
            points2d.append((xi, yi))
            found_2d[key] = (xi, yi)

    for key, value in MIDPOINTS.items():
        sp = key.split("_")
        p1 = found_2d.get(sp[0], None)
        p2 = found_2d.get(sp[1], None)
        if p1 and p2:
            points2d.append(mid_points([p1, p2]))
            points3d.append(value)
            x = points2d[-1]
            firstp = np.array(INTERSECTIONS[sp[0]])
            secondp = np.array(INTERSECTIONS[sp[1]])

            points2d.append(mid_points([p1, x]))
            points3d.append(tuple(np.mean((firstp, value), 0)))

            points2d.append(mid_points([p2, x]))
            points3d.append(tuple(np.mean((secondp, value), 0)))

    return points2d, points3d
