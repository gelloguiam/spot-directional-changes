import math

def get_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2

    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return d

p1 = (506, 869)
p2 = (416, 966)

(x1, y1) = p1
(x2, y2) = p2
x3 = x1
y3 = y2

p3 = (x3, y3)

a = get_distance(p1, p2)
b = get_distance(p1, p3)
c = get_distance(p2, p3)

z = (c**2 - a**2 - b**2)/(-2*b*a)
rad = math.acos(z)
deg = math.degrees(rad)

print(p1, p2, p3, a, b, c, z, deg)