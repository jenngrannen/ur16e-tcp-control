import numpy as np

# # input data left
# ins = [[-0.62048,0.03015,-0.32585], [-0.58158, -0.03185, -0.36940], [-0.63311, 0.12783, -0.36940], [-0.46867, 0.23780, -0.47524]]  # <- points
# out = [[0.62052, -0.25083, 0.61001], [0.57594, -0.32265, 0.59979], [0.64239,-0.21371,0.70872], [0.48911, -0.20011, 0.87198]] # <- mapped to

# test_ins = [[-0.41557,0.31340,-0.40466],[-0.46860,0.09410,-0.32770]]
# test_outs = [[0.44345,-0.09342,0.87885],[0.47540,-0.19693,0.66653]]

ins = [[-0.52497, -0.19873, -0.28406], [-0.61570, -0.38688, -0.21733], [-0.48611, -0.51371, -0.36740], [-0.51996, -0.67218, -0.45191]]
out = [[0.52496743, -0.34138827, 0.4603461], [0.61570977, -0.42723104, 0.28011543], [0.4861251, -0.62305884, 0.2965394], [0.51996743, -0.7948641, 0.24427336]]
test_outs = [[0.52633607, -0.62968755, 0.42984542], [0.47329373, -0.46826591, 0.50465731]]
test_ins = [[-0.52633, -0.42412, -0.46638], [-0.47329, -0.25713, -0.40510]]

# # input data right
# ins = [[0.75780, 0.01926, -0.03268], [0.80826, 0.06097, -0.08499], [0.86943, -0.16581, -0.13630], [0.58340, -0.21483, -0.26278]] # <- points (view/world)
# out = [[-0.757407, -0.001312, 0.444915], [-0.80723567, -0.00827929, 0.51195435], [-0.87185053, -0.20418717, 0.38857162], [-0.5865983, -0.3314091, 0.44025964]] # <- mapped to (base)

# test_ins = [[0.62424, -0.15410, -0.09205], [0.78503, -0.05769, -0.21757]]
# test_outs = [[-0.62652927, -0.16732076, 0.36286983], [-0.78582362, -0.18616926, 0.52156958]]

# calculations
l = len(ins)
B = np.vstack([np.transpose(ins), np.ones(l)])
D = 1.0 / np.linalg.det(B)
entry = lambda r,d: np.linalg.det(np.delete(np.vstack([r, B]), (d+1), axis=0))
M = [[(-1)**i * D * entry(R, i) for i in range(l)] for R in np.transpose(out)]
A, t = np.hsplit(np.array(M), [l-1])
t = np.transpose(t)[0]
# output
print("Affine transformation matrix:\n", A)
print("Affine transformation translation vector:\n", t)
# unittests
print("TESTING:")
for p, P in zip(np.array(ins), np.array(out)):
  image_p = np.dot(A, p) + t
  result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
  print(p, " mapped to: ", image_p, " ; expected: ", P, result)
for p, P in zip(np.array(test_ins), np.array(test_outs)):
  image_p = np.dot(A, p) + t
  result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
  print(p, " mapped to: ", image_p, " ; expected: ", P, result)

## inverse transform:
print('calculate inverse transform')
for p,P in zip(np.array(test_outs), np.array(test_ins)):
    image_p = np.dot(np.linalg.inv(A), p-t)
    result = "[OK]" if np.allclose(image_p, P) else "[ERROR]"
    print(p, " mapped to: ", image_p, " ; expected: ", P, result)


## transformation matrix is listed here
## the transformation from point in view frame to base frame
## A*p_view + t = p_b

# Left arm
# Affine transformation matrix:
#  [[-9.99832626e-01 -8.28114314e-05  1.79002770e-04]
#  [-2.53652477e-04  7.07137744e-01  7.07061032e-01]
#  [-4.14350147e-04  7.07193057e-01 -7.07476252e-01]]
# Affine transformation translation vector:
#  [ 1.19686640e-04 -1.44189272e-04  3.99703351e-01]

# Right arm
# Affine transformation matrix:
#  [[-9.99930239e-01  1.51850848e-02  1.06289609e-04]
#  [ 1.09051164e-02  7.06784829e-01  7.07275041e-01]
#  [ 1.08878878e-02  7.07033318e-01 -7.07313850e-01]]
# Affine transformation translation vector:
#  [ 5.11442081e-05 -7.48246846e-05  3.99931680e-01]

