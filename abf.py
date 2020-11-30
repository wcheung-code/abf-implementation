import igl
import meshplot as mp
import numpy as np
import argparse
from scipy import sparse as sps
from scipy.sparse.linalg import qmr

def dot(A, B):
    return np.sum(np.multiply(A.T.conj(), B.T), axis = 0)

def angle(A, B):
    return np.arccos(np.divide(-1 * dot(A, B), np.multiply(np.sqrt(dot(A, A)), np.sqrt(dot(B, B)))))

def optimal_angles(angles, face_indices, ring_sum):
    return 2*np.pi*np.divide(angles, np.take(ring_sum, face_indices))

def right(mat): 
    return np.concatenate(np.roll(mat, 1, axis = 1).T)

def left(mat): 
    return np.concatenate(np.roll(mat, -1, axis = 1).T)

class MeshObject:
    def __init__(self, source):
        self.source = source
        self.vertices, self.faces = igl.read_triangle_mesh(self.source)
        (self.num_faces, self.num_angles) = self.faces.shape
        self.max_face_index = np.max(self.faces) + 1

    def get_boundary_points(self):        
        return np.unique(igl.boundary_facets(self.faces))
    
    def get_interior_points(self):
        return np.array(list(set(np.concatenate(self.faces)) - set(self.get_boundary_points())))
    
    def get_faces(self):
        return self.faces[:,0], self.faces[:,1], self.faces[:,2]
    
    def get_vertices(self):
        a1, a2, a3 = self.get_faces()
        return self.vertices[tuple(a1),:], self.vertices[tuple(a2),:], self.vertices[tuple(a3),:]

    def compute_angles(self):
        v1, v2, v3 = self.get_vertices()
        r23, r31, r12 = v3 - v2, v1 - v3, v2 - v1
        return angle(r12, r31), angle(r12, r23), angle(r31, r23)
    
class AngleBasedFlattening(MeshObject):
    
    def __init__(self, source):
        super().__init__(source)

    def preprocess_angles(self):
        (ang1, ang2, ang3), (a1, a2, a3) = self.compute_angles(), self.get_faces()
        Mang = sps.csr_matrix((np.concatenate([ang1, ang2, ang3]), 
                               (np.concatenate([a1, a2, a3]), np.concatenate([a2, a3, a1]))))
        ring_sum = np.squeeze(np.asarray(Mang.sum(axis = 1)))
        ang1opt = optimal_angles(ang1, a1, ring_sum)
        ang2opt = optimal_angles(ang2, a2, ring_sum)
        ang3opt = optimal_angles(ang3, a3, ring_sum)

        deltas = 2*np.pi - ring_sum
        delta_indices = np.arange(deltas.shape[0])[np.abs(deltas) > 1]
        delta_indices = delta_indices[~np.isin(delta_indices, self.get_boundary_points())]

        ang1[np.isin(a1, delta_indices)] = ang1opt[np.isin(a1, delta_indices)]
        ang2[np.isin(a2, delta_indices)] = ang2opt[np.isin(a2, delta_indices)]
        ang3[np.isin(a3, delta_indices)] = ang3opt[np.isin(a3, delta_indices)]

        return np.vstack((ang1, ang2, ang3)).T.flatten()

    def triangle_consistency(self):
        ones = np.ones(self.num_faces * self.num_angles)
        arange = np.concatenate(np.tile(np.arange(self.num_faces),(self.num_angles,1)))
        content = np.concatenate(np.arange(self.num_faces * self.num_angles).reshape(
            self.num_faces, self.num_angles).T)
        return sps.csr_matrix((ones, (arange, content))), np.pi - np.sum(self.preprocess_angles().reshape(
            (self.num_faces, self.num_angles)), axis = 1)
    
    def vertex_consistency(self):
        ones = np.ones(self.num_faces * self.num_angles)
        content = np.concatenate(np.arange(self.num_faces * self.num_angles).reshape(
            self.num_faces, self.num_angles).T)
        indicator = sps.csr_matrix((ones, (np.concatenate(self.get_faces()), content)))
        indicator = indicator[np.isin(np.arange(self.num_faces * self.num_angles),
                                      self.get_interior_points()), :]
        return indicator, 2*np.pi - indicator * self.preprocess_angles()
    
    def wheel_consistency(self):
        cot = np.reciprocal(np.tan(self.preprocess_angles())).reshape(self.num_faces, self.num_angles)
        logsin = np.log(np.sin(self.preprocess_angles())).reshape(self.num_faces, self.num_angles)
        arange = np.arange(self.num_faces * self.num_angles).reshape(self.num_faces, self.num_angles)

        indicator = sps.csr_matrix((right(cot), (np.concatenate(self.get_faces()), right(arange))))
        indicator -= sps.csr_matrix((left(cot), (np.concatenate(self.get_faces()), left(arange))))
        indicator = indicator[np.isin(np.arange(self.num_faces * self.num_angles), self.get_interior_points()), :]
        
        CC = sps.csr_matrix((right(logsin), (np.concatenate(self.get_faces()), right(arange))))
        CC -= sps.csr_matrix((left(logsin), (np.concatenate(self.get_faces()), left(arange))))
        
        return indicator, np.concatenate(np.array(
            np.sum(CC, axis = 1)[np.isin(np.arange(self.max_face_index), self.get_interior_points())]).T)
    
    def solve(self):
        A, b = list(zip(self.triangle_consistency(), self.vertex_consistency(), self.wheel_consistency()))
        A, b = sps.vstack(A), np.concatenate(b)
        W = sps.diags(self.preprocess_angles()); A = A*W
        x, success = qmr(A * A.T, b); delta = W*A.T*x
        error = np.linalg.norm(delta)/(3*self.num_faces)
        return (self.preprocess_angles() + delta).reshape(self.faces.shape), True if not success else False, error

if __name__=="__main__":

    parser = argparse.ArgumentParser(description = "Angle Based Flattening Implementation")
    parser.add_argument('-s', '--source', type = str, metavar = '', help = "relative path of source file")
    args = parser.parse_args()

    abf = AngleBasedFlattening(args.source)
    solution, success, error = abf.solve()

    if success:
        print('Mesh Characteristics: {} vertices and {} faces'.format(len(abf.vertices), abf.num_faces))
        print('ABF Solution = {}'.format(solution))
        print('ABF Error = {}'.format(error))
    else:
        print('ERROR: Failure to determine a convergent solution.')