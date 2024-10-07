import tensorflow as tf
import numpy as np
import math
import Gfiltertf
import imageio
import os

def fdct_wrapping_var(x, is_real, finest, nbscales, nbangles_coarse, D0, fre_per, status):

    X = tf.signal.fftshift(tf.signal.fft2d(tf.cast((tf.signal.ifftshift(x)),dtype=tf.complex64))) / tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(x)), dtype=tf.complex64))
    Xtmp = X
    nbangles = tf.cast(nbangles_coarse * 2**(tf.math.ceil((nbscales-tf.range(nbscales,1,-1))/2)),dtype=tf.double)
    C = list()    
    N1, N2 = x.shape
    M1, M2 = N1/3, N2/3
    bigN1, bigN2 = int(2*math.floor(2*M1)+1), int(2*math.floor(2*M2)+1)
    equiv_index_1 = tf.math.floormod(tf.range(tf.math.floor(N1/2)-tf.math.floor(2*M1), tf.math.floor(N1/2)-tf.math.floor(2*M1)+bigN1), N1)
    equiv_index_2 = tf.math.floormod(tf.range(tf.math.floor(N2/2)-tf.math.floor(2*M2), tf.math.floor(N2/2)-tf.math.floor(2*M2)+bigN2), N2)
    equiv_index_1 = tf.cast(equiv_index_1,dtype=tf.int32)
    equiv_index_2 = tf.cast(equiv_index_2,dtype=tf.int32)
    mesh = tf.stack(tf.meshgrid(equiv_index_1, equiv_index_2))
    X = tf.reshape(tf.gather_nd(X,tf.transpose(tf.reshape(mesh,[2,-1]))),[bigN2, bigN1])
    X = tf.transpose(X)
    lowpass = Gfiltertf.Gfilter(X, (D0[status*nbscales] + sum(2 ** (i - 1) * D0[status*nbscales + nbscales - i] for i in range(1, nbscales)))/nbscales, X.shape[0], X.shape[1])
    Xlow = X * lowpass
    scales = tf.range(nbscales,1,-1)

    for j in scales.eval():
        M1 *= fre_per
        M2 *= fre_per
        lowpass = Gfiltertf.Gfilter(X, D0[j-1+status*nbscales], int(2*math.floor(2*M1)+1), int(2*math.floor(2*M2)+1))
        hipass = tf.sqrt(1 - lowpass**2)
        Xhi = Xlow
        Xlow = tf.slice( Xlow, [ tf.cast(tf.math.floor(2*M1/fre_per)-tf.math.floor(2*M1), dtype=tf.int32) , tf.cast(tf.math.floor(2*M2/fre_per)-tf.math.floor(2*M2), dtype=tf.int32) ], [ tf.cast(2*(tf.math.floor(2*M1))+1, dtype=tf.int32), tf.cast(2*(tf.math.floor(2*M2))+1, dtype=tf.int32)] )
        Xlow_index_1 = tf.range(-tf.math.floor(2*M1)-1, tf.math.floor(2*M1), dtype=tf.int32) + tf.cast(tf.math.floor(2*M1/fre_per), dtype=tf.int32) + 1
        Xlow_index_2 = tf.range(-tf.math.floor(2*M2)-1, tf.math.floor(2*M2), dtype=tf.int32) + tf.cast(tf.math.floor(2*M2/fre_per), dtype=tf.int32) + 1        
        Xhi = tf.tensor_scatter_nd_update(Xhi, tf.transpose(tf.reshape(tf.stack(tf.meshgrid(Xlow_index_1, Xlow_index_2)),[2,2*(tf.math.floor(2*M1))+1,2*(tf.math.floor(2*M2))+1]),[1,2,0]), Xlow * hipass)
        Xlow = Xlow * lowpass

        l = 0
        nbquadrants = 2 + 2*(not is_real)
        nbangles_perquad = nbangles[j-2] // 4
        for quadrant in range(1, nbquadrants+1):
            M_horiz = M2 * (quadrant%2==1) + M1 * (quadrant%2==0)
            M_vert = M1 * (quadrant%2==1) + M2 * (quadrant%2==0)
            M_horiz /= (2*fre_per)
            M_vert /= (2*fre_per)
            if nbangles_perquad%2 == 1:
                wedge_ticks_left = tf.math.round(tf.range(0, 0.5+(1/(2*nbangles_perquad)), 1/(2*nbangles_perquad))*2*tf.math.floor(4*M_horiz) + 1).numpy().astype(int)
                wedge_ticks_right = 2*tf.math.floor(4*M_horiz) + 2 - wedge_ticks_left
                wedge_ticks = np.concatenate((wedge_ticks_left, wedge_ticks_right[::-1]))
            else:
                tmp1 = tf.range(0, 0.5001, 1/(2*nbangles_perquad))
                wedge_ticks_left = tf.round(tf.cast(tmp1,dtype=tf.double) * 2 * tf.cast(tf.floor(4*M_horiz),dtype=tf.double) + 1)
                wedge_ticks_right = 2 * tf.cast(tf.floor(4*M_horiz),dtype=tf.double) + 2 - wedge_ticks_left
                wedge_ticks = tf.concat([wedge_ticks_left, tf.reverse(wedge_ticks_right[:-1], axis=[0])], axis=0)
            wedge_endpoints = wedge_ticks[1:-1:2]
            wedge_midpoints = (wedge_endpoints[:-1] + wedge_endpoints[1:]) / 2

            l += 1
            #left
            first_wedge_endpoint_vert = tf.cast(2 * tf.cast(tf.floor(4 * M_vert),dtype=tf.double) / (2 * tf.cast(nbangles_perquad,dtype=tf.double)) + 1, dtype=tf.int32)
            length_corner_wedge = tf.cast(4 * M_vert - tf.floor(M_vert) + tf.cast(tf.math.ceil(first_wedge_endpoint_vert / 4),dtype=tf.float32), dtype=tf.int32)
            Y_corner = tf.range(1, length_corner_wedge + 1, dtype=tf.int32)
            width_wedge = tf.cast(wedge_endpoints[1] + wedge_endpoints[0] - 1, dtype=tf.int32)
            slope_wedge = (tf.cast(tf.floor(4 * M_horiz),dtype=tf.double) + 1 - wedge_endpoints[0]) / tf.cast(tf.floor(4 * M_vert),dtype=tf.double)
            left_line = tf.cast(tf.round(2 - wedge_endpoints[0] + slope_wedge * tf.cast((Y_corner - 1),dtype=tf.double)), dtype=tf.int32)
            wrapped_data = tf.zeros((length_corner_wedge, width_wedge), dtype=Xhi.dtype)
            first_row = tf.cast( tf.cast(tf.floor(4*M_vert),dtype=tf.double) + 2 - tf.math.ceil((length_corner_wedge + 1)/2) + tf.cast(tf.math.mod(length_corner_wedge + 1, 2),dtype=tf.double) * tf.cast((quadrant - 2 == tf.math.mod(quadrant - 2, 2)),dtype=tf.double), dtype=tf.int32)
            first_col = tf.cast( tf.cast(tf.floor(4*M_horiz),dtype=tf.double) + 2 - tf.math.ceil((width_wedge + 1)/2) + tf.cast(tf.math.mod(width_wedge + 1, 2),dtype=tf.double) * tf.cast((quadrant - 3 == tf.math.mod(quadrant - 3, 2)), dtype=tf.double), dtype=tf.int32)

            for row in range(1, (length_corner_wedge + 1).eval()):
                cols = left_line[row-1] + tf.math.mod((tf.range(width_wedge, dtype=tf.int32) - (left_line[row-1] - first_col)), width_wedge)
                cols_bool = list()
                admissible_cols = tf.cast(tf.round(1/2 * tf.cast((cols + 1 + tf.abs(cols - 1)),dtype=tf.double)), dtype=tf.int32)
                new_row = 1 + tf.math.mod(row - first_row, length_corner_wedge)#60yes
                nr = (new_row-1) * tf.cast(tf.ones(width_wedge),dtype=tf.int32)
                tr = tf.range(width_wedge)
                wrapped_data = tf.tensor_scatter_nd_update(wrapped_data, tf.transpose(tf.stack([nr,tr])), tf.gather(Xhi[row-1], admissible_cols-1) * tf.cast(cols > 0, dtype=Xhi.dtype))
                
            if not is_real:
                if quadrant == 2:
                    wrapped_data = tf.transpose(wrapped_data)
                    wrapped_data = tf.reverse(wrapped_data, axis=[0])
                elif quadrant == 3:
                    wrapped_data = tf.reverse(wrapped_data, axis=[0])
                    wrapped_data = tf.reverse(wrapped_data, axis=[1])
                elif quadrant == 4:
                    wrapped_data = tf.transpose(wrapped_data)
                    wrapped_data = tf.reverse(wrapped_data, axis=[1])
                C.append(tf.signal.fftshift(tf.signal.ifft2d(tf.cast((tf.signal.ifftshift(wrapped_data)),dtype=tf.complex64))) * tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(wrapped_data)), dtype=tf.complex64)))
                print('left wedge appended, scale = ',j,', direction =',l)

            else:
                wrapped_data = tf.signal.ifftshift(wrapped_data, axes=[0, 1])
                wrapped_data = tf.signal.ifft2d(wrapped_data)
                wrapped_data = tf.signal.fftshift(wrapped_data, axes=[0, 1])
                x = wrapped_data * tf.sqrt(tf.cast(tf.size(wrapped_data), dtype=tf.float32))
                C[j][l] = tf.sqrt(tf.constant(2.0)) * tf.math.real(x)
                C[j][l+nbangles(j)//2] = tf.sqrt(tf.constant(2.0)) * tf.math.imag(x)


            #regular
            length_wedge = tf.math.floor(4*M_vert) - tf.math.floor(M_vert)
            Y = tf.range(1, tf.cast(length_wedge, tf.float32)+1)
            first_row = tf.math.floor(4*M_vert)+2-tf.math.ceil((length_wedge+1)/2)+ tf.math.mod(length_wedge+1,2)*tf.cast((quadrant-2 == tf.math.mod(quadrant-2,2)), dtype=tf.float32)
            for subl in (tf.cast(tf.range(1, nbangles_perquad-1),dtype=tf.int32)).eval():
                l += 1
                width_wedge = tf.cast(wedge_endpoints[subl+1] - wedge_endpoints[subl-1] + 1, dtype=tf.int32)
                slope_wedge = (tf.cast(tf.math.floor(4*M_horiz),dtype=tf.double) + 1 - wedge_endpoints[subl]) / tf.cast(tf.math.floor(4*M_vert),dtype=tf.double)
                left_line = tf.cast(tf.math.round(wedge_endpoints[subl-1] + slope_wedge*tf.cast((Y - 1),dtype=tf.double)), dtype=tf.int32)
                wrapped_data = tf.zeros((length_wedge, width_wedge), dtype=Xhi.dtype)
                first_col = tf.cast(tf.cast(tf.math.floor(4 * M_horiz), dtype=tf.float32) + 2 - tf.cast(tf.math.ceil((width_wedge + 1) / 2), dtype=tf.float32) + tf.cast(tf.math.mod(width_wedge + 1, 2), dtype=tf.float32) * tf.cast((quadrant - 3 == tf.math.mod(quadrant - 3, 2)), dtype=tf.float32),dtype=tf.float32)
                width_wedge = tf.cast(width_wedge,dtype=tf.int32)
                for row in Y.eval():
                    cols = tf.cast(
                        tf.math.mod(
                            (tf.range(tf.cast(width_wedge, dtype=tf.float32)) - (tf.cast(left_line[tf.cast(row-1, tf.int32)], dtype=tf.float32) - first_col)),
                            tf.cast(width_wedge, dtype=tf.float32)
                        ) + tf.cast(left_line[tf.cast(row-1, tf.int32)], dtype=tf.float32),
                        dtype=tf.int32
                    )
                    new_row = tf.cast(1 + tf.math.mod(row - first_row, length_wedge),dtype=tf.int32)
                    wrapped_data = tf.tensor_scatter_nd_update(wrapped_data, tf.transpose(tf.stack([(new_row-1) * tf.cast(tf.ones(width_wedge),dtype=tf.int32),tf.range(width_wedge)])), tf.gather(Xhi[tf.cast(row-1, tf.int32)], cols-1))                    

                if is_real == 0:
                    if quadrant == 2:
                        wrapped_data = tf.transpose(wrapped_data)
                        wrapped_data = tf.reverse(wrapped_data, axis=[0])
                    elif quadrant == 3:
                        wrapped_data = tf.reverse(wrapped_data, axis=[0])
                        wrapped_data = tf.reverse(wrapped_data, axis=[1])
                    elif quadrant == 4:
                        wrapped_data = tf.transpose(wrapped_data)
                        wrapped_data = tf.reverse(wrapped_data, axis=[1])

                    C.append(tf.signal.fftshift(tf.signal.ifft2d(tf.cast((tf.signal.ifftshift(wrapped_data)),dtype=tf.complex64))) * tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(wrapped_data)), dtype=tf.complex64)))
                    print('regular wedge appended, scale = ',j,', direction =',l)
                    
                elif is_real == 1:
                    wrapped_data = tf.image.rot90(wrapped_data, k=-(quadrant-1))
                    x = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(wrapped_data))) * tf.sqrt(tf.math.reduce_prod(tf.shape(wrapped_data)))
                    C[j][l] = tf.sqrt(2) * tf.math.real(x)
                    C[j][l+nbangles[j]//2] = tf.sqrt(2) * tf.math.imag(x)

            wedge_midpoints = tf.cast(wedge_midpoints, dtype=tf.float32)
            wedge_endpoints = tf.cast(wedge_endpoints, dtype=tf.float32)
            #right
            l += 1
            width_wedge = 4*tf.cast(tf.floor(4*M_horiz), dtype=tf.int32) + 3 - tf.cast((wedge_endpoints[-1] + wedge_endpoints[-2]), dtype=tf.int32)
            slope_wedge = ((tf.cast(tf.floor(4*M_horiz), dtype=tf.float32)+1) - tf.cast(wedge_endpoints[-1], dtype=tf.float32))/tf.cast(tf.floor(4*M_vert), dtype=tf.float32)
            left_line = tf.cast(tf.round(wedge_endpoints[-2] + slope_wedge*tf.cast((Y_corner - 1), dtype=tf.float32)), dtype=tf.int32)
            wrapped_data = tf.zeros((length_corner_wedge, width_wedge), dtype=Xhi.dtype)
            first_row = tf.cast(tf.floor(4*M_vert)+2-tf.cast(tf.math.ceil((length_corner_wedge+1)/2), dtype=tf.float32)+tf.cast(tf.math.mod(length_corner_wedge+1,2)*tf.cast((quadrant-2 == tf.math.mod(quadrant-2,2)), dtype=tf.int32), dtype=tf.float32), dtype=tf.int32)
            first_col = tf.cast(tf.floor(4*M_horiz)+2-tf.cast(tf.math.ceil((width_wedge+1)/2), dtype=tf.float32)+tf.cast(tf.math.mod(width_wedge+1,2)*tf.cast((quadrant-3 == tf.math.mod(quadrant-3,2)), dtype=tf.int32), dtype=tf.float32), dtype=tf.int32)

            for row in Y_corner.eval():
                cols = left_line[row-1] + tf.math.mod(tf.range(width_wedge)-(left_line[row-1]-first_col), width_wedge)
                admissible_cols = tf.cast(tf.round(1/2*tf.cast((cols+2*tf.cast(tf.floor(4*M_horiz), dtype=tf.int32)+1-tf.math.abs(cols-(2*tf.cast(tf.floor(4*M_horiz), dtype=tf.int32)+1))), dtype=tf.float32)), dtype=tf.int32)
                new_row = 1 + tf.math.mod(Y_corner[row-1] - first_row, length_corner_wedge)                
                wrapped_data = tf.tensor_scatter_nd_update(wrapped_data, tf.transpose(tf.stack([(new_row-1) * tf.cast(tf.ones(width_wedge),dtype=tf.int32),tf.range(width_wedge)])), tf.gather(Xhi[tf.cast(row-1, tf.int32)], admissible_cols-1) * tf.cast(tf.cast(cols, dtype=tf.float32) <= (2*tf.floor(4*M_horiz)+1), dtype=Xhi.dtype))               

            if is_real == 0:
                if quadrant == 2:
                    wrapped_data = tf.transpose(wrapped_data)
                    wrapped_data = tf.reverse(wrapped_data, axis=[0])
                elif quadrant == 3:
                    wrapped_data = tf.reverse(wrapped_data, axis=[0])
                    wrapped_data = tf.reverse(wrapped_data, axis=[1])
                elif quadrant == 4:
                    wrapped_data = tf.transpose(wrapped_data)
                    wrapped_data = tf.reverse(wrapped_data, axis=[1])

                C.append(tf.signal.fftshift(tf.signal.ifft2d(tf.cast((tf.signal.ifftshift(wrapped_data)),dtype=tf.complex64))) * tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(wrapped_data)), dtype=tf.complex64)))
                print('right wedge appended, scale = ',j,', direction =',l)

            elif is_real == 1:
                # Rotate wrapped_data by -(quadrant-1)
                wrapped_data = tf.image.rot90(wrapped_data, k=-(quadrant-1))
                x = tf.signal.fftshift(tf.signal.ifft2d(tf.signal.ifftshift(wrapped_data))) * tf.math.sqrt(tf.math.reduce_prod(tf.shape(wrapped_data)))
                C[j][l] = tf.math.sqrt(2) * tf.math.real(x)
                C[j][l + nbangles[j] // 2] = tf.math.sqrt(2) * tf.math.imag(x)

            if quadrant < nbquadrants:
                Xhi = tf.transpose(Xhi)
                Xhi = tf.reverse(Xhi, axis=[1])

    C.append(tf.signal.fftshift(tf.signal.ifft2d(tf.cast((tf.signal.ifftshift(Xlow)),dtype=tf.complex64))) * tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(Xlow)), dtype=tf.complex64)))
    C.reverse()
    return C


