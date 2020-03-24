import tensorflow as tf
import numpy as np
import os
import glob
import time
import matplotlib.pyplot as plt
import utilsICME
import warnings

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ----------------- Parameters setting -----------------------
sceneName = 'Development_dataset_2'  # name of the LF
input_path = '../Datasets/ICME/'  # Path to the ICME dataset
result_path = 'Results/' + sceneName + 'x16/'
EVALUATION = 1
ang_res_out = 193
up_scale = 16  # The up scale is fixed, since we use a FROZEN model (Tensorflow),
down_scale = up_scale  # the dimensions of the placeholder are fixed
modelPath1 = "./Model/model_DANx16_1.pb"
modelPath2 = "./Model/model_DANx16_2.pb"  # The two models are exactly the same except for the size of the input placeholder

logWritePath = result_path + 'Log.txt'

model_up_scale = 4
num_iter = int(np.ceil(np.log(up_scale) / np.log(model_up_scale)))
ang_res_in = (ang_res_out - 1)//down_scale + 1
batch = [23, 10, 6]

utilsICME.mkdir(result_path + 'images/')

# -------------- Load light field -----------------
print("Loading light field: %s ..." % sceneName)

lf_files = glob.glob(input_path + sceneName + '/*.png')
im = plt.imread(lf_files[0])
[hei, wid, chn] = im.shape

inputLF = np.zeros([hei, wid, chn, ang_res_in])
n = 0
for i in range(0, ang_res_out, up_scale):
    cur_im = input_path + sceneName + '/%04d.png' % (i + 1)
    im = plt.imread(cur_im)
    inputLF[:, :, :, n] = im
    n += 1

wid = wid // 4 * 4
hei = hei // 4 * 4
inputLF = inputLF[0:hei, 0:wid, :, :]

if EVALUATION == 1:
    fullLF = np.zeros([hei, wid, chn, ang_res_out])
    for i in range(0, ang_res_out):
        cur_im = input_path + sceneName + '/%04d.png' % (i + 1)
        im = plt.imread(cur_im)
        fullLF[:, :, :, i] = im[0:hei, 0:wid, :]

ang_res_out = (ang_res_in - 1) * up_scale + 1

with open(logWritePath, 'a') as f:
    f.write("Input (scene name: %s) is a 1 X %d light field. The output will be a 1 X %d light field.\n" %
            (sceneName, ang_res_in, ang_res_out))


# -------------- Light field reconstruction -----------------
print('Reconstructing light field ...')
start = time.time()
global ang_cur_in, lf_in, lf_cur
for i_iter in range(num_iter):
    print('Iteration %d' % (i_iter + 1))
    if i_iter == 0:
        ang_cur_in = ang_res_in
        lf_in = inputLF
        ang_cur_out = (ang_res_in - 1) * model_up_scale + 1
        cur_model_path = modelPath1
    else:
        cur_model_path = modelPath2

    if i_iter == num_iter - 1:
        ang_cur_out = ang_res_out
    else:
        ang_cur_out = (ang_cur_in - 1) * model_up_scale + 1

    # -------------- Restore graph ----------------
    reconstruction_graph = tf.Graph()
    with reconstruction_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(cur_model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with reconstruction_graph.as_default():
        with tf.Session() as sess:
            tensor_dict = tf.get_default_graph().get_tensor_by_name('Blender/add_10:0')
            input_tensor = tf.get_default_graph().get_tensor_by_name('Placeholder:0')

            # -------------- Reconstruction ----------------
            lf_cur = np.zeros([hei, wid, chn, ang_cur_out])

            n = int(np.ceil(hei / batch[i_iter]))
            for i in range(0, n):
                h_start = i * batch[i_iter]
                if i == n - 1:
                    h_end = hei
                else:
                    h_end = (i + 1) * batch[i_iter]
                slice3D = lf_in[h_start:h_end, :, :, :]
                slice3D = np.transpose(slice3D, (0, 3, 1, 2))

                slice3D = utilsICME.rgb2ycbcr(slice3D)
                slice_r = slice3D[:, :, :, 0:1]
                slice_g = slice3D[:, :, :, 1:2]
                slice_b = slice3D[:, :, :, 2:3]

                slice_r = sess.run(tensor_dict, feed_dict={input_tensor: slice_r})
                slice_g = sess.run(tensor_dict, feed_dict={input_tensor: slice_g})
                slice_b = sess.run(tensor_dict, feed_dict={input_tensor: slice_b})

                slice3D = np.concatenate((slice_r, slice_g, slice_b), axis=-1)
                slice3D = tf.image.resize_bicubic(slice3D, [ang_cur_out, wid])
                slice3D = sess.run(slice3D)

                slice3D = utilsICME.ycbcr2rgb(slice3D)
                slice3D = np.minimum(np.maximum(slice3D, 0), 1)

                lf_cur[h_start:h_end, :, :, :] = np.transpose(slice3D, (0, 2, 3, 1))
            sess.close()

    lf_in = lf_cur
    ang_cur_in = ang_cur_out

out_lf = lf_cur
elapsed = (time.time() - start)
print("Light field reconstruction consumes %.2f seconds, %.3f seconds per view." % (elapsed, elapsed / ang_res_out))

with open(logWritePath, 'a') as f:
    f.write("Reconstruction completed within %.2f seconds (%.3f seconds averaged on each view).\n"
            % (elapsed, elapsed / ang_res_out))

# -------------- Evaluation -----------------
psnr = [0 for _ in range(ang_res_out-ang_res_in)]
border_cut = 0

n = 0
for s in range(0, ang_res_out):
    cur_im = out_lf[:, :, :, s]
    if EVALUATION == 1:
        if np.mod(s, up_scale) != 0 and down_scale == up_scale:
            cur_gt = fullLF[:, :, :, s]
            psnr[n] = utilsICME.metric(cur_im, cur_gt, border_cut)
            n += 1
    plt.imsave(result_path + 'images/' + 'out_' + str(s + 1) + '.png', np.uint8(out_lf[:, :, :, s] * 255))

psnr_avg = np.average(psnr)
psnr_min = np.min(psnr)
print("Mean PSNR and minimum PSNR are %2.3f and %2.3f." % (psnr_avg, psnr_min))
with open(logWritePath, 'a') as f:
    f.write("Mean PSNR and minimum PSNR are %2.3f and %2.3f.\n" % (psnr_avg, psnr_min))
