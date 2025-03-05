import os
import cv2
import glob
import time
import math
import numpy as np
from tqdm import tqdm
from pywt import dwt2, wavedec2



def get_file_name(path):
    basename = os.path.basename(path)
    onlyname = os.path.splitext(basename)[0]
    return onlyname


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    # r = 60
    r = 30
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)
    return t


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * im + mean_b
    return q


def normalize_img(img):
    return (img - img.min()) / (img.max() - img.min())


def get_atmosphere_w_location_drop(I, darkch, p, drop_rate):
    M, N = darkch.shape
    flatI = I.reshape(M * N, 3)
    flatdark = darkch.ravel()
    searchidx = (-flatdark).argsort()[ int(M*N*drop_rate): int(M*N*drop_rate + M*N*p)]  # find top M * N * p indexes
    collected_pixels = flatI.take(searchidx, axis=0)
    avg_pixels = np.average(collected_pixels, axis=1)
    max_avg_pixel_id = np.argmax(avg_pixels)
    location_in_flat = searchidx[max_avg_pixel_id]
    y = (location_in_flat)//N + 1
    x = (location_in_flat)%N + 1
    return collected_pixels[max_avg_pixel_id], x, y


def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]
    return res


def BrightChannel(im,sz):
    b,g,r = cv2.split(im)
    bc = cv2.max(cv2.max(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    bright = cv2.dilate(bc,kernel)
    return bright


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma
    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)
    return cv2.LUT(src, table)


def sobel_edge_extractor(img):
    sobel_horizontal = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_vertical = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    absx= cv2.convertScaleAbs(sobel_horizontal)
    absy = cv2.convertScaleAbs(sobel_vertical)
    edge = cv2.addWeighted(absx, 0.5, absy, 0.5,0)
    return edge


def norm(x):
    # normalise x to range [-1,1]
    nom = (x - x.min()) * 2.0
    denom = x.max() - x.min()
    return  nom/denom - 1.0


def sigmoid(x, t=0.3, k=0.15):
    # sigmoid function
    # use k to adjust the slope
    s = 1 / (1 + np.exp(-(x-t) / k)) 
    return s



if __name__ == '__main__':


	path = 'test/tiananmen1.png'
	# path = 'test/newyork.png'


	# Configurations:
	# -------------------
	a = 2
	R = 3
	# -------------------

	total_time = 0
	fname = get_file_name(path)

	# Start:
	start_time = time.time()

	hazy_img_orig = cv2.imread(path)
	img_h, img_w, _ = hazy_img_orig.shape
	# hazy_img_orig = cv2.resize(hazy_img_orig, (512,512))
	hazy_img = cv2.cvtColor(hazy_img_orig, cv2.COLOR_BGR2RGB)

	src = hazy_img.copy()
	I = src.astype("float64") / 255

	dark2 = DarkChannel(I, 3)
	t2 = TransmissionRefine(src, dark2)

	bright1 = BrightChannel(I, 30)
	tb1 = TransmissionRefine(src, bright1)

	stretch = 'multiply'
	# stretch = 'sigmoid'

	if stretch == 'multiply':
		# multiplied = (t2**a)*(tb1**b)
		multiplied = (t2*tb1)**a
		mul1_clip = multiplied.copy()

	else:
		multiplied = normalize_img(t2)*normalize_img(tb1)
		mul_norm = norm(multiplied)
		mul_sigmoid = sigmoid(mul_norm, 0.3, 0.15)
		# mul_sigmoid = sigmoid(mul_norm, 0.2, 0.4)
		mul1_clip = mul_sigmoid.copy()

	mul1_clip[np.where(mul1_clip < 0.01)] = 0


	gray = cv2.cvtColor(hazy_img_orig, cv2.COLOR_BGR2GRAY)
	sobel = 1 - sobel_edge_extractor(gray)/255
	cHVD_inv = sobel.copy()


	multi = 2*mul1_clip*cHVD_inv
	multi = np.clip(multi, 0.0, 1.0)


	A2, Ax2, Ay2 = get_atmosphere_w_location_drop(I, multi, 0.001, 0.00)
	A = np.array([A2])
	te = TransmissionEstimate(I, A, 5)
	t = TransmissionRefine(src, te)
	J2 = Recover(I, t, A, 0.1)
	J_clip_A2 = np.clip(J2, 0.0, 1.0)
	J_clip_A2_8bit = (J_clip_A2*255).astype('uint8')

	sky_mask = multi.copy()
	fore_mask = 1 - sky_mask

	A2_8bit = hazy_img_orig.copy()
	A2_8bit[:,:,0] = int(A2[0]*255)
	A2_8bit[:,:,1] = int(A2[1]*255)
	A2_8bit[:,:,2] = int(A2[2]*255)
	A2_8bit_to_hsv = cv2.cvtColor(A2_8bit, cv2.COLOR_RGB2HSV)
	A2_8bit_hsv_values = np.mean(A2_8bit_to_hsv, axis=0)[0]

	hazy_hsv = cv2.cvtColor(hazy_img, cv2.COLOR_RGB2HSV)
	h,s,v = cv2.split(hazy_hsv)

	t_s = TransmissionEstimate(I, A, 3)

	translation_ratio = 1/(R*t_s) - 1

	s_inters_img = s.copy()
	s_inters_img[:,:] = A2_8bit_hsv_values[1]
	s = s.astype('float64')
	s_inters_img = s_inters_img.astype('float64')
	enh_s = s + translation_ratio*(s - s_inters_img)
	enh_s = np.clip(enh_s, 0, 255).astype('uint8')


	v_inters_img = v.copy()
	v_inters_img[:,:] = A2_8bit_hsv_values[2]
	v = v.astype('float64')
	v_inters_img = v_inters_img.astype('float64')
	enh_v = v + translation_ratio*(v - v_inters_img)
	enh_v = np.clip(enh_v, 0, 255).astype('uint8')

	new_hsv = cv2.merge([h,enh_s,enh_v])
	rgb_back = cv2.cvtColor(new_hsv, cv2.COLOR_HSV2RGB)

	fore_only = (cv2.merge([fore_mask,fore_mask,fore_mask])*(J_clip_A2_8bit)).astype('uint8')
	back_only = (cv2.merge([sky_mask,sky_mask,sky_mask])*rgb_back).astype('uint8')
	added = fore_only + back_only

	proc_time = time.time() - start_time
	print('Processing Time (in seconds):', proc_time)

	added_for_display = cv2.cvtColor(added, cv2.COLOR_RGB2BGR)
	# added_for_display = gammaCorrection(added_for_display, 1.5)


	# Display
	cv2.imshow('hazy', hazy_img_orig)
	cv2.imshow('dehazed', added_for_display)
	cv2.waitKey(0)