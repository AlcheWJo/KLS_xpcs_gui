#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:57:57 2024

@author: wonhyukjo
"""

# [FIX - 크로스플랫폼 호환성] 아래 azimuthal_integration() 함수의 타입힌트는
# `np.ndarray | None` 같은 PEP 604 문법을 쓰는데, 이 문법은 Python 3.10부터만
# 지원됩니다. `from __future__ import annotations` 를 추가하면 모든 타입힌트를
# 문자열로 지연 평가해서 Python 3.7+ 에서도 (실제로 3.10 미만인 학생 PC에서도)
# import 시 TypeError 없이 정상 동작합니다. 이 한 줄 외에는 아무 로직도 바꾸지
# 않았습니다.
from __future__ import annotations

import numpy as np
import random, copy
from skimage.transform import resize

def particle_gen(xPos,yPos, r, images):
    for initx_, inity_ in zip(xPos, yPos):
        for xx in range(initx_-r*2, initx_+r*2):
            for yy in range(inity_-r*2, inity_+r*2):
                r_ = np.sqrt((xx-initx_)**2 + (yy-inity_)**2)
                if r_ < r:
                    try:
                        images[xx,yy] = 1
                    except:
                        pass

    return images

def particle_gen_polydisp(xPos,yPos, r, images):
    for initx_, inity_ in zip(xPos, yPos):
        r_ = int(np.round(r*np.random.uniform(0.8, 1.2)))
        for xx in range(initx_-r_*2, initx_+r_*2):
            for yy in range(inity_-r_*2, inity_+r_*2):
                _r_ = np.sqrt((xx-initx_)**2 + (yy-inity_)**2)
                if _r_ < r_:
                    try:
                        images[xx,yy] = 1
                    except:
                        pass

    return images

def speckle_gen(image_):
    ftimage = np.fft.fft2(image_)
    ftimage = np.fft.fftshift(ftimage)
    return np.abs(ftimage)


def BC_conti(initx_, inity_,obj):# dimx, dimy):
    
    dimx = obj['dimx']; dimy = obj['dimy']
    
    if obj['BC'] == 'bound':

        if initx_ >= dimx:
            shift_ = initx_ - dimx
            initx_ = shift_

        if initx_ <= 0:
            initx_ = dimx + initx_-1

        if inity_ >= dimy:
            shift_ = inity_ - dimy
            inity_ =shift_
        if inity_ <= 0:
            inity_ = dimy + inity_-1
            
            
    if obj['BC'] == 'unbound':
        if initx_ >= dimx:
            shift_ = initx_ - dimx
            initx_ = shift_
            inity_ = random.randint(0, obj['dimy']-1)

        if initx_ <= 0:
            initx_ = dimx + initx_-1
            inity_ = random.randint(0, obj['dimy']-1)


        if inity_ >= dimy:
            shift_ = inity_ - dimy
            inity_ =shift_
            initx_ = random.randint(0, obj['dimx']-1)


        if inity_ <= 0:
            inity_ = dimy + inity_-1
            initx_ = random.randint(0, obj['dimx']-1)
    return initx_, inity_





def Speckle_calculator(xPos_tot, yPos_tot,obj):
    # [FIX - SDD가 speckle contrast에 반영되지 않던 문제]
    # 원래 코드는 scale2pix (speckle FFT 패턴을 검출기 픽셀 격자로 리샘플링하는
    # 배율) 를 obj['scalefactor'] 라는 임의의 상수로만 정했다. 이 상수는
    # obj['SDD'] / obj['Lambda'] 와 아무 관계가 없어서, SDD(시료-검출기 거리)를
    # 바꿔도 speckle 패턴이 검출기 픽셀 몇 개에 걸쳐 나타나는지가 전혀 변하지
    # 않았고, 그 결과 speckle contrast(beta) 도 SDD에 반응하지 않았다.
    #
    # 실제 XPCS에서는 Fraunhofer(far-field) 회절 관계식에 따라, 실공간에서
    # 격자간격 dx로 샘플링된 시료를 파장 lambda, 거리 L(=SDD)로 전파했을 때
    # 검출기 평면에서의 전체 패턴 폭이 X_det = lambda*L/dx 이다. 이를 검출기
    # 픽셀 크기(pixelsize)로 나누면 "검출기 픽셀 몇 개짜리 패턴인지"(scale2pix)가
    # 나온다:
    #
    #     scale2pix = lambda * L / (dx * pixelsize)
    #
    # 즉 SDD(L)가 커지거나 시료 평면 격자간격 dx가 작아질수록 하나의 speckle
    # 알갱이가 더 많은 검출기 픽셀에 걸쳐 나타나서(oversampling 증가) 픽셀당
    # 측정되는 contrast가 증가한다 - 실제 실험에서 관찰되는 물리적 거동과 일치.
    #
    # obj['sample_pixelsize'] (실공간 격자간격 dx [m]) 가 있으면 이 물리적으로
    # 올바른 식을 쓰고, 없으면(예: 원본 Speckle_sim.py 를 그대로 쓰는 경우)
    # 기존 scalefactor 방식으로 자동 fallback 한다 - 기존 사용법과 호환됨.
    Speckle_img = []
    Particle_img = []
    for dt in range(obj['dimt']-1):
        xPos_ = xPos_tot[dt]
        yPos_ = yPos_tot[dt]

        images_blank = np.zeros((obj['dimx'],obj['dimy'])).astype('int8')
        #Particle_img_ = particle_gen(xPos_, yPos_, obj['r'], images_blank)
        Particle_img_ = particle_gen_polydisp(xPos_, yPos_, obj['r'], images_blank)
        Speckle_roi = speckle_gen(Particle_img_)#[4000:6000,4000:6000]
        lengths = len(Speckle_roi)

        dx = obj.get('sample_pixelsize', None)
        Lambda = obj.get('Lambda', None)
        if dx is not None and Lambda is not None:
            scale2pix = int(round(Lambda * obj['SDD'] / (dx * obj['pixelsize'])))
        else:
            scale2pix = int(lengths*obj['scalefactor']*1e-3/obj['pixelsize'])
        scale2pix = max(scale2pix, 4)  # 너무 작아져서 resize가 깨지는 것 방지

        Speckle_img.append(resize(Speckle_roi,(scale2pix,scale2pix)))
        Particle_img.append(Particle_img_)
    return np.array(Particle_img), np.array(Speckle_img)



def Cross_cut(Speckle_ori):
    

    Speckle_h = copy.deepcopy(Speckle_ori)
    Speckle_v = copy.deepcopy(Speckle_ori)

    lengh_ = int(len(Speckle_h[0,0])/2)
    width = 50

    Speckle_h[:,:lengh_-width,:] = np.nan
    Speckle_h[:,lengh_+width:,:] = np.nan

    Speckle_v[:,:,:lengh_-width] = np.nan
    Speckle_v[:,:,lengh_+width:] = np.nan
    
    return Speckle_h,  Speckle_v




def ROI_matrix(ImgMat_Data,obj):
    xwidth, ywidth = np.shape(ImgMat_Data)
    BinNum = obj['BinNum']; Center = obj['Beamcenter']
    StartPixel = obj['StartPixel']; EndPixel = obj['EndPixel'];
    pixy = np.arange(xwidth)-Center[0]
    pixz = np.arange(ywidth)-Center[1]
    pixz, pixy = np.meshgrid(pixz, pixy)
    Distance_M = np.sqrt(pixy**2 + pixz**2)
    Pixel_range = np.linspace(StartPixel,EndPixel,BinNum+1)
    IndexMat = []
    Ave_R_Pix = []
    CalQlist = range(BinNum)
    for kk in CalQlist:
        Choosed = []
        for ii in range(xwidth):
            for jj in range(ywidth):
                if Pixel_range[kk]<Distance_M[ii,jj]<Pixel_range[kk+1]:
                    Choosed.append([ii,jj])
        R_Pix = (Pixel_range[kk]+Pixel_range[kk+1])/2
        Ave_R_Pix.append(R_Pix)
        Choosed = np.array(Choosed)
        IndexMat.append(Choosed)
    pix_x = Pixel_range[:-1]+np.abs(Pixel_range[0]-Pixel_range[1])/2
    return IndexMat




def cal_g2(spec,dellist = 20):
    try:
        fr, d1_, d2_ = np.shape(spec)
    except:
        fr, d1_ = np.shape(spec)
    g2_ = []
#     denom = np.mean(spec)**2
    for ct, del_ in enumerate(dellist):
        g2_del = []
        del_ = int(del_)

        for frame in range(fr-del_):
            denom = np.nanmean(spec[frame])*np.nanmean(spec[frame+del_])
            g2_del.append(np.nanmean(spec[frame]*spec[frame+del_])/denom)

        g2_.append(np.nanmean(g2_del))
    return g2_


def Extract_g2_TTC(C2t):
    t1list = range(len(C2t))
    t2list = range(len(C2t))
    t1 = t1list[0]
    g2_ = np.empty((len(t2list),len(t1list)))
    g2_[:] = np.nan
    n = 0
    for t1 in t1list:#,900,1000,1999]:
        g2_temp = []; dt_temp = []
        for t2 in range(len(C2t)):
            dt = (t2-t1)
            if dt > 0:
                g2_[t1,dt]=C2t[t1,t1+dt]
        n +=1
    g2 = np.nanmean(g2_,axis = 0)
    return g2


#%%

def azimuthal_integration(
    image: np.ndarray,
    x0: float,
    y0: float,
    *,
    r_edges: np.ndarray | None = None,   # 직접 경계 지정시 우선
    bins: int = 200,                     # r_edges가 없을 때만 사용
    r_min: float = 0.0,
    r_max: float | None = None,
    phi_range: tuple[float, float] | None = None,  # degrees, 예 (0, 360) 또는 (350, 20)
    mask: np.ndarray | None = None,      # True면 제외
    weights: np.ndarray | None = None,   # 가중 평균에 사용 (예 노출 시간, solid angle 등)
    return_std: bool = True,
    # 아래 세 개를 모두 주면 r을 q로 변환해 q축을 함께 반환
    pixel_size: float | None = None,     # m per pixel
    distance: float | None = None,       # sample to detector center distance [m]
    wavelength: float | None = None,     # meter
):
    """
    Azimuthal integration about (x0, y0). x는 열 인덱스, y는 행 인덱스 기준.

    Returns
    -------
    r_or_q : 1D array
        r bin center (pixels). pixel_size, distance, wavelength를 모두 주면 q [1/m]
    I_mean : 1D array
        각 반지름 bin의 평균 강도
    I_std : 1D array (optional)
        각 bin의 표준편차 (return_std=True일 때)
    counts : 1D array
        각 bin에 기여한 픽셀 수 (또는 가중치 합계)
    """
    if image.ndim != 2:
        raise ValueError("image must be 2D")

    ny, nx = image.shape
    yy, xx = np.indices((ny, nx), dtype=float)

    # 중심 기준 좌표, 반지름 r, 방위각 phi(0~360)
    dx = xx - x0
    dy = yy - y0
    r = np.hypot(dx, dy)
    phi = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0

    # 유효성 마스크
    valid = np.isfinite(image)
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError("mask shape must match image")
        valid &= ~mask

    # 방위각 구간 선택
    if phi_range is not None:
        pmin, pmax = phi_range
        pmin %= 360.0
        pmax %= 360.0
        if pmin <= pmax:
            valid &= (phi >= pmin) & (phi < pmax)
        else:
            # 래핑 구간 예 (350, 20)
            valid &= (phi >= pmin) | (phi < pmax)

    if r_max is None:
        r_max = float(r[valid].max()) if np.any(valid) else 0.0

    # 최종 선택
    valid &= (r >= r_min) & (r <= r_max)
    if not np.any(valid):
        raise ValueError("no valid pixels in the requested region")

    r_sel = r[valid]
    I_sel = image[valid]
    w_sel = np.ones_like(I_sel) if weights is None else weights[valid]

    # bin 경계
    if r_edges is None:
        if bins <= 0:
            raise ValueError("bins must be a positive integer")
        edges = np.linspace(r_min, r_max, bins + 1, dtype=float)
    else:
        edges = np.asarray(r_edges, dtype=float)
        if edges.ndim != 1 or np.any(np.diff(edges) <= 0):
            raise ValueError("r_edges must be 1D strictly increasing")

    nbins = len(edges) - 1
    # 픽셀을 bin에 할당
    idx = np.digitize(r_sel, edges) - 1
    in_range = (idx >= 0) & (idx < nbins)
    if not np.any(in_range):
        raise ValueError("no samples fell into the bins")
    idx = idx[in_range]
    I_sel = I_sel[in_range]
    w_sel = w_sel[in_range]

    # 가중 합, 가중 제곱합, 가중치 합
    sum_w = np.bincount(idx, weights=w_sel, minlength=nbins)
    sum_Iw = np.bincount(idx, weights=I_sel * w_sel, minlength=nbins)
    with np.errstate(invalid="ignore", divide="ignore"):
        I_mean = sum_Iw / sum_w

    if return_std:
        sum_I2w = np.bincount(idx, weights=(I_sel * I_sel) * w_sel, minlength=nbins)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (sum_I2w / sum_w) - I_mean * I_mean
        var[var < 0] = 0.0
        I_std = np.sqrt(var)
    else:
        I_std = None

    # bin center
    r_centers = 0.5 * (edges[:-1] + edges[1:])

    # 비어 있는 bin 제거
    nonzero = sum_w > 0
    r_centers = r_centers[nonzero]
    I_mean = I_mean[nonzero]
    counts = sum_w[nonzero]
    if I_std is not None:
        I_std = I_std[nonzero]

    # 필요하면 q로 변환
    if (pixel_size is not None) and (distance is not None) and (wavelength is not None):
        # 일반 기하: 2θ = arctan( r*px / L ), θ = 0.5 * arctan(...)
        two_theta = np.arctan(r_centers * pixel_size / distance)
        theta = 0.5 * two_theta
        q = (4.0 * np.pi / wavelength) * np.sin(theta)
        x_axis = q
    else:
        x_axis = r_centers

    if return_std:
        return x_axis, I_mean, I_std, counts
    else:
        return x_axis, I_mean, counts
