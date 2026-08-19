"""
xpcs_core.py

Particle_Sim_module.py 의 함수(BC_conti, Speckle_calculator, ROI_matrix, cal_g2)를
그대로 호출하는 "얇은" 오케스트레이션 레이어입니다.

원래 Speckle_sim.py 스크립트에 순서대로 나열되어 있던 다음 두 단계를
함수로 정리한 것뿐이며, 알고리즘 자체는 전혀 바꾸지 않았습니다.

    1) run_brownian_and_speckle : obj -> Brownian 위치 시계열 -> Particle/Speckle 이미지
    2) run_roi_and_g2           : Speckle 이미지 -> ROI(Q-bin) -> g2(tau)

GUI(PyQt5) 코드와 분리해 두었기 때문에, Qt 없이도 이 로직만 따로 실행/테스트할 수 있습니다.
"""

import copy
import random

import numpy as np

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

from Particle_Sim_module import (
    BC_conti,
    Speckle_calculator,
    ROI_matrix,
    cal_g2,
    azimuthal_integration,
)

K_BOLTZMANN = 1.380649e-23  # [J/K]


def build_default_obj():
    """Speckle_sim.py 의 obj = {...} 셋업과 동일한 키를 쓰되, 사용자가 지정한 실공간
    조건(SDD=8m, pixelsize=75um, pix2real=10nm/px, FOV=10um, r=200nm, jump=20nm/frame,
    seed=100, timestep=500ns, T_total=30us)을 픽셀 단위로 환산해 넣은 기본값.

    obj['sample_pixelsize'] : 실공간(샘플 평면) 격자 간격 dx = pix2real [m]. Speckle_calculator
    가 Fraunhofer 관계식 scale2pix = Lambda*SDD/(sample_pixelsize*pixelsize) 으로 speckle
    이미지를 검출기 픽셀로 리샘플링하는 데 쓰인다 (SDD 변화가 speckle contrast에 반영되도록
    하는 물리적 fix, 자세한 설명은 Particle_Sim_module.py 참고).

    StartPixel=40, EndPixel=120, BinNum=5 는 위 조건에서 g2(τ) decay가 관측 가능한
    지연시간(tau) 범위 안에 clearly 들어오도록 확인한 값이다 (StartPixel을 더 줄이면
    beam center에 가까워져 다분산성 노이즈가 섞이고, EndPixel을 더 늘리면 고-q bin이
    1 프레임 안에 다 decay 해버려서 Gamma(q) 피팅이 무너진다).
    """
    return dict(
        BinNum=5,
        StartPixel=40,
        EndPixel=120,
        SDD=8.0,                   # sample-to-detector distance [m]
        Energy=9.0,                 # X-ray energy [keV]
        pixelsize=75e-6,            # detector pixel size [m]
        sample_pixelsize=1e-8,      # pix2real: 샘플 평면 격자 간격 dx [m] (10 nm/px)
        seed=100,                   # 입자 개수
        jump=2,                     # Brownian jump 크기 [grid px / frame] (= 20 nm/frame)
        dimx=1000,                  # = 10 um / 10 nm
        dimy=1000,
        dimt=61,                    # 총 프레임 수 + 1 (= 30 us / 500 ns + 1)
        timestep=5e-7,              # 프레임당 시간 [s] (500 ns)
        r=20,                       # 입자 반경 [pixel] (= 200 nm / 10 nm)
        BC="unbound",
    )


def run_brownian_and_speckle(obj, progress_cb=None):
    """Speckle_sim.py 42~79번 줄 로직 그대로 (progress 콜백만 추가).

    Parameters
    ----------
    obj : dict
        BinNum, StartPixel, EndPixel, SDD, Energy, pixelsize, scalefactor,
        seed, jump, dimx, dimy, dimt, timestep, r, BC 키를 포함해야 함.
    progress_cb : callable(str) or None

    Returns
    -------
    Particle_img : ndarray (dimt-1, dimx, dimy)
    Spec_img     : ndarray (dimt-1, scale2pix, scale2pix)
    obj          : dict (obj['Lambda'] 가 채워져서 반환됨)
    """
    obj = copy.deepcopy(obj)
    obj["Lambda"] = 12.34 / obj["Energy"] * 1e-10  # wavelength [m]

    xPos, yPos = [], []
    for _ in range(obj["seed"]):
        xPos.append(random.randint(0, obj["dimx"] - 1))
        yPos.append(random.randint(0, obj["dimy"] - 1))

    xPos_Brw, yPos_Brw = [], []
    n_steps = obj["dimt"] - 1
    for ct, _dt in enumerate(range(1, obj["dimt"])):
        xPos_Brw_dummy, yPos_Brw_dummy = [], []

        if ct == 0:
            xPos_Brw_time, yPos_Brw_time = copy.deepcopy(xPos), copy.deepcopy(yPos)
        else:
            xPos_Brw_time = copy.deepcopy(xPos_Brw[ct - 1])
            yPos_Brw_time = copy.deepcopy(yPos_Brw[ct - 1])

        for ii in range(obj["seed"]):
            initx_Brw = xPos_Brw_time[ii]
            inity_Brw = yPos_Brw_time[ii]

            jumpx = np.round(random.randint(-100, 100) / 100 * obj["jump"]).astype("int")
            jumpy = np.round(random.randint(-100, 100) / 100 * obj["jump"]).astype("int")

            initx_Brw += jumpx
            inity_Brw += jumpy
            initx_Brw, inity_Brw = BC_conti(initx_Brw, inity_Brw, obj)

            xPos_Brw_dummy.append(initx_Brw)
            yPos_Brw_dummy.append(inity_Brw)

        xPos_Brw.append(xPos_Brw_dummy)
        yPos_Brw.append(yPos_Brw_dummy)

        if progress_cb is not None and n_steps > 0 and (ct % max(1, n_steps // 20) == 0):
            progress_cb(f"Brownian motion 계산 중... {ct}/{n_steps}")

    if progress_cb is not None:
        progress_cb("Speckle 패턴 계산 중 (particle 이미지 생성 -> FFT)... 파라미터에 따라 시간이 걸릴 수 있습니다.")

    Particle_img, Spec_img = Speckle_calculator(xPos_Brw, yPos_Brw, obj)
    return Particle_img, Spec_img, obj


def run_roi_and_g2(Spec_img, obj, baseline_subtract=True, n_lag_points=100, progress_cb=None):
    """Speckle_sim.py 100~135번 줄 로직 그대로 (progress 콜백만 추가).

    Parameters
    ----------
    Spec_img : ndarray (N_frames, H, W)
    obj      : dict (run_brownian_and_speckle 가 반환한 obj, Lambda 포함)
    baseline_subtract : bool
        True면 원본 코드처럼 g2_ -= nanmedian(g2_[-30:]) 로 baseline을 0으로 맞춤
        (여러 bin을 같은 y축에서 비교하기 쉬움). False면 원시 g2 값을 그대로 반환.
    n_lag_points : int
        로그 스케일 지연시간(lag) 샘플 개수.

    Returns
    -------
    dict with keys: Indexmat, dellist, dtlist, g2_list, npix_list,
                     q_centers_nm, r_centers, obj(Beamcenter 포함)
    """
    obj = copy.deepcopy(obj)
    obj["Beamcenter"] = [len(Spec_img[0, :, 0]) / 2.0, len(Spec_img[0, 0, :]) / 2.0]

    if progress_cb is not None:
        progress_cb("ROI(Q-bin) 계산 중...")
    Indexmat = ROI_matrix(Spec_img[0], obj)

    n_frames = Spec_img.shape[0]
    dellist = np.unique(np.round(np.logspace(0, np.log10(max(n_frames - 1, 1)), n_lag_points)))
    dtlist = dellist * obj["timestep"]

    # ROI 중심 pixel radius -> q 변환 [1/nm] (Particle_Sim_module.azimuthal_integration 과 동일한 식)
    edges = np.linspace(obj["StartPixel"], obj["EndPixel"], obj["BinNum"] + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    Lambda = obj.get("Lambda", 12.34 / obj["Energy"] * 1e-10)
    two_theta = np.arctan(r_centers * obj["pixelsize"] / obj["SDD"])
    theta = 0.5 * two_theta
    q_centers_nm = (4.0 * np.pi / Lambda) * np.sin(theta) * 1e-9  # [1/nm]

    g2_list, npix_list = [], []
    for ii in range(obj["BinNum"]):
        if progress_cb is not None:
            progress_cb(f"g2 계산 중... bin {ii + 1}/{obj['BinNum']}")
        Indx_ = Indexmat[ii]
        npix_list.append(len(Indx_))
        if len(Indx_) == 0:
            g2_list.append(np.full(len(dellist), np.nan))
            continue
        g2_ = np.array(cal_g2(Spec_img[:, Indx_[:, 0], Indx_[:, 1]], dellist))
        if baseline_subtract:
            tail = g2_[-30:] if len(g2_) >= 30 else g2_
            g2_ = g2_ - np.nanmedian(tail)
        g2_list.append(g2_)

    return {
        "Indexmat": Indexmat,
        "dellist": dellist,
        "dtlist": dtlist,
        "g2_list": g2_list,
        "npix_list": npix_list,
        "q_centers_nm": q_centers_nm,
        "r_centers": r_centers,
        "obj": obj,
    }


def compute_iq(Spec_img, obj, bins=100, r_min=0.0, r_max=None, progress_cb=None):
    """시간 평균한 speckle 패턴에 Particle_Sim_module.azimuthal_integration() 을
    그대로 적용해서 I(q) 를 구한다 (pixel binning = bins).

    Parameters
    ----------
    Spec_img : ndarray (N_frames, H, W)
    obj      : dict (SDD, pixelsize, Energy 또는 Lambda 포함)
    bins     : int, azimuthal_integration 의 반경 방향 bin 개수 (pixel binning)
    r_min, r_max : float or None, 적분할 반경 범위 [pixel] (r_max=None 이면 이미지 끝까지)

    Returns
    -------
    dict: q_nm (1/nm), I_mean, I_std, counts, avg_img
    """
    if progress_cb is not None:
        progress_cb("프레임 평균 speckle 패턴 계산 중...")
    avg_img = np.mean(Spec_img, axis=0)

    y0 = avg_img.shape[0] / 2.0  # row center
    x0 = avg_img.shape[1] / 2.0  # col center
    Lambda = obj.get("Lambda", 12.34 / obj["Energy"] * 1e-10)

    if progress_cb is not None:
        progress_cb(f"azimuthal integration 계산 중 (bins={bins})...")
    q_1_m, I_mean, I_std, counts = azimuthal_integration(
        avg_img, x0, y0,
        bins=int(bins), r_min=float(r_min), r_max=(None if r_max in (None, "", 0) else float(r_max)),
        pixel_size=obj["pixelsize"], distance=obj["SDD"], wavelength=Lambda,
        return_std=True,
    )
    return {
        "q_nm": np.asarray(q_1_m) * 1e-9,
        "I_mean": np.asarray(I_mean),
        "I_std": np.asarray(I_std),
        "counts": np.asarray(counts),
        "avg_img": avg_img,
    }


def _single_exp_model(tau, baseline, beta, gamma):
    return baseline + beta * np.exp(-2.0 * gamma * tau)


def fit_gamma_per_bin(dtlist, g2_list, q_centers_nm, npix_list=None):
    """각 Q-bin의 g2(tau) 곡선을 baseline + beta*exp(-2*Gamma*tau) 로 "개별" 피팅해서
    Gamma(q) 를 얻는다 (선형피팅/D 계산은 fit_D_from_gamma 에서 별도로 수행 -
    어떤 bin을 D 피팅에 쓸지 사용자가 고를 수 있도록 단계를 분리했다).

    scipy 가 없으면 RuntimeError.

    Returns
    -------
    dict: q_nm, q2_nm2, gamma, gamma_err, beta, fit_ok (각 bin별 배열)
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("scipy 가 설치되어 있지 않습니다. `pip install scipy` 후 다시 시도하세요.")

    n_bins = len(g2_list)
    gamma = np.full(n_bins, np.nan)
    gamma_err = np.full(n_bins, np.nan)
    beta_arr = np.full(n_bins, np.nan)
    fit_ok = np.zeros(n_bins, dtype=bool)

    for ii, g2_ in enumerate(g2_list):
        g2_ = np.asarray(g2_, dtype=float)
        valid = np.isfinite(g2_) & np.isfinite(dtlist) & (dtlist > 0)
        if npix_list is not None and npix_list[ii] == 0:
            continue
        if valid.sum() < 4:
            continue

        x = np.asarray(dtlist)[valid]
        y = g2_[valid]

        tail = y[-max(1, len(y) // 5):]
        baseline0 = float(np.nanmedian(tail))
        beta0 = float(y[0] - baseline0) if (y[0] - baseline0) != 0 else 1.0
        # 대략적인 half-decay 지점으로 Gamma 초기값 추정
        half_level = baseline0 + beta0 / 2.0
        try:
            idx_half = np.argmin(np.abs(y - half_level))
            tau_half = max(x[idx_half], x[0])
        except Exception:
            tau_half = x[len(x) // 4]
        gamma0 = 1.0 / (2.0 * tau_half) if tau_half > 0 else 1.0

        try:
            popt, pcov = curve_fit(
                _single_exp_model, x, y,
                p0=[baseline0, beta0, gamma0],
                maxfev=20000,
                bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
            )
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan] * 3
            gamma[ii] = popt[2]
            gamma_err[ii] = perr[2]
            beta_arr[ii] = popt[1]
            fit_ok[ii] = np.isfinite(gamma[ii]) and gamma[ii] > 0
        except Exception:
            continue

    q_nm = np.asarray(q_centers_nm, dtype=float)
    q2_nm2 = q_nm ** 2

    return {
        "q_nm": q_nm,
        "q2_nm2": q2_nm2,
        "gamma": gamma,
        "gamma_err": gamma_err,
        "beta": beta_arr,
        "fit_ok": fit_ok,
    }


def fit_D_from_gamma(q2_nm2, gamma, gamma_err, include_mask):
    """Gamma vs q^2 선형피팅(가중치 = 1/gamma_err, weighted least squares)으로
    확산계수 D = slope 를 구한다. include_mask=False 인 bin(예: outlier로 판단해서
    사용자가 제외한 bin)은 피팅에서 빠진다.

    Returns
    -------
    dict: n_used, D_fit_nm2_s, D_fit_err_nm2_s, intercept_1_s,
          q2_fit_line, gamma_fit_line (플롯용)
    """
    q2_nm2 = np.asarray(q2_nm2, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    gamma_err = np.asarray(gamma_err, dtype=float)
    include_mask = np.asarray(include_mask, dtype=bool)

    good = include_mask & np.isfinite(gamma) & np.isfinite(q2_nm2)
    result = {"n_used": int(good.sum())}

    if good.sum() < 2:
        result.update(D_fit_nm2_s=np.nan, D_fit_err_nm2_s=np.nan, intercept_1_s=np.nan,
                       q2_fit_line=np.array([]), gamma_fit_line=np.array([]))
        return result

    x_fit = q2_nm2[good]
    y_fit = gamma[good]
    yerr = gamma_err[good]
    use_weights = np.all(np.isfinite(yerr)) and np.all(yerr > 0)
    w = 1.0 / yerr if use_weights else None

    coeffs, cov = np.polyfit(x_fit, y_fit, 1, w=w, cov=True)
    slope, intercept = coeffs[0], coeffs[1]
    slope_err = np.sqrt(cov[0, 0]) if cov is not None else np.nan

    x_line = np.linspace(0, x_fit.max() * 1.1, 50)
    y_line = slope * x_line + intercept

    result.update(
        D_fit_nm2_s=slope,
        D_fit_err_nm2_s=slope_err,
        intercept_1_s=intercept,
        q2_fit_line=x_line,
        gamma_fit_line=y_line,
    )
    return result


def fit_gamma_of_q(dtlist, g2_list, q_centers_nm, npix_list=None):
    """편의용 wrapper: fit_gamma_per_bin() 으로 개별 bin을 피팅한 뒤,
    fit_ok 인 bin 전부를 사용해서 fit_D_from_gamma() 를 한 번에 수행한다.
    (GUI는 outlier를 사용자가 직접 뺄 수 있도록 두 함수를 별도로 호출한다.)
    """
    per_bin = fit_gamma_per_bin(dtlist, g2_list, q_centers_nm, npix_list=npix_list)
    d_fit = fit_D_from_gamma(per_bin["q2_nm2"], per_bin["gamma"], per_bin["gamma_err"], per_bin["fit_ok"])
    result = dict(per_bin)
    result["n_good"] = d_fit["n_used"]
    result.update({k: v for k, v in d_fit.items() if k != "n_used"})
    return result


def stokes_einstein_radius_nm(D_nm2_s, T_K=298.0, eta_Pa_s=1.0e-3):
    """D [nm^2/s] -> Stokes-Einstein 유체역학적 반경 a [nm].
        D = kB*T / (6*pi*eta*a)  =>  a = kB*T / (6*pi*eta*D)
    """
    if D_nm2_s is None or not np.isfinite(D_nm2_s) or D_nm2_s <= 0:
        return np.nan
    D_m2_s = D_nm2_s * 1e-18
    a_m = K_BOLTZMANN * T_K / (6.0 * np.pi * eta_Pa_s * D_m2_s)
    return a_m * 1e9  # [nm]


def theoretical_D_pixel_model(obj):
    """이 toy 시뮬레이션에서 'jump'(프레임당 무작위 이동 폭)와 timestep, sample_pixelsize
    로부터 기대되는 확산계수를 직접 계산한다 (Gamma(q) 피팅 결과와 비교/검증용).

    jumpx, jumpy ~ Uniform(-jump, jump) (독립), 프레임당 2D 변위 분산:
        Var(jumpx)+Var(jumpy) = 2 * jump^2/3   [grid px^2]
    2D 확산의 MSD = 4*D*t 이므로 한 스텝(timestep)당:
        D_pixel = (2*jump^2/3) / (4*timestep)  [grid px^2 / s]
        D_phys  = D_pixel * sample_pixelsize^2  [m^2/s]
    """
    jump = obj["jump"]
    timestep = obj["timestep"]
    dx = obj.get("sample_pixelsize", None)
    D_pixel = (2.0 * jump ** 2 / 3.0) / (4.0 * timestep)  # [px^2/s]
    if dx is None:
        return {"D_pixel_px2_s": D_pixel, "D_phys_m2_s": np.nan, "D_phys_nm2_s": np.nan}
    D_phys_m2_s = D_pixel * dx ** 2
    return {
        "D_pixel_px2_s": D_pixel,
        "D_phys_m2_s": D_phys_m2_s,
        "D_phys_nm2_s": D_phys_m2_s * 1e18,
    }
