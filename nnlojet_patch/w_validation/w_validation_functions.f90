  real(kind=wp) function eval_wval_pt(npar, n) result(res)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    res = 0.0_wp
    if (npar < 2) return
    pV = kin(npar)%p(:,npar-1) + kin(npar)%p(:,npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*log(x/25.0_wp)/log(500.0_wp/25.0_wp)-1.0_wp
    res = eval_w_ptw(npar)*cheb_T(u,n)
  end function eval_wval_pt

  real(kind=wp) function eval_wval_yctw(npar, code) result(res)
    integer, intent(in) :: npar, code
    integer :: ilep, k, n, packed
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: uy, c
    res = 0.0_wp
    if (npar < 2) return
    packed = abs(code)
    ilep = packed/1000
    k = mod(packed,1000)/10
    n = mod(packed,10)
    pV = kin(npar)%p(:,npar-1) + kin(npar)%p(:,npar)
    uy = abs(v1_y(pV))/2.0_wp - 1.0_wp
    c = eval_costhCS_w(npar,ilep)
    res = eval_w_ctw(npar,ilep)*cheb_T(uy,k)*cheb_T(c,n)
  end function eval_wval_yctw

  real(kind=wp) function eval_wval_eta(npar, code) result(res)
    integer, intent(in) :: npar, code
    integer :: ilep, ptcut, idx
    real(kind=wp), dimension(4) :: pl
    res = -1.0_wp
    if (npar < 2) return
    ilep = abs(code)/100
    ptcut = mod(abs(code),100)
    idx = npar-2+ilep
    pl = kin(npar)%p(:,idx)
    if (v1_pt(pl) <= real(ptcut,wp)) return
    res = abs(v1_y(pl))
  end function eval_wval_eta
