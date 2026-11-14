module check_mod
implicit none
integer,parameter::wp=kind(1d0)
type state
real(wp)::p(4,2)
end type
type(state)::kin(2)
contains
real(wp) function v1_pt(p)
real(wp),intent(in)::p(4)
v1_pt=sqrt(p(1)**2+p(2)**2)
end function
real(wp) function v1_minv(p)
real(wp),intent(in)::p(4)
v1_minv=sqrt(max(p(4)**2-sum(p(1:3)**2),0d0))
end function
real(wp) function v1_y(p)
real(wp),intent(in)::p(4)
v1_y=0.5d0*log((p(4)+p(3))/(p(4)-p(3)))
end function
  pure function cheb_T(u_in, n) result(res)
    real(kind=wp), intent(in) :: u_in
    integer, intent(in) :: n
    real(kind=wp) :: res, u, t0, t1, t2
    integer :: k
    u = max(-1.0_wp, min(1.0_wp, u_in))
    if (n <= 0) then
      res = 1.0_wp
    else if (n == 1) then
      res = u
    else
      t0 = 1.0_wp
      t1 = u
      do k = 2, n
        t2 = 2.0_wp*u*t1 - t0
        t0 = t1
        t1 = t2
      end do
      res = t1
    end if
  end function cheb_T

  real(kind=wp) function bw_u(x, lo, hi, mres, gres) result(u)
    real(kind=wp), intent(in) :: x, lo, hi, mres, gres
    real(kind=wp) :: t, tl, th
    t  = atan((x*x   - mres*mres)/(mres*gres))
    tl = atan((lo*lo - mres*mres)/(mres*gres))
    th = atan((hi*hi - mres*mres)/(mres*gres))
    u  = 2.0_wp*(t - tl)/(th - tl) - 1.0_wp
  end function bw_u

  pure function prof_w(x, a, b, c, d) result(res)
    real(kind=wp), intent(in) :: x, a, b, c, d
    real(kind=wp) :: res, t, up, dn, lx
    lx = log(max(x, 1.0e-30_wp))
    if (b > a) then
      t = max(0.0_wp, min(1.0_wp, (lx - log(a))/(log(b) - log(a))))
      up = t*t*t*(t*(t*6.0_wp - 15.0_wp) + 10.0_wp)
    else
      up = merge(1.0_wp, 0.0_wp, x >= a)
    end if
    if (d > c) then
      t = max(0.0_wp, min(1.0_wp, (lx - log(c))/(log(d) - log(c))))
      dn = 1.0_wp - t*t*t*(t*(t*6.0_wp - 15.0_wp) + 10.0_wp)
    else
      dn = merge(1.0_wp, 0.0_wp, x < c)
    end if
    res = up * dn
  end function prof_w

  real(kind=wp) function eval_chebT_mw(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 40.0_wp, hi = 1000.0_wp
    real(kind=wp), parameter :: mw = 80.385_wp, gw = 2.085_wp
    if (npar < 1) then; eval_chebT_mw = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = v1_minv(pV)
    u = bw_u(x, lo, hi, mw, gw)
    eval_chebT_mw = cheb_T(u, n)
  end function eval_chebT_mw

  real(kind=wp) function eval_wpred(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep, key, iq, ia, im
    real(kind=wp), dimension(4) :: pl,pn,pv
    real(kind=wp), dimension(3) :: wg
    real(kind=wp), dimension(5) :: hh,hs
    real(kind=wp) :: mass,qt,mtv,c,uy,wc,gate,bornpt,tt,sgn
    real(kind=wp) :: rho,ax,ay,sx,sy,ez,dot,cross,ptl,ptnu,mtlep,p2,wh
    logical,save :: cached_valid=.false.
    real(kind=wp),save :: cached_pl(4),cached_pn(4),cached_values(157)
    res=0.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep)
    pn=kin(npar)%p(:,npar+1-ilep)
    if(key<1.or.key>157) return
    if(cached_valid) then
      if(all(pl==cached_pl).and.all(pn==cached_pn)) then
        res=cached_values(key)
        return
      endif
    endif
    ! Exact equality with a previously finite key already proves finiteness.
    ! New keys are checked explicitly before any kinematic arithmetic.
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) then
      cached_valid=.false.
      error stop 'nonfinite four-vector in W predictivity booking'
    endif
    if(key>=152) then
      pv=pl+pn;qt=v1_pt(pv)
      ptl=v1_pt(pl);ptnu=v1_pt(pn)
      mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
      select case(key)
      case(152)
        res=merge(1.0_wp,0.0_wp,ptl>28.0_wp)
      case(153)
        res=merge(1.0_wp,0.0_wp,ptl>32.0_wp)
      case(154)
        res=merge(1.0_wp,0.0_wp,qt>50.0_wp)
      case(155)
        res=merge(1.0_wp,0.0_wp,ptl>32.0_wp.and.qt>50.0_wp)
      case(156)
        res=merge(1.0_wp,0.0_wp,mtlep>60.0_wp)
      case(157)
        res=merge(1.0_wp,0.0_wp,mtlep>60.0_wp.and.ptnu>30.0_wp.and.abs(v1_y(pl))<1.7_wp)
      end select
      return
    endif
    pv=pl+pn; mass=v1_minv(pv); qt=v1_pt(pv); mtv=sqrt(mass*mass+qt*qt)
    if(mass<=0.0_wp.or.mtv<=0.0_wp) return
    sgn=1.0_wp; if(pv(3)<0.0_wp) sgn=-1.0_wp
    c=sgn*((pl(4)+pl(3))*(pn(4)-pn(3))-(pl(4)-pl(3))*(pn(4)+pn(3)))/(mass*mtv)
    c=max(-1.0_wp,min(1.0_wp,c))
    uy=abs(v1_y(pv))/2.0_wp-1.0_wp
    wc=prof_w(c+1.0_wp,0.3_wp,0.4_wp,1.6_wp,1.7_wp)
    bornpt=0.5_wp*mass*sqrt(max(1.0_wp-c*c,0.0_wp))
    gate=0.0_wp
    if(bornpt>25.0_wp) then
      tt=min(log(bornpt/25.0_wp)/log(28.0_wp/25.0_wp),1.0_wp)
      gate=6.0_wp*tt**5-15.0_wp*tt**4+10.0_wp*tt**3
    endif
    wg(1)=gate*prof_w(c+1.0_wp,0.4_wp,0.5_wp,1.5_wp,1.6_wp)
    wg(2)=gate*wc
    wg(3)=gate*prof_w(c+1.0_wp,0.25_wp,0.35_wp,1.65_wp,1.75_wp)
    rho=qt/mtv
    dot=pl(1)*pv(1)+pl(2)*pv(2)
    cross=(-pl(1)*pv(2)+pl(2)*pv(1))*sgn
    ez=(pl(4)*pv(4)-pl(3)*pv(3))/mtv
    ! ax=rho*sx and ay=rho*sy remain finite and vanish at qT=0.
    ax=2.0_wp*(mtv*dot-qt*qt*ez)/(mass*mass*mtv)
    ay=2.0_wp*cross/(mass*mtv)
    sx=0.0_wp; sy=0.0_wp
    if(qt>1.0e-12_wp) then
      sx=ax/rho; sy=ay/rho
    endif
    p2=0.5_wp*(1.0_wp-3.0_wp*c*c)
    wh=prof_w(max(qt,1.0e-12_wp),25.0_wp,50.0_wp,500.0_wp,500.0_wp)
    hh=wh*[p2,c,2.0_wp*c*sx,sx*sx-sy*sy,sx]
    hs=[rho*rho*p2,rho*rho*c,2.0_wp*c*ax,ax*ax-ay*ay,ax]
    ptl=v1_pt(pl); ptnu=v1_pt(pn)
    mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
    iq=0; if(qt>=25.0_wp) iq=1; if(qt>=50.0_wp) iq=2
    ia=0; if(abs(c)>0.6_wp) ia=1; if(abs(c)>=0.7_wp) ia=2
    im=0; if(mass>=76.0_wp) im=1; if(mass>=84.0_wp) im=2
    cached_values(1)=wc*cheb_T(c,7)
    cached_values(2)=wc*cheb_T(c,8)
    cached_values(3)=wc*cheb_T(c,9)
    cached_values(4)=wc*cheb_T(c,10)
    cached_values(5)=wc*eval_chebT_mw(npar,1)*cheb_T(c,0)
    cached_values(6)=wc*eval_chebT_mw(npar,1)*cheb_T(c,1)
    cached_values(7)=wc*eval_chebT_mw(npar,1)*cheb_T(c,2)
    cached_values(8)=wc*eval_chebT_mw(npar,2)*cheb_T(c,0)
    cached_values(9)=wc*eval_chebT_mw(npar,2)*cheb_T(c,1)
    cached_values(10)=wc*eval_chebT_mw(npar,2)*cheb_T(c,2)
    cached_values(11)=wg(1)*cheb_T(c,0)
    cached_values(12)=wg(1)*cheb_T(c,1)
    cached_values(13)=wg(1)*cheb_T(c,2)
    cached_values(14)=wg(1)*cheb_T(c,3)
    cached_values(15)=wg(1)*cheb_T(c,4)
    cached_values(16)=wg(1)*cheb_T(c,5)
    cached_values(17)=wg(1)*cheb_T(c,6)
    cached_values(18)=wg(1)*cheb_T(uy,1)*cheb_T(c,0)
    cached_values(19)=wg(1)*cheb_T(uy,1)*cheb_T(c,1)
    cached_values(20)=wg(1)*cheb_T(uy,1)*cheb_T(c,2)
    cached_values(21)=wg(1)*cheb_T(uy,2)*cheb_T(c,0)
    cached_values(22)=wg(1)*cheb_T(uy,2)*cheb_T(c,1)
    cached_values(23)=wg(1)*cheb_T(uy,2)*cheb_T(c,2)
    cached_values(24)=wg(2)*cheb_T(c,0)
    cached_values(25)=wg(2)*cheb_T(c,1)
    cached_values(26)=wg(2)*cheb_T(c,2)
    cached_values(27)=wg(2)*cheb_T(c,3)
    cached_values(28)=wg(2)*cheb_T(c,4)
    cached_values(29)=wg(2)*cheb_T(c,5)
    cached_values(30)=wg(2)*cheb_T(c,6)
    cached_values(31)=wg(2)*cheb_T(uy,1)*cheb_T(c,0)
    cached_values(32)=wg(2)*cheb_T(uy,1)*cheb_T(c,1)
    cached_values(33)=wg(2)*cheb_T(uy,1)*cheb_T(c,2)
    cached_values(34)=wg(2)*cheb_T(uy,2)*cheb_T(c,0)
    cached_values(35)=wg(2)*cheb_T(uy,2)*cheb_T(c,1)
    cached_values(36)=wg(2)*cheb_T(uy,2)*cheb_T(c,2)
    cached_values(37)=wg(3)*cheb_T(c,0)
    cached_values(38)=wg(3)*cheb_T(c,1)
    cached_values(39)=wg(3)*cheb_T(c,2)
    cached_values(40)=wg(3)*cheb_T(c,3)
    cached_values(41)=wg(3)*cheb_T(c,4)
    cached_values(42)=wg(3)*cheb_T(c,5)
    cached_values(43)=wg(3)*cheb_T(c,6)
    cached_values(44)=wg(3)*cheb_T(uy,1)*cheb_T(c,0)
    cached_values(45)=wg(3)*cheb_T(uy,1)*cheb_T(c,1)
    cached_values(46)=wg(3)*cheb_T(uy,1)*cheb_T(c,2)
    cached_values(47)=wg(3)*cheb_T(uy,2)*cheb_T(c,0)
    cached_values(48)=wg(3)*cheb_T(uy,2)*cheb_T(c,1)
    cached_values(49)=wg(3)*cheb_T(uy,2)*cheb_T(c,2)
    cached_values(50)=hh(1)*cheb_T(uy,0)
    cached_values(51)=hh(2)*cheb_T(uy,0)
    cached_values(52)=hh(3)*cheb_T(uy,0)
    cached_values(53)=hh(4)*cheb_T(uy,0)
    cached_values(54)=hh(5)*cheb_T(uy,0)
    cached_values(55)=hh(1)*cheb_T(uy,1)
    cached_values(56)=hh(2)*cheb_T(uy,1)
    cached_values(57)=hh(3)*cheb_T(uy,1)
    cached_values(58)=hh(4)*cheb_T(uy,1)
    cached_values(59)=hh(5)*cheb_T(uy,1)
    cached_values(60)=hh(1)*cheb_T(uy,2)
    cached_values(61)=hh(2)*cheb_T(uy,2)
    cached_values(62)=hh(3)*cheb_T(uy,2)
    cached_values(63)=hh(4)*cheb_T(uy,2)
    cached_values(64)=hh(5)*cheb_T(uy,2)
    cached_values(65)=hs(1)*cheb_T(uy,0)
    cached_values(66)=hs(2)*cheb_T(uy,0)
    cached_values(67)=hs(3)*cheb_T(uy,0)
    cached_values(68)=hs(4)*cheb_T(uy,0)
    cached_values(69)=hs(5)*cheb_T(uy,0)
    cached_values(70)=hs(1)*cheb_T(uy,1)
    cached_values(71)=hs(2)*cheb_T(uy,1)
    cached_values(72)=hs(3)*cheb_T(uy,1)
    cached_values(73)=hs(4)*cheb_T(uy,1)
    cached_values(74)=hs(5)*cheb_T(uy,1)
    cached_values(75)=hs(1)*cheb_T(uy,2)
    cached_values(76)=hs(2)*cheb_T(uy,2)
    cached_values(77)=hs(3)*cheb_T(uy,2)
    cached_values(78)=hs(4)*cheb_T(uy,2)
    cached_values(79)=hs(5)*cheb_T(uy,2)
    cached_values(80)=merge(1.0_wp,0.0_wp,iq==0.and.ia==0)
    cached_values(81)=merge(1.0_wp,0.0_wp,iq==0.and.ia==0.and.ptl>27.0_wp)
    cached_values(82)=merge(1.0_wp,0.0_wp,iq==0.and.ia==0.and.ptl>30.0_wp)
    cached_values(83)=merge(1.0_wp,0.0_wp,iq==0.and.ia==0.and.bornpt>27.0_wp)
    cached_values(84)=merge(1.0_wp,0.0_wp,iq==0.and.ia==0.and.bornpt>30.0_wp)
    cached_values(85)=merge(1.0_wp,0.0_wp,iq==0.and.ia==1)
    cached_values(86)=merge(1.0_wp,0.0_wp,iq==0.and.ia==1.and.ptl>27.0_wp)
    cached_values(87)=merge(1.0_wp,0.0_wp,iq==0.and.ia==1.and.ptl>30.0_wp)
    cached_values(88)=merge(1.0_wp,0.0_wp,iq==0.and.ia==1.and.bornpt>27.0_wp)
    cached_values(89)=merge(1.0_wp,0.0_wp,iq==0.and.ia==1.and.bornpt>30.0_wp)
    cached_values(90)=merge(1.0_wp,0.0_wp,iq==0.and.ia==2)
    cached_values(91)=merge(1.0_wp,0.0_wp,iq==0.and.ia==2.and.ptl>27.0_wp)
    cached_values(92)=merge(1.0_wp,0.0_wp,iq==0.and.ia==2.and.ptl>30.0_wp)
    cached_values(93)=merge(1.0_wp,0.0_wp,iq==0.and.ia==2.and.bornpt>27.0_wp)
    cached_values(94)=merge(1.0_wp,0.0_wp,iq==0.and.ia==2.and.bornpt>30.0_wp)
    cached_values(95)=merge(1.0_wp,0.0_wp,iq==1.and.ia==0)
    cached_values(96)=merge(1.0_wp,0.0_wp,iq==1.and.ia==0.and.ptl>27.0_wp)
    cached_values(97)=merge(1.0_wp,0.0_wp,iq==1.and.ia==0.and.ptl>30.0_wp)
    cached_values(98)=merge(1.0_wp,0.0_wp,iq==1.and.ia==0.and.bornpt>27.0_wp)
    cached_values(99)=merge(1.0_wp,0.0_wp,iq==1.and.ia==0.and.bornpt>30.0_wp)
    cached_values(100)=merge(1.0_wp,0.0_wp,iq==1.and.ia==1)
    cached_values(101)=merge(1.0_wp,0.0_wp,iq==1.and.ia==1.and.ptl>27.0_wp)
    cached_values(102)=merge(1.0_wp,0.0_wp,iq==1.and.ia==1.and.ptl>30.0_wp)
    cached_values(103)=merge(1.0_wp,0.0_wp,iq==1.and.ia==1.and.bornpt>27.0_wp)
    cached_values(104)=merge(1.0_wp,0.0_wp,iq==1.and.ia==1.and.bornpt>30.0_wp)
    cached_values(105)=merge(1.0_wp,0.0_wp,iq==1.and.ia==2)
    cached_values(106)=merge(1.0_wp,0.0_wp,iq==1.and.ia==2.and.ptl>27.0_wp)
    cached_values(107)=merge(1.0_wp,0.0_wp,iq==1.and.ia==2.and.ptl>30.0_wp)
    cached_values(108)=merge(1.0_wp,0.0_wp,iq==1.and.ia==2.and.bornpt>27.0_wp)
    cached_values(109)=merge(1.0_wp,0.0_wp,iq==1.and.ia==2.and.bornpt>30.0_wp)
    cached_values(110)=merge(1.0_wp,0.0_wp,iq==2.and.ia==0)
    cached_values(111)=merge(1.0_wp,0.0_wp,iq==2.and.ia==0.and.ptl>27.0_wp)
    cached_values(112)=merge(1.0_wp,0.0_wp,iq==2.and.ia==0.and.ptl>30.0_wp)
    cached_values(113)=merge(1.0_wp,0.0_wp,iq==2.and.ia==0.and.bornpt>27.0_wp)
    cached_values(114)=merge(1.0_wp,0.0_wp,iq==2.and.ia==0.and.bornpt>30.0_wp)
    cached_values(115)=merge(1.0_wp,0.0_wp,iq==2.and.ia==1)
    cached_values(116)=merge(1.0_wp,0.0_wp,iq==2.and.ia==1.and.ptl>27.0_wp)
    cached_values(117)=merge(1.0_wp,0.0_wp,iq==2.and.ia==1.and.ptl>30.0_wp)
    cached_values(118)=merge(1.0_wp,0.0_wp,iq==2.and.ia==1.and.bornpt>27.0_wp)
    cached_values(119)=merge(1.0_wp,0.0_wp,iq==2.and.ia==1.and.bornpt>30.0_wp)
    cached_values(120)=merge(1.0_wp,0.0_wp,iq==2.and.ia==2)
    cached_values(121)=merge(1.0_wp,0.0_wp,iq==2.and.ia==2.and.ptl>27.0_wp)
    cached_values(122)=merge(1.0_wp,0.0_wp,iq==2.and.ia==2.and.ptl>30.0_wp)
    cached_values(123)=merge(1.0_wp,0.0_wp,iq==2.and.ia==2.and.bornpt>27.0_wp)
    cached_values(124)=merge(1.0_wp,0.0_wp,iq==2.and.ia==2.and.bornpt>30.0_wp)
    cached_values(125)=merge(1.0_wp,0.0_wp,iq==0.and.im==0)
    cached_values(126)=merge(1.0_wp,0.0_wp,iq==0.and.im==0.and.ptl>27.0_wp)
    cached_values(127)=merge(1.0_wp,0.0_wp,iq==0.and.im==0.and.ptl>30.0_wp)
    cached_values(128)=merge(1.0_wp,0.0_wp,iq==0.and.im==1)
    cached_values(129)=merge(1.0_wp,0.0_wp,iq==0.and.im==1.and.ptl>27.0_wp)
    cached_values(130)=merge(1.0_wp,0.0_wp,iq==0.and.im==1.and.ptl>30.0_wp)
    cached_values(131)=merge(1.0_wp,0.0_wp,iq==0.and.im==2)
    cached_values(132)=merge(1.0_wp,0.0_wp,iq==0.and.im==2.and.ptl>27.0_wp)
    cached_values(133)=merge(1.0_wp,0.0_wp,iq==0.and.im==2.and.ptl>30.0_wp)
    cached_values(134)=merge(1.0_wp,0.0_wp,iq==1.and.im==0)
    cached_values(135)=merge(1.0_wp,0.0_wp,iq==1.and.im==0.and.ptl>27.0_wp)
    cached_values(136)=merge(1.0_wp,0.0_wp,iq==1.and.im==0.and.ptl>30.0_wp)
    cached_values(137)=merge(1.0_wp,0.0_wp,iq==1.and.im==1)
    cached_values(138)=merge(1.0_wp,0.0_wp,iq==1.and.im==1.and.ptl>27.0_wp)
    cached_values(139)=merge(1.0_wp,0.0_wp,iq==1.and.im==1.and.ptl>30.0_wp)
    cached_values(140)=merge(1.0_wp,0.0_wp,iq==1.and.im==2)
    cached_values(141)=merge(1.0_wp,0.0_wp,iq==1.and.im==2.and.ptl>27.0_wp)
    cached_values(142)=merge(1.0_wp,0.0_wp,iq==1.and.im==2.and.ptl>30.0_wp)
    cached_values(143)=merge(1.0_wp,0.0_wp,iq==2.and.im==0)
    cached_values(144)=merge(1.0_wp,0.0_wp,iq==2.and.im==0.and.ptl>27.0_wp)
    cached_values(145)=merge(1.0_wp,0.0_wp,iq==2.and.im==0.and.ptl>30.0_wp)
    cached_values(146)=merge(1.0_wp,0.0_wp,iq==2.and.im==1)
    cached_values(147)=merge(1.0_wp,0.0_wp,iq==2.and.im==1.and.ptl>27.0_wp)
    cached_values(148)=merge(1.0_wp,0.0_wp,iq==2.and.im==1.and.ptl>30.0_wp)
    cached_values(149)=merge(1.0_wp,0.0_wp,iq==2.and.im==2)
    cached_values(150)=merge(1.0_wp,0.0_wp,iq==2.and.im==2.and.ptl>27.0_wp)
    cached_values(151)=merge(1.0_wp,0.0_wp,iq==2.and.im==2.and.ptl>30.0_wp)
    cached_values(152)=merge(1.0_wp,0.0_wp,ptl>28.0_wp)
    cached_values(153)=merge(1.0_wp,0.0_wp,ptl>32.0_wp)
    cached_values(154)=merge(1.0_wp,0.0_wp,qt>50.0_wp)
    cached_values(155)=merge(1.0_wp,0.0_wp,ptl>32.0_wp.and.qt>50.0_wp)
    cached_values(156)=merge(1.0_wp,0.0_wp,mtlep>60.0_wp)
    cached_values(157)=merge(1.0_wp,0.0_wp,mtlep>60.0_wp.and.ptnu>30.0_wp.and.abs(v1_y(pl))<1.7_wp)
    cached_pl=pl; cached_pn=pn; cached_valid=.true.
    res=cached_values(key)
  end function eval_wpred

  real(kind=wp) function eval_wpred_eta(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep,key
    real(kind=wp), dimension(4) :: pl,pn
    real(kind=wp) :: ptl,ptnu,mtlep
    res=-1.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep); pn=kin(npar)%p(:,npar+1-ilep)
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) error stop 'nonfinite four-vector in W eta booking'
    ptl=v1_pt(pl); ptnu=v1_pt(pn)
    mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
    if(key==1.and.ptl<=28.0_wp) return
    if(key==2.and.ptl<=32.0_wp) return
    if(key==3.and.mtlep<=60.0_wp) return
    res=abs(v1_y(pl))
  end function eval_wpred_eta

  real(kind=wp) function eval_wpred_region(npar,code) result(res)
    use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
    integer, intent(in) :: npar,code
    integer :: ilep,key,iq,ia,im
    real(kind=wp), dimension(4) :: pl,pn,pv
    real(kind=wp) :: mass,qt,mtv,c,bornpt,ptl
    res=-1.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep); pn=kin(npar)%p(:,npar+1-ilep)
    if(.not.all(ieee_is_finite(pl)).or..not.all(ieee_is_finite(pn))) error stop 'nonfinite four-vector in W region booking'
    pv=pl+pn;mass=v1_minv(pv);qt=v1_pt(pv);mtv=sqrt(mass*mass+qt*qt)
    if(mass<=0.0_wp) return
    c=((pl(4)+pl(3))*(pn(4)-pn(3))-(pl(4)-pl(3))*(pn(4)+pn(3)))/(mass*mtv)
    c=max(-1.0_wp,min(1.0_wp,c))
    bornpt=0.5_wp*mass*sqrt(max(1.0_wp-c*c,0.0_wp));ptl=v1_pt(pl)
    if((key==2.or.key==7).and.ptl<=27.0_wp) return
    if((key==3.or.key==8).and.ptl<=30.0_wp) return
    if(key==4.and.bornpt<=27.0_wp) return
    if(key==5.and.bornpt<=30.0_wp) return
    iq=0;if(qt>=25.0_wp) iq=1;if(qt>=50.0_wp) iq=2
    ia=0;if(abs(c)>0.6_wp) ia=1;if(abs(c)>=0.7_wp) ia=2
    im=0;if(mass>=76.0_wp) im=1;if(mass>=84.0_wp) im=2
    if(key<=5) then
      res=real(3*iq+ia,wp)+0.5_wp
    else
      res=real(3*iq+im,wp)+0.5_wp
    endif
  end function eval_wpred_region

end module
program check
use check_mod
implicit none
integer::i,j,n,ios
real(wp)::pl(4),pn(4),vals(157+11),tmp(4),early(6)
read(*,*)n
do i=1,n
read(*,*)pl,pn
kin(2)%p(:,1)=pl;kin(2)%p(:,2)=pn
do j=1,6
early(j)=eval_wpred(2,1151+j)
enddo
do j=1,157
vals(j)=eval_wpred(2,1000+j)
enddo
if(any(early/=vals(152:157))) error stop 'cut fast path differs from cached full vector'
do j=1,3
vals(size(vals)-11+j)=eval_wpred_eta(2,1000+j)
enddo
do j=1,8
vals(size(vals)-8+j)=eval_wpred_region(2,1000+j)
enddo
write(*,'(*(ES25.16E3,1X))')vals
kin(2)%p(:,1)=pn;kin(2)%p(:,2)=pl
do j=1,157
vals(j)=eval_wpred(2,2000+j)
enddo
do j=1,3
vals(size(vals)-11+j)=eval_wpred_eta(2,2000+j)
enddo
do j=1,8
vals(size(vals)-8+j)=eval_wpred_region(2,2000+j)
enddo
write(*,'(*(ES25.16E3,1X))')vals
enddo
end program
