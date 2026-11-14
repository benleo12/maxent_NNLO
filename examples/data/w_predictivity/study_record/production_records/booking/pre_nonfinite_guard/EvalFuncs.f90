!-------------------------------------------------------------------------------
! NNLOJET
!-------------------------------------------------------------------------------
!
! MODULE: EvalFuncs
!
! DESCRIPTION:
!> Module collecting the implementation of various observables
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
module EvalFuncs_mod
!-------------------------------------------------------------------------------
  use Kinds_mod, only: wp
  use ObsHelper_mod
  use KinData_mod
  implicit none
  public

contains

  real(kind=wp) function eval_njets(npar)
    integer, intent(in) :: npar
    eval_njets = kin(npar)%njets
  end function eval_njets

  real(kind=wp) function eval_npartons(npar)
    integer, intent(in) :: npar
    eval_npartons = kin(npar)%nproto
  end function eval_npartons

  real(kind=wp) function eval_nphotons(npar)
    integer, intent(in) :: npar
    eval_nphotons = kin(npar)%nphotons
  end function eval_nphotons

  real(kind=wp) function eval_flav_njets(npar)
    integer, intent(in) :: npar
    eval_flav_njets = kin(npar)%flav_njets
  end function eval_flav_njets

  !--------------------------!
  !----- leptons (1-4)  -----!
  !--------------------------!
  !> ordering:
  !>   l1: npar-1  (next-to-last entry)
  !>   l2: npar    (last entry)
  !>   l3: npar-3  (next-to-next-to-next-to-last entry)
  !>   l4: npar-2  (next-to-next-to-last entry)
  !
  real(kind=wp) function eval_pt_lep(npar, i)
    integer, intent(in) :: npar, i
    select case (i)
    case (1)
      eval_pt_lep = v1_pt(kin(npar)%p(:, npar - 1))
    case (2)
      eval_pt_lep = v1_pt(kin(npar)%p(:, npar))
    case (3)
      eval_pt_lep = v1_pt(kin(npar)%p(:, npar - 3))
    case (4)
      eval_pt_lep = v1_pt(kin(npar)%p(:, npar - 2))
    case default
      print *, "eval_pt_lep: invalid index: ", i
      stop
    end select
  end function eval_pt_lep
  !
  real(kind=wp) function eval_E_lep(npar, i)
    integer, intent(in) :: npar, i
    select case (i)
    case (1)
      eval_E_lep = kin(npar)%p(4, npar - 1)
    case (2)
      eval_E_lep = kin(npar)%p(4, npar)
    case (3)
      eval_E_lep = kin(npar)%p(4, npar - 3)
    case (4)
      eval_E_lep = kin(npar)%p(4, npar - 2)
    case default
      print *, "eval_E_lep: invalid index: ", i
      stop
    end select
  end function eval_E_lep
  !
  real(kind=wp) function eval_y_lep(npar, i) ! same as eta
    integer, intent(in) :: npar, i
    select case (i)
    case (1)
      eval_y_lep = v1_y(kin(npar)%p(:, npar - 1))
    case (2)
      eval_y_lep = v1_y(kin(npar)%p(:, npar))
    case (3)
      eval_y_lep = v1_y(kin(npar)%p(:, npar - 3))
    case (4)
      eval_y_lep = v1_y(kin(npar)%p(:, npar - 2))
    case default
      print *, "eval_y_lep: invalid index: ", i
      stop
    end select
  end function eval_y_lep
  !
  real(kind=wp) function eval_phi_lep(npar, i) ! same as eta
    integer, intent(in) :: npar, i
    select case (i)
    case (1)
      eval_phi_lep = v1_phi(kin(npar)%p(:, npar - 1))
    case (2)
      eval_phi_lep = v1_phi(kin(npar)%p(:, npar))
    case (3)
      eval_phi_lep = v1_phi(kin(npar)%p(:, npar - 3))
    case (4)
      eval_phi_lep = v1_phi(kin(npar)%p(:, npar - 2))
    case default
      print *, "eval_phi_lep: invalid index: ", i
      stop
    end select
  end function eval_phi_lep
  !
  real(kind=wp) function eval_dR_j1_lep(npar, i)
    integer, intent(in) :: npar, i
    select case (i)
    case (1)
      eval_dR_j1_lep = v2_delta_R(kin(npar)%pjets(:, 1), &
                                  kin(npar)%p(:, npar - 1))
    case (2)
      eval_dR_j1_lep = v2_delta_R(kin(npar)%pjets(:, 1), &
                                  kin(npar)%p(:, npar))
    case (3)
      eval_dR_j1_lep = v2_delta_R(kin(npar)%pjets(:, 1), &
                                  kin(npar)%p(:, npar - 3))
    case (4)
      eval_dR_j1_lep = v2_delta_R(kin(npar)%pjets(:, 1), &
                                  kin(npar)%p(:, npar - 2))
    case default
      print *, "eval_dR_j1_lep: invalid index: ", i
      stop
    end select
  end function eval_dR_j1_lep

  !----- lepton 1 (next-to-last entry)
  real(kind=wp) function eval_lp_l1(npar)
    ! Lepton projection variable Eq. 3 [1104.3829]
    integer, intent(in) :: npar
    real(kind=wp), dimension(2) :: pt_V, pt_l
    integer :: i
    do i = 1, 2
      pt_V(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
      pt_l(i) = kin(npar)%p(i, npar - 1)
    end do
    eval_lp_l1 = (pt_V(1)*pt_l(1) + pt_V(2)*pt_l(2))/(pt_V(1)**2 + pt_V(2)**2)
  end function eval_lp_l1

  !----- lepton 2 (last entry)
  real(kind=wp) function eval_lp_l2(npar)
    ! Lepton projection variable Eq. 3 [1104.3829]
    integer, intent(in) :: npar
    real(kind=wp), dimension(2) :: pt_V, pt_l
    integer :: i
    do i = 1, 2
      pt_V(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
      pt_l(i) = kin(npar)%p(i, npar)
    end do
    eval_lp_l2 = (pt_V(1)*pt_l(1) + pt_V(2)*pt_l(2))/(pt_V(1)**2 + pt_V(2)**2)
  end function eval_lp_l2

  real(kind=wp) function eval_epem_costh_ej1(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pi(1:4), pj(1:4), cos_ij, Emax, E
    integer i, imax

    !> Find the highest energy jet
    Emax = -1._wp
    do i = 1, kin(npar)%njets
      E = kin(npar)%pjets(4, i)
      if (E .gt. Emax) then
        Emax = E
        imax = i
      end if
    end do

    !> leading jet
    pi = kin(npar)%pjets(:, imax)
    !> electron momentum
    pj = kin(npar)%p(:, 2)
    !> the angle
    cos_ij = dot_eucl(pi(1:3), pj(1:3)) &
             /sqrt(dot_eucl(pi(1:3), pi(1:3))) &
             /sqrt(dot_eucl(pj(1:3), pj(1:3)))
    !> only positive values
    eval_epem_costh_ej1 = abs(cos_ij)
  end function eval_epem_costh_ej1

  real(kind=wp) function eval_epem_costh_en3j(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pj1(1:4), pj2(1:4), pi(1:3), pj(1:4), cos_ij
    !> leading jet
    pj1 = kin(npar)%pjets(:, 1)
    !> subleading jet
    pj2 = kin(npar)%pjets(:, 2)
    !> cross product j1xj2
    pi(1) = pj1(2)*pj2(3) - pj1(3)*pj2(2)
    pi(2) = pj1(3)*pj2(1) - pj1(1)*pj2(3)
    pi(3) = pj1(1)*pj2(2) - pj1(2)*pj2(1)
    !> electron momentum
    pj = kin(npar)%p(:, 2)
    !> the angle
    cos_ij = dot_eucl(pi(1:3), pj(1:3)) &
             /sqrt(dot_eucl(pi(1:3), pi(1:3))) &
             /sqrt(dot_eucl(pj(1:3), pj(1:3)))
    eval_epem_costh_en3j = abs(cos_ij)
  end function eval_epem_costh_en3j

  real(kind=wp) function eval_epem_chi(npar)
    use Constants_mod
    integer, intent(in) :: npar
    real(kind=wp) :: pj1(1:4), pj2(1:4), pe(1:4), pin1(1:3), pj(1:3), cos_ij
    real(kind=wp) :: val, Emax, E
    integer i, imax, j

    !> Find the highest energy jet
    Emax = -1._wp
    do i = 1, kin(npar)%njets
      E = kin(npar)%pjets(4, i)
      if (E .gt. Emax) then
        Emax = E
        imax = i
      end if
    end do

    if (imax .eq. 1) then
      j = 2
    else
      j = 1
    end if

    !> leading jet
    pj1 = kin(npar)%pjets(:, imax)
    !> subleading jet
    pj2 = kin(npar)%pjets(:, j)
    !> cross product j1xj2
    pin1(1) = pj1(2)*pj2(3) - pj1(3)*pj2(2)
    pin1(2) = pj1(3)*pj2(1) - pj1(1)*pj2(3)
    pin1(3) = pj1(1)*pj2(2) - pj1(2)*pj2(1)
    !> electron momentum
    pe = kin(npar)%p(:, 2)
    !> cross product j1xe
    pj(1) = pj1(2)*pe(3) - pj1(3)*pe(2)
    pj(2) = pj1(3)*pe(1) - pj1(1)*pe(3)
    pj(3) = pj1(1)*pe(2) - pj1(2)*pe(1)

    !> the angle
    cos_ij = dot_eucl(pin1(1:3), pj(1:3)) &
             /sqrt(dot_eucl(pin1(1:3), pin1(1:3))) &
             /sqrt(dot_eucl(pj(1:3), pj(1:3)))

    val = acos(cos_ij)
    !> range 0<val<pi/2
    if (val .gt. pi/2._wp) val = pi - val
    if (val .gt. pi/2._wp .or. val .lt. 0._wp) then
      write (*, *) val
      stop
    end if
    eval_epem_chi = val
  end function eval_epem_chi

  !----- C parameter in ep em
  real(kind=wp) function eval_epem_C(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: t11, t12, t13, t22, t23, t33
    real(kind=wp) :: Cpar
    t11 = theta(1, 1, npar)
    t12 = theta(1, 2, npar)
    t13 = theta(1, 3, npar)
    t22 = theta(2, 2, npar)
    t23 = theta(2, 3, npar)
    t33 = theta(3, 3, npar)
    Cpar = -t12**2 - t13**2 - t23**2 + t11*t22 + t11*t33 + t22*t33
    eval_epem_C = 3._wp*Cpar
  end function eval_epem_C

  !----- D parameter in ep em
  real(kind=wp) function eval_epem_D(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: t11, t12, t13, t22, t23, t33
    real(kind=wp) :: Dpar
    t11 = theta(1, 1, npar)
    t12 = theta(1, 2, npar)
    t13 = theta(1, 3, npar)
    t22 = theta(2, 2, npar)
    t23 = theta(2, 3, npar)
    t33 = theta(3, 3, npar)
    Dpar = t11*t22*t33 - t11*t23**2 - t22*t13**2 - t33*t12**2 + 2._wp*t12*t13*t23
    Dpar = 27._wp*Dpar
    eval_epem_D = Dpar
  end function eval_epem_D

  !----- jet C parameter in ep em
  real(kind=wp) function eval_epem_jC(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: t11, t12, t13, t22, t23, t33
    real(kind=wp) :: Cpar
    t11 = thetajet(1, 1, npar)
    t12 = thetajet(1, 2, npar)
    t13 = thetajet(1, 3, npar)
    t22 = thetajet(2, 2, npar)
    t23 = thetajet(2, 3, npar)
    t33 = thetajet(3, 3, npar)
    Cpar = -t12**2 - t13**2 - t23**2 + t11*t22 + t11*t33 + t22*t33
    eval_epem_jC = 3._wp*Cpar
  end function eval_epem_jC

  !----- solve the equation to find the three eigenvalues of sphericity tensor (the theta or thetajet matrix)
    !> Shengjin Formula
  function eval_Sphericitylambda(M) result(lambda)

    real(kind=wp), intent(in) :: M(3, 3)

    real(kind=wp) :: X(3), lambda(3)
    real(kind=wp) :: A, B, C, D, A0, B0, C0, D0, Y1, Y2

    A = 1._wp
    B = -M(1,1)-M(2,2)-M(3,3)
    C = M(1,1)*M(2,2)+M(1,1)*M(3,3)+M(2,2)*M(3,3)-M(1,3)**2-M(1,2)**2-M(2,3)**2
    D = M(1,1)*(M(2,3)**2)+M(2,2)*(M(1,3)**2)+M(3,3)*(M(1,2)**2)-M(1,1)*M(2,2)*M(3,3)-2*M(1,2)*M(2,3)*M(3,1)



    A0 = B*B - 3._wp*A*C
    B0 = B*C - 9._wp*A*D
    C0 = C*C - 3._wp*B*D
    D0 = B0*B0 - 4._wp*A0*C0

      if(abs(A0) < 0._wp .And. abs(B0) < 0._wp)then

                X(1:3) = -B/(3._wp*A)

       else if(D0 < 0._wp)then

        Y1 = (2._wp*A0*B-3._wp*A*B0)/(2._wp*(A0**1.5_wp))

        Y2 = acos(Y1)/3._wp


        X(1) = (-B-2._wp*sqrt(A0)*cos(Y2))/(3._wp*A)

        X(2) = (-B+sqrt(A0)*(cos(Y2)+sqrt(3._wp)*sin(Y2)))/(3._wp*A)

        X(3) = (-B+sqrt(A0)*(cos(Y2)-sqrt(3._wp)*sin(Y2)))/(3._wp*A)

       else


        X(1) = B0/A0 - B/A

        X(2:3) = -0.5_wp*B0/A0



       endif

    lambda(1)=maxval(X)
    lambda(3)=minval(X)
    lambda(2)=1._wp-lambda(1)-lambda(3)

    if(lambda(3)<0._wp)then

        lambda(3) = 0._wp

    endif


  end function eval_Sphericitylambda

  !----- parton lambda: the three eigenvalues of sphericity tensor( the theta matrix) and some event shape observables

  function eval_partonlambda(npar, imem) result(res)

    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [lambda1 ,lambda2 ,lambda3 ,Sphericity , aplanarity , planarity, C parameter]
    real(kind=wp), dimension(7), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j
    real(kind=wp) :: M(3, 3), lambda(3)
    real(kind=wp) :: a1, s1, c1, p

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    !> solve the equation to find the three eigenvalues of the theta matrix
    do i=1,3
      do j=1,3
       M(i,j)=theta(i,j,npar)
      end do
    end do

    lambda = eval_Sphericitylambda(M)

    !> Sphericity for parton
    s1 = 1.5_wp*(lambda(2)+lambda(3))

    !> aplanarity for parton
    a1 = 1.5_wp*lambda(3)

    !> C parameter (a different method than before: use three eigenvalues of the theta matrix )
    c1 = 3._wp*(lambda(1)*lambda(2) + lambda(1)*lambda(3)  + lambda(2)*lambda(3))

    !> planarity for parton
    p = (lambda(2)-lambda(3))/(lambda(2)+lambda(3))


    !> store in cache & return
    cache_ = [lambda(1), &                           ! lambda1
              lambda(2), &                           ! lambda2
              lambda(3), &                           ! lambda3
              s1, &                                  ! Sphericity for parton
              a1, &                                  ! aplanarity for parton
              p, &                                   ! planarity for parton
              c1]                                    ! C parameter


    qcached_ = .true.
    res = cache_(imem)



  end function eval_partonlambda

  !----- use power iteration method to get Sphericity axis:the longest axis corresponding to lambda1(the max eigenvalue
        ! of sphericity tensor)
  ! and regard the longest axis as the event axis

  function eval_Sphericityaxis(M) result(Laxis)

    real(kind=wp), intent(in) :: M(3, 3)
    integer :: z, i, j
    integer ::max_iterations =100
    real(kind=wp) :: uu(3), vv(3), llambda, Laxis(3)

    !> power iteration method
    uu = [1._wp,1._wp,1._wp]

    z =0
    llambda=1._wp
    do while(z<max_iterations)
     vv = matmul(M, uu)
     llambda = maxval(vv)
     uu = vv/llambda
     z=z+1
    end do

    !> Sphericity axis
    Laxis(1)= uu(1)/sqrt(uu(1)**2+uu(2)**2+uu(3)**2)
    Laxis(2)= uu(2)/sqrt(uu(1)**2+uu(2)**2+uu(3)**2)
    Laxis(3)= uu(3)/sqrt(uu(1)**2+uu(2)**2+uu(3)**2)

    !> choose the positive direction: closer to the positive lab z-direction
    if(Laxis(3) < 0._wp)then
     Laxis(1)= -Laxis(1)
     Laxis(2)= -Laxis(2)
     Laxis(3)= -Laxis(3)

    endif

  end function eval_Sphericityaxis

  !----- Sphericity axis for parton

  function eval_partonaxis(npar, imem) result(res)

    real(kind=wp) :: res
    integer, intent(in) :: npar,imem
    !> cache for memoization:
    !>   [Laxis(1), Laxis(2), Laxis(3)]
    real(kind=wp), dimension(3), save :: &
      cache_ = [0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j
    real(kind=wp) :: Laxis(3), M(3,3)

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

     do i=1,3
       do j=1,3
        M(i,j)=theta(i,j,npar)
       end do
     end do

    Laxis = eval_Sphericityaxis(M)

    !> store in cache & return
    cache_ = [Laxis(1), &
              Laxis(2), &
              Laxis(3)]


    qcached_ = .true.
    res = cache_(imem)

  end function eval_partonaxis

  !----- Sphericity axis for jets
  function eval_jetaxis(npar, imem) result(res)

    real(kind=wp) :: res
    integer, intent(in) :: npar,imem
    !> cache for memoization:
    !>  [Laxis(1), Laxis(2), Laxis(3)]
    real(kind=wp), dimension(3), save :: &
      cache_ = [0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j
    real(kind=wp) :: Laxis(3), M(3,3)

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

     do i=1,3
       do j=1,3
        M(i,j)=thetajet(i,j,npar)
       end do
     end do

    Laxis = eval_Sphericityaxis(M)

    !> store in cache & return
    cache_ = [Laxis(1), &
              Laxis(2), &
              Laxis(3)]


    qcached_ = .true.
    res = cache_(imem)

  end function eval_jetaxis

  !----- parton ridge: long-pseudorapidity-range correlation between parton n1 and parton n2

  function eval_partonridge(npar, n1, n2) result(ridge)

    integer, intent(in) :: npar, n1, n2
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j, k,  z
    real(kind=wp) :: M(3, 3), Laxis(3), pt1(3), pt2(3), ridge(2)
    real(kind=wp) :: paxis1, p1, Thetaa1, ptabs1, eta1, deta, dphi, h, ptdot
    real(kind=wp) :: paxis2, p2, Thetaa2, ptabs2, eta2

    kd => kin(npar)

    !> call partonaxis
    Laxis(1) = eval_partonaxis(npar, 1)
    Laxis(2) = eval_partonaxis(npar, 2)
    Laxis(3) = eval_partonaxis(npar, 3)



    !> correlations between two parton : n1 & n2
    !> parton n1

    !> dot product of parton n1 momentum and Sphericity axis
    paxis1 = kin(npar)%p(1,n1+2)*Laxis(1)+kin(npar)%p(2,n1+2)*Laxis(2)+kin(npar)%p(3,n1+2)*Laxis(3)

    p1 = sqrt(kin(npar)%p(1,n1+2)**2+kin(npar)%p(2,n1+2)**2+kin(npar)%p(3,n1+2)**2)

    !> the angle between parton momentum and Sphericity axis
    Thetaa1=acos(paxis1/p1)

    !> parton pt in the new coordinate system(regard Sphericity axis as Z-axis)
    ptabs1 =  p1*sin(Thetaa1)

    !> pseudorapidity
    eta1 = -log(tan(Thetaa1/2._wp))

    !> the three components of parton pt
    do k = 1,3

      pt1(k) = kin(npar)%p(k,n1+2)-Laxis(k)*paxis1

    enddo

    !> parton n2

    paxis2 = kin(npar)%p(1,n2+2)*Laxis(1)+kin(npar)%p(2,n2+2)*Laxis(2)+kin(npar)%p(3,n2+2)*Laxis(3)
    p2 = sqrt(kin(npar)%p(1,n2+2)**2+kin(npar)%p(2,n2+2)**2+kin(npar)%p(3,n2+2)**2)

    Thetaa2=acos(paxis2/p2)
    ptabs2 =  p2*sin(Thetaa2)
    eta2 = -log(tan(Thetaa2/2._wp))

    do k = 1,3

      pt2(k) = kin(npar)%p(k,n2+2)-Laxis(k)*paxis2

    enddo

    !> delta pseudorapidity between parton n1 & n2
    deta = eta1 - eta2

    !> pt dot of parton n1 and n2
    ptdot = pt1(1)*pt2(1) + pt1(2)*pt2(2) + pt1(3)*pt2(3)

    !> to get azimuthal angle between parton n1 & n2
    h = ptdot/(ptabs1 * ptabs2)

    !> dphi : azimuthal angle between parton
    if( h < -1._wp )then
       h = -1._wp
       dphi = acos(h)

    else if(h > 1._wp)then
       h = 1._wp
       dphi = acos(h)

    else

       dphi = acos(h)
    endif

    ridge(1)=deta
    ridge(2)=dphi


  end function eval_partonridge

  !----- delta_eta and delta_phi of parton n1 & n2
    !> delta pseudorapidity between parton1 & parton2
  real(kind=wp) function eval_epem_dEta12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,2)

    eval_epem_dEta12=ridge(1)
  end function eval_epem_dEta12

    !> azimuthal angle between parton1 & parton2
  real(kind=wp) function eval_epem_dPhi12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,2)

    eval_epem_dPhi12=ridge(2)
  end function eval_epem_dPhi12

    !> delta pseudorapidity between parton1 & parton3
  real(kind=wp) function eval_epem_dEta13(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,3)

    eval_epem_dEta13=ridge(1)
  end function eval_epem_dEta13

    !> azimuthal angle between parton1 & parton3
  real(kind=wp) function eval_epem_dPhi13(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,3)

    eval_epem_dPhi13=ridge(2)
  end function eval_epem_dPhi13

    !> delta pseudorapidity between parton2 & parton3
  real(kind=wp) function eval_epem_dEta23(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,3)

    eval_epem_dEta23=ridge(1)
  end function eval_epem_dEta23

    !> azimuthal angle between parton2 & parton3
  real(kind=wp) function eval_epem_dPhi23(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,3)

    eval_epem_dPhi23=ridge(2)
  end function eval_epem_dPhi23

    !> delta pseudorapidity between parton1 & parton4
  real(kind=wp) function eval_epem_dEta14(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,4)

    eval_epem_dEta14=ridge(1)
  end function eval_epem_dEta14

    !> azimuthal angle between parton1 & parton4
  real(kind=wp) function eval_epem_dPhi14(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,4)

    eval_epem_dPhi14=ridge(2)
  end function eval_epem_dPhi14

    !> delta pseudorapidity between parton2 & parton4
  real(kind=wp) function eval_epem_dEta24(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,4)

    eval_epem_dEta24=ridge(1)
  end function eval_epem_dEta24

    !> azimuthal angle between parton2 & parton4
  real(kind=wp) function eval_epem_dPhi24(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,4)

    eval_epem_dPhi24=ridge(2)
  end function eval_epem_dPhi24

    !> delta pseudorapidity between parton3 & parton4
  real(kind=wp) function eval_epem_dEta34(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,3,4)

    eval_epem_dEta34=ridge(1)
  end function eval_epem_dEta34

    !> azimuthal angle between parton3 & parton4
  real(kind=wp) function eval_epem_dPhi34(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,3,4)

    eval_epem_dPhi34=ridge(2)
  end function eval_epem_dPhi34

    !> delta pseudorapidity between parton1 & parton5
  real(kind=wp) function eval_epem_dEta15(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,5)

    eval_epem_dEta15=ridge(1)
  end function eval_epem_dEta15

    !> azimuthal angle between parton1 & parton5
  real(kind=wp) function eval_epem_dPhi15(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,1,5)

    eval_epem_dPhi15=ridge(2)
  end function eval_epem_dPhi15

    !> delta pseudorapidity between parton2 & parton5
  real(kind=wp) function eval_epem_dEta25(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,5)

    eval_epem_dEta25=ridge(1)
  end function eval_epem_dEta25

    !> azimuthal angle between parton2 & parton5
  real(kind=wp) function eval_epem_dPhi25(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,2,5)

    eval_epem_dPhi25=ridge(2)
  end function eval_epem_dPhi25

    !> delta pseudorapidity between parton3 & parton5
  real(kind=wp) function eval_epem_dEta35(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,3,5)

    eval_epem_dEta35=ridge(1)
  end function eval_epem_dEta35

    !> azimuthal angle between parton3 & parton5
  real(kind=wp) function eval_epem_dPhi35(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,3,5)

    eval_epem_dPhi35=ridge(2)
  end function eval_epem_dPhi35

    !> delta pseudorapidity between parton4 & parton5
  real(kind=wp) function eval_epem_dEta45(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,4,5)

    eval_epem_dEta45=ridge(1)
  end function eval_epem_dEta45

    !> azimuthal angle between parton4 & parton5
  real(kind=wp) function eval_epem_dPhi45(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_partonridge(npar,4,5)

    eval_epem_dPhi45=ridge(2)
  end function eval_epem_dPhi45

  !----- lambda jet: the three eigenvalues of sphericity tensor( the thetajet matrix) and some event shape observables

  function eval_jetlambda(npar, imem) result(res)

    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ lambda1 ,lambda2 ,lambda3 ,Sphericity , aplanarity , planarity, C parameter ]
    real(kind=wp), dimension(7), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j
    real(kind=wp) :: M(3, 3), lambda(3)
    real(kind=wp) :: a1, s1, c1, p

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)



    !> solve the equation to find the three eigenvalues of the thetajet matrix
    do i=1,3
      do j=1,3
       M(i,j)=thetajet(i,j,npar)
      end do
    end do

    lambda = eval_Sphericitylambda(M)

    !> Sphericity for jets
    s1 = 1.5_wp*(lambda(2)+lambda(3))

    !> aplanarity for jets
    a1 = 1.5_wp*lambda(3)

    !> C parameter  (a different method than before: use three eigenvalues of the thetajet matrix )
    c1 = 3._wp*(lambda(1)*lambda(2) + lambda(1)*lambda(3)  + lambda(2)*lambda(3))

    !> planarity for jets
    p = (lambda(2)-lambda(3))/(lambda(2)+lambda(3))


    !> store in cache & return
    cache_ = [lambda(1), &                           ! lambda1
              lambda(2), &                           ! lambda2
              lambda(3), &                           ! lambda3
              s1, &                                  ! Sphericity for jets
              a1, &                                  ! aplanarity for jets
              p, &                                   ! planarity for jets
              c1]                                    ! C parameter


    qcached_ = .true.
    res = cache_(imem)



  end function eval_jetlambda

  !----- ridge of jets : long-pseudorapidity-range correlation between jetn1 and jetn2

  function eval_jetridge(npar, n1, n2) result(ridge)

    integer, intent(in) :: npar, n1, n2
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j, k,  z
    real(kind=wp) :: M(3, 3), Laxis(3), pt1(3), pt2(3), ridge(2)
    real(kind=wp) :: paxis1, p1, Thetaa1, ptabs1, eta1, deta, dphi, h, ptdot
    real(kind=wp) :: paxis2, p2, Thetaa2, ptabs2, eta2

    kd => kin(npar)

    !> call jetaxis
    Laxis(1) = eval_jetaxis(npar, 1)
    Laxis(2) = eval_jetaxis(npar, 2)
    Laxis(3) = eval_jetaxis(npar, 3)

    !> correlations between two jets : n1 & n2

    !> jet n1

    !> dot product of jetn1 momentum and Sphericity axis
    paxis1 = kin(npar)%pjets(1,n1)*Laxis(1)+kin(npar)%pjets(2,n1)*Laxis(2)+kin(npar)%pjets(3,n1)*Laxis(3)


    p1 = sqrt(kin(npar)%pjets(1,n1)**2+kin(npar)%pjets(2,n1)**2+kin(npar)%pjets(3,n1)**2)

    !> the angle between jet momentum and Sphericity axis
    Thetaa1=acos(paxis1/p1)

    !> jet pt in the new coordinate system(regard Sphericity axis as Z-axis)
    ptabs1 =  p1*sin(Thetaa1)

    !> pseudorapidity
    eta1 = -log(tan(Thetaa1/2._wp))

    !> the three components of jet pt
    do k = 1,3

      pt1(k) = kin(npar)%pjets(k,n1)-Laxis(k)*paxis1

    enddo

    !> jet n2

    paxis2 = kin(npar)%pjets(1,n2)*Laxis(1)+kin(npar)%pjets(2,n2)*Laxis(2)+kin(npar)%pjets(3,n2)*Laxis(3)
    p2 = sqrt(kin(npar)%pjets(1,n2)**2+kin(npar)%pjets(2,n2)**2+kin(npar)%pjets(3,n2)**2)

    Thetaa2=acos(paxis2/p2)
    ptabs2 =  p2*sin(Thetaa2)
    eta2 = -log(tan(Thetaa2/2._wp))


    do k = 1,3

      pt2(k) = kin(npar)%pjets(k,n2)-Laxis(k)*paxis2

    enddo

    !> delta pseudorapidity between jetn1 & jetn2
    deta = eta1 - eta2

    !> pt dot of jetn1 and jetn2
    ptdot = pt1(1)*pt2(1) + pt1(2)*pt2(2) + pt1(3)*pt2(3)

    !> to get azimuthal angle between jetn1 & jetn2
    h = ptdot/(ptabs1 * ptabs2)

    !> dphi : azimuthal angle between jetn1 & jetn2
             if( h < -1._wp )then
                      h = -1._wp
                      dphi = acos(h)

              else if(h > 1._wp)then
                      h = 1._wp
                      dphi = acos(h)
              else

                      dphi = acos(h)

             endif

    ridge(1)=deta
    ridge(2)=dphi

  end function eval_jetridge

  !----- delta_eta and delta_phi between jetn1 and jetn2
    !> delta pseudorapidity between j1 & j2
  real(kind=wp) function eval_epem_jdEta12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,2)

    eval_epem_jdEta12=ridge(1)
  end function eval_epem_jdEta12

    !> azimuthal angle between j1 & j2
  real(kind=wp) function eval_epem_jdPhi12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,2)

    eval_epem_jdPhi12=ridge(2)
  end function eval_epem_jdPhi12

    !> delta pseudorapidity between j1 & j3
  real(kind=wp) function eval_epem_jdEta13(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,3)

    eval_epem_jdEta13=ridge(1)
  end function eval_epem_jdEta13

    !> azimuthal angle between j1 & j3
  real(kind=wp) function eval_epem_jdPhi13(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,3)

    eval_epem_jdPhi13=ridge(2)
  end function eval_epem_jdPhi13

    !> delta pseudorapidity between j2 & j3
  real(kind=wp) function eval_epem_jdEta23(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,3)

    eval_epem_jdEta23=ridge(1)
  end function eval_epem_jdEta23

    !> azimuthal angle between j2 & j3
  real(kind=wp) function eval_epem_jdPhi23(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,3)

    eval_epem_jdPhi23=ridge(2)
  end function eval_epem_jdPhi23

    !> delta pseudorapidity between j1 & j4
  real(kind=wp) function eval_epem_jdEta14(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,4)

    eval_epem_jdEta14=ridge(1)
  end function eval_epem_jdEta14

    !> azimuthal angle between j1 & j4
  real(kind=wp) function eval_epem_jdPhi14(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,4)

    eval_epem_jdPhi14=ridge(2)
  end function eval_epem_jdPhi14

    !> delta pseudorapidity between j2 & j4
  real(kind=wp) function eval_epem_jdEta24(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,4)

    eval_epem_jdEta24=ridge(1)
  end function eval_epem_jdEta24

    !> azimuthal angle between j2 & j4
  real(kind=wp) function eval_epem_jdPhi24(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,4)

    eval_epem_jdPhi24=ridge(2)
  end function eval_epem_jdPhi24

    !> delta pseudorapidity between j3 & j4
  real(kind=wp) function eval_epem_jdEta34(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,3,4)

    eval_epem_jdEta34=ridge(1)
  end function eval_epem_jdEta34

    !> azimuthal angle between j3 & j4
  real(kind=wp) function eval_epem_jdPhi34(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,3,4)

    eval_epem_jdPhi34=ridge(2)
  end function eval_epem_jdPhi34

    !> delta pseudorapidity between j1 & j5
  real(kind=wp) function eval_epem_jdEta15(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,5)

    eval_epem_jdEta15=ridge(1)
  end function eval_epem_jdEta15

    !> azimuthal angle between j1 & j5
  real(kind=wp) function eval_epem_jdPhi15(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,1,5)

    eval_epem_jdPhi15=ridge(2)
  end function eval_epem_jdPhi15

    !> delta pseudorapidity between j2 & j5
  real(kind=wp) function eval_epem_jdEta25(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,5)

    eval_epem_jdEta25=ridge(1)
  end function eval_epem_jdEta25

    !> azimuthal angle between j2 & j5
  real(kind=wp) function eval_epem_jdPhi25(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,2,5)

    eval_epem_jdPhi25=ridge(2)
  end function eval_epem_jdPhi25

    !> delta pseudorapidity between j3 & j5
  real(kind=wp) function eval_epem_jdEta35(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,3,5)

    eval_epem_jdEta35=ridge(1)
  end function eval_epem_jdEta35

    !> azimuthal angle between j3 & j5
  real(kind=wp) function eval_epem_jdPhi35(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,3,5)

    eval_epem_jdPhi35=ridge(2)
  end function eval_epem_jdPhi35

    !> delta pseudorapidity between j4 & j5
  real(kind=wp) function eval_epem_jdEta45(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,4,5)

    eval_epem_jdEta45=ridge(1)
  end function eval_epem_jdEta45

    !> azimuthal angle between j4 & j5
  real(kind=wp) function eval_epem_jdPhi45(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ridge(2)
    ridge=eval_jetridge(npar,4,5)

    eval_epem_jdPhi45=ridge(2)
  end function eval_epem_jdPhi45

  !----- ordering the four lepton system in H --> 4l according to pt of leptons

  function eval_4l_nth(npar, imem) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ ptl1(1), yl1(2), dRj1l1(3), ptl2(4),  yl2(5),  dRj1l2(6),
    !>     ptl3(7), yl3(8), dRj1l3(9), ptl4(10), yl4(11), dRj1l4(12) ]
    real(kind=wp), dimension(12), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, &
                0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    real(kind=wp) :: pt_l1, y_l1, dRj1l1, pt_l2, y_l2, dRj1l2
    real(kind=wp) :: pt_l3, y_l3, dRj1l3, pt_l4, y_l4, dRj1l4
    real(kind=wp) :: pt_tmp, y_tmp, dR_tmp
    integer :: i, j

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    !> lepton observables
    pt_l1 = eval_pt_lep(npar, 1)
    pt_l2 = eval_pt_lep(npar, 2)
    pt_l3 = eval_pt_lep(npar, 3)
    pt_l4 = eval_pt_lep(npar, 4)
    y_l1 = eval_y_lep(npar, 1)
    y_l2 = eval_y_lep(npar, 2)
    y_l3 = eval_y_lep(npar, 3)
    y_l4 = eval_y_lep(npar, 4)
    dRj1l1 = eval_dR_j1_lep(npar, 1)
    dRj1l2 = eval_dR_j1_lep(npar, 2)
    dRj1l3 = eval_dR_j1_lep(npar, 3)
    dRj1l4 = eval_dR_j1_lep(npar, 4)

    cache_ = [pt_l1, y_l1, dRj1l1, pt_l2, y_l2, dRj1l2, &
              pt_l3, y_l3, dRj1l3, pt_l4, y_l4, dRj1l4]
    do i = 1, 4
    do j = i + 1, 4
      if (cache_(3*i - 2) >= cache_(3*j - 2)) cycle
      !> swap i & j
      pt_tmp = cache_(3*i - 2)
      y_tmp = cache_(3*i - 1)
      dR_tmp = cache_(3*i)
      cache_(3*i - 2) = cache_(3*j - 2)
      cache_(3*i - 1) = cache_(3*j - 1)
      cache_(3*i) = cache_(3*j)
      cache_(3*j - 2) = pt_tmp
      cache_(3*j - 1) = y_tmp
      cache_(3*j) = dR_tmp
    end do
    end do

    ! print*, "eval_4l_nth: pt: ", cache(1), cache(4), cache(7), cache(10)
    ! print*, "eval_4l_nth: y:  ", cache(2), cache(5), cache(8), cache(11)
    ! print*, "eval_4l_nth: dR: ", cache(3), cache(6), cache(9), cache(12)
    ! read(*,*)
    qcached_ = .true.
    res = cache_(imem)

  end function eval_4l_nth

  !----- ordering the two lepton system in H --> Z/gamma* gamma --> 2l gamma according to pt of leptons

  function eval_2l1p_nth(npar, imem) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ ptl1(1), yl1(2), ptl2(3), yl2(4) ]
    real(kind=wp), dimension(4), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    real(kind=wp) :: pt_l1, y_l1, pt_l2, y_l2

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    !> lepton observables [(npar-1),(npar-2)]
    pt_l1 = eval_pt_lep(npar, 1)
    pt_l2 = eval_pt_lep(npar, 4)
    y_l1 = eval_y_lep(npar, 1)
    y_l2 = eval_y_lep(npar, 4)

    !> no *exact* degenerate configurations
    if (pt_l1 > pt_l2) then
      cache_ = [pt_l1, y_l1, pt_l2, y_l2]
    else
      cache_ = [pt_l2, y_l2, pt_l1, y_l1]
    end if

    qcached_ = .true.
    res = cache_(imem)

  end function eval_2l1p_nth

  !----- leading & sub-leading photon / leptons in Z (last two entries)

  function eval_2l_nth(npar, imem) result(res)
    use Constants_mod, only: pi, twopi
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ ptl1(1), yl1(2), ptl2(3), yl2(4),
    !>     del_y_l1l2(5), del_phi_l1l2(6) ]
    real(kind=wp), dimension(6), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    real(kind=wp) :: pt_l1, y_l1, pt_l2, y_l2, del_y_12, del_phi_12
    !> threshold = 1d-X -> require at least ~ X digit agreement before rand
    real(kind=wp), parameter :: threshold = 1d-12
    real(kind=wp) :: rnd

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    !> lepton observables
    pt_l1 = eval_pt_lep(npar, 1)
    pt_l2 = eval_pt_lep(npar, 2)
    y_l1 = eval_y_lep(npar, 1)
    y_l2 = eval_y_lep(npar, 2)
    del_y_12 = y_l1 - y_l2  ! no abs(...) !
    del_phi_12 = eval_phi_lep(npar, 1) - eval_phi_lep(npar, 2)
    !> get the orientation & range right:  [-pi,+pi] relative to l1
    if (del_phi_12 < -pi) del_phi_12 = twopi + del_phi_12
    if (del_phi_12 > +pi) del_phi_12 = del_phi_12 - twopi

    if (abs((pt_l1 - pt_l2)/(pt_l1 + pt_l2)) > threshold) then
      !> have hierarcy above threshold
      if (pt_l1 > pt_l2) then
        cache_ = [pt_l1, y_l1, pt_l2, y_l2, +del_y_12, +del_phi_12]
      else
        cache_ = [pt_l2, y_l2, pt_l1, y_l1, -del_y_12, -del_phi_12]
      end if
    else
      !> randomize
      call random_number(rnd)
      if (rnd < 0.5_wp) then
        cache_ = [pt_l1, y_l1, pt_l2, y_l2, +del_y_12, +del_phi_12]
      else
        cache_ = [pt_l2, y_l2, pt_l1, y_l1, -del_y_12, -del_phi_12]
      end if
    end if

    qcached_ = .true.
    res = cache_(imem)

  end function eval_2l_nth

  real(kind=wp) function eval_deltaphi_l1l2(npar)
    integer, intent(in) :: npar
    eval_deltaphi_l1l2 = v2_delta_phi(kin(npar)%p(:, npar), &
                                      kin(npar)%p(:, npar - 1))
  end function eval_deltaphi_l1l2

  real(kind=wp) function eval_abscosthetastar_eta(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y_g1, y_g2
    y_g1 = eval_y_gam(npar, 1)  ! leading
    y_g2 = eval_y_gam(npar, 2)  ! sub-leading
    eval_abscosthetastar_eta = tanh(abs((y_g1 - y_g2))/2._wp)
  end function eval_abscosthetastar_eta

  real(kind=wp) function eval_ptt_l1l2(npar)
    integer, intent(in) :: npar
    !> in the ATLAS paper, it sounds like taking the 3-momenta but that's wrong.
    !> only the transverse component (x,y)
    real(kind=wp) :: pP(2), pM(2), vpPM, vpMM
    integer :: i
    pP = kin(npar)%p(1:2, npar - 1) + kin(npar)%p(1:2, npar)
    pM = kin(npar)%p(1:2, npar - 1) - kin(npar)%p(1:2, npar)
    vpPM = 0._wp
    vpMM = 0._wp
    do i = 1, 2
      vpPM = vpPM + pP(i)*pM(i)
      vpMM = vpMM + pM(i)**2
    end do
    pP = pP - (vpPM/vpMM)*pM  ! orthogonal component w.r.t. pM axis
    eval_ptt_l1l2 = 0._wp
    do i = 1, 2
      eval_ptt_l1l2 = eval_ptt_l1l2 + pP(i)**2
    end do
    eval_ptt_l1l2 = sqrt(eval_ptt_l1l2)
  end function eval_ptt_l1l2

  ! transverse component of p_Tgg wrt thrust axis of diphoton system - this is what ATLAS calls a_T
  real(kind=wp) function eval_absptt_l1l2(npar)
    integer, intent(in) :: npar
    integer :: i
    type(Kin_t), pointer :: kd
    real(kind=wp) :: abs_t
    kd => kin(npar)
    ! i = id of first photon (ordering gives relative minus sign; taking absolute value at end)
    i = npar - 1
    abs_t = sqrt((kd%p(1, i) - kd%p(1, i + 1))**2 + (kd%p(2, i) - kd%p(2, i + 1))**2)
    eval_absptt_l1l2 = 2._wp*(kd%p(1, i)*kd%p(2, i + 1) - kd%p(2, i)*kd%p(1, i + 1))
    eval_absptt_l1l2 = abs(eval_absptt_l1l2)
    eval_absptt_l1l2 = eval_absptt_l1l2/abs_t

  end function eval_absptt_l1l2

  ! absolute value of a_T variable defined in arXiv:0807.4956 - i.e. |ptt| (above) for dphigg > pi/2, and |pt_gg| for dphigg < pi/2
  real(kind=wp) function eval_absat_l1l2(npar)
    use Constants_mod
    integer, intent(in) :: npar
    real(kind=wp) :: dphi
    dphi = eval_deltaphi_l1l2(npar)
    if (dphi .gt. pi/2._wp) then
      eval_absat_l1l2 = eval_absptt_l1l2(npar)
    else
      eval_absat_l1l2 = eval_pt_l1l2(npar)
    end if
  end function eval_absat_l1l2

  real(kind=wp) function eval_deltaR_l1l2(npar)
    integer, intent(in) :: npar
    eval_deltaR_l1l2 = v2_delta_R(kin(npar)%p(:, npar), kin(npar)%p(:, npar - 1))
  end function eval_deltaR_l1l2

  real(kind=wp) function eval_pt_l1l2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_pt_l1l2 = v1_pt(pV)
  end function eval_pt_l1l2

  real(kind=wp) function eval_ptavg_l1l2(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt1, pt2
    pt1 = eval_pt_lep(npar, 1)
    pt2 = eval_pt_lep(npar, 2)
    eval_ptavg_l1l2 = (pt1 + pt2)/2._wp
  end function eval_ptavg_l1l2

  real(kind=wp) function eval_ptavg_geom_l1l2(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt1, pt2
    pt1 = eval_pt_lep(npar, 1)
    pt2 = eval_pt_lep(npar, 2)
    eval_ptavg_geom_l1l2 = sqrt(pt1*pt2)
  end function eval_ptavg_geom_l1l2

  pure integer function idx_lep_jet(lep, jet)
    integer, intent(in) :: lep, jet
    idx_lep_jet = 4*(jet - 1) + lep
  end function idx_lep_jet

  function eval_deltaphi_lepton_jet(npar, idx) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    res = v2_delta_phi(kin(npar)%p(:, plep), kin(npar)%pjets(:, jet))
  end function eval_deltaphi_lepton_jet

  real(kind=wp) function eval_deltaphi_photon_j1(npar)

    integer, intent(in) :: npar
    eval_deltaphi_photon_j1 = v2_delta_phi(kin(npar)%p(:, npar), kin(npar)%pjets(:, 1))
  end function eval_deltaphi_photon_j1

  ! real(kind=wp) function eval_deltaphi_Vj1(npar)
  !   integer, intent(in) :: npar
  !   real(kind=wp), dimension(4) :: pV
  !   integer :: i
  !   do i=1,4
  !     pV(i) = kin(npar)%p(i,npar-1) + kin(npar)%p(i,npar)
  !   end do
  !   eval_deltaphi_Vj1 = v2_delta_phi(pV , kin(npar)%pjets(:,1) )
  ! end function eval_deltaphi_Vj1

  real(kind=wp) function eval_deltaphi_V_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_deltaphi_V_jet = v2_delta_phi(pV, kin(npar)%pjets(:, ijet))
  end function eval_deltaphi_V_jet

  real(kind=wp) function eval_deltaphi_Vj12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), dimension(4) :: pj12
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
      pj12(i) = kin(npar)%pjets(i, 1) + kin(npar)%pjets(i, 2)
    end do
    eval_deltaphi_Vj12 = v2_delta_phi(pV, pj12)
  end function eval_deltaphi_Vj12

  real(kind=wp) function eval_h_zep(npar)
    integer, intent(in) :: npar
    eval_h_zep = abs(eval_eta_V(npar) - 0.5_wp*(eval_eta_jet(npar, 1) + eval_eta_jet(npar, 2)))
    !abs(cat.higgs.eta() - 0.5 * (cat.jets30[0].eta() + cat.jets30[1].eta()))
  end function eval_h_zep

  real(kind=wp) function eval_h_zstar(npar)
    integer, intent(in) :: npar
    eval_h_zstar = eval_h_zep(npar)/eval_deltaeta_j12(npar)
  end function eval_h_zstar

  real(kind=wp) function eval_deltay_Vj1(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yV, yj1
    yV = eval_y_V(npar)
    yj1 = eval_y_jet(npar, 1)
    eval_deltay_Vj1 = abs(yV - yj1)
  end function eval_deltay_Vj1

  real(kind=wp) function eval_dR_Vj1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_dR_Vj1 = v2_delta_R(pV, kin(npar)%pjets(:, 1))
  end function eval_dR_Vj1

  real(kind=wp) function eval_dR_Vj2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_dR_Vj2 = v2_delta_R(pV, kin(npar)%pjets(:, 2))
  end function eval_dR_Vj2

  real(kind=wp) function eval_ydiff_Vj1(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: deltay
    deltay = eval_deltay_Vj1(npar)
    eval_ydiff_Vj1 = deltay/2._wp
  end function eval_ydiff_Vj1

  real(kind=wp) function eval_ysum_Vj1(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yV, yj1
    yV = eval_y_V(npar)
    yj1 = eval_y_jet(npar, 1)
    eval_ysum_Vj1 = abs(yV + yj1)/2._wp
  end function eval_ysum_Vj1

  function eval_Collins_Soper(npar, imem) result(res)
    use Constants_mod, only: twopi
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ cos_theta(1), theta(2), phi(3),
    !>     A0(4), A1(5), A2(6), A3(7), A4(8), A5(9), A6(10), A7(11),
    !>     A0mA2(12), A0mA2alt(13) ]
    real(kind=wp), dimension(13), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, &
                0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    real(kind=wp) :: cos_th, th, phi
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: QV, QV2, PTV, PTV2, XTV
    real(kind=wp) :: pl_x, pl_y, pl_z
    !> for the angular coefficients
    real(kind=wp) :: cos2_th, sin2_th, sin_th, sin_2th
    real(kind=wp) :: cos_phi, sin_phi, cos_2phi, sin_2phi

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    !> momentum of intermediate gauge boson
    pV(1:4) = kd%p(1:4, npar - 1) + kd%p(1:4, npar)

    !> Eq.(23) of hep-ph/9406381
    !> with corrected signs and generalised to pV(2) /= 0
    !> 'n-1':  l1 = lm (for Z)
    QV2 = dot(pV, pV)
    PTV2 = pV(1)**2 + pV(2)**2
    QV = sqrt(QV2)
    PTV = sqrt(PTV2)
    XTV = sqrt(QV2 + PTV2)
    pl_x = +PTV/QV/XTV*( &
           -pV(4)*kd%p(4, npar - 1) &
           + pV(3)*kd%p(3, npar - 1)) &
           + XTV/QV/PTV*( &
           +pV(1)*kd%p(1, npar - 1) &
           + pV(2)*kd%p(2, npar - 1))
    pl_y = sign(1._wp, pV(3))/PTV*( &
           -pV(2)*kd%p(1, npar - 1) &
           + pV(1)*kd%p(2, npar - 1))
    pl_z = sign(1._wp, pV(3))/XTV*( &
           -pV(3)*kd%p(4, npar - 1) &
           + pV(4)*kd%p(3, npar - 1))

    cos_th = pl_z/sqrt(pl_x**2 + pl_y**2 + pl_z**2)
    th = atan2(sqrt(pl_x**2 + pl_y**2), pl_z)
    phi = atan2(pl_y, pl_x)
    if (phi < 0._wp) phi = phi + twopi

    !> angular coefficients; projectors from [1606.00689]
    cos2_th = cos_th**2
    sin2_th = (1._wp - cos_th)*(1._wp + cos_th)
    sin_th = sqrt(sin2_th)
    sin_2th = 2._wp*cos_th*sin_th
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cos_2phi = (cos_phi - sin_phi)*(cos_phi + sin_phi)
    sin_2phi = 2._wp*cos_phi*sin_phi

    cache_ = [cos_th, th, phi, &
              4._wp - 10._wp*cos2_th, &  ! A0
              5._wp*sin_2th*cos_phi, &  ! A1
              10._wp*sin2_th*cos_2phi, &  ! A2
              4._wp*sin_th*cos_phi, &  ! A3
              4._wp*cos_th, &  ! A4
              5._wp*sin2_th*sin_2phi, &  ! A5
              5._wp*sin_2th*sin_phi, &  ! A6
              4._wp*sin_th*sin_phi, &  ! A7
              (4._wp - 10._wp*cos2_th) &  ! (A0-A2)
              - (10._wp*sin2_th*cos_2phi), &
              -6._wp + 20._wp*sin2_th*sin_phi**2]  ! (A0-A2) [alt]

    qcached_ = .true.
    res = cache_(imem)

  end function eval_Collins_Soper

  ! Generic subroutine to set up Helicity frame observables, independent(ish) of lepton position
  function eval_Helicity_Frame(npar, imem) result(res)
    use Constants_mod, only: twopi
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    integer :: lep, lep_pos, imem_
    !> cache for memoization:
    !>   [ cos_theta(1), theta(2), phi(3),
    !>     A0(4), A1(5), A2(6), A3(7), A4(8), A5(9), A6(10), A7(11),
    !>     A0mA2(12), f0(13), f1(14), f2(15), cos_theta_2D(16) ]
    real(kind=wp), dimension(16), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, &
                0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    real(kind=wp), dimension(4) :: pV, recoil_system
    real(kind=wp) :: QV, QV2, PTV, PTV2, XTV
    real(kind=wp) :: pl_x, pl_y, pl_z
    real(kind=wp) :: cos_th, th, phi
    real(kind=wp) :: cos_theta_2D
    integer :: i
    real(kind=wp), dimension(4) :: bp, bm, pl
    real(kind=wp), dimension(3) :: ep, em, ex, ey, ez
    !> for the angular coefficients
    real(kind=wp) :: cos2_th, sin2_th, sin_th, sin_2th
    real(kind=wp) :: cos_phi, sin_phi, cos_2phi, sin_2phi
    real(kind=wp) :: A0, A2, A4

    res = 0._wp

    !> we concatenate the `imem` index to map to the same cache
    lep = int((imem - 1)/size(cache_)) + 1
    select case (lep)
    case (1)
      lep_pos = npar - 1  ! l^-
      imem_ = imem
    case (2)
      lep_pos = npar      ! l^+
      imem_ = imem - size(cache_)
    end select

    if (qcached_ .and. (imem_ > 0)) then
      res = cache_(imem_)
      return
    end if
    if (imem_ == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    !> momentum of intermediate gauge boson
    pV(1:4) = kd%p(1:4, npar - 1) + kd%p(1:4, npar)

    !----- BEGIN: naive implementation
    !> beam directions
    ! plus (+)
    bp(1) = 0._wp
    bp(2) = 0._wp
    bp(3) = 1._wp
    bp(4) = 1._wp
    ! minus (-)
    bm(1) = 0._wp
    bm(2) = 0._wp
    bm(3) = -1._wp
    bm(4) = 1._wp
    !> lepton momentum
    pl(1:4) = kd%p(1:4, lep_pos)
    !> boost into the gauge-boson rest frame
    call boost_rest(bp, pV)
    call boost_rest(bm, pV)
    call boost_rest(pl, pV)
    !> only interested in directions
    ep(1:3) = bp(1:3)
    em(1:3) = bm(1:3)
    ! normalise
    ep = ep/sqrt(dot_eucl(ep, ep))
    em = em/sqrt(dot_eucl(em, em))
    !> set up the Collins-Soper coordinate system

    ! z axis = direction of W boson in the lab frame
    ez = pV(1:3)/sqrt(dot_eucl(pV(1:3), pV(1:3)))
    ! ex = orthogonal to event plane in hemisphere opposite to recoil system
    ex(1) = ep(2)*em(3) - ep(3)*em(2)
    ex(2) = ep(3)*em(1) - ep(1)*em(3)
    ex(3) = ep(1)*em(2) - ep(2)*em(1)
    ex = ex/sqrt(dot_eucl(ex, ex))

    recoil_system(1:4) = 0._wp
    do i = 1, kd%njets
      recoil_system(1:4) = recoil_system(1:4) + kd%pjets(1:4, i)
    end do

    call boost_rest(recoil_system, pV)

    ! Same hemisphere?
    if (dot_eucl(recoil_system(1:3), ex) > 0._wp) then
      ex = -1._wp*ex
    end if
    ! y-axis to get a right-handed coordinate system: ey = ez x ex
    ey(1) = ez(2)*ex(3) - ez(3)*ex(2)
    ey(2) = ez(3)*ex(1) - ez(1)*ex(3)
    ey(3) = ez(1)*ex(2) - ez(2)*ex(1)

    !> project the lepton momentum onto the axes
    pl_x = dot_eucl(pl(1:3), ex)
    pl_y = dot_eucl(pl(1:3), ey)
    pl_z = dot_eucl(pl(1:3), ez)

    ! !>----- END: naive implementation

    ! print *, "Orthogonality checks"
    ! print *, dot_eucl(ex,ey)
    ! print *, dot_eucl(ex,ez)
    ! print *, dot_eucl(ez,ey) ! Check
    ! print *, "Unit vector checks"
    ! print *, dot_eucl(ex,ex)
    ! print *, dot_eucl(ey,ey)
    ! print *, dot_eucl(ez,ez) ! Check
    ! print *, "ez check"
    ! print *, pV(1:3)
    ! print *, ez
    ! print *, pv(1:3)/ez
    ! read *,

    cos_th = pl_z/sqrt(pl_x**2 + pl_y**2 + pl_z**2)
    th = atan2(sqrt(pl_x**2 + pl_y**2), pl_z)
    phi = atan2(pl_y, pl_x)
    if (phi < 0._wp) phi = phi + twopi

    !> angular coefficients; projectors from [1606.00689]
    cos2_th = cos_th**2
    sin2_th = (1._wp - cos_th)*(1._wp + cos_th)
    sin_th = sqrt(sin2_th)
    sin_2th = 2._wp*cos_th*sin_th
    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cos_2phi = (cos_phi - sin_phi)*(cos_phi + sin_phi)
    sin_2phi = 2._wp*cos_phi*sin_phi

    A0 = 4._wp - 10._wp*cos2_th
    A2 = 10._wp*sin2_th*cos_2phi
    A4 = 4._wp*cos_th

    cache_ = [cos_th, th, phi, &
              !> projectors from [1606.00689]
              A0, &                      ! A0
              5._wp*sin_2th*cos_phi, &   ! A1
              A2, &                      ! A2
              4._wp*sin_th*cos_phi, &    ! A3
              A4, &                      ! A4
              5._wp*sin2_th*sin_2phi, &  ! A5
              5._wp*sin_2th*sin_phi, &   ! A6
              4._wp*sin_th*sin_phi, &    ! A7
              A0 - A2, &                 ! (A0-A2)
              !> Helicity projections [Eqn 4. 1203.2165]
              0.5_wp*A0, &                  ! f0
              0.25_wp*(2._wp - A0 - A4), &  ! f1 = fl (W+) = fr (W-)
              0.25_wp*(2._wp - A0 + A4), &  ! f2 = fr (W+) = fl (W-)
              !> COS THETA 2D = ptl.ptW/(|ptl||ptw|) [Eqn 6. 1203.2165]
              (pl_x*pV(1) + pl_y*pV(2)) &
              /(sqrt(pl_x**2 + pl_y**2)*sqrt(pV(1)**2 + pV(2)**2))]

    qcached_ = .true.
    res = cache_(imem_)

  end function eval_Helicity_Frame

  !> routine to find the thrust axis for ep em collisions

  function eval_Taxis(npar, imem) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ T(1), 1mT(2), cosT(3), WJB(4), TJB(5), HJM(6), SHM(7), angP(8) ]
    real(kind=wp), dimension(8), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j, k, n, ithrust, jthrust
    real(kind=wp), dimension(5) :: Taxis
    real(kind=wp) :: pp(6, 15), Energy, Eijsq, sij
    real(kind=wp) :: bplus, bminus, bot, bsum, bmax, bmin
    real(kind=wp) :: em2h, em2l, t, omt, tmp, tmp1, tmp2, Tpar, OmTpar, absp, cosT

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    !> vector of possible thrust directions + thrust[5] + (one-thrust)[6]
    pp(:, :) = 0._wp
    pp(6, :) = sqrt(kd%s(1, 2))
    n = 0
    !> sum of final state energy (== scalar 3-moms, since p^2=0)
    Energy = 0._wp

    do j = 3, npar
      n = n + 1
      Energy = Energy + kd%p(4, j)
      pp(1:4, n) = kd%p(1:4, j)
      pp(5, n) = pp(4, n)
      !> directly compute (1-T) for stability
      pp(6, n) = -kd%p(4, j)
      do i = 3, npar
        if (i /= j) pp(6, n) = pp(6, n) + kd%p(4, i)
      end do
      !if (pp(6,n) < 0._wp) pp(6,n) = 0._wp  ! safety
    end do

    if (npar .eq. 6) then
      !print*, "2 -> 4"
      do k = 4, npar
        n = n + 1
        pp(1:4, n) = kd%p(1:4, 3) + kd%p(1:4, k)
        pp(5, n) = sqrt(pp(1, n)**2 + pp(2, n)**2 + pp(3, n)**2)
        !> directly compute (1-T) for stability
        sij = kd%s(3, k)
        Eijsq = pp(4, n)**2
        if (sij/Eijsq < 1d-3) then
          !print*, "< expansion >"
          !> expand:   Energy - 2 sqrt( Eijsq - sij )
          pp(6, n) = -pp(4, n)
          do i = 4, npar
            if (i /= k) pp(6, n) = pp(6, n) + kd%p(4, i)
          end do
          sij = sij/Eijsq  ! the expansion parameter
          !> seven orders with sij < 1d-3 should be more than enough for double
          pp(6, n) = pp(6, n) + pp(4, n)*( &
                     sij &
                     + sij**2/4._wp &
                     + sij**3/8._wp &
                     + (5*sij**4)/64._wp &
                     + (7*sij**5)/128._wp &
                     + (21*sij**6)/512._wp &
                     + (33*sij**7)/1024._wp &
                     )
        else
          pp(6, n) = Energy - 2._wp*sqrt(Eijsq - sij)
        end if
        !if (pp(6,n) < 0._wp) pp(6,n) = 0._wp  ! safety
      end do
    elseif (npar .eq. 7) then
      !print*, "2 -> 5"
      do j = 3, npar - 1
        do k = j + 1, npar
          n = n + 1
          pp(1:4, n) = kd%p(1:4, j) + kd%p(1:4, k)
          pp(5, n) = sqrt(pp(1, n)**2 + pp(2, n)**2 + pp(3, n)**2)
          !> directly compute (1-T) for stability
          sij = kd%s(j, k)
          Eijsq = pp(4, n)**2
          if (sij/Eijsq < 1d-3) then
            !print*, "< expansion >"
            !> expand:   Energy - 2 sqrt( Eijsq - sij )
            pp(6, n) = -pp(4, n)
            do i = 3, npar
              if ((i /= k) .and. (i /= j)) pp(6, n) = pp(6, n) + kd%p(4, i)
            end do
            sij = sij/Eijsq  ! the expansion parameter
            !> seven orders with sij < 1d-3 should be more than enough for double
            pp(6, n) = pp(6, n) + pp(4, n)*( &
                       sij &
                       + sij**2/4._wp &
                       + sij**3/8._wp &
                       + (5*sij**4)/64._wp &
                       + (7*sij**5)/128._wp &
                       + (21*sij**6)/512._wp &
                       + (33*sij**7)/1024._wp &
                       )
          else
            pp(6, n) = Energy - 2._wp*sqrt(Eijsq - sij)
          end if
          !if (pp(6,n) < 0._wp) pp(6,n) = 0._wp  ! safety
        end do
      end do
    end if

    !> safety measure (assuming < 0 only from numerical rounding)
    !pp(6,:) = abs(pp(6,:))

    !>
    !> Find thrust axis and calculate thrust
    !> ithrust gives index for thrust axis
    ithrust = 0
    jthrust = 0
    Tpar = 0._wp
    OmTpar = Energy
    do i = 1, 15
      t = 2._wp*pp(5, i)
      if (t .gt. Tpar) then
        Tpar = t
        ithrust = i
      end if
      omt = pp(6, i)
      !if ((omt>=0._wp).and.(omt<OmTpar)) then
      if (omt < OmTpar) then
        OmTpar = omt
        jthrust = i
      end if
      !print*, i, (1._wp-t/Energy), omt/Energy
    end do

    !> thrust axis (unit vector)
    Taxis(4) = Tpar/Energy
    Taxis(5) = OmTpar/Energy
    do j = 1, 3
      Taxis(j) = 2._wp*pp(j, jthrust)/Tpar
    end do

    !> angle of thrust axis
    cosT = abs(Taxis(3))

    ! if (ithrust /= jthrust) then
    !   print*, "eval_Taxis: min != max?!", ithrust, jthrust
    !   print*, "1mT:   ", 1._wp-Taxis(4)
    !   print*, "1mT':  ", Taxis(5)
    !   print*, "----"
    !   do i=1,15
    !     t=2._wp*pp(5,i)
    !     omt = pp(6,i)
    !     print*, i, (1._wp-t/Energy), omt/Energy
    !   enddo
    !   !call print_momenta(npar)
    !   !call print_xvegas()
    !   !read(*,*)
    ! end if

    ! print*, "thrust:", Taxis(4)
    ! print*, "1mT:   ", 1._wp-Taxis(4)
    ! print*, "1mT':  ", pp(6,ithrust) / Energy, (ithrust>npar-2), ithrust
    ! print*, ""
    ! !if ((npar.eq.6).and.(ithrust>npar-2).and.(pp(6,ithrust)/Energy<1d-5)) then
    ! if ((npar.eq.7).and.(ithrust>npar-2)) then
    !   call print_momenta(npar)
    !   call print_xvegas()
    !   read(*,*)
    ! end if

    !> evaluate all related results
    !> HJM (heavy jet mass)
    em2h = pp(4, ithrust)**2 - pp(5, ithrust)**2
    em2l = (Energy - pp(4, ithrust))**2 - pp(5, ithrust)**2
    if (em2l .gt. em2h) then
      tmp = em2l
      em2l = em2h
      em2h = tmp
    end if

    !>
    !> Hemisphere broadening
    !>
    bplus = 0._wp
    bminus = 0._wp
    bot = 0._wp
    absp = 0._wp
    !> 1 to npar-2 are the normal momenta
    do i = 1, npar - 2
      absp = sqrt(pp(1, i)**2 + pp(2, i)**2 + pp(3, i)**2)
      bot = bot + absp
      if (i .ne. ithrust) then
        !> |p|cos(theta_(iT))
        tmp1 = pp(1, i)*Taxis(1) + pp(2, i)*Taxis(2) + pp(3, i)*Taxis(3)
        !> |p|sin(theta_(iT))
        tmp2 = sqrt(absp**2 - tmp1**2)
        if (tmp1 .gt. 0._wp) then
          bplus = bplus + tmp2
        elseif (tmp1 .lt. 0._wp) then
          bminus = bminus + tmp2
        end if
      end if
    end do
    bplus = bplus/bot/2._wp
    bminus = bminus/bot/2._wp
    bmax = max(bplus, bminus)
    bmin = min(bplus, bminus)
    bsum = bmax + bmin

    !> store in cache & return
    cache_ = [Taxis(4), &                           ! T
              Taxis(5), &                           ! 1mT
              cosT, &                               ! cosT
              bmax, &                               ! WJB
              bsum, &                               ! TJB
              em2h/energy**2, &                     ! HJM
              (em2h + em2l)/energy**2, &            ! SHM
              3._wp/8._wp*(2._wp - 5._wp*cosT**2)]  ! angP

    qcached_ = .true.
    res = cache_(imem)

  end function eval_Taxis

  !----- EFP_G3b: triangle energy flow polynomial with sqrt-edge weights
  !> EFP_{G3b}(p_1,...,p_n) = 6 * sum_{i<j<k} z_i z_j z_k * sqrt(chi_{ij} chi_{ik} chi_{jk})
  !> with z_i = E_i / Q  and  chi_{ab} = (1 - cos theta_{ab})/2 = (p_a . p_b) / (2 E_a E_b)
  !> Final-state partons in NNLOJET kinematics are indices 3..npar.
  !> The matrix kin(npar)%s(i,j) stores 2 p_i.p_j (Mandelstam-like), so
  !>     chi_{ij} = kd%s(i,j) / (4 * E_i * E_j)
  !> For 2-parton (Born) state -> identically 0 (no triangle).
  real(kind=wp) function eval_epem_EFPG3b(npar)
    integer, intent(in) :: npar
    type(Kin_t), pointer :: kd
    integer :: i, j, k, nfs
    real(kind=wp) :: Q, Q3, Ei, Ej, Ek
    real(kind=wp) :: chi_ij, chi_ik, chi_jk
    real(kind=wp) :: prod, total

    eval_epem_EFPG3b = 0._wp

    nfs = npar - 2  ! number of final-state partons (initial e+e- are entries 1,2)
    if (nfs < 3) return  ! Born and below: no triangle

    kd => kin(npar)
    Q = sqrt(kd%s(1, 2))
    Q3 = Q*Q*Q

    total = 0._wp
    do i = 3, npar - 2
      Ei = kd%p(4, i)
      do j = i + 1, npar - 1
        Ej = kd%p(4, j)
        chi_ij = kd%s(i, j) / (4._wp*Ei*Ej)
        do k = j + 1, npar
          Ek = kd%p(4, k)
          chi_ik = kd%s(i, k) / (4._wp*Ei*Ek)
          chi_jk = kd%s(j, k) / (4._wp*Ej*Ek)
          prod = chi_ij*chi_ik*chi_jk
          if (prod < 0._wp) prod = 0._wp  ! numerical safety
          total = total + (Ei*Ej*Ek) * sqrt(prod)
        end do
      end do
    end do

    !> Note: explicit factor of 6 from sum-over-distinct-triples vs. ordered sum is
    !> NOT applied here because the loops above already enumerate i<j<k (each
    !> distinct triple once). The user-side definition uses "6 * sum over distinct
    !> triples", so we restore that factor:
    eval_epem_EFPG3b = 6._wp * total / Q3

  end function eval_epem_EFPG3b

  !----- EFP_K3^(2): triangle with double edges
  !> Edges: {(1,2),(1,2),(1,3),(1,3),(2,3),(2,3)}.
  !> Each doubled edge contributes chi (= sqrt(chi)^2). 3 vertices.
  !> EFP = sum_{phi: {1,2,3} -> {1..nfs}} z_{phi(1)} z_{phi(2)} z_{phi(3)}
  !>       * chi_{phi(1),phi(2)} * chi_{phi(1),phi(3)} * chi_{phi(2),phi(3)}
  !> Maps with phi(a)=phi(b) for any edge contribute 0 (chi_ii=0).
  !> Equivalent to 6 * sum_{i<j<k} z_i z_j z_k chi_ij chi_ik chi_jk.
  real(kind=wp) function eval_epem_EFPK32(npar)
    integer, intent(in) :: npar
    type(Kin_t), pointer :: kd
    integer :: i, j, k
    real(kind=wp) :: Q, Q3, Ei, Ej, Ek
    real(kind=wp) :: chi_ij, chi_ik, chi_jk
    real(kind=wp) :: total

    eval_epem_EFPK32 = 0._wp
    if (npar - 2 < 3) return  ! need >=3 final-state partons

    kd => kin(npar)
    Q = sqrt(kd%s(1, 2))
    Q3 = Q*Q*Q

    total = 0._wp
    do i = 3, npar - 2
      Ei = kd%p(4, i)
      do j = i + 1, npar - 1
        Ej = kd%p(4, j)
        chi_ij = kd%s(i, j) / (4._wp*Ei*Ej)
        do k = j + 1, npar
          Ek = kd%p(4, k)
          chi_ik = kd%s(i, k) / (4._wp*Ei*Ek)
          chi_jk = kd%s(j, k) / (4._wp*Ej*Ek)
          total = total + (Ei*Ej*Ek) * chi_ij * chi_ik * chi_jk
        end do
      end do
    end do

    !> factor of 6 = 3! permutations of {i,j,k} (the unordered i<j<k loop counts each
    !> distinct triple once, but the EFP sums over all 3! ordered maps).
    eval_epem_EFPK32 = 6._wp * total / Q3

  end function eval_epem_EFPK32

  !----- EFP_C5: five-cycle
  !> Edges: {(1,2),(2,3),(3,4),(4,5),(5,1)}. 5 vertices.
  !> EFP = sum_{phi: {1..5} -> {1..nfs}} prod_v z_{phi(v)}
  !>       * sqrt(chi_{phi(1),phi(2)} chi_{phi(2),phi(3)} chi_{phi(3),phi(4)}
  !>              chi_{phi(4),phi(5)} chi_{phi(5),phi(1)})
  !> Generic combinatorial loop over all maps.
  real(kind=wp) function eval_epem_EFPC5(npar)
    integer, intent(in) :: npar
    type(Kin_t), pointer :: kd
    integer :: nfs, a, b, c, d, e, ia, ib, ic, id, ie
    real(kind=wp) :: Q, Q5
    real(kind=wp), allocatable :: z(:), chi(:,:)
    real(kind=wp) :: prod, total, w

    eval_epem_EFPC5 = 0._wp
    nfs = npar - 2
    if (nfs < 3) return

    kd => kin(npar)
    Q = sqrt(kd%s(1, 2))
    Q5 = Q**5

    allocate(z(nfs), chi(nfs, nfs))
    do a = 1, nfs
      z(a) = kd%p(4, a + 2) / Q
    end do
    do a = 1, nfs
      chi(a, a) = 0._wp
      do b = a + 1, nfs
        chi(a, b) = kd%s(a + 2, b + 2) / (4._wp * kd%p(4, a + 2) * kd%p(4, b + 2))
        if (chi(a, b) < 0._wp) chi(a, b) = 0._wp
        chi(b, a) = chi(a, b)
      end do
    end do

    total = 0._wp
    do ia = 1, nfs
      do ib = 1, nfs
        if (ib == ia) cycle
        do ic = 1, nfs
          if (ic == ib) cycle
          do id = 1, nfs
            if (id == ic) cycle
            do ie = 1, nfs
              if (ie == id) cycle
              if (ie == ia) cycle  ! edge (5,1)
              w = chi(ia, ib) * chi(ib, ic) * chi(ic, id) * chi(id, ie) * chi(ie, ia)
              if (w <= 0._wp) cycle
              prod = z(ia) * z(ib) * z(ic) * z(id) * z(ie)
              total = total + prod * sqrt(w)
            end do
          end do
        end do
      end do
    end do

    eval_epem_EFPC5 = total
    deallocate(z, chi)

  end function eval_epem_EFPC5

  !----- EFP_theta_{1,2,2}: theta graph
  !> Edges: {(1,2),(1,3),(3,2),(1,4),(4,2)}. 4 vertices.
  !> EFP = sum_{phi: {1..4} -> {1..nfs}} prod_v z_{phi(v)}
  !>       * sqrt(chi_{phi(1),phi(2)} chi_{phi(1),phi(3)} chi_{phi(3),phi(2)}
  !>              chi_{phi(1),phi(4)} chi_{phi(4),phi(2)})
  real(kind=wp) function eval_epem_EFPTHETA122(npar)
    integer, intent(in) :: npar
    type(Kin_t), pointer :: kd
    integer :: nfs, a, b, ia, ib, ic, id
    real(kind=wp) :: Q
    real(kind=wp), allocatable :: z(:), chi(:,:)
    real(kind=wp) :: prod, total, w

    eval_epem_EFPTHETA122 = 0._wp
    nfs = npar - 2
    if (nfs < 3) return

    kd => kin(npar)
    Q = sqrt(kd%s(1, 2))

    allocate(z(nfs), chi(nfs, nfs))
    do a = 1, nfs
      z(a) = kd%p(4, a + 2) / Q
    end do
    do a = 1, nfs
      chi(a, a) = 0._wp
      do b = a + 1, nfs
        chi(a, b) = kd%s(a + 2, b + 2) / (4._wp * kd%p(4, a + 2) * kd%p(4, b + 2))
        if (chi(a, b) < 0._wp) chi(a, b) = 0._wp
        chi(b, a) = chi(a, b)
      end do
    end do

    total = 0._wp
    !> phi(1)=ia, phi(2)=ib, phi(3)=ic, phi(4)=id
    !> Edges: (1,2),(1,3),(3,2),(1,4),(4,2) require ia/=ib, ia/=ic, ic/=ib, ia/=id, id/=ib
    do ia = 1, nfs
      do ib = 1, nfs
        if (ib == ia) cycle
        do ic = 1, nfs
          if (ic == ia .or. ic == ib) cycle
          do id = 1, nfs
            if (id == ia .or. id == ib) cycle
            w = chi(ia, ib) * chi(ia, ic) * chi(ic, ib) * chi(ia, id) * chi(id, ib)
            if (w <= 0._wp) cycle
            prod = z(ia) * z(ib) * z(ic) * z(id)
            total = total + prod * sqrt(w)
          end do
        end do
      end do
    end do

    eval_epem_EFPTHETA122 = total
    deallocate(z, chi)

  end function eval_epem_EFPTHETA122

  !> Calculate entries in theta matrix to determine C parameter
  pure real(kind=wp) function theta(ii, ij, npar)
    integer, intent(in) :: ii, ij, npar
    real(kind=wp) :: top, bot, absp
    integer :: i

    top = 0._wp
    bot = 0._wp
    !> magnitude of impulse
    absp = 0._wp

    do i = 3, npar
      absp = sqrt(kin(npar)%p(1, i)**2 + &
                  kin(npar)%p(2, i)**2 + kin(npar)%p(3, i)**2)

      top = top + kin(npar)%p(ii, i)*kin(npar)%p(ij, i)/absp
      bot = bot + absp
    end do
    theta = top/bot
    return
  end function theta


  !> Calculate entries in theta matrix to determine jetC parameter
  pure real(kind=wp) function thetajet(ii, ij, npar)
    integer, intent(in) :: ii, ij, npar
    real(kind=wp) :: top, bot, absp
    integer :: i

    top = 0._wp
    bot = 0._wp
    !> magnitude of impulse
    absp = 0._wp

    do i = 1, kin(npar)%njets
      absp = sqrt(kin(npar)%pjets(1, i)**2 + &
                  kin(npar)%pjets(2, i)**2 + kin(npar)%pjets(3, i)**2)

      top = top + kin(npar)%pjets(ii, i)*kin(npar)%pjets(ij, i)/absp
      bot = bot + absp
    end do
    thetajet = top/bot
    return
  end function thetajet




  !> Calculate entries in theta matrix to determine C parameter for DIS
  pure real(kind=wp) function thetadis(ii, ij, npar)
    integer, intent(in) :: ii, ij, npar
    real(kind=wp) :: top, bot, absp, ehtot
    integer :: i, ncount

    top = 0._wp
    bot = 0._wp
    !> magnitude of impulse
    absp = 0._wp
    ehtot = 0._wp
    ncount = 0

    do i = 4, npar
      ! Select only particles in the current hemisphere (pz > 0)
      if (kin(npar)%p(3, i) > 0._wp) then
        ncount = ncount + 1
        absp = sqrt(kin(npar)%p(1, i)**2 + &
                    kin(npar)%p(2, i)**2 + kin(npar)%p(3, i)**2)
        top = top + kin(npar)%p(ii, i)*kin(npar)%p(ij, i)/absp
        bot = bot + absp
      end if
    end do
    if (ncount .eq. 0) then
      thetadis = -1._wp
    else
      thetadis = top/bot
    end if
    return
  end function thetadis

  real(kind=wp) function eval_cos_theta_CS_0PT(npar)
    !> Definition of cos(theta*) that is also valid at ptZ = 0
    !> so can be used for Z+0j production/ cases where ptZ = 0
    !> Tested for ZJ(J)LO against eval_cos_theta_CS and agrees perfectly
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: mll, pzll, ptll, p1p, p1m, p2p, p2m
    !> 'n-1':  l1 = lm (for Z)
    !> 3 = z component
    !> 4 = E component
    do i = 1, 4 ! Only need first 3 components
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    mll = v1_minv(pV)
    pzll = pV(3)
    ptll = sqrt(pV(1)**2 + pV(2)**2)
    p1p = kin(npar)%p(4, npar - 1) + kin(npar)%p(3, npar - 1)
    p1m = kin(npar)%p(4, npar - 1) - kin(npar)%p(3, npar - 1)
    p2p = kin(npar)%p(4, npar) + kin(npar)%p(3, npar)
    p2m = kin(npar)%p(4, npar) - kin(npar)%p(3, npar)
    eval_cos_theta_CS_0PT = sign(1._wp, pzll)*((p1p*p2m) - (p1m*p2p))/mll/sqrt(mll**2 + ptll**2)
  end function eval_cos_theta_CS_0PT

  real(kind=wp) function eval_costhetastar(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y_g1, y_g2
    y_g1 = eval_y_gam(npar, 1)  ! leading
    y_g2 = eval_y_gam(npar, 2)  ! sub-leading
    eval_costhetastar = abs(tanh((y_g1 - y_g2)/2._wp))
  end function eval_costhetastar

  ! !which one???
  ! real(kind=wp) function eval_costhetastar(npar)
  !   integer, intent(in) :: npar
  !   real(kind=wp), dimension(4) :: pg1,pg2,pH
  !   real(kind=wp) :: ptg1,ptg2,ptH
  !   integer :: i
  !   do i=1,4
  !     pg1(i) = kin(npar)%p(i,npar)
  !     pg2(i) = kin(npar)%p(i,npar-1)
  !     pH(i)  = kin(npar)%p(i,npar)+kin(npar)%p(i,npar-1)
  !   end do
  !     ptg1=v1_pt(pg1)
  !     ptg2=v1_pt(pg2)
  !     ptH =v1_pt(pH)
  !     eval_costhetastar= abs(dsinh(v1_y(pg1)-v1_y(pg2)))*2._wp*ptg1*ptg2 &
  !                      & /v1_minv2(pH)/dsqrt(1._wp+(ptH/v1_minv(pH))**2._wp)
  ! end function eval_costhetastar

  !----- gauge boson (sum of the related leptons)

  real(kind=wp) function eval_pt_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_pt_V = v1_pt(pV)
  end function eval_pt_V

  real(kind=wp) function eval_pt_V2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2)
    eval_pt_V2 = v1_pt(pV)
  end function eval_pt_V2

  real(kind=wp) function eval_pt_V4l(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_pt_V4l = v1_pt(pV)
  end function eval_pt_V4l

  real(kind=wp) function eval_pt_V2l1p(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_pt_V2l1p = v1_pt(pV)
  end function eval_pt_V2l1p

  real(kind=wp) function eval_pt_2l1p_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) + kin(npar)%p(1:4, npar - 1)
    eval_pt_2l1p_V = v1_pt(pV)
  end function eval_pt_2l1p_V

  real(kind=wp) function eval_minv_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_minv_V = v1_minv(pV)
  end function eval_minv_V

  !===========================================================================
  !  Chebyshev-moment observables (event-level moments via PROFILE histograms)
  !---------------------------------------------------------------------------
  !  A profile single-bin histogram of chebT_<x>_n accumulates the literal
  !  per-phase-space-point sum  Sum_j w_j T_n(u(x_j)).  Divided by the rate
  !  this is <T_n(u(x))>, computed event-level with no binning of x.
  !===========================================================================

  !> Chebyshev polynomial T_n(u) with u clipped to [-1,1] (stable recurrence).
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

  !> T_n of the dilepton invariant mass, linear map on [66,116] GeV.
  !> imem = n (Chebyshev order).  Bind as chebT_mll_n.
  !> u(m) on the BREIT-WIGNER variable, not on the mass.
  !>
  !> A linear map over [66,116] spends its resolution where there is no rate:
  !> 70% of the cross section lies in 88-94 GeV, a 6 GeV sliver of a 50 GeV
  !> window, and six Chebyshev polynomials resolve ~8 GeV -- 3.3 times the Z
  !> width, so the peak is unresolved by construction.  Mapping through
  !> arctan((m^2-M^2)/(M*Gamma)) instead makes a pure Breit-Wigner FLAT in u,
  !> so the same moments describe the resonance rather than smearing it: four
  !> moments on this map beat twelve on the linear one by a factor ~20.
  real(kind=wp) function bw_u(x, lo, hi, mres, gres) result(u)
    real(kind=wp), intent(in) :: x, lo, hi, mres, gres
    real(kind=wp) :: t, tl, th
    t  = atan((x*x   - mres*mres)/(mres*gres))
    tl = atan((lo*lo - mres*mres)/(mres*gres))
    th = atan((hi*hi - mres*mres)/(mres*gres))
    u  = 2.0_wp*(t - tl)/(th - tl) - 1.0_wp
  end function bw_u

  real(kind=wp) function eval_chebT_mll(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 66.0_wp, hi = 116.0_wp
    real(kind=wp), parameter :: mz = 91.1876_wp, gz = 2.4952_wp
    !> defensive: never touch kin() for the reset sentinel (npar<1)
    if (npar < 1) then
      eval_chebT_mll = 0.0_wp
      return
    end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = v1_minv(pV)
    u = bw_u(x, lo, hi, mz, gz)
    eval_chebT_mll = cheb_T(u, n)
  end function eval_chebT_mll

  !> |cos theta*| of the leptons in the Collins-Soper frame, lin [0,1].
  !> At Born level pT_l = (m/2) sin(theta*), so the pair (m, cos theta*)
  !> DETERMINES the lepton transverse momenta -- which is why the cross moments
  !> below are the right object for predicting them.
  real(kind=wp) function eval_abscosthstCS_ll(npar) result(res)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: p1, p2, pV
    real(kind=wp) :: p1p, p1m, p2p, p2m, m, qt, den
    if (npar < 1) then; res = 0.0_wp; return; end if
    p1(1:4) = kin(npar)%p(1:4, npar - 1)
    p2(1:4) = kin(npar)%p(1:4, npar)
    pV(1:4) = p1(1:4) + p2(1:4)
    m  = v1_minv(pV)
    qt = v1_pt(pV)
    p1p = (p1(4) + p1(3)); p1m = (p1(4) - p1(3))
    p2p = (p2(4) + p2(3)); p2m = (p2(4) - p2(3))
    den = m*sqrt(m*m + qt*qt)
    if (den <= 0.0_wp) then; res = 0.0_wp; return; end if
    res = min(abs(p1p*p2m - p1m*p2p)/den, 1.0_wp)
  end function eval_abscosthstCS_ll

  real(kind=wp) function eval_chebT_ctl(npar, n)   ! T_n(|cos th*_CS|), lin [0,1]
    integer, intent(in) :: npar, n
    real(kind=wp) :: u
    if (npar < 1) then; eval_chebT_ctl = 0.0_wp; return; end if
    u = 2.0_wp*eval_abscosthstCS_ll(npar) - 1.0_wp
    eval_chebT_ctl = cheb_T(u, n)
  end function eval_chebT_ctl

  !> CROSS moments  < T_m(u_BW(m_ll)) T_n(u(|cos th*|)) >.  Separate moments of
  !> the two constrain the marginals; only the cross moments carry the JOINT
  !> distribution, and it is the joint distribution that fixes pT_l1.
  !> Orders packed as imem = -(10*m + n), 1 <= m,n <= 9.
  real(kind=wp) function eval_chebTT_mllct(npar, mn)
    integer, intent(in) :: npar, mn
    real(kind=wp) :: um, uc, x
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), parameter :: lo = 66.0_wp, hi = 116.0_wp
    real(kind=wp), parameter :: mz = 91.1876_wp, gz = 2.4952_wp
    integer :: m, n
    if (npar < 1) then; eval_chebTT_mllct = 0.0_wp; return; end if
    m = abs(mn)/10
    n = mod(abs(mn), 10)
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x  = v1_minv(pV)
    um = bw_u(x, lo, hi, mz, gz)
    uc = 2.0_wp*eval_abscosthstCS_ll(npar) - 1.0_wp
    eval_chebTT_mllct = cheb_T(um, m)*cheb_T(uc, n)
  end function eval_chebTT_mllct

  !> T_n of the dilepton rapidity |y_ll|, linear map on [0,2.4].
  !> imem = n.  Bind as chebT_absyz_n.
  real(kind=wp) function eval_chebT_absyz(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 0.0_wp, hi = 2.4_wp
    if (npar < 1) then
      eval_chebT_absyz = 0.0_wp
      return
    end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = abs(v1_y(pV))
    u = 2.0_wp*(x - lo)/(hi - lo) - 1.0_wp
    u = max(-1.0_wp, min(1.0_wp, u))   ! out-of-range events sit at the edge
    eval_chebT_absyz = cheb_T(u, n)
  end function eval_chebT_absyz

  !===========================================================================
  !  WEIGHTED Chebyshev observables  T_n(u)/sqrt(1-u^2)  (basis test, 2026-09).
  !  The classical Chebyshev orthogonality weight, accumulated EVENT-LEVEL like
  !  the plain moments, no regulator: a phase-space point with u^2 >= 1 exactly
  !  (measure zero) contributes 0.  n = 0 is the pure weight 1/sqrt(1-u^2).
  !===========================================================================
  pure function cheb_W(u_in, n) result(res)
    real(kind=wp), intent(in) :: u_in
    integer, intent(in) :: n
    real(kind=wp) :: res, u, s2
    integer :: nn
    u  = max(-1.0_wp, min(1.0_wp, u_in))
    s2 = 1.0_wp - u*u
    nn = abs(n)
    if (nn == 10) nn = 0            ! imem=-10 encodes order 0 (imem must be nonzero)
    if (s2 <= 0.0_wp) then
      res = 0.0_wp
    else
      res = cheb_T(u, nn)/sqrt(s2)
    end if
  end function cheb_W

  !> T_n(u)/sqrt(1-u^2) of m_ll on the Breit-Wigner map [66,116].  Bind as chebW_mll_n.
  real(kind=wp) function eval_chebW_mll(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 66.0_wp, hi = 116.0_wp
    real(kind=wp), parameter :: mz = 91.1876_wp, gz = 2.4952_wp
    if (npar < 1) then
      eval_chebW_mll = 0.0_wp
      return
    end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = v1_minv(pV)
    u = bw_u(x, lo, hi, mz, gz)
    eval_chebW_mll = cheb_W(u, n)
  end function eval_chebW_mll

  !> T_n(u)/sqrt(1-u^2) of |y_ll| on the linear map [0,2.4].  Bind as chebW_absyz_n.
  real(kind=wp) function eval_chebW_absyz(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 0.0_wp, hi = 2.4_wp
    if (npar < 1) then
      eval_chebW_absyz = 0.0_wp
      return
    end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = abs(v1_y(pV))
    u = 2.0_wp*(x - lo)/(hi - lo) - 1.0_wp
    eval_chebW_absyz = cheb_W(u, n)
  end function eval_chebW_absyz

  !> T_n of the dilepton transverse momentum pT_ll, LOG map on the composite
  !> recoil axis [soft_lo, x_hi] = [30,2500] GeV (MUST match the engine map
  !> u = umap(x, soft_lo, x_hi, 'log') so the FO moment composes with the soft
  !> and tail pieces).  The FO REGION is imposed only inside the window: book
  !> with a per-histogram selector ptz in [x_match, 500] (e.g. [30,500]); below
  !> the seam the shower is preserved.  Map floor is soft_lo, NOT x_match.
  !> imem = n.  Bind as chebT_ptz_n.
  real(kind=wp) function eval_chebT_ptz(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u, lnlo, lnhi
    real(kind=wp), parameter :: lo = 30.0_wp, hi = 2500.0_wp
    if (npar < 1) then
      eval_chebT_ptz = 0.0_wp
      return
    end if
    lnlo = log(lo)
    lnhi = log(hi)
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - lnlo)/(lnhi - lnlo) - 1.0_wp
    eval_chebT_ptz = cheb_T(u, n)
  end function eval_chebT_ptz

  !> C2 smootherstep recoil profile w(x) in ln x: 0 below a, C2 ramp to 1 over
  !> [a,b], 1, then (optional) C2 ramp back to 0 over [c,d] (hard cut if d<=c).
  !> a==b gives the hard lower step.  MUST match maxent_upgrade.profile_w.
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

  !> recoil profile weight w(pT_ll) alone (= chebTw n=0, the w-rate numerator).
  real(kind=wp) function eval_w_ptz(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then
      eval_w_ptz = 0.0_wp
      return
    end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_w_ptz = prof_w(v1_pt(pV), pa, pb, pc, pd)
  end function eval_w_ptz

  !> w(pT_ll) * T_n(u(pT_ll)), LOG map [30,2500], smooth profile [30,60](->500).
  !> (map [0.5,500] until 2026-09-18; [30,2500] matches Rene Poncelet's Stripper run)
  !> Book as a single-bin PROFILE histogram (no selector; the profile windows it).
  !> <T_n>_FO(w) = prof(chebTw_ptz_n) / prof(w_ptz);  R = prof(w_ptz)/sigma_fid.
  real(kind=wp) function eval_chebTw_ptz(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u, lnlo, lnhi
    real(kind=wp), parameter :: lo = 30.0_wp, hi = 2500.0_wp
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then
      eval_chebTw_ptz = 0.0_wp
      return
    end if
    lnlo = log(lo)
    lnhi = log(hi)
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - lnlo)/(lnhi - lnlo) - 1.0_wp
    eval_chebTw_ptz = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_ptz

  !===========================================================================
  !  W production (WP / WM) event-level Chebyshev moments.  The W is the
  !  colour singlet built from the last two particles (charged lepton and
  !  neutrino), exactly as ptw/mln/yw are bound (eval_pt_V, eval_minv_V,
  !  eval_y_V).  Same construction as Drell-Yan with the W in place of the Z:
  !    Born   : m_W   Breit-Wigner map on [40,1000] GeV around (M_W, Gamma_W)
  !             |y_W| linear map on [0,4]
  !    recoil : pT_W  log map [25,2500], smooth profile [25,50] -> 500 (hard)
  !  matching_scale(80.385) = 24.9 GeV -> seam 25 (DY: 27.9 -> 30, H: 37.1 -> 37)
  !===========================================================================
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

  real(kind=wp) function eval_chebT_absyw(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: u
    real(kind=wp), parameter :: hi = 4.0_wp
    if (npar < 1) then; eval_chebT_absyw = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    u = 2.0_wp*abs(v1_y(pV))/hi - 1.0_wp
    eval_chebT_absyw = cheb_T(u, n)
  end function eval_chebT_absyw

  real(kind=wp) function eval_w_ptw(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), parameter :: pa = 25.0_wp, pb = 50.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then; eval_w_ptw = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_w_ptw = prof_w(v1_pt(pV), pa, pb, pc, pd)
  end function eval_w_ptw

  real(kind=wp) function eval_chebTw_ptw(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u, lnlo, lnhi
    real(kind=wp), parameter :: lo = 25.0_wp, hi = 2500.0_wp
    real(kind=wp), parameter :: pa = 25.0_wp, pb = 50.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then; eval_chebTw_ptw = 0.0_wp; return; end if
    lnlo = log(lo); lnhi = log(hi)
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - lnlo)/(lnhi - lnlo) - 1.0_wp
    eval_chebTw_ptw = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_ptw

  !> Signed Collins-Soper decay angle of the CHARGED lepton in W production:
  !>   cos th* = sign(p_z^W) * [(E_l+p_zl)(E_nu-p_znu) - (E_l-p_zl)(E_nu+p_znu)] / (m sqrt(m^2+q_T^2)),
  !> with the CS z-axis along the W longitudinal momentum.  It is the Born-level
  !> variable the lepton acceptance depends on and the V-A structure makes
  !> asymmetric, so unlike |cos th*| of the Z it must keep its sign.  ilep is the
  !> index of the charged lepton in the last pair (WM: 1 = npar-1, WP: 2 = npar).
  real(kind=wp) function eval_costhCS_w(npar, ilep) result(res)
    integer, intent(in) :: npar, ilep
    real(kind=wp), dimension(4) :: pl, pn, pV
    real(kind=wp) :: lp, lm, np_, nm, m, qt, den
    if (npar < 1) then; res = 0.0_wp; return; end if
    if (ilep == 1) then
      pl(1:4) = kin(npar)%p(1:4, npar - 1); pn(1:4) = kin(npar)%p(1:4, npar)
    else
      pl(1:4) = kin(npar)%p(1:4, npar);     pn(1:4) = kin(npar)%p(1:4, npar - 1)
    end if
    pV(1:4) = pl(1:4) + pn(1:4)
    m  = v1_minv(pV)
    qt = v1_pt(pV)
    lp = pl(4) + pl(3); lm = pl(4) - pl(3)
    np_ = pn(4) + pn(3); nm = pn(4) - pn(3)
    den = m*sqrt(m*m + qt*qt)
    if (den <= 0.0_wp) then; res = 0.0_wp; return; end if
    res = (lp*nm - lm*np_)/den
    if (pV(3) < 0.0_wp) res = -res
    res = max(-1.0_wp, min(1.0_wp, res))
  end function eval_costhCS_w

  real(kind=wp) function eval_chebT_ctwm(npar, n)   ! T_n(cos th*), W-: charged lepton first
    integer, intent(in) :: npar, n
    if (npar < 1) then; eval_chebT_ctwm = 0.0_wp; return; end if
    eval_chebT_ctwm = cheb_T(eval_costhCS_w(npar, 1), n)
  end function eval_chebT_ctwm

  real(kind=wp) function eval_chebT_ctwp(npar, n)   ! T_n(cos th*), W+: charged lepton second
    integer, intent(in) :: npar, n
    if (npar < 1) then; eval_chebT_ctwp = 0.0_wp; return; end if
    eval_chebT_ctwp = cheb_T(eval_costhCS_w(npar, 2), n)
  end function eval_chebT_ctwp

  !> WINDOWED decay-angle tower.  The fiducial cos th* distribution has a
  !> kinematic edge at |cos th*| ~ sqrt(1-(2 pT_nu,min/m_W)^2) = 0.78 where only
  !> boosted W's survive the cuts; imposing moments over the full range writes
  !> that (recoil-driven) edge into the tilt.  Constrain the polarization where
  !> it lives, in the bulk: profile w(c) = 1 for |c| < 0.5, C2 ramp to 0 by 0.7,
  !> written on x = c + 1 in (0,2) so the standard ln-space prof_w applies
  !> (a=0.3, b=0.4 rising; c=1.6, d=1.7 falling).  Moments T_n(c) on the linear
  !> map [-1,1].  Book prof_wctw_0 (= w) and prof_wctw_n (= w T_n) as profiles.
  real(kind=wp) function eval_w_ctw(npar, ilep)
    integer, intent(in) :: npar, ilep
    real(kind=wp), parameter :: pa = 0.3_wp, pb = 0.4_wp, pc = 1.6_wp, pd = 1.7_wp
    if (npar < 1) then; eval_w_ctw = 0.0_wp; return; end if
    eval_w_ctw = prof_w(eval_costhCS_w(npar, ilep) + 1.0_wp, pa, pb, pc, pd)
  end function eval_w_ctw

  real(kind=wp) function eval_chebTw_ctwm(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp) :: c
    if (npar < 1) then; eval_chebTw_ctwm = 0.0_wp; return; end if
    c = eval_costhCS_w(npar, 1)
    eval_chebTw_ctwm = eval_w_ctw(npar, 1) * cheb_T(c, n)
  end function eval_chebTw_ctwm

  real(kind=wp) function eval_chebTw_ctwp(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp) :: c
    if (npar < 1) then; eval_chebTw_ctwp = 0.0_wp; return; end if
    c = eval_costhCS_w(npar, 2)
    eval_chebTw_ctwp = eval_w_ctw(npar, 2) * cheb_T(c, n)
  end function eval_chebTw_ctwp

  !> JOINT recoil x decay-angle moments  w_ptw(pT_W) w_ctw(c) T_m(u_log(pT_W)) T_n(c),
  !> orders packed as imem = -(10 m + n), 1 <= m,n <= 9 (chebTT_ pattern).  Separate
  !> towers fix the two marginals only; the m_T tail above M_W lives at small
  !> |cos th*| AND large pT_W, so it depends on their correlation, which the
  !> leading-order prior gets wrong at large pT_W (LO polarization).
  real(kind=wp) function eval_w_ptwctw(npar, ilep)
    integer, intent(in) :: npar, ilep
    if (npar < 1) then; eval_w_ptwctw = 0.0_wp; return; end if
    eval_w_ptwctw = eval_w_ptw(npar) * eval_w_ctw(npar, ilep)
  end function eval_w_ptwctw

  real(kind=wp) function eval_chebTT_ptwctw_i(npar, mn, ilep) result(res)
    integer, intent(in) :: npar, mn, ilep
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u, c
    real(kind=wp), parameter :: lo = 25.0_wp, hi = 2500.0_wp
    integer :: m, n
    if (npar < 1) then; res = 0.0_wp; return; end if
    m = abs(mn)/10; n = mod(abs(mn), 10)
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    c = eval_costhCS_w(npar, ilep)
    res = eval_w_ptwctw(npar, ilep) * cheb_T(u, m) * cheb_T(c, n)
  end function eval_chebTT_ptwctw_i

  real(kind=wp) function eval_chebTT_ptwctwm(npar, mn)
    integer, intent(in) :: npar, mn
    eval_chebTT_ptwctwm = eval_chebTT_ptwctw_i(npar, mn, 1)
  end function eval_chebTT_ptwctwm

  real(kind=wp) function eval_chebTT_ptwctwp(npar, mn)
    integer, intent(in) :: npar, mn
    eval_chebTT_ptwctwp = eval_chebTT_ptwctw_i(npar, mn, 2)
  end function eval_chebTT_ptwctwp

  !===========================================================================
  !  DY lepton Chebyshev moments (leading/subleading by pT).  Born-type: full
  !  range, no window.  Leptons are the last two particles (npar-1, npar).
  !===========================================================================
  real(kind=wp) function eval_chebT_ptl1(npar, n)   ! leading lepton pT, log [27,200]
    integer, intent(in) :: npar, n
    real(kind=wp) :: p1, p2, u
    real(kind=wp), parameter :: lo = 27.0_wp, hi = 200.0_wp
    if (npar < 1) then; eval_chebT_ptl1 = 0.0_wp; return; end if
    p1 = v1_pt(kin(npar)%p(:, npar - 1)); p2 = v1_pt(kin(npar)%p(:, npar))
    u = 2.0_wp*(log(max(max(p1, p2), 1.0e-6_wp)) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebT_ptl1 = cheb_T(u, n)
  end function eval_chebT_ptl1

  real(kind=wp) function eval_chebT_ptl2(npar, n)   ! subleading lepton pT, log [27,150]
    integer, intent(in) :: npar, n
    real(kind=wp) :: p1, p2, u
    real(kind=wp), parameter :: lo = 27.0_wp, hi = 150.0_wp
    if (npar < 1) then; eval_chebT_ptl2 = 0.0_wp; return; end if
    p1 = v1_pt(kin(npar)%p(:, npar - 1)); p2 = v1_pt(kin(npar)%p(:, npar))
    u = 2.0_wp*(log(max(min(p1, p2), 1.0e-6_wp)) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebT_ptl2 = cheb_T(u, n)
  end function eval_chebT_ptl2

  real(kind=wp) function eval_chebT_yl1(npar, n)    ! |y| of leading-pT lepton, lin [0,2.5]
    integer, intent(in) :: npar, n
    real(kind=wp) :: p1, p2, y, u
    real(kind=wp), parameter :: hi = 2.5_wp
    if (npar < 1) then; eval_chebT_yl1 = 0.0_wp; return; end if
    p1 = v1_pt(kin(npar)%p(:, npar - 1)); p2 = v1_pt(kin(npar)%p(:, npar))
    if (p1 >= p2) then; y = abs(v1_y(kin(npar)%p(:, npar - 1)))
    else; y = abs(v1_y(kin(npar)%p(:, npar))); end if
    u = 2.0_wp*y/hi - 1.0_wp
    eval_chebT_yl1 = cheb_T(u, n)
  end function eval_chebT_yl1

  real(kind=wp) function eval_chebT_yl2(npar, n)    ! |y| of subleading-pT lepton, lin [0,2.5]
    integer, intent(in) :: npar, n
    real(kind=wp) :: p1, p2, y, u
    real(kind=wp), parameter :: hi = 2.5_wp
    if (npar < 1) then; eval_chebT_yl2 = 0.0_wp; return; end if
    p1 = v1_pt(kin(npar)%p(:, npar - 1)); p2 = v1_pt(kin(npar)%p(:, npar))
    if (p1 < p2) then; y = abs(v1_y(kin(npar)%p(:, npar - 1)))
    else; y = abs(v1_y(kin(npar)%p(:, npar))); end if
    u = 2.0_wp*y/hi - 1.0_wp
    eval_chebT_yl2 = cheb_T(u, n)
  end function eval_chebT_yl2

  !===========================================================================
  !  gg -> H Chebyshev moments.  NNLOJET binds pth/yh to eval_pt_V/eval_y_V, so
  !  the colour-singlet accessor p(:,npar-1)+p(:,npar) is the right one here too.
  !    Born   : |y_H|  (lin [0,4])          -- the Higgs mass is fixed
  !    recoil : pT_H   (log [1,1000], smooth profile [30,60]->1000)
  !===========================================================================
  real(kind=wp) function eval_chebT_yh(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: u
    real(kind=wp), parameter :: hi = 4.0_wp
    if (npar < 1) then; eval_chebT_yh = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    u = 2.0_wp*abs(v1_y(pV))/hi - 1.0_wp
    u = max(-1.0_wp, min(1.0_wp, u))   ! out-of-range events sit at the edge
    eval_chebT_yh = cheb_T(u, n)
  end function eval_chebT_yh

  real(kind=wp) function eval_w_pth(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), parameter :: pa = 37.0_wp, pb = 74.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_w_pth = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_w_pth = prof_w(v1_pt(pV), pa, pb, pc, pd)
  end function eval_w_pth

  real(kind=wp) function eval_chebTw_pth(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 1.0_wp, hi = 1000.0_wp
    real(kind=wp), parameter :: pa = 37.0_wp, pb = 74.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_chebTw_pth = 0.0_wp; return; end if
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_pth = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_pth

  !===========================================================================
  !  DY+jet (ZJ) Chebyshev moments, all with SMOOTH profiles:
  !    pT_j1 : constrain only ABOVE a resolution cut   w = prof_w(pTj1, 30,60,..)
  !    pT_j2 : same, NLO-accurate in ZJ@NNLO           w = prof_w(pTj2, 30,60,..)
  !    dphi(l1,l2) : constrain only AWAY from pi, i.e. in x = pi - dphi, so the
  !                  back-to-back (Sudakov) region x->0 is left to the shower.
  !  Bind pT_j1 with min_njets=1, pT_j2 with min_njets=2.
  !===========================================================================
  real(kind=wp) function eval_w_ptj1(npar)
    integer, intent(in) :: npar
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_w_ptj1 = 0.0_wp; return; end if
    eval_w_ptj1 = prof_w(v1_pt(kin(npar)%pjets(:, 1)), pa, pb, pc, pd)
  end function eval_w_ptj1

  real(kind=wp) function eval_chebTw_ptj1(npar, n)   ! w*T_n(pT_j1), log [10,1000]
    integer, intent(in) :: npar, n
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 10.0_wp, hi = 1000.0_wp
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_chebTw_ptj1 = 0.0_wp; return; end if
    x = max(v1_pt(kin(npar)%pjets(:, 1)), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_ptj1 = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_ptj1

  real(kind=wp) function eval_w_ptj2(npar)
    integer, intent(in) :: npar
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_w_ptj2 = 0.0_wp; return; end if
    eval_w_ptj2 = prof_w(v1_pt(kin(npar)%pjets(:, 2)), pa, pb, pc, pd)
  end function eval_w_ptj2

  real(kind=wp) function eval_chebTw_ptj2(npar, n)   ! w*T_n(pT_j2), log [10,1000]
    integer, intent(in) :: npar, n
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 10.0_wp, hi = 1000.0_wp
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_chebTw_ptj2 = 0.0_wp; return; end if
    x = max(v1_pt(kin(npar)%pjets(:, 2)), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_ptj2 = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_ptj2

  !===========================================================================
  !  MIXED (two-observable) moments:  < T_m(u(pT_j1)) T_n(u(pT_j2)) >_w
  !
  !  Separate moments of pT_j1 and pT_j2 constrain the two SPECTRA but say
  !  nothing about their CORRELATION, and it is the correlation that fixes
  !  dphi_ll: for the diphoton case, constraining the recoil alone left the
  !  shower's pT<->dphi correlation wrong and dphi overshot by ~60%.  A mixed
  !  moment constrains the joint distribution directly.
  !
  !  The two orders are packed into the single imem slot as  imem = -(10*m + n)
  !  with 1 <= m,n <= 9, so bind these as chebTT_ptj12_<m><n>.  Both profiles
  !  multiply, so the feature vanishes unless BOTH jets are above their seam --
  !  which is what makes this a well-defined two-jet observable.
  !===========================================================================
  real(kind=wp) function eval_w_ptj12(npar)      ! joint profile w1*w2
    integer, intent(in) :: npar
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    if (npar < 1) then; eval_w_ptj12 = 0.0_wp; return; end if
    eval_w_ptj12 = prof_w(v1_pt(kin(npar)%pjets(:, 1)), pa, pb, pc, pd) &
                 * prof_w(v1_pt(kin(npar)%pjets(:, 2)), pa, pb, pc, pd)
  end function eval_w_ptj12

  real(kind=wp) function eval_chebTT_ptj12(npar, mn)
    integer, intent(in) :: npar, mn
    real(kind=wp) :: x1, x2, u1, u2, lnlo, lnhi
    real(kind=wp), parameter :: lo = 10.0_wp, hi = 1000.0_wp
    real(kind=wp), parameter :: pa = 30.0_wp, pb = 60.0_wp, pc = 1000.0_wp, pd = 1000.0_wp
    integer :: m, n
    if (npar < 1) then; eval_chebTT_ptj12 = 0.0_wp; return; end if
    m = abs(mn) / 10
    n = mod(abs(mn), 10)
    x1 = max(v1_pt(kin(npar)%pjets(:, 1)), 1.0e-6_wp)
    x2 = max(v1_pt(kin(npar)%pjets(:, 2)), 1.0e-6_wp)
    lnlo = log(lo); lnhi = log(hi)
    u1 = 2.0_wp*(log(x1) - lnlo)/(lnhi - lnlo) - 1.0_wp
    u2 = 2.0_wp*(log(x2) - lnlo)/(lnhi - lnlo) - 1.0_wp
    eval_chebTT_ptj12 = prof_w(x1, pa, pb, pc, pd) * prof_w(x2, pa, pb, pc, pd) &
                      * cheb_T(u1, m) * cheb_T(u2, n)
  end function eval_chebTT_ptj12

  !> x = pi - dphi(l1,l2) in (0,pi]; small x = back-to-back (Sudakov) region.
  pure function pimdphi_ll(npar) result(x)
    integer, intent(in) :: npar
    real(kind=wp) :: x, ph1, ph2, d
    real(kind=wp), parameter :: pi = 3.14159265358979323846_wp
    ph1 = atan2(kin(npar)%p(2, npar - 1), kin(npar)%p(1, npar - 1))
    ph2 = atan2(kin(npar)%p(2, npar), kin(npar)%p(1, npar))
    d = abs(ph1 - ph2)
    if (d > pi) d = 2.0_wp*pi - d
    x = pi - d
  end function pimdphi_ll

  real(kind=wp) function eval_w_dphill(npar)   ! 0 near dphi=pi, 1 away from it
    integer, intent(in) :: npar
    real(kind=wp), parameter :: pa = 0.05_wp, pb = 0.20_wp, pc = 4.0_wp, pd = 4.0_wp
    if (npar < 1) then; eval_w_dphill = 0.0_wp; return; end if
    eval_w_dphill = prof_w(pimdphi_ll(npar), pa, pb, pc, pd)
  end function eval_w_dphill

  real(kind=wp) function eval_chebTw_dphill(npar, n)  ! w*T_n(pi-dphi), log [0.01,pi]
    integer, intent(in) :: npar, n
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 0.01_wp, hi = 3.14159265358979323846_wp
    real(kind=wp), parameter :: pa = 0.05_wp, pb = 0.20_wp, pc = 4.0_wp, pd = 4.0_wp
    if (npar < 1) then; eval_chebTw_dphill = 0.0_wp; return; end if
    x = max(pimdphi_ll(npar), 1.0e-8_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_dphill = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_dphill

  !> SMOOTH-PROFILE recoil moments of (pi - dphi_gg).  pi - dphi is a RECOIL
  !> observable: it vanishes at Born level and is Sudakov-divergent as dphi->pi,
  !> so it may only be constrained ABOVE a resolution scale.  The profile turns
  !> on over pi-dphi in [0.3, 0.6], which is where the pT_gg seam at 28 GeV maps
  !> to for a back-to-back pair of ~45 GeV photons.  Imposing it over the FULL
  !> range instead makes the MaxEnt dual infeasible; see aa_angular_test.py.
  real(kind=wp) function eval_w_dpa(npar)
    integer, intent(in) :: npar
    real(kind=wp), parameter :: pa = 0.3_wp, pb = 0.6_wp, pc = 4.0_wp, pd = 4.0_wp
    if (npar < 1) then; eval_w_dpa = 0.0_wp; return; end if
    eval_w_dpa = prof_w(eval_pi_deltaphi_gam12(npar), pa, pb, pc, pd)
  end function eval_w_dpa

  real(kind=wp) function eval_chebTw_dpa(npar, n)   ! w*T_n(pi-dphi_gg), log [0.01,pi]
    integer, intent(in) :: npar, n
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 0.01_wp, hi = 3.14159265358979323846_wp
    real(kind=wp), parameter :: pa = 0.3_wp, pb = 0.6_wp, pc = 4.0_wp, pd = 4.0_wp
    if (npar < 1) then; eval_chebTw_dpa = 0.0_wp; return; end if
    x = max(eval_pi_deltaphi_gam12(npar), 1.0e-8_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_dpa = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_dpa

  !===========================================================================
  !  Diphoton Chebyshev moments (GG process).  Photons are pT-ORDERED in
  !  kin(npar)%pphotons(:,1..2) -- the same accessor eval_minv_gam12 uses.
  !  Bind with min_nphotons=2.
  !  m_aa Born (log [80,700]); pt_aa recoil with smooth profile [28,56]->500.
  !===========================================================================

  !> T_n of |cos theta*_CS| (diphoton), linear map on [0,1].  imem = n.
  real(kind=wp) function eval_chebT_cts(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp) :: u
    if (npar < 1) then; eval_chebT_cts = 0.0_wp; return; end if
    u = 2.0_wp*eval_abscosthstCS_gam12(npar) - 1.0_wp
    eval_chebT_cts = cheb_T(u, n)
  end function eval_chebT_cts

  !> T_n of (pi - dphi_gg), LOG map on [0.01, pi]: the back-to-back region is
  !> pi-dphi -> 0, so a log map resolves it.  imem = n.
  real(kind=wp) function eval_chebT_dpa(npar, n)
    integer, intent(in) :: npar, n
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 0.01_wp, hi = 3.14159265358979323846_wp
    if (npar < 1) then; eval_chebT_dpa = 0.0_wp; return; end if
    x = max(eval_pi_deltaphi_gam12(npar), 1.0e-8_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    u = max(-1.0_wp, min(1.0_wp, u))   ! out-of-range events sit at the edge
    eval_chebT_dpa = cheb_T(u, n)
  end function eval_chebT_dpa

  real(kind=wp) function eval_chebT_maa(npar, n)    ! m_gg, log [80,700]
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 80.0_wp, hi = 700.0_wp
    if (npar < 1) then; eval_chebT_maa = 0.0_wp; return; end if
    pV(:) = kin(npar)%pphotons(:, 1) + kin(npar)%pphotons(:, 2)
    x = v1_minv(pV)
    u = 2.0_wp*(log(max(x, 1.0e-6_wp)) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    u = max(-1.0_wp, min(1.0_wp, u))   ! out-of-range events sit at the edge
    eval_chebT_maa = cheb_T(u, n)
  end function eval_chebT_maa

  real(kind=wp) function eval_w_ptaa(npar)          ! gg recoil profile w(pt_aa)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp), parameter :: pa = 28.0_wp, pb = 56.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then; eval_w_ptaa = 0.0_wp; return; end if
    pV(:) = kin(npar)%pphotons(:, 1) + kin(npar)%pphotons(:, 2)
    eval_w_ptaa = prof_w(v1_pt(pV), pa, pb, pc, pd)
  end function eval_w_ptaa

  real(kind=wp) function eval_chebTw_ptaa(npar, n)  ! w*T_n(pt_aa), log [1,500], prof [28,56]->500
    integer, intent(in) :: npar, n
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: x, u
    real(kind=wp), parameter :: lo = 1.0_wp, hi = 500.0_wp
    real(kind=wp), parameter :: pa = 28.0_wp, pb = 56.0_wp, pc = 500.0_wp, pd = 500.0_wp
    if (npar < 1) then; eval_chebTw_ptaa = 0.0_wp; return; end if
    pV(:) = kin(npar)%pphotons(:, 1) + kin(npar)%pphotons(:, 2)
    x = max(v1_pt(pV), 1.0e-6_wp)
    u = 2.0_wp*(log(x) - log(lo))/(log(hi) - log(lo)) - 1.0_wp
    eval_chebTw_ptaa = prof_w(x, pa, pb, pc, pd) * cheb_T(u, n)
  end function eval_chebTw_ptaa

  real(kind=wp) function eval_r_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_r_V = v1_pt(pV)/v1_minv(pV)
  end function eval_r_V

  real(kind=wp) function eval_minv_V2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2)
    eval_minv_V2 = v1_minv(pV)
  end function eval_minv_V2

  real(kind=wp) function eval_minv_V4l(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_minv_V4l = v1_minv(pV)
  end function eval_minv_V4l

  real(kind=wp) function eval_minv_V23(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) + kin(npar)%p(1:4, npar - 1)
    eval_minv_V23 = v1_minv(pV)
  end function eval_minv_V23

  real(kind=wp) function eval_minv_V14(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar) + kin(npar)%p(1:4, npar - 3)
    eval_minv_V14 = v1_minv(pV)
  end function eval_minv_V14

  real(kind=wp) function eval_minv_mZ1(npar)
    use Parameters_mod, only: MZ_param
    integer, intent(in) :: npar
    real(kind=wp) :: mV1, mV2
    mV1 = eval_minv_V(npar)
    mV2 = eval_minv_V2(npar)
    if (abs(mV1 - MZ_param()) < abs(mV2 - MZ_param())) then
      eval_minv_mZ1 = mV1
    else
      eval_minv_mZ1 = mV2
    end if
  end function eval_minv_mZ1

  real(kind=wp) function eval_minv_mZ2(npar)
    use Parameters_mod, only: MZ_param
    integer, intent(in) :: npar
    real(kind=wp) :: mV1, mV2
    mV1 = eval_minv_V(npar)
    mV2 = eval_minv_V2(npar)
    if (abs(mV1 - MZ_param()) > abs(mV2 - MZ_param())) then
      eval_minv_mZ2 = mV1
    else
      eval_minv_mZ2 = mV2
    end if
  end function eval_minv_mZ2

  real(kind=wp) function eval_minvSFOS_vs_mZ(npar)
    use Parameters_mod, only: MZ_param
    integer, intent(in) :: npar
    real(kind=wp) :: m2e, m2mu
    m2mu = eval_minv_V(npar)
    m2e = eval_minv_V2(npar)
    if (abs(m2mu - MZ_param()) < abs(m2e - MZ_param())) then
      eval_minvSFOS_vs_mZ = 2
    else
      eval_minvSFOS_vs_mZ = 1
    end if
  end function eval_minvSFOS_vs_mZ

  function eval_minv_mZSFOS(npar, imem) result(res)
    use Parameters_mod, only: MZ_param
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ mZSFOS1st(1), mZSFOS2nd(2), mZSFOS3rd(3), mZSFOS4th(4), mZ2SFOS(5) ]
    real(kind=wp), dimension(5), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    real(kind=wp) :: input_minv_4l(1:4), diff, diff_max
    integer :: i, j, order(1:4), icomp4th

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    !> get all inv. masses
    input_minv_4l(1) = eval_minv_V(npar)
    input_minv_4l(2) = eval_minv_V2(npar)
    input_minv_4l(3) = eval_minv_V23(npar)
    input_minv_4l(4) = eval_minv_V14(npar)
    !> sort
    do i = 1, 4
      diff_max = -1._wp
      order(i) = -1
      do j = 1, 4
        if (any(order(1:i) == j)) cycle
        diff = abs(input_minv_4l(j) - MZ_param())
        if (diff > diff_max) then
          diff_max = diff
          order(i) = j
        end if
      end do
    end do
    !> get the complement pair to "4th"
    select case (order(4))
    case (1)
      icomp4th = 2
    case (2)
      icomp4th = 1
    case (3)
      icomp4th = 4
    case (4)
      icomp4th = 3
    end select

    cache_ = [input_minv_4l(order(1)), input_minv_4l(order(2)), &
              input_minv_4l(order(3)), input_minv_4l(order(4)), &
              input_minv_4l(icomp4th)]

    qcached_ = .true.
    res = cache_(imem)

  end function eval_minv_mZSFOS

  real(kind=wp) function eval_y_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_y_V = v1_y(pV)
  end function eval_y_V

  real(kind=wp) function eval_y_V2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2)
    eval_y_V2 = v1_y(pV)
  end function eval_y_V2

  real(kind=wp) function eval_y_V4l(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_y_V4l = v1_y(pV)
  end function eval_y_V4l

  real(kind=wp) function eval_y_V2l1p(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_y_V2l1p = v1_y(pV)
  end function eval_y_V2l1p

  real(kind=wp) function eval_eta_V(npar) ! not same as y_eta
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_eta_V = v1_eta(pV)
  end function eval_eta_V

  real(kind=wp) function eval_eta_V2(npar) ! not same as y_eta
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2)
    eval_eta_V2 = v1_eta(pV)
  end function eval_eta_V2

  real(kind=wp) function eval_eta_V4l(npar) ! not same as y_eta
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_eta_V4l = v1_eta(pV)
  end function eval_eta_V4l

  real(kind=wp) function eval_eta_V2l1p(npar) ! not same as y_eta
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_eta_V2l1p = v1_eta(pV)
  end function eval_eta_V2l1p

  real(kind=wp) function eval_mt_V(npar)
    integer, intent(in) :: npar
    !> we assume massless daugthers (lepton, neutrino) here
    eval_mt_V = sqrt(2._wp*v1_pt(kin(npar)%p(:, npar - 1))*v1_pt(kin(npar)%p(:, npar)) &
                     *(1._wp - cos(v2_delta_phi(kin(npar)%p(:, npar - 1), kin(npar)%p(:, npar)))))
  end function eval_mt_V

  real(kind=wp) function eval_mt_V2(npar)
    integer, intent(in) :: npar
    !> we assume massless daugthers (lepton, neutrino) here
    eval_mt_V2 = sqrt(2._wp*v1_pt(kin(npar)%p(:, npar - 3))*v1_pt(kin(npar)%p(:, npar - 2)) &
                      *(1._wp - cos(v2_delta_phi(kin(npar)%p(:, npar - 3), kin(npar)%p(:, npar - 2)))))
  end function eval_mt_V2

  ! ET = sqrt(mll^2 + ptz^2)
  real(kind=wp) function eval_et_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_et_V = sqrt(v1_minv2(pV) + v1_pt2(pV))
  end function eval_et_V

  ! ET = sqrt(mll^2 + ptz^2)
  real(kind=wp) function eval_et_V2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2)
    eval_et_V2 = sqrt(v1_minv2(pV) + v1_pt2(pV))
  end function eval_et_V2

  ! ET = sqrt(m3l^2 + pt3l^2)
  real(kind=wp) function eval_et_3l(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_et_3l = sqrt(v1_minv2(pV) + v1_pt2(pV))
  end function eval_et_3l

  ! ET = sqrt(m4l^2 + pt4l^2)
  real(kind=wp) function eval_et_4l(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 3) + kin(npar)%p(1:4, npar - 2) &
              + kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_et_4l = sqrt(v1_minv2(pV) + v1_pt2(pV))
  end function eval_et_4l

  ! xF = 2*pz/sqrt(s)
  real(kind=wp) function eval_xF_V(npar)
    use Process_mod, only: sqrts_proc
    integer, intent(in) :: npar
    real(kind=wp) :: pVz
    pVz = kin(npar)%p(3, npar - 1) + kin(npar)%p(3, npar)
    eval_xF_V = 2d0*pVz/sqrts_proc
  end function eval_xF_V

  ! E = pV(4)
  real(kind=wp) function eval_E_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_E_V = pV(4)
  end function eval_E_V

  ! ptsq=ptz**2
  real(kind=wp) function eval_ptsq_V(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_ptsq_V = v1_pt2(pV)
  end function eval_ptsq_V

  real(kind=wp) function eval_pt_Vjj(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pVjj
    pVjj(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar) &
                + kin(npar)%pjets(1:4, 1) + kin(npar)%pjets(1:4, 2)
    eval_pt_Vjj = v1_pt(pVjj)
  end function eval_pt_Vjj

  real(kind=wp) function eval_pt_jl1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pjl1
    integer :: i, j
    pjl1 = 0._wp
    do i = 1, 4
      do j = 1, kin(npar)%njets
        pjl1(i) = pjl1(i) + kin(npar)%pjets(i, j)
      end do
      pjl1(i) = pjl1(i) + kin(npar)%p(i, npar - 1)
    end do
    eval_pt_jl1 = v1_pt(pjl1)
  end function eval_pt_jl1

  real(kind=wp) function eval_pt_jl2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pjl2
    integer :: i, j
    pjl2 = 0._wp
    do i = 1, 4
      do j = 1, kin(npar)%njets
        pjl2(i) = pjl2(i) + kin(npar)%pjets(i, j)
      end do
      pjl2(i) = pjl2(i) + kin(npar)%p(i, npar)
    end do
    eval_pt_jl2 = v1_pt(pjl2)
  end function eval_pt_jl2

  real(kind=wp) function eval_ht_jl1(npar)
    integer, intent(in) :: npar
    eval_ht_jl1 = eval_pt_jet(npar, 1) + eval_pt_lep(npar, 1)
  end function eval_ht_jl1

  real(kind=wp) function eval_ht_jl2(npar)
    integer, intent(in) :: npar
    eval_ht_jl2 = eval_pt_jet(npar, 1) + eval_pt_lep(npar, 2)
  end function eval_ht_jl2

  real(kind=wp) function eval_pt_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pVj
    pVj(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar) &
               + kin(npar)%pjets(1:4, 1)
    eval_pt_Vj = v1_pt(pVj)
  end function eval_pt_Vj

  real(kind=wp) function eval_ptavg_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ptV, ptj1
    ptV = eval_pt_V(npar)
    ptj1 = eval_pt_jet(npar, 1)
    eval_ptavg_Vj = (ptV + ptj1)/2._wp
  end function eval_ptavg_Vj

  real(kind=wp) function eval_dpt_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ptV, ptj1
    ptV = eval_pt_V(npar)
    ptj1 = eval_pt_jet(npar, 1)
    eval_dpt_Vj = ptV - ptj1
  end function eval_dpt_Vj

  real(kind=wp) function eval_minv_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pVj
    integer :: i
    do i = 1, 4
      pVj(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar) &
               + kin(npar)%pjets(i, 1)
    end do
    eval_minv_Vj = v1_minv(pVj)
  end function eval_minv_Vj

  real(kind=wp) function eval_ystar_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yV, yj1
    yV = eval_y_V(npar)
    yj1 = eval_y_jet(npar, 1)
    eval_ystar_Vj = abs(yV - yj1)/2._wp
  end function eval_ystar_Vj

  real(kind=wp) function eval_yboost_Vj(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yV, yj1
    yV = eval_y_V(npar)
    yj1 = eval_y_jet(npar, 1)
    eval_yboost_Vj = abs(yV + yj1)/2._wp
  end function eval_yboost_Vj

  function eval_ptrat_Vclj(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: dR_Vi, dRmin, ptV, ptj
    integer :: i, idx
    res = 0._wp
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    dRmin = -1._wp
    idx = -1
    do i = 1, kin(npar)%njets
      dR_Vi = v2_delta_R(pV, kin(npar)%pjets(:, i))
      if ((dR_Vi < dRmin) .or. (dRmin < 0._wp)) then
        dRmin = dR_Vi
        idx = i
      end if
    end do
    if (dRmin < 0._wp) return
    ptV = eval_pt_V(npar)
    ptj = eval_pt_jet(npar, idx)
    res = ptV/ptj
  end function eval_ptrat_Vclj

  !----- Compare pt[V+j] with pt[j]

  real(kind=wp) function eval_pt_Vj_vs_pt_j1(npar, ijet)
    integer, intent(in) :: npar, ijet
    real(kind=wp) :: pt_j1, pt_Vj1
    real(kind=wp), dimension(4) :: p_Vj1
    integer :: i
    pt_j1 = v1_pt(kin(npar)%pjets(:, ijet))
    do i = 1, 4
      p_Vj1(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar) &
                 + kin(npar)%pjets(i, ijet)
    end do
    pt_Vj1 = v1_pt(p_Vj1)
    eval_pt_Vj_vs_pt_j1 = pt_Vj1 - pt_j1
  end function eval_pt_Vj_vs_pt_j1

  !----- jets: 1,2,...  <-->  leading, sub-leading, ...

  real(kind=wp) function eval_pt_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    eval_pt_jet = v1_pt(kin(npar)%pjets(:, ijet))
  end function eval_pt_jet

  real(kind=wp) function eval_e_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    eval_e_jet = kin(npar)%pjets(4, ijet)
  end function eval_e_jet

  real(kind=wp) function eval_y_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    eval_y_jet = v1_y(kin(npar)%pjets(:, ijet))
  end function eval_y_jet

  ! after recombination eta_j[i] in general not the same as y_j[i]
  real(kind=wp) function eval_eta_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    eval_eta_jet = v1_eta(kin(npar)%pjets(:, ijet))
  end function eval_eta_jet

  real(kind=wp) function eval_phi_jet(npar, ijet)
    integer, intent(in) :: npar, ijet
    eval_phi_jet = v1_phi(kin(npar)%pjets(:, ijet))
  end function eval_phi_jet

  real(kind=wp) function eval_sum_pt_jets(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_sum_pt_jets = 0._wp
    do i = 1, kin(npar)%njets
      eval_sum_pt_jets = eval_sum_pt_jets &
                         + v1_pt(kin(npar)%pjets(:, i))
    end do
  end function eval_sum_pt_jets

  real(kind=wp) function eval_deltapt_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ptj1, ptj2
    ptj1 = eval_pt_jet(npar, 1)
    ptj2 = eval_pt_jet(npar, 2)
    eval_deltapt_j12 = abs(ptj1 - ptj2)
  end function eval_deltapt_j12

  real(kind=wp) function eval_dR_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj12
    eval_dR_j12 = v2_delta_R(kin(npar)%pjets(1:4, 1), kin(npar)%pjets(1:4, 2))
  end function eval_dR_j12

  real(kind=wp) function eval_pt_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj12
    pj12(1:4) = kin(npar)%pjets(1:4, 1) + kin(npar)%pjets(1:4, 2)
    eval_pt_j12 = v1_pt(pj12)
  end function eval_pt_j12

  real(kind=wp) function eval_deltaeta_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: etaj1, etaj2
    etaj1 = eval_eta_jet(npar, 1)
    etaj2 = eval_eta_jet(npar, 2)
    eval_deltaeta_j12 = abs(etaj1 - etaj2)
  end function eval_deltaeta_j12

  real(kind=wp) function eval_avgeta_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: etaj1, etaj2
    etaj1 = eval_eta_jet(npar, 1)
    etaj2 = eval_eta_jet(npar, 2)
    eval_avgeta_j12 = (etaj1 + etaj2)/2._wp
  end function eval_avgeta_j12

  real(kind=wp) function eval_avgeta_j123(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: etaj1, etaj2, etaj3
    etaj1 = eval_eta_jet(npar, 1)
    etaj2 = eval_eta_jet(npar, 2)
    etaj3 = eval_eta_jet(npar, 3)
    eval_avgeta_j123 = (etaj1 + etaj2 + etaj3)/3._wp
  end function eval_avgeta_j123

  real(kind=wp) function eval_deltay_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yj1, yj2
    yj1 = eval_y_jet(npar, 1)
    yj2 = eval_y_jet(npar, 2)
    eval_deltay_j12 = abs(yj1 - yj2)
  end function eval_deltay_j12

  real(kind=wp) function eval_deltay_j13(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yj1, yj3
    yj1 = eval_y_jet(npar, 1)
    yj3 = eval_y_jet(npar, 3)
    eval_deltay_j13 = abs(yj1 - yj3)
  end function eval_deltay_j13

  real(kind=wp) function eval_deltay_j23(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yj2, yj3
    yj2 = eval_y_jet(npar, 2)
    yj3 = eval_y_jet(npar, 3)
    eval_deltay_j23 = abs(yj2 - yj3)
  end function eval_deltay_j23

  real(kind=wp) function eval_deltaphi_j12(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j12 = v2_delta_phi(kin(npar)%pjets(:, 1), &
                                     kin(npar)%pjets(:, 2))
  end function eval_deltaphi_j12

  function eval_deltaphi_j12_rap_ord(npar) result(res)
    use Constants_mod, only: pi, twopi
    real(kind=wp) :: res
    integer, intent(in) :: npar
    real(kind=wp) :: y1, y2
    y1 = eval_y_jet(npar, 1)
    y2 = eval_y_jet(npar, 2)
    res = eval_phi_jet(npar, 1) - eval_phi_jet(npar, 2)
    if (y2 > y1) res = -res
    !> correct mapping onto [-pi,pi]
    if (res > +pi) res = res - twopi
    if (res < -pi) res = res + twopi
  end function eval_deltaphi_j12_rap_ord

  real(kind=wp) function eval_deltaphi_j13(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j13 = v2_delta_phi(kin(npar)%pjets(:, 1), &
                                     kin(npar)%pjets(:, 3))
  end function eval_deltaphi_j13

  real(kind=wp) function eval_deltaphi_j14(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j14 = v2_delta_phi(kin(npar)%pjets(:, 1), &
                                     kin(npar)%pjets(:, 4))
  end function eval_deltaphi_j14

  real(kind=wp) function eval_deltaphi_j15(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j15 = v2_delta_phi(kin(npar)%pjets(:, 1), &
                                     kin(npar)%pjets(:, 5))
  end function eval_deltaphi_j15

  real(kind=wp) function eval_deltaphi_j23(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j23 = v2_delta_phi(kin(npar)%pjets(:, 2), &
                                     kin(npar)%pjets(:, 3))
  end function eval_deltaphi_j23

  real(kind=wp) function eval_deltaphi_j24(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j24 = v2_delta_phi(kin(npar)%pjets(:, 2), &
                                     kin(npar)%pjets(:, 4))
  end function eval_deltaphi_j24

  real(kind=wp) function eval_deltaphi_j25(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j25 = v2_delta_phi(kin(npar)%pjets(:, 2), &
                                     kin(npar)%pjets(:, 5))
  end function eval_deltaphi_j25

  real(kind=wp) function eval_deltaphi_j34(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j34 = v2_delta_phi(kin(npar)%pjets(:, 3), &
                                     kin(npar)%pjets(:, 4))
  end function eval_deltaphi_j34

  real(kind=wp) function eval_deltaphi_j35(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j35 = v2_delta_phi(kin(npar)%pjets(:, 3), &
                                     kin(npar)%pjets(:, 5))
  end function eval_deltaphi_j35

  real(kind=wp) function eval_deltaphi_j45(npar)
    integer, intent(in) :: npar
    eval_deltaphi_j45 = v2_delta_phi(kin(npar)%pjets(:, 4), &
                                     kin(npar)%pjets(:, 5))
  end function eval_deltaphi_j45

  real(kind=wp) function eval_chi_j12(npar)
    integer, intent(in) :: npar
    eval_chi_j12 = exp(eval_deltay_j12(npar))
  end function eval_chi_j12

  real(kind=wp) function eval_ystar_j12(npar)
    integer, intent(in) :: npar
    eval_ystar_j12 = eval_deltay_j12(npar)/2._wp
  end function eval_ystar_j12

  real(kind=wp) function eval_yboost_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yj1, yj2
    yj1 = eval_y_jet(npar, 1)
    yj2 = eval_y_jet(npar, 2)
    eval_yboost_j12 = abs(yj1 + yj2)/2._wp
  end function eval_yboost_j12

  real(kind=wp) function eval_max_abs_y_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: yj1, yj2
    yj1 = abs(eval_y_jet(npar, 1))
    yj2 = abs(eval_y_jet(npar, 2))
    eval_max_abs_y_j12 = max(yj1, yj2)
  end function eval_max_abs_y_j12

  real(kind=wp) function eval_minv_j12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj12
    pj12(1:4) = kin(npar)%pjets(1:4, 1) + kin(npar)%pjets(1:4, 2)
    eval_minv_j12 = v1_minv(pj12)
  end function eval_minv_j12

  real(kind=wp) function eval_minv_j45(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj45
    pj45(1:4) = kin(npar)%pjets(1:4, 4) + kin(npar)%pjets(1:4, 5)
    eval_minv_j45 = v1_minv(pj45)
  end function eval_minv_j45

  function eval_jet_fb(npar, imem) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ del_y(1), del_phi(2), del_eta(3) ]
    real(kind=wp), dimension(3), save :: &
      cache_ = [0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    real(kind=wp) :: del_y, del_phi, del_eta
    integer :: i, if, ib
    real(kind=wp) :: y, yf, yb

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    !> find the most forward (largest y) & backward (smallest y) jets
    if = -1
    ib = -1
    do i = 1, kd%njets
      y = eval_y_jet(npar, i)
      if (if < 0 .or. y > yf) then
        if = i
        yf = y
      end if
      if (ib < 0 .or. y < yb) then
        ib = i
        yb = y
      end if
    end do

    del_y = abs(yf - yb)
    del_phi = v2_delta_phi(kd%pjets(:, if), kd%pjets(:, ib))
    del_eta = abs(eval_eta_jet(npar, if) - eval_eta_jet(npar, ib))

    cache_ = [del_y, del_phi, del_eta]

    qcached_ = .true.
    res = cache_(imem)

  end function eval_jet_fb

  real(kind=wp) function eval_pt_j1expYstar(npar)
    integer, intent(in) :: npar
    eval_pt_j1expYstar = v1_pt(kin(npar)%pjets(:, 1))*exp(0.3_wp*eval_ystar_j12(npar))
  end function eval_pt_j1expYstar

  real(kind=wp) function eval_minv_j123(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj123
    pj123(1:4) = kin(npar)%pjets(1:4, 1) &
                 + kin(npar)%pjets(1:4, 2) &
                 + kin(npar)%pjets(1:4, 3)
    eval_minv_j123 = v1_minv(pj123)
  end function eval_minv_j123

  !> definition from 1411.1855
  real(kind=wp) function eval_ystar_j123(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: abs_del_y_j12, abs_del_y_j23, abs_del_y_j13
    abs_del_y_j12 = eval_deltay_j12(npar)
    abs_del_y_j23 = eval_deltay_j23(npar)
    abs_del_y_j13 = eval_deltay_j13(npar)
    eval_ystar_j123 = abs_del_y_j12 + abs_del_y_j23 + abs_del_y_j13
  end function eval_ystar_j123

  !> definition from 1412.1633
  real(kind=wp) function eval_ymax_j123(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y_j1, y_j2, y_j3
    real(kind=wp) :: max_abs_y, max_y, min_y
    y_j1 = eval_y_jet(npar, 1)
    y_j2 = eval_y_jet(npar, 2)
    y_j3 = eval_y_jet(npar, 3)
    max_abs_y = max(abs(y_j1), abs(y_j2), abs(y_j3))
    max_y = max(y_j1, y_j2, y_j3)
    min_y = min(y_j1, y_j2, y_j3)
    eval_ymax_j123 = sign(max_abs_y, abs(max_y) - abs(min_y))
  end function eval_ymax_j123

  real(kind=wp) function eval_ptavg_j12(npar)
    integer, intent(in) :: npar
    eval_ptavg_j12 = (v1_pt(kin(npar)%pjets(:, 1)) &
                      + v1_pt(kin(npar)%pjets(:, 2)))/2._wp
  end function eval_ptavg_j12

  real(kind=wp) function eval_ptavg_j123(npar)
    integer, intent(in) :: npar
    eval_ptavg_j123 = (v1_pt(kin(npar)%pjets(:, 1)) &
                       + v1_pt(kin(npar)%pjets(:, 2)) &
                       + v1_pt(kin(npar)%pjets(:, 3)))/3._wp
  end function eval_ptavg_j123

  real(kind=wp) function eval_ptavg_jall(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ptavg_jall = 0._wp
    do i = 1, kin(npar)%njets
      eval_ptavg_jall = eval_ptavg_jall + v1_pt(kin(npar)%pjets(:, i))
    end do
    eval_ptavg_jall = eval_ptavg_jall/DBLE(kin(npar)%njets)
  end function eval_ptavg_jall

  real(kind=wp) function eval_ptavg_geom_j12(npar)
    integer, intent(in) :: npar
    eval_ptavg_geom_j12 = sqrt(v1_pt(kin(npar)%pjets(:, 1)) &
                               *v1_pt(kin(npar)%pjets(:, 2)))
  end function eval_ptavg_geom_j12

  real(kind=wp) function eval_ptavg_geom_jall(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ptavg_geom_jall = 1._wp
    do i = 1, kin(npar)%njets
      eval_ptavg_geom_jall = eval_ptavg_geom_jall*v1_pt(kin(npar)%pjets(:, i))
    end do
    eval_ptavg_geom_jall = eval_ptavg_geom_jall**(1._wp/DBLE(kin(npar)%njets))
  end function eval_ptavg_geom_jall

  real(kind=wp) function eval_etavg_j12(npar)
    integer, intent(in) :: npar
    eval_etavg_j12 = (v1_et(kin(npar)%pjets(:, 1)) &
                      + v1_et(kin(npar)%pjets(:, 2)))/2._wp
  end function eval_etavg_j12

  real(kind=wp) function eval_etavg_j123(npar)
    integer, intent(in) :: npar
    eval_etavg_j123 = (v1_et(kin(npar)%pjets(:, 1)) &
                       + v1_et(kin(npar)%pjets(:, 2)) &
                       + v1_et(kin(npar)%pjets(:, 3)))/3._wp
  end function eval_etavg_j123

  real(kind=wp) function eval_etavg_jall(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_etavg_jall = 0._wp
    do i = 1, kin(npar)%njets
      eval_etavg_jall = eval_etavg_jall + v1_et(kin(npar)%pjets(:, i))
    end do
    eval_etavg_jall = eval_etavg_jall/DBLE(kin(npar)%njets)
  end function eval_etavg_jall

  real(kind=wp) function eval_etavg_geom_jall(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_etavg_geom_jall = 1._wp
    do i = 1, kin(npar)%njets
      eval_etavg_geom_jall = eval_etavg_geom_jall*v1_et(kin(npar)%pjets(:, i))
    end do
    eval_etavg_geom_jall = eval_etavg_geom_jall**(1._wp/DBLE(kin(npar)%njets))
  end function eval_etavg_geom_jall

  real(kind=wp) function eval_et2_j12(npar)
    integer, intent(in) :: npar
    eval_et2_j12 = (v1_et(kin(npar)%pjets(:, 1))**2 &
                    + v1_et(kin(npar)%pjets(:, 2))**2)
  end function eval_et2_j12

  !> photon observables

  !----- photons: 1,2,...  <-->  leading, sub-leading, ...

  real(kind=wp) function eval_pt_gam(npar, ith)
    integer, intent(in) :: npar, ith
    eval_pt_gam = v1_pt(kin(npar)%pphotons(:, ith))
  end function eval_pt_gam

  real(kind=wp) function eval_e_gam(npar, ith)
    integer, intent(in) :: npar, ith
    eval_e_gam = kin(npar)%pphotons(4, ith)
  end function eval_e_gam

  real(kind=wp) function eval_y_gam(npar, ith)
    integer, intent(in) :: npar, ith
    eval_y_gam = v1_y(kin(npar)%pphotons(:, ith))
  end function eval_y_gam

  real(kind=wp) function eval_minv_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, 1) &
              + kin(npar)%pphotons(:, 2)
    eval_minv_gam12 = v1_minv(psum)
  end function eval_minv_gam12

  real(kind=wp) function eval_pt_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, 1) &
              + kin(npar)%pphotons(:, 2)
    eval_pt_gam12 = v1_pt(psum)
  end function eval_pt_gam12

!> MM: commenting this out to avoid circular dependence
!   real(kind=wp) function eval_pt_iso_gam12(npar)
!     use PhotonIsolation_mod
!     integer, intent(in) :: npar
!     real(kind=wp), dimension(4) :: psum
!     integer :: j, j_kd
!     type(Isol_t), pointer :: isol
!
!     psum(:) = kin(npar)%pphotons(:, 1) &
!               + kin(npar)%pphotons(:, 2)
!
!     !> WF: start loop over partons
!     do j = 1, kin(npar)%nproto
!       j_kd = kin(npar)%ijets_lower + j - 1
!       !> WF: check if the parton is in either isolation cone
!       !> WF: if yes, add the parton to the qT_gammagamma
!       !> WF: the following logic assumes the event has passed the isolation criteria
!       if ( (v2_delta_R( kin(npar)%pphotons(:, 1) , kin(npar)%p(:,j_kd) ) < isol%R)) then
!         psum(:) = psum(:) + kin(npar)%p(:,j_kd)
!       end if
!       if ( (v2_delta_R( kin(npar)%pphotons(:, 2) , kin(npar)%p(:,j_kd) ) < isol%R)) then
!         psum(:) = psum(:) + kin(npar)%p(:,j_kd)
!       end if
!     end do  ! j <-> partons
!
!     eval_pt_iso_gam12 = v1_pt(psum)
!   end function eval_pt_iso_gam12

  real(kind=wp) function eval_r_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, 1) &
              + kin(npar)%pphotons(:, 2)
    eval_r_gam12 = v1_pt(psum)/v1_minv(psum)
  end function eval_r_gam12

  real(kind=wp) function eval_ptavg_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt1, pt2
    pt1 = eval_pt_gam(npar, 1)
    pt2 = eval_pt_gam(npar, 2)
    eval_ptavg_gam12 = (pt1 + pt2)/2._wp
  end function eval_ptavg_gam12

  real(kind=wp) function eval_ptavg_geom_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt1, pt2
    pt1 = eval_pt_gam(npar, 1)
    pt2 = eval_pt_gam(npar, 2)
    eval_ptavg_geom_gam12 = sqrt(pt1*pt2)
  end function eval_ptavg_geom_gam12

  real(kind=wp) function eval_y_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, 1) &
              + kin(npar)%pphotons(:, 2)
    eval_y_gam12 = v1_y(psum)
  end function eval_y_gam12

  ! Common mur and muf scale choice for di-photon
  real(kind=wp) function eval_minv_pt_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: minv, pt
    minv = eval_minv_gam12(npar)
    pt = eval_pt_gam12(npar)
    eval_minv_pt_gam12 = sqrt(minv**2 + pt**2)/4._wp
  end function eval_minv_pt_gam12

  real(kind=wp) function eval_Et_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: minv, pt
    minv = eval_minv_gam12(npar)
    pt = eval_pt_gam12(npar)
    eval_Et_gam12 = sqrt(minv**2 + pt**2)
  end function eval_Et_gam12

  ! \phi^*_\eta
  ! defined in Sect. 3 of arXiv:1009.1580 [hep-ex]
  real(kind=wp) function eval_phi_star_gam12(npar)
    use Constants_mod
    integer, intent(in) :: npar
    real(kind=wp) :: del_phi, del_eta, cos_th, sin_th
    del_phi = eval_deltaphi_gam12(npar)
    del_eta = eval_deltay_gam12(npar)  ! massless: eta = y
    cos_th = tanh(del_eta/2._wp)
    sin_th = sqrt((1._wp + cos_th)*(1._wp - cos_th))
    eval_phi_star_gam12 = tan((pi - del_phi)/2._wp)*sin_th
  end function eval_phi_star_gam12

  real(kind=wp) function eval_deltaphi_gam12(npar)
    integer, intent(in) :: npar
    eval_deltaphi_gam12 = &
      v2_delta_phi(kin(npar)%pphotons(:, 1), &
                   kin(npar)%pphotons(:, 2))
  end function eval_deltaphi_gam12

  real(kind=wp) function eval_pi_deltaphi_gam12(npar)
    use Constants_mod, only: pi
    integer, intent(in) :: npar
    eval_pi_deltaphi_gam12 = pi - eval_deltaphi_gam12(npar)
  end function eval_pi_deltaphi_gam12

  real(kind=wp) function eval_deltay_gam12(npar)
    integer, intent(in) :: npar
    eval_deltay_gam12 = &
      v2_delta_y(kin(npar)%pphotons(:, 1), &
                 kin(npar)%pphotons(:, 2))
  end function eval_deltay_gam12

  real(kind=wp) function eval_abs_deltay_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: deltay_gam12
    deltay_gam12 = eval_deltay_gam12(npar)
    eval_abs_deltay_gam12 = abs(deltay_gam12)
  end function eval_abs_deltay_gam12

  real(kind=wp) function eval_deltaR_gam12(npar)
    integer, intent(in) :: npar
    eval_deltaR_gam12 = &
      v2_delta_R(kin(npar)%pphotons(:, 1), &
                 kin(npar)%pphotons(:, 2))
  end function eval_deltaR_gam12

  real(kind=wp) function eval_costhst_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y_g1, y_g2
    y_g1 = eval_y_gam(npar, 1)  ! leading
    y_g2 = eval_y_gam(npar, 2)  ! sub-leading
    eval_costhst_gam12 = abs(tanh((y_g1 - y_g2)/2._wp))
  end function eval_costhst_gam12

  real(kind=wp) function eval_abscosthst_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y_g1, y_g2
    y_g1 = eval_y_gam(npar, 1)  ! leading
    y_g2 = eval_y_gam(npar, 2)  ! sub-leading
    eval_abscosthst_gam12 = tanh(abs((y_g1 - y_g2))/2._wp)
  end function eval_abscosthst_gam12

  real(kind=wp) function eval_costhstCS_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt12, m12
    real(kind=wp) :: p1p, p1m, p2p, p2m
    pt12 = eval_pt_gam12(npar)
    m12 = eval_minv_gam12(npar)
    p1p = kin(npar)%pphotons(4, 1) + kin(npar)%pphotons(3, 1)
    p1m = kin(npar)%pphotons(4, 1) - kin(npar)%pphotons(3, 1)
    p2p = kin(npar)%pphotons(4, 2) + kin(npar)%pphotons(3, 2)
    p2m = kin(npar)%pphotons(4, 2) - kin(npar)%pphotons(3, 2)
    eval_costhstCS_gam12 = ((p1p*p2m) - (p1m*p2p))/m12/sqrt(m12**2 + pt12**2)
  end function eval_costhstCS_gam12

  real(kind=wp) function eval_abscosthstCS_gam12(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pt12, m12
    real(kind=wp) :: p1p, p1m, p2p, p2m
    pt12 = eval_pt_gam12(npar)
    m12 = eval_minv_gam12(npar)
    p1p = kin(npar)%pphotons(4, 1) + kin(npar)%pphotons(3, 1)
    p1m = kin(npar)%pphotons(4, 1) - kin(npar)%pphotons(3, 1)
    p2p = kin(npar)%pphotons(4, 2) + kin(npar)%pphotons(3, 2)
    p2m = kin(npar)%pphotons(4, 2) - kin(npar)%pphotons(3, 2)
    eval_abscosthstCS_gam12 = abs((p1p*p2m) - (p1m*p2p))/m12/sqrt(m12**2 + pt12**2)
  end function eval_abscosthstCS_gam12

  real(kind=wp) function eval_ptt_gam12(npar)
    integer, intent(in) :: npar
    !> in the ATLAS paper, it sounds like taking the 3-momenta but that's wrong.
    !> only the transverse component (x,y)
    real(kind=wp) :: pP(2), pM(2), vpPM, vpMM
    integer :: i
    pP = kin(npar)%pphotons(1:2, 1) + kin(npar)%pphotons(1:2, 2)
    pM = kin(npar)%pphotons(1:2, 1) - kin(npar)%pphotons(1:2, 2)
    vpPM = 0._wp
    vpMM = 0._wp
    do i = 1, 2
      vpPM = vpPM + pP(i)*pM(i)
      vpMM = vpMM + pM(i)**2
    end do
    pP = pP - (vpPM/vpMM)*pM  ! orthogonal component w.r.t. pM axis
    eval_ptt_gam12 = 0._wp
    do i = 1, 2
      eval_ptt_gam12 = eval_ptt_gam12 + pP(i)**2
    end do
    eval_ptt_gam12 = sqrt(eval_ptt_gam12)
  end function eval_ptt_gam12

  ! transverse component of p_Tgg wrt thrust axis of diphoton system - this is what ATLAS calls a_T
  real(kind=wp) function eval_absptt_gam12(npar)
    integer, intent(in) :: npar
    type(Kin_t), pointer :: kd
    real(kind=wp) :: abs_t
    kd => kin(npar)
    ! ordering gives relative minus sign; taking absolute value at end
    abs_t = sqrt((kd%pphotons(1, 1) - kd%pphotons(1, 2))**2 + (kd%pphotons(2, 1) - kd%pphotons(2, 2))**2)
    eval_absptt_gam12 = 2._wp*(kd%pphotons(1, 1)*kd%pphotons(2, 2) - kd%pphotons(2, 1)*kd%pphotons(1, 2))
    eval_absptt_gam12 = abs(eval_absptt_gam12)/abs_t
  end function eval_absptt_gam12

  ! absolute value of a_T variable defined in arXiv:0807.4956 - i.e. |ptt| (above) for dphigg > pi/2, and |pt_gg| for dphigg < pi/2
  real(kind=wp) function eval_absat_gam12(npar)
    use Constants_mod
    integer, intent(in) :: npar
    real(kind=wp) :: dphi
    dphi = eval_deltaphi_gam12(npar)
    if (dphi .gt. pi/2._wp) then
      eval_absat_gam12 = eval_absptt_gam12(npar)
    else
      eval_absat_gam12 = eval_pt_gam12(npar)
    end if
  end function eval_absat_gam12

  !>---- photon(s) - j1 observables

  real(kind=wp) function eval_deltaphi_gamj1(npar, ith)
    integer, intent(in) :: npar, ith
    eval_deltaphi_gamj1 = &
      v2_delta_phi(kin(npar)%pphotons(:, ith), kin(npar)%pjets(:, 1))
  end function eval_deltaphi_gamj1

  real(kind=wp) function eval_minv_gamj1(npar, ith)
    integer, intent(in) :: npar, ith
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, ith) &
              + kin(npar)%pjets(:, 1)
    eval_minv_gamj1 = v1_minv(psum)
  end function eval_minv_gamj1

  real(kind=wp) function eval_costhstar_gamj1(npar, ith)
    integer, intent(in) :: npar, ith
    real(kind=wp) :: y_gam, y_j1
    y_gam = eval_y_gam(npar, ith)
    y_j1 = eval_y_jet(npar, 1)
    eval_costhstar_gamj1 = tanh((y_j1 - y_gam)/2._wp)
  end function eval_costhstar_gamj1

  real(kind=wp) function eval_yPlus_gamj1(npar, ith)
    integer, intent(in) :: npar, ith
    real(kind=wp) :: y_gam, y_j1
    y_gam = eval_y_gam(npar, ith)
    y_j1 = eval_y_jet(npar, 1)
    eval_yPlus_gamj1 = y_gam + y_j1
  end function eval_yPlus_gamj1

  real(kind=wp) function eval_yMinus_gamj1(npar, ith)
    integer, intent(in) :: npar, ith
    real(kind=wp) :: y_gam, y_j1
    y_gam = eval_y_gam(npar, ith)
    y_j1 = eval_y_jet(npar, 1)
    eval_yMinus_gamj1 = y_gam - y_j1
  end function eval_yMinus_gamj1

  real(kind=wp) function eval_min_dR_gamj(npar)
    integer, intent(in) :: npar
    integer :: igam, ijet
    real(kind=wp) :: dR
    eval_min_dR_gamj = -1._wp
    do igam = 1, kin(npar)%nphotons
    do ijet = 1, kin(npar)%njets
      dR = v2_delta_R(kin(npar)%pphotons(:, igam), &
                      kin(npar)%pjets(:, ijet))
      if ((eval_min_dR_gamj < 0._wp) .or. (dR < eval_min_dR_gamj)) eval_min_dR_gamj = dR
    end do
    end do
  end function eval_min_dR_gamj

  real(kind=wp) function eval_min_dR_exp_gam1j1(npar)
    integer, intent(in) :: npar
    eval_min_dR_exp_gam1j1 = &
      v2_delta_R_exp(kin(npar)%pphotons(:, 1), kin(npar)%pjets(:, 1))
  end function eval_min_dR_exp_gam1j1

  ! scalar pt sum over photon and all final state particles
  real(kind=wp) function eval_htpart_photon(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_htpart_photon = 0._wp
    do i = 3, npar
      eval_htpart_photon = eval_htpart_photon + v1_pt(kin(npar)%p(:, i))
    end do
  end function eval_htpart_photon

  real(kind=wp) function eval_ht_g1g2(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ht_g1g2 = 4._wp*eval_minv_pt_gam12(npar)
    do i = kin(npar)%ijets_lower, kin(npar)%ijets_upper
      eval_ht_g1g2 = eval_ht_g1g2 + v1_pt(kin(npar)%p(:, i))
    end do
  end function eval_ht_g1g2

  real(kind=wp) function eval_imbalance_pt_gam1j1(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ptj, ptgam
    eval_imbalance_pt_gam1j1 = 0._wp
    ptj = eval_pt_jet(npar, 1)
    ptgam = eval_pt_gam(npar, 1)
    eval_imbalance_pt_gam1j1 = ptgam/ptj
  end function eval_imbalance_pt_gam1j1

  real(kind=wp) function eval_tau_j1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj1
    real(kind=wp), dimension(4) :: ph
    pj1(1:4) = kin(npar)%pjets(1:4, 1)
    ph(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_tau_j1 = sqrt(v1_pt2(pj1) + v1_minv2(pj1))/2/dcosh(v1_y(pj1) - v1_y(ph))
  end function eval_tau_j1

  real(kind=wp) function eval_sum_tau_jets(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp), dimension(4) :: pjet
    real(kind=wp), dimension(4) :: ph
    ph(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_sum_tau_jets = 0._wp
    do i = 1, kin(npar)%njets
      pjet(1:4) = kin(npar)%pjets(1:4, i)
      eval_sum_tau_jets = eval_sum_tau_jets &
                          + sqrt(v1_pt2(pjet) + v1_minv2(pjet))/2/dcosh(v1_y(pjet) - v1_y(ph))
    end do
  end function eval_sum_tau_jets

  real(kind=wp) function eval_max_tau_jet(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: eval_tau_jet
    real(kind=wp), dimension(4) :: pjet
    real(kind=wp), dimension(4) :: ph
    ph(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    eval_max_tau_jet = 0._wp
    eval_tau_jet = 0._wp
    do i = 1, kin(npar)%njets
      pjet(1:4) = kin(npar)%pjets(1:4, kin(npar)%njets)
      eval_tau_jet = sqrt(v1_pt2(pjet) + v1_minv2(pjet))/2/dcosh(v1_y(pjet) - v1_y(ph))
      if (eval_max_tau_jet .le. eval_tau_jet) then
        eval_max_tau_jet = eval_tau_jet
      end if
    end do
  end function eval_max_tau_jet

  !> y_{n n+1} ycut transitions in the JADE algorithm
  !> this is *before* any jet cuts are applied!!!
  function eval_jet_ytrans(npar, imem) result(res)
    use Jets_mod
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ y01(1), y12(2), y23(3), y34(4), y45(5) ]
    real(kind=wp), dimension(5), save :: &
      cache_ = [0._wp, 0._wp, 0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    !> initialise the recombination algorithm
    call initRecomb_jet(npar)

    !> run clustering till the end & compute y_{n n+1}
    call cluster_jade_jet(compute_ytrans=.true.)
    cache_ = [value_y01, value_y12, value_y23, value_y34, value_y45]

    !> reset the Jets_mod local state for good measure
    call initRecomb_jet(npar)

    qcached_ = .true.
    res = cache_(imem)

  end function eval_jet_ytrans

  !> determine the momentum fractions entering the PDFs

  function eval_x1_pdf(npar) result(res)
    use Process_mod, only: nPDFs_proc
    real(kind=wp) :: res
    integer, intent(in) :: npar
    integer :: imethod
    common/method/imethod
    integer :: igx
    common/gridix/igx
!$omp threadprivate(/gridix/)
    res = 0._wp
    if (nPDFs_proc > 0) then
      if (imethod == 2) then
        select case (igx)
        case (1, 2)
          res = x1_kin
        case (3, 4)
          res = x1_kin/xreg1_kin
        end select
      else
        res = x1_kin
      end if
    end if
  end function eval_x1_pdf

  function eval_x2_pdf(npar) result(res)
    use Process_mod, only: nPDFs_proc
    real(kind=wp) :: res
    integer, intent(in) :: npar
    integer :: imethod
    common/method/imethod
    integer :: igx
    common/gridix/igx
!$omp threadprivate(/gridix/)
    res = 0._wp
    if (nPDFs_proc > 1) then
      if (imethod == 2) then
        select case (igx)
        case (1, 3)
          res = x2_kin
        case (2, 4)
          res = x2_kin/xreg2_kin
        end select
      else
        res = x2_kin
      end if
    end if
  end function eval_x2_pdf

  !----- more complicated guys

  ! scalar pt sum over reconstructed jets
  real(kind=wp) function eval_ht_jets(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ht_jets = 0._wp
    do i = 1, kin(npar)%njets
      eval_ht_jets = eval_ht_jets + v1_pt(kin(npar)%pjets(:, i))
    end do
  end function eval_ht_jets

  ! scalar pt sum over the partons
  real(kind=wp) function eval_ht_part(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ht_part = 0._wp
    do i = kin(npar)%ijets_lower, kin(npar)%ijets_upper
      eval_ht_part = eval_ht_part + v1_pt(kin(npar)%p(:, i))
    end do
  end function eval_ht_part

  ! pt-weighted parton rapidities
  real(kind=wp) function eval_yb_part(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: ht_part
    ht_part = eval_ht_part(npar)
    eval_yb_part = 0._wp
    do i = kin(npar)%ijets_lower, kin(npar)%ijets_upper
      eval_yb_part = eval_yb_part + v1_pt(kin(npar)%p(:, i))*v1_y(kin(npar)%p(:, i))
    end do
    eval_yb_part = eval_yb_part/ht_part
  end function eval_yb_part

  ! pt-weighted parton rapidities in part CMS
  real(kind=wp) function eval_ys_part(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: ht_part, yb_part
    ht_part = eval_ht_part(npar)
    yb_part = eval_yb_part(npar)
    eval_ys_part = 0._wp
    do i = kin(npar)%ijets_lower, kin(npar)%ijets_upper
      eval_ys_part = eval_ys_part &
                     + v1_pt(kin(npar)%p(:, i))*abs(v1_y(kin(npar)%p(:, i)) - yb_part)
    end do
    eval_ys_part = eval_ys_part/ht_part
  end function eval_ys_part

  ! scalar pt sum over leptons and reconstructed jets
  real(kind=wp) function eval_ht_full(npar)
    integer, intent(in) :: npar
    integer :: i
    eval_ht_full = v1_pt(kin(npar)%p(:, npar - 1)) + v1_pt(kin(npar)%p(:, npar))
    do i = 1, kin(npar)%njets
      eval_ht_full = eval_ht_full + v1_pt(kin(npar)%pjets(:, i))
    end do
  end function eval_ht_full

  ! smallest angular separation between lepton lm and photon p3
  real(kind=wp) function eval_min_dR_l1p3(npar)
    integer, intent(in) :: npar
    eval_min_dR_l1p3 = v2_delta_R(kin(npar)%p(:, npar - 2), kin(npar)%p(:, npar))
  end function eval_min_dR_l1p3

  ! smallest angular separation between lepton lp and photon p3
  real(kind=wp) function eval_min_dR_l2p3(npar)
    integer, intent(in) :: npar
    eval_min_dR_l2p3 = v2_delta_R(kin(npar)%p(:, npar - 1), kin(npar)%p(:, npar))
  end function eval_min_dR_l2p3

  ! smallest angular separation between reconstructed leading jet and l1
  real(kind=wp) function eval_min_dR_l1j1(npar)
    integer, intent(in) :: npar
    eval_min_dR_l1j1 = v2_delta_R(kin(npar)%p(:, npar - 1), kin(npar)%pjets(:, 1))
  end function eval_min_dR_l1j1

  ! smallest angular separation between reconstructed leading jet and l2
  real(kind=wp) function eval_min_dR_l2j1(npar)
    integer, intent(in) :: npar
    eval_min_dR_l2j1 = v2_delta_R(kin(npar)%p(:, npar), kin(npar)%pjets(:, 1))
  end function eval_min_dR_l2j1

  ! smallest angular separation between reconstructed jets and l1
  real(kind=wp) function eval_min_dR_l1j(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: dR_l1i
    eval_min_dR_l1j = -1._wp
    do i = 1, kin(npar)%njets
      dR_l1i = v2_delta_R(kin(npar)%p(:, npar - 1), kin(npar)%pjets(:, i))
      if ((dR_l1i < eval_min_dR_l1j) .or. (eval_min_dR_l1j < 0._wp)) then
        eval_min_dR_l1j = dR_l1i
      end if
    end do
  end function eval_min_dR_l1j

  ! smallest angular separation between reconstructed jets and l2
  real(kind=wp) function eval_min_dR_l2j(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: dR_l2i
    eval_min_dR_l2j = -1._wp
    do i = 1, kin(npar)%njets
      dR_l2i = v2_delta_R(kin(npar)%p(:, npar), kin(npar)%pjets(:, i))
      if ((dR_l2i < eval_min_dR_l2j) .or. (eval_min_dR_l2j < 0._wp)) then
        eval_min_dR_l2j = dR_l2i
      end if
    end do
  end function eval_min_dR_l2j

  ! smallest angular separation between reconstructed jets and l3
  real(kind=wp) function eval_min_dR_l3j(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: dR_l3i
    eval_min_dR_l3j = -1._wp
    do i = 1, kin(npar)%njets
      dR_l3i = v2_delta_R(kin(npar)%p(:, npar - 3), kin(npar)%pjets(:, i))
      if ((dR_l3i < eval_min_dR_l3j) .or. (eval_min_dR_l3j < 0._wp)) then
        eval_min_dR_l3j = dR_l3i
      end if
    end do
  end function eval_min_dR_l3j

  ! smallest angular separation between reconstructed jets and l4
  real(kind=wp) function eval_min_dR_l4j(npar)
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: dR_l4i
    eval_min_dR_l4j = -1._wp
    do i = 1, kin(npar)%njets
      dR_l4i = v2_delta_R(kin(npar)%p(:, npar - 2), kin(npar)%pjets(:, i))
      if ((dR_l4i < eval_min_dR_l4j) .or. (eval_min_dR_l4j < 0._wp)) then
        eval_min_dR_l4j = dR_l4i
      end if
    end do
  end function eval_min_dR_l4j

  ! smallest angular separation between leptons lm/lp and photon p3
  real(kind=wp) function eval_min_dR_l12p3(npar)
    integer, intent(in) :: npar
    eval_min_dR_l12p3 = min(eval_min_dR_l1p3(npar), eval_min_dR_l2p3(npar))
  end function eval_min_dR_l12p3

  ! smallest angular separation between reconstructed leading jet and l1/l2
  real(kind=wp) function eval_min_dR_l12j1(npar)
    integer, intent(in) :: npar
    eval_min_dR_l12j1 = min(eval_min_dR_l1j1(npar), eval_min_dR_l2j1(npar))
  end function eval_min_dR_l12j1

  ! smallest angular separation between reconstructed jets and l1/l2
  real(kind=wp) function eval_min_dR_l12j(npar)
    integer, intent(in) :: npar
    eval_min_dR_l12j = min(eval_min_dR_l1j(npar), eval_min_dR_l2j(npar))
  end function eval_min_dR_l12j

  ! smallest angular separation between reconstructed jets and l3/l4
  real(kind=wp) function eval_min_dR_l34j(npar)
    integer, intent(in) :: npar
    eval_min_dR_l34j = min(eval_min_dR_l3j(npar), eval_min_dR_l4j(npar))
  end function eval_min_dR_l34j

  ! smallest angular separation between reconstructed jets and l1/l2/l3/l4
  real(kind=wp) function eval_min_dR_l1234j(npar)
    integer, intent(in) :: npar
    eval_min_dR_l1234j = min(eval_min_dR_l1j(npar), eval_min_dR_l2j(npar) &
                             , eval_min_dR_l3j(npar), eval_min_dR_l4j(npar))
  end function eval_min_dR_l1234j

  !> smallest angular separation between reconstructed jets and V
  !> also sets the index of the jet that is closes to "the V"
  function eval_min_dR_Vj(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i, idx
    real(kind=wp) :: dR_Vi
    pV(1:4) = kin(npar)%p(1:4, npar - 1) + kin(npar)%p(1:4, npar)
    res = -1._wp
    idx = -1
    do i = 1, kin(npar)%njets
      dR_Vi = v2_delta_R(pV, kin(npar)%pjets(:, i))
      if ((dR_Vi < res) .or. (res < 0._wp)) then
        res = dR_Vi
        idx = i
      end if
    end do
  end function eval_min_dR_Vj

  ! angular separation between l1 & l2
  real(kind=wp) function eval_dR_l12(npar)
    integer, intent(in) :: npar
    eval_dR_l12 = v2_delta_R(kin(npar)%p(:, npar), kin(npar)%p(:, npar - 1))
  end function eval_dR_l12

  ! angular separation between l3 & l4
  real(kind=wp) function eval_dR_l34(npar)
    integer, intent(in) :: npar
    eval_dR_l34 = v2_delta_R(kin(npar)%p(:, npar - 2), kin(npar)%p(:, npar - 3))
  end function eval_dR_l34

  ! smallest angular separation between same-flavour leptons (l1,l2), (l3,l4)
  real(kind=wp) function eval_min_dR_l_sf(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: dR_l12, dR_l34
    dR_l12 = eval_dR_l12(npar)
    dR_l34 = eval_dR_l34(npar)
    eval_min_dR_l_sf = min(dR_l12, dR_l34)
  end function eval_min_dR_l_sf

  ! smallest angular separation between different-flavour leptons (l1,l3), (l1,l4), (l2,l3), (l2,l4)
  real(kind=wp) function eval_min_dR_l_df(npar)
    integer, intent(in) :: npar
    integer :: i, j
    real(kind=wp) :: dR_lij
    eval_min_dR_l_df = -1._wp
    do i = 0, 1
    do j = 2, 3
      dR_lij = v2_delta_R(kin(npar)%p(:, npar - i), kin(npar)%p(:, npar - j))
      if ((dR_lij < eval_min_dR_l_df) .or. (eval_min_dR_l_df < 0._wp)) then
        eval_min_dR_l_df = dR_lij
      end if
    end do
    end do
  end function eval_min_dR_l_df

  ! \phi^*_\eta
  ! defined in Sect. 3 of arXiv:1009.1580 [hep-ex]
  real(kind=wp) function eval_phi_star_eta(npar)
    use Constants_mod
    integer, intent(in) :: npar
    real(kind=wp) :: del_phi, del_eta, cos_th, sin_th
    del_phi = v2_delta_phi(kin(npar)%p(:, npar - 1), kin(npar)%p(:, npar))
    del_eta = v2_delta_y(kin(npar)%p(:, npar - 1), kin(npar)%p(:, npar)) ! massless: eta = y
    cos_th = tanh(del_eta/2._wp)
    sin_th = sqrt((1._wp + cos_th)*(1._wp - cos_th))
    eval_phi_star_eta = tan((pi - del_phi)/2._wp)*sin_th
  end function eval_phi_star_eta

  ! dynamical scale for ZJ production
  ! sqrt(mll^2 + sum ptj{i}^2) [arXiv:1512.01291]
  real(kind=wp) function eval_ZJ_dscale1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_ZJ_dscale1 = v1_minv2(pV)
    do i = 1, kin(npar)%njets
      eval_ZJ_dscale1 = eval_ZJ_dscale1 + v1_pt2(kin(npar)%pjets(:, i))
    end do
    eval_ZJ_dscale1 = sqrt(eval_ZJ_dscale1)
  end function eval_ZJ_dscale1

  ! dynamical scale for ZJ production
  ! sqrt(mV^2 + ptV^2) + sum pt{partons}
  function eval_ZJ_HT(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    integer :: i
    res = eval_et_V(npar)
    do i = kin(npar)%ijets_lower, kin(npar)%ijets_upper
      res = res + v1_pt(kin(npar)%p(:, i))
    end do
  end function eval_ZJ_HT

  ! dynamical scale for ZJ production
  ! ZpT + sum ptj{i} [Post LH19 study]
  function eval_ZJ_HTpT(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    real(kind=wp) :: V_pt
    integer :: i
    res = eval_pt_V(npar)
    do i = 1, kin(npar)%njets
      res = res + v1_pt(kin(npar)%pjets(:, i))
    end do
  end function eval_ZJ_HTpT

  ! dynamical scale for ZJ production
  ! sqrt(m4l^2 + sum ptj{i}^2) [arXiv:1512.01291]
  real(kind=wp) function eval_H4lJ_dscale1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 3) + kin(npar)%p(i, npar - 2) &
              + kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_H4lJ_dscale1 = v1_minv2(pV)
    do i = 1, kin(npar)%njets
      eval_H4lJ_dscale1 = eval_H4lJ_dscale1 + v1_pt2(kin(npar)%pjets(:, i))
    end do
    eval_H4lJ_dscale1 = sqrt(eval_H4lJ_dscale1)
  end function eval_H4lJ_dscale1

  ! dynamical scale for inclusive Z production
  ! sqrt(m4l^2 + ptz^2)
  real(kind=wp) function eval_H4lJ_dscale2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 3) + kin(npar)%p(i, npar - 2) &
              + kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_H4lJ_dscale2 = sqrt(v1_minv2(pV) + v1_pt2(pV))
  end function eval_H4lJ_dscale2

  ! dynamical scale for Z(H)J production
  ! etz + ptj1
  real(kind=wp) function eval_ZJ_dscale2(npar)
    integer, intent(in) :: npar
    eval_ZJ_dscale2 = eval_et_V(npar) + eval_pt_jet(npar, 1)
  end function eval_ZJ_dscale2

  ! dynamical scale for Z(H)J production
  ! sqrt(mll^2 + ptZ^2) + sum ptj{i} [for Les Houches 2017]
  real(kind=wp) function eval_ZJ_dscale3(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    integer :: i
    do i = 1, 4
      pV(i) = kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_ZJ_dscale3 = sqrt(v1_minv2(pV) + v1_pt2(pV))
    do i = 1, kin(npar)%njets
      eval_ZJ_dscale3 = eval_ZJ_dscale3 + v1_pt(kin(npar)%pjets(:, i))
    end do
  end function eval_ZJ_dscale3

  ! VFH selectors for VBF cuts
  real(kind=wp) function eval_VFH_deltay(npar)
    integer, intent(in) :: npar
    eval_VFH_deltay = abs(v2_delta_y(kin(npar)%pjets(:, 1), kin(npar)%pjets(:, 2)))
  end function eval_VFH_deltay

  real(kind=wp) function eval_VFH_y1xy2(npar)
    integer, intent(in) :: npar
    eval_VFH_y1xy2 = v1_y(kin(npar)%pjets(:, 1))*v1_y(kin(npar)%pjets(:, 2))
  end function eval_VFH_y1xy2

  ! dynamical scale for VFH production
  ! mu0^2 = mH/2 * sqrt(mH^2/4 + ptH^2)
  real(kind=wp) function eval_VFH_dscale(npar)
    integer, intent(in)             :: npar
    real(kind=wp)                :: mHo2, ptH2
    real(kind=wp), dimension(4) :: pV
    pV = kin(npar)%p(:, npar - 1) + kin(npar)%p(:, npar)
    mHo2 = v1_minv(pV)/2._wp
    ptH2 = v1_pt2(pV)
    eval_VFH_dscale = dsqrt(mHo2*dsqrt(mHo2**2 + ptH2))
  end function eval_VFH_dscale

  ! normalised central rapidity
  real(kind=wp) function eval_VFH_z3(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: y1, y2, y3
    y1 = v1_y(kin(npar)%pjets(:, 1))
    y2 = v1_y(kin(npar)%pjets(:, 2))
    y3 = v1_y(kin(npar)%pjets(:, 3))
    eval_VFH_z3 = (y3 - (y1 + y2)/2._wp)/(y1 - y2)
  end function eval_VFH_z3

  ! max pt of all jets in the gap
  function eval_VFH_max_pt_in_gap(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    real(kind=wp) :: y1, y2, y_min, y_max, yi, pti
    integer :: i
    y1 = eval_y_jet(npar, 1)
    y2 = eval_y_jet(npar, 2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    res = 0._wp
    do i = 3, kin(npar)%njets
      yi = eval_y_jet(npar, i)
      if ((yi < y_min) .or. (yi > y_max)) continue
      pti = eval_pt_jet(npar, i)
      if (pti > res) res = pti
    end do
  end function eval_VFH_max_pt_in_gap

  real(kind=wp) function eval_VFH_Q2_1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q1
    q1(:) = kin(npar)%p(:, 1) - kin(npar)%p(:, 4)
    eval_VFH_Q2_1 = -dot(q1, q1)
  end function eval_VFH_Q2_1

  real(kind=wp) function eval_VFH_Q2_2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q2
    q2(:) = kin(npar)%p(:, 2) - kin(npar)%p(:, 3)
    eval_VFH_Q2_2 = -dot(q2, q2)
  end function eval_VFH_Q2_2

  real(kind=wp) function eval_VFH_Q1P1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q1, p1
    q1(:) = kin(npar)%p(:, 1) - kin(npar)%p(:, 4)
    p1(:) = kin(npar)%p(:, 1)
    eval_VFH_Q1P1 = dot(q1, p1)
  end function eval_VFH_Q1P1

  real(kind=wp) function eval_VFH_Q1P2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q1, p2
    q1(:) = kin(npar)%p(:, 1) - kin(npar)%p(:, 4)
    p2(:) = kin(npar)%p(:, 2)
    eval_VFH_Q1P2 = dot(q1, p2)
  end function eval_VFH_Q1P2

  real(kind=wp) function eval_VFH_Q2P1(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q2, p1
    q2(:) = kin(npar)%p(:, 2) - kin(npar)%p(:, 3)
    p1(:) = kin(npar)%p(:, 1)
    eval_VFH_Q2P1 = dot(q2, p1)
  end function eval_VFH_Q2P1

  real(kind=wp) function eval_VFH_Q2P2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q2, p2
    q2(:) = kin(npar)%p(:, 2) - kin(npar)%p(:, 3)
    p2(:) = kin(npar)%p(:, 2)
    eval_VFH_Q2P2 = dot(q2, p2)
  end function eval_VFH_Q2P2

  real(kind=wp) function eval_VFH_Q1Q2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: q1, q2
    q1(:) = kin(npar)%p(:, 1) - kin(npar)%p(:, 4)
    q2(:) = kin(npar)%p(:, 2) - kin(npar)%p(:, 3)
    eval_VFH_Q1Q2 = -dot(q1, q2)
  end function eval_VFH_Q1Q2

  real(kind=wp) function eval_VFH_P1P2(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: p1, p2
    p1(:) = kin(npar)%p(:, 1)
    p2(:) = kin(npar)%p(:, 2)
    eval_VFH_P1P2 = -dot(p1, p2)
  end function eval_VFH_P1P2

  !>----- Spike plots

  !> (1,2) & (2,1)  -->  1
  !> (1,3) & (3,1)  -->  2
  !> (2,3) & (3,2)  -->  3
  !> (1,4) & (4,1)  -->  4
  !> (2,4) & (4,2)  -->  5
  !> (3,4) & (4,3)  -->  6, ...
  pure integer function idx_sij(i, j)
    integer, intent(in) :: i, j
    integer :: imin, imax, ioffset
    idx_sij = 0  ! invalid index
    if (i == j) return
    if (i < j) then
      imin = i; imax = j;
    else
      imin = j; imax = i;
    end if
    ioffset = 0
    if (imax > 2) ioffset = ((imax - 1)*(imax - 2))/2
    idx_sij = ioffset + imin
  end function idx_sij

  function eval_sij(npar, idx) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, idx
    integer :: n, i, j
    real(kind=wp) :: shat
    !> get the highest momentum set by traversing npar_from
    n = npar
    do while (n /= kin(n)%npar_from)
      n = kin(n)%npar_from
    end do
    call idx_to_sij(idx, i, j)
    res = abs(kin(n)%s(i, j)/kin(n)%s(1, 2))

  contains

    !> inverse mapping of idx_sij
    pure subroutine idx_to_sij(idx_, i_, j_)
      use Kinds_mod, only: sp
      integer, intent(in) :: idx_
      integer, intent(out) :: i_, j_
      integer :: ioffset_
      j_ = int(0.5_sp*(sqrt(1._sp + 8._sp*(idx_ - 0.5_sp)) - 1._sp)) + 2
      ioffset_ = 0
      if (j_ > 2) ioffset_ = ((j_ - 1)*(j_ - 2))/2
      i_ = idx_ - ioffset_
    end subroutine idx_to_sij

  end function eval_sij


  !> WF: bijective map (i,j,k) -> N^2*i + N*j + k with N=10. Need switching to higher N when npar>10.
  pure integer function idx_sijk(i, j, k)
    integer, intent(in) :: i, j, k
    integer :: N
    idx_sijk = 0  ! invalid index
    if ((i == j).or.(i == k).or.(k == j)) return
    N = 10
    idx_sijk = N*N*i + N*j + k
  end function idx_sijk

  function eval_sijk(npar, idx) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, idx
    integer :: n, i, j, k
    real(kind=wp) :: shat
    !> get the highest momentum set by traversing npar_from
    n = npar
    do while (n /= kin(n)%npar_from)
      n = kin(n)%npar_from
    end do
    call idx_to_sijk(idx, i, j, k)
    res = abs((kin(n)%s(i, j) + kin(n)%s(i, k) + kin(n)%s(j, k))/kin(n)%s(1, 2))

  contains

    !> inverse mapping of idx_sijk
    pure subroutine idx_to_sijk(idx_, i_, j_, k_)
      use Kinds_mod, only: sp
      integer, intent(in) :: idx_
      integer, intent(out) :: i_, j_, k_
      integer :: N, idxtmp
      N = 10
      if (N*N*N < idx_) return
      idxtmp = idx_
      k_ = mod(idxtmp, N)
      idxtmp = idxtmp/10
      j_ = mod(idxtmp, N)
      idxtmp = idxtmp/10
      i_ = mod(idxtmp, N)
      if ((i_ == j_).or.(i_ == k).or.(k_ == j_)) then
        i_ = 0; j_ = 0; k_ = 0;
        return
      endif
    end subroutine idx_to_sijk

  end function eval_sijk

  !> VH
  !> MM: assumes VH system given by 4 last momenta (first H, then V)
  real(kind=wp) function eval_pt_VH(npar)
    integer, intent(in) :: npar
    real(8) :: pVH(4)
    integer :: i

    do i = 1, 4
      pVH(i) = kin(npar)%p(i, npar - 3) + kin(npar)%p(i, npar - 2) + kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_pt_VH = v1_pt(pVH)
  end function eval_pt_VH

  !> MM: assumes VH system given by 4 last momenta (first H, then V)
  real(kind=wp) function eval_pth_VH(npar)
    integer, intent(in) :: npar
    real(8) :: pH(4)
    integer :: i

    do i = 1, 4
      pH(i) = kin(npar)%p(i, npar - 3) + kin(npar)%p(i, npar - 2)
    end do
    eval_pth_VH = v1_pt(pH)
  end function eval_pth_VH

  !> MM: assumes VH system given by 4 last momenta (first H, then V)
  real(kind=wp) function eval_minv_VH(npar)
    integer, intent(in) :: npar
    real(8) :: pVH(4)
    integer :: i

    do i = 1, 4
      pVH(i) = kin(npar)%p(i, npar - 3) + kin(npar)%p(i, npar - 2) + kin(npar)%p(i, npar - 1) + kin(npar)%p(i, npar)
    end do
    eval_minv_VH = v1_minv(pVH)
  end function eval_minv_VH

  !> flavour tagging
  function eval_flav_pt_ji(npar, i) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, i
    integer :: j, k
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        res = v1_pt(kin(npar)%pjets(:, j))
        return
      end if
    end do

  end function eval_flav_pt_ji

  real(kind=wp) function eval_flav_sumq_ji(npar, i)
    integer, intent(in) :: npar, i
    integer :: j, k
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_sumq_ji = kin(npar)%sumqflavjet(j)
        return
      end if
    end do
  end function eval_flav_sumq_ji

  !> eta
  real(kind=wp) function eval_flav_eta_ji(npar, i)
    integer, intent(in) :: npar, i
    integer :: j, k
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_eta_ji = v1_eta(kin(npar)%pjets(:, j))
        return
      end if
    end do
  end function eval_flav_eta_ji

  !> y
  real(kind=wp) function eval_flav_y_ji(npar, i)
    integer, intent(in) :: npar, i
    integer :: j, k
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_y_ji = v1_y(kin(npar)%pjets(:, j))
        return
      end if
    end do
  end function eval_flav_y_ji

  !> deltay(V,jeti)
  real(kind=wp) function eval_flav_deltay_Vji(npar, i)
    integer, intent(in) :: npar, i
    real(kind=wp), dimension(4) :: pV
    integer :: j, k
    do j = 1, 4
      pV(j) = kin(npar)%p(j, npar - 1) + kin(npar)%p(j, npar)
    end do
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_deltay_Vji = v2_delta_y(pV, kin(npar)%pjets(:, j))
        return
      end if
    end do
  end function eval_flav_deltay_Vji

  !> deltaphi(V,jeti)
  real(kind=wp) function eval_flav_deltaphi_Vji(npar, i)
    integer, intent(in) :: npar, i
    real(kind=wp), dimension(4) :: pV
    integer :: j, k
    do j = 1, 4
      pV(j) = kin(npar)%p(j, npar - 1) + kin(npar)%p(j, npar)
    end do
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_deltaphi_Vji = v2_delta_phi(pV, kin(npar)%pjets(:, j))
        return
      end if
    end do
  end function eval_flav_deltaphi_Vji

  !> deltaR(V,jeti)
  real(kind=wp) function eval_flav_deltaR_Vji(npar, i)
    integer, intent(in) :: npar, i
    real(kind=wp), dimension(4) :: pV
    integer :: j, k
    do j = 1, 4
      pV(j) = kin(npar)%p(j, npar - 1) + kin(npar)%p(j, npar)
    end do
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        eval_flav_deltaR_Vji = v2_delta_R(pV, kin(npar)%pjets(:, j))
        return
      end if
    end do
  end function eval_flav_deltaR_Vji

  !> |yZ + yj_i|/2
  real(kind=wp) function eval_flav_yboost_Vji(npar, i)
    integer, intent(in) :: npar, i
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: yV, yj1
    integer :: j, k
    do j = 1, 4
      pV(j) = kin(npar)%p(j, npar - 1) + kin(npar)%p(j, npar)
    end do
    k = 0
    do j = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(j)) k = k + 1
      if (k == i) then
        yV = v1_y(pV)
        yj1 = v1_y(kin(npar)%pjets(:, j))
        eval_flav_yboost_Vji = abs(yV + yj1)/2._wp
        return
      end if
    end do
  end function eval_flav_yboost_Vji

  ! dynamical scale for V(H)J production
  ! etV + ptj1_flav
  function eval_flav_VJ_dscale2(npar) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar
    res = eval_et_V(npar) + eval_flav_pt_ji(npar, 1)
  end function eval_flav_VJ_dscale2

  ! same as eval_flav_VJ_dscale2 but etV defined with static mass MV
  ! sqrt(MV**2 + ptV**2) + ptj1_flav
  real(kind=wp) function eval_flav_VJ_dscale3(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: ptv
    !> common block to get mass parameter
    real(kind=wp) :: rm12, rg12
    integer :: inwa
    common/nwa/rm12, rg12, inwa
    eval_flav_VJ_dscale3 = sqrt(rm12**2 + eval_pt_V(npar)**2) &
                           + eval_flav_pt_ji(npar, 1)
  end function eval_flav_VJ_dscale3

  !> invariant mass of a two flavoured jet system that best approximates a
  !> given mass
  real(kind=wp) function eval_flav_minv_jj(npar)
    use Parameters_mod, only: MH_param
    integer, intent(in) :: npar
    real(8)          :: pj_kl(4), inv_mass
    integer          :: k, l
    real(8), target  :: dmass
    real(8), pointer :: dmass_min

    dmass_min => null()

    do l = 1, kin(npar)%njets
      if (.not. kin(npar)%qflavjet(l)) cycle
      do k = 1, l - 1
        if (.not. kin(npar)%qflavjet(k)) cycle
        pj_kl(:) = kin(npar)%pjets(:, k) + kin(npar)%pjets(:, l)
        inv_mass = v1_minv(pj_kl)
        dmass = abs(MH_param() - inv_mass)
        if (associated(dmass_min)) then
          if (dmass < dmass_min) dmass_min => dmass
        else
          dmass_min => dmass
        end if
      end do
    end do

    if (.not. (associated(dmass_min))) stop "eval_flav_minv_jj: this should not happen"

    eval_flav_minv_jj = inv_mass

  end function eval_flav_minv_jj

  !> transverse momentum of a two flavoured jet system whose invariant mass best
  !> approximates a given mass
  real(kind=wp) function eval_flav_pt_jj(npar)
    use Parameters_mod, only: MH_param
    integer, intent(in) :: npar
    integer          :: k, l
    real(8), target  :: pj_kl(4), dmass
    real(8), pointer :: pj_kl_best(:), dmass_min

    dmass_min => null()
    pj_kl_best => null()

    do l = 1, kin(npar)%njets
      if (.not. kin(npar)%qflavjet(l)) cycle
      do k = 1, l - 1
        if (.not. kin(npar)%qflavjet(k)) cycle
        pj_kl(:) = kin(npar)%pjets(:, k) + kin(npar)%pjets(:, l)
        dmass = abs(MH_param() - v1_minv(pj_kl))
        if (associated(dmass_min)) then
          if (dmass < dmass_min) then
            dmass_min => dmass
            pj_kl_best => pj_kl
          end if
        else
          dmass_min => dmass
          pj_kl_best => pj_kl
        end if
      end do
    end do
    if (.not. (associated(dmass_min))) stop "eval_flav_pt_jj: this should not happen"
    eval_flav_pt_jj = v1_pt(pj_kl_best)

  end function eval_flav_pt_jj

  !> angular separation of a two flavoured jet system whose invariant mass best
  !> approximates a given mass
  real(kind=wp) function eval_flav_dR_jj(npar)
    use Parameters_mod, only: MH_param
    integer, intent(in) :: npar
    integer          :: k, l
    real(8)          :: pj_kl(4)
    real(8), target  :: pj_k(4), pj_l(4), dmass
    real(8), pointer :: pj_k_best(:), pj_l_best(:), dmass_min

    dmass_min => null()
    pj_k_best => null()
    pj_l_best => null()

    do l = 1, kin(npar)%njets
      if (.not. kin(npar)%qflavjet(l)) cycle
      do k = 1, l - 1
        if (.not. kin(npar)%qflavjet(k)) cycle
        pj_k(:) = kin(npar)%pjets(:, k)
        pj_l(:) = kin(npar)%pjets(:, l)
        pj_kl = pj_k + pj_l
        dmass = abs(MH_param() - v1_minv(pj_kl))
        if (associated(dmass_min)) then
          if (dmass < dmass_min) then
            dmass_min => dmass
            pj_k_best => pj_k
            pj_l_best => pj_l
          end if
        else
          dmass_min => dmass
          pj_k_best => pj_k
          pj_l_best => pj_l
        end if
      end do
    end do
    if (.not. (associated(dmass_min))) stop "eval_flav_dR_jj: this should not happen"
    eval_flav_dR_jj = v2_delta_R(pj_k_best, pj_l_best)

  end function eval_flav_dR_jj

  real(kind=wp) function eval_flav_deltaphi_lep_jet(npar, idx)
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep, pjet, iflav
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == jet) then
        eval_flav_deltaphi_lep_jet = v2_delta_phi(kin(npar)%p(:, plep), kin(npar)%pjets(:, pjet))
        return
      end if
    end do
    eval_flav_deltaphi_lep_jet = 0._wp
    stop "eval_flav_deltaphi_lep_jet: should not get here"
  end function eval_flav_deltaphi_lep_jet

  real(kind=wp) function eval_flav_deltaR_lep_jet(npar, idx)
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep, pjet, iflav
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == jet) then
        eval_flav_deltaR_lep_jet = v2_delta_R(kin(npar)%p(:, plep), kin(npar)%pjets(:, pjet))
        return
      end if
    end do
    eval_flav_deltaR_lep_jet = 0._wp
    stop "eval_flav_deltaR_lep_jet: should not get here"
  end function eval_flav_deltaR_lep_jet

  real(kind=wp) function eval_flav_minv_lep_jet(npar, idx)
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep, pjet, iflav
    real(kind=wp), dimension(4) :: plj
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == jet) then
        plj(:) = kin(npar)%p(:, plep) + kin(npar)%pjets(:, pjet)
        eval_flav_minv_lep_jet = v1_minv(plj)
        return
      end if
    end do
    eval_flav_minv_lep_jet = 0._wp
    stop "eval_flav_minv_lep_jet: should not get here"
  end function eval_flav_minv_lep_jet

  real(kind=wp) function eval_flav_pt_lep_jet(npar, idx)
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep, pjet, iflav
    real(kind=wp), dimension(4) :: plj
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == jet) then
        plj(:) = kin(npar)%p(:, plep) + kin(npar)%pjets(:, pjet)
        eval_flav_pt_lep_jet = v1_pt(plj)
        return
      end if
    end do
    eval_flav_pt_lep_jet = 0._wp
    stop "eval_flav_pt_lep_jet: should not get here"
  end function eval_flav_pt_lep_jet

  real(kind=wp) function eval_flav_eta_lep_jet(npar, idx)
    integer, intent(in) :: npar, idx
    integer :: lep, jet
    integer :: plep, pjet, iflav
    real(kind=wp), dimension(4) :: plj
    !> idx_lep_jet:
    !> idx = 4*(jet-1) + lep
    lep = mod((idx - 1), 4) + 1
    jet = int((idx - 1)/4) + 1
    select case (lep)
    case (1)
      plep = npar - 1
    case (2)
      plep = npar
    case (3)
      plep = npar - 3
    case (4)
      plep = npar - 2
    end select
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == jet) then
        plj(:) = kin(npar)%p(:, plep) + kin(npar)%pjets(:, pjet)
        eval_flav_eta_lep_jet = v1_eta(plj)
        return
      end if
    end do
    eval_flav_eta_lep_jet = 0._wp
    stop "eval_flav_eta_lep_jet: should not get here"
  end function eval_flav_eta_lep_jet

  !> xFeynman of a flavoured jet "i" according to ATLAS Z+HF measurement
  real(kind=wp) function eval_flav_xF_ptjet_i(npar, i)
    use Process_mod, only: sqrts_proc
    integer, intent(in) :: npar, i
    integer pjet, iflav
    real(kind=wp) pzj
    iflav = 0
    do pjet = 1, kin(npar)%njets
      if (kin(npar)%qflavjet(pjet)) iflav = iflav + 1
      if (iflav == i) then
        ! ATLAS have now informed me the previous definition was incorrect
        ! They actually use: xF = 2 |pz| / sqrts_proc
        pzj = kin(npar)%pjets(3, pjet)
        eval_flav_xF_ptjet_i = 2._wp*dabs(pzj)/sqrts_proc
        return
      end if
    end do
    eval_flav_xF_ptjet_i = 0._wp
    stop "eval_flav_xF_ptjet_i: should not get here"
  end function eval_flav_xF_ptjet_i

  real(kind=wp) function eval_smin_over_shat(npar)
    use KinData_mod
    integer, intent(in) :: npar
    integer :: ii, jj, npar_curr
    real(kind=wp) :: smin

    eval_smin_over_shat = 1._wp  ! init to largest possible value

    npar_curr = npar
    do while (.true.)
      if (npar_curr == kin(npar_curr)%npar_from) exit
      npar_curr = kin(npar_curr)%npar_from
    end do
    !> find min of npar_curr
    smin = kin(npar_curr)%s(1, 2)
    do ii = 1, npar_curr
      do jj = ii + 1, npar_curr
        smin = min(smin, abs(kin(npar_curr)%s(ii, jj)))
      end do
    end do
    smin = smin/kin(npar_curr)%s(1, 2)
    eval_smin_over_shat = smin
  end function eval_smin_over_shat

  !> identified hadrons
  real(kind=wp) function eval_pt_had(npar)
    integer, intent(in) :: npar
    eval_pt_had = kin(npar)%zfrag*v1_pt(kin(npar)%p(:, kin(npar)%ifrag))
  end function eval_pt_had

  real(kind=wp) function eval_p_had(npar)
    integer, intent(in) :: npar
    eval_p_had = kin(npar)%zfrag*v1_p(kin(npar)%p(:, kin(npar)%ifrag))
  end function eval_p_had

  real(kind=wp) function eval_pt_had_vs_ptj1(npar)
    integer, intent(in) :: npar
    integer :: hadinjet, ijet
    real(kind=wp), dimension(4) ::ptjet, pthad, cross_product
    real(kind=wp) :: dR
    ptjet = kin(npar)%pjets(:, 1)
    pthad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    cross_product(1) = pthad(2)*ptjet(3) - pthad(3)*ptjet(2)
    cross_product(2) = pthad(3)*ptjet(1) - pthad(1)*ptjet(3)
    cross_product(3) = pthad(1)*ptjet(2) - pthad(2)*ptjet(1)
    cross_product(4) = 0._wp
    eval_pt_had_vs_ptj1 = v1_p(cross_product(:))/v1_p(ptjet)
  end function eval_pt_had_vs_ptj1

  real(kind=wp) function eval_pt_had_vs_ptj(npar)
    integer, intent(in) :: npar
    integer :: hadinjet, ijet
    real(kind=wp), dimension(4) ::ptjet, pthad, cross_product
    real(kind=wp) :: dR, comparator
    do ijet = 1, kin(npar)%njets
      dR = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), kin(npar)%pjets(:, ijet))
      if ((comparator < 0._wp) .or. (dR < comparator)) then
        hadinjet = ijet
        comparator = dR
      end if
    end do
    ptjet = kin(npar)%pjets(:, hadinjet)
    pthad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    cross_product(1) = pthad(2)*ptjet(3) - pthad(3)*ptjet(2)
    cross_product(2) = pthad(3)*ptjet(1) - pthad(1)*ptjet(3)
    cross_product(3) = pthad(1)*ptjet(2) - pthad(2)*ptjet(1)
    cross_product(4) = 0._wp
    eval_pt_had_vs_ptj = v1_p(cross_product(:))/v1_p(ptjet)
  end function eval_pt_had_vs_ptj

  real(kind=wp) function eval_pl_frac_had1(npar)
    integer, intent(in) :: npar
    integer :: hadinjet, ijet
    real(kind=wp) :: phad(1:4), pjet(1:4)
    real(kind=wp) :: dR
    !> hadron and leading jet momenta
    phad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    pjet = kin(npar)%pjets(:, 1)
    !> longitudinal momentum fraction
    eval_pl_frac_had1 = dot_eucl(pjet(1:3), phad(1:3))/v1_p2(pjet(:))
  end function eval_pl_frac_had1

  real(kind=wp) function eval_pl_frac_had(npar)
    integer, intent(in) :: npar
    integer :: hadinjet, ijet
    real(kind=wp) :: phad(1:4), pjet(1:4)
    real(kind=wp) :: dR, comparator
    comparator = -1._wp
    !> hadron momentum
    phad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    !> find closest jet to hadron
    do ijet = 1, kin(npar)%njets
      dR = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), kin(npar)%pjets(:, ijet))
      if ((comparator < 0._wp) .or. (dR < comparator)) then
        hadinjet = ijet
        comparator = dR
      end if
    end do
    pjet = kin(npar)%pjets(:, hadinjet)
    !> longitudinal momentum fraction
    eval_pl_frac_had = dot_eucl(pjet(1:3), phad(1:3))/v1_p2(pjet(:))
  end function eval_pl_frac_had

  real(kind=wp) function eval_E_had(npar)
    integer, intent(in) :: npar
    eval_E_had = kin(npar)%zfrag*kin(npar)%p(4, kin(npar)%ifrag)
  end function eval_E_had

  real(kind=wp) function eval_x_had(npar)
    use Process_mod, only: sqrts_proc
    integer, intent(in) :: npar
    eval_x_had = 2._wp*eval_E_had(npar)/sqrts_proc
  end function eval_x_had

  real(kind=wp) function eval_y_had(npar)
    integer, intent(in) :: npar
    eval_y_had = v1_y(kin(npar)%p(:, kin(npar)%ifrag))
  end function eval_y_had

  real(kind=wp) function eval_costh_had(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pi(1:4), pj(1:4), cos_ij
    !> hadron momentum
    pi = kin(npar)%p(:, kin(npar)%ifrag)
    !> electron momentum
    pj = kin(npar)%p(:, 2)
    !> the angle
    cos_ij = dot_eucl(pi(1:3), pj(1:3)) &
             /sqrt(dot_eucl(pi(1:3), pi(1:3))) &
             /sqrt(dot_eucl(pj(1:3), pj(1:3)))
    eval_costh_had = cos_ij
  end function eval_costh_had

  real(kind=wp) function eval_deltaR_hadj1(npar)
    integer, intent(in) :: npar
    integer :: ijet
    real(kind=wp) :: phad(1:4), pjet(1:4)
    real(kind=wp) :: dR
    eval_deltaR_hadj1 = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), kin(npar)%pjets(:, 1))
  end function eval_deltaR_hadj1

  real(kind=wp) function eval_deltaR_hadj(npar)
    integer, intent(in) :: npar
    integer :: ijet
    real(kind=wp) :: phad(1:4), pjet(1:4)
    real(kind=wp) :: dR
    eval_deltaR_hadj = -1._wp
    !> find closest jet to hadron
    do ijet = 1, kin(npar)%njets
      dR = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), kin(npar)%pjets(:, ijet))
      if ((eval_deltaR_hadj < 0._wp) .or. (dR < eval_deltaR_hadj)) eval_deltaR_hadj = dR
    end do
  end function eval_deltaR_hadj

  real(kind=wp) function eval_deltaR_hadZ(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pZ(1:4)
    pZ = kin(npar)%p(:, npar) + kin(npar)%p(:, npar - 1)
    eval_deltaR_hadZ = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), pZ)
  end function eval_deltaR_hadZ

  real(kind=wp) function eval_deltaphi_hadZ(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: pZ(1:4)
    pZ = kin(npar)%p(:, npar) + kin(npar)%p(:, npar - 1)
    eval_deltaphi_hadZ = v2_delta_phi(kin(npar)%p(:, kin(npar)%ifrag), pZ)
  end function eval_deltaphi_hadZ

  real(kind=wp) function eval_xit_hadZ(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: p_had(1:4), p_z(1:4), pt2_z
    !> hadron momentum
    p_had = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    !> Z boson momentum and transverse momentum
    p_z = kin(npar)%p(:, npar - 1) + kin(npar)%p(:, npar)
    pt2_z = v1_pt2(p_z)
    !> (log of) pt hadron fraction along the Z transverse momentum
    eval_xit_hadZ = log(-1._wp*((pt2_z)/(p_z(1)*p_had(1) + p_z(2)*p_had(2))))
  end function eval_xit_hadZ

  real(kind=wp) function eval_pt_ratio_hadj(npar)
    integer :: ijet, hadinjet
    real(kind=wp) :: phad(1:4), pjet(1:4)
    integer, intent(in) :: npar
    real(kind=wp) :: dR, comparator
    comparator = -1._wp
    !> hadron momentum
    phad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    !> find closest jet to hadron
    do ijet = 1, kin(npar)%njets
      dR = v2_delta_R(kin(npar)%p(:, kin(npar)%ifrag), kin(npar)%pjets(:, ijet))
      if ((comparator < 0._wp) .or. (dR < comparator)) then
        hadinjet = ijet
        comparator = dR
      end if
    end do
    !> jet momentum
    pjet = kin(npar)%pjets(:, hadinjet)
    !> ratio of the two transverse momentum
    eval_pt_ratio_hadj = v1_pt(phad)/v1_pt(pjet)
  end function eval_pt_ratio_hadj

  real(kind=wp) function eval_eta_had(npar)
    integer, intent(in) :: npar
    real(kind=wp) :: phad(1:4)
    phad = kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    eval_eta_had = v1_eta(phad)
  end function eval_eta_had

  real(kind=wp) function eval_minv_gamhad(npar)
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: psum
    psum(:) = kin(npar)%pphotons(:, 1) &
              + kin(npar)%zfrag*kin(npar)%p(:, kin(npar)%ifrag)
    eval_minv_gamhad = v1_minv(psum)
  end function eval_minv_gamhad

!!  !-----------------------------------------------------------------------------
!!  !> @brief
!!  !> compute N-jettiness (dimensionless)
!!  !> arXiv:1004.2489 [hep-ph]
!!  !> our normalisation follows arXiv:1711.09984 [hep-ph]
!!  !
!!  !> @param[in] npar momentum set
!!  !> @param[in] N number of jets in the Born process: "N"-jettiness (tau_N)
!!  !> @return tau_N
!!  !-----------------------------------------------------------------------------
!!  function eval_jettiness(npar, N) result(res)
!!    use Jets_mod, only: jetalg
!!    real(kind=wp) :: res
!!    integer, intent(in) :: npar, N
!!    integer :: i, j
!!    type(Kin_t), pointer :: kd
!!    real(kind=wp) :: tauN, pnorm, di_min
!!    real(kind=wp) :: qref_in(4,2), qref_out(4,3)  ! up to 3-jettiness
!!
!!    if (jetalg==0) then
!!      print*, "eval_jettiness: need to have a jet algorithm active for this observable"
!!      stop
!!    end if
!!
!!    if ( (N < 0) .or. (N > 3) ) then
!!      print*, "eval_jettiness: only N = 0,1,2,3 supported:", N
!!      stop
!!    end if
!!
!!    !# (1) perform a jet clustering
!!    call initRecomb_jet(npar)
!!    if ( nproto_jet < N ) then
!!      print*, "eval_jettiness: not enough partons:", nproto_jet, "<", N
!!      stop
!!    end if
!!    !> cluster the jets
!!    call cluster_jet()
!!    if (njets_jet < N) stop "eval_jettiness: clustered too much?!"
!!    !> sort in pt
!!    call sort_jet()
!!
!!    !# (2) compute (light-like) reference axes (normalised -> dimensionless)
!!    !> prettify code
!!    kd => kin(npar)
!!    !> initial-state axes
!!    !  (normalisation removes dependence on initial-state collinear splittings)
!!    qref_in  = 0d0
!!    do i=1,2
!!      qref_in(:,i) = kd%p(:,i) / kd%p(4,i)
!!    end do
!!    !> take the N leading jets to determine final-state axes
!!    qref_out = 0d0
!!    do i=1,N
!!      pnorm = sqrt( dot_eucl(jets(pt_sort_jet(i))%p(1:3), jets(pt_sort_jet(i))%p(1:3)) )
!!      qref_out(1:3,i) = jets(pt_sort_jet(i))%p(1:3) / pnorm
!!      qref_out(4,i)   = 1d0
!!      ! print*, pt_sort_jet(i), "pjet:", jets(pt_sort_jet(i))%p(:)
!!      ! print*, "> norm", pnorm, "pt", sqrt(jets(pt_sort_jet(i))%pt2), "y", jets(pt_sort_jet(i))%y
!!      ! print*, i, "qref_out:", qref_out(:,i), ", norm:", dot(qref_out(:,i),qref_out(:,i))
!!    end do
!!
!!    !# (3) compute jettiness observable
!!    !> accumulate smallest scalar products
!!    tauN = 0d0
!!    do i = kd%ijets_lower, kd%ijets_upper
!!      ! print*, "i = ", i
!!      ! first deal with the initial-state reference axes
!!      di_min = min( dot(qref_in(:,1),kd%p(:,i)),  &
!!                  & dot(qref_in(:,2),kd%p(:,i)) )
!!      ! print*, "j = ", -1, ":", dot(qref_in(:,1),kd%p(:,i))
!!      ! print*, "j = ", -2, ":", dot(qref_in(:,2),kd%p(:,i))
!!      do j = 1, N
!!        di_min = min( di_min, dot(qref_out(:,j),kd%p(:,i)) )
!!        ! print*, "j = ", j, ":", dot(qref_out(:,j),kd%p(:,i))
!!      end do  ! j
!!      ! print*, "min:", di_min
!!      !> force positive in case numerical cancellation give "-epsilon"
!!      if ( di_min < 0d0 ) then
!!        if ( abs(di_min)/kd%p(4,i) < 1d-6 ) then
!!          di_min = abs(di_min)
!!        else
!!          print*, "eval_jettiness: encountered large negative scalar product", di_min
!!          print*, "final-state parton", i, kd%p(:,i)
!!          print*, "qref", -2, qref_in(:,2), dot(qref_in(:,2),kd%p(:,i))
!!          print*, "qref", -1, qref_in(:,1), dot(qref_in(:,1),kd%p(:,i))
!!          do j = 1, N
!!            print*, "qref", j, qref_out(:,j), dot(qref_out(:,j),kd%p(:,i))
!!          end do
!!          stop
!!        end if
!!      end if
!!      tauN = tauN + di_min
!!    end do  ! i
!!    !> so far, definition of arXiv:1711.09984 [hep-ph]
!!    !  => [tauN] = GeV
!!    !> make dimensionless using shat
!!    tauN = tauN  / sqrt(kd%s(1,2))
!!
!!    !# (4) store and return
!!    select case(N)
!!    case(0)
!!      vcache_(id_jettiness_tau0)   = tauN
!!      qcache_(id_jettiness_tau0) = .true.
!!    case(1)
!!      vcache_(id_jettiness_tau1)   = tauN
!!      qcache_(id_jettiness_tau1) = .true.
!!    case(2)
!!      vcache_(id_jettiness_tau2)   = tauN
!!      qcache_(id_jettiness_tau2) = .true.
!!    case(3)
!!      vcache_(id_jettiness_tau3)   = tauN
!!      qcache_(id_jettiness_tau3) = .true.
!!    case default
!!      print*, 'eval_jettiness: invalid value', N
!!      stop
!!    end select
!!
!!    ! ! !> check the unresolved limits
!!    ! ! if (kd%njets <= N .and. nproto_jet > N) then
!!    ! ! !> check single-unresolved limits
!!    ! ! if (kd%njets == N .and. nproto_jet == N+1) then
!!    ! ! !> check double-unresolved limits
!!    ! ! if (kd%njets == N .and. nproto_jet == N+2) then
!!    ! !> after clustering
!!    ! if (njets_jet < nproto_jet) then
!!    !   call print_momenta(npar)
!!    !   print*, "KinData:  ", kd%njets
!!    !   print*, "RefClust: ", njets_jet, "[",nproto_jet,"]"
!!    !   print*, "tau", N, " = ", tauN
!!    !   read(*,*)
!!    ! end if
!!
!!  end function eval_jettiness

  function eval_born_xi(npar, imem) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: npar, imem
    !> cache for memoization:
    !>   [ xib1(1), xib2(2), min_xib(3) ]
    real(kind=wp), dimension(3), save :: &
      cache_ = [0._wp, 0._wp, 0._wp]
    logical, save :: qcached_ = .false.
!$OMP THREADPRIVATE(cache_, qcached_)
    !>-----
    type(Kin_t), pointer :: kd
    integer :: i, j
    real(kind=wp) :: pX(4)
    real(kind=wp) :: s1X, s2X, xi1Oxi2, xi1Txi2, xi1, xi2

    res = 0._wp

    if (qcached_ .and. (imem > 0)) then
      res = cache_(imem)
      return
    end if
    if (imem == 0) then
      qcached_ = .false.
      return
    end if

    kd => kin(npar)

    pX(:) = 0._wp
    s1X = 0._wp
    s2X = 0._wp
    !> loop over all X-decay products (not QCD partons)
    do i = 3, npar
      if (i >= kd%ijets_lower .and. i <= kd%ijets_upper) cycle
      pX(:) = pX(:) + kd%p(:, i)
      ! attention: s(a,i) are negative: s(a,i) = (pa-pi)^2 = -2pa.pi
      s1X = s1X + kd%s(1, i)
      s2X = s2X + kd%s(2, i)
    end do
    xi1Oxi2 = s2X/s1X

    xi1Txi2 = kd%s(1, 2)
    !> loop over all QCD partons
    do i = kd%ijets_lower, kd%ijets_upper
      ! for initial-final s(a,i): sign = -1!
      xi1Txi2 = xi1Txi2 + kd%s(1, i) + kd%s(2, i)
      do j = i + 1, kd%ijets_upper
        ! for final-final s(i,j): sign = +1!
        xi1Txi2 = xi1Txi2 + kd%s(i, j)
      end do
    end do
    xi1Txi2 = xi1Txi2/kd%s(1, 2)

    !> splitting
    xi1 = sqrt(xi1Txi2*xi1Oxi2)
    xi2 = sqrt(xi1Txi2/xi1Oxi2)

    cache_ = [(1._wp - xi1), (1._wp - xi2), min(1._wp - xi1, 1._wp - xi2)]

    qcached_ = .true.
    res = cache_(imem)

  end function eval_born_xi

  !>
  !> 0-jettiness for VJ, in the lab frame
  !>
  function eval_VJ_tau0(npar) result(tau0)
    real(kind=wp) :: tau0
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: yV, etaplus, etaminus, pplus, pminus
    integer :: j

    tau0 = 0._wp
    if (npar.lt.5) return

    pV(1:4) = kin(npar)%p(1:4, npar-1) + kin(npar)%p(1:4, npar)
    yV = v1_y(pV)
    etaplus = exp(yV)
    etaminus = 1._wp/etaplus
    do j=3,npar-2
       pplus = etaplus * (kin(npar)%p(4,j) - kin(npar)%p(3,j))
       pminus = etaminus * (kin(npar)%p(4,j) + kin(npar)%p(3,j))
       if (pplus < pminus) then
          tau0 = tau0 + pplus
       else
          tau0 = tau0 + pminus
       endif
    enddo

  end function eval_VJ_tau0

  !>
  !> 1-jettiness for VJ, in the colour-singlet frame
  !> as done in 2312.06496
  !>
  function eval_VJ_tau1(npar) result(tau1)
    real(kind=wp) :: tau1
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pV
    real(kind=wp) :: yV, etaplus, etaminus, pplus, pminus
    real(kind=wp) :: bfac
    real(kind=wp), dimension(4) :: pj1,pj2,pj3
    real(kind=wp), dimension(4) :: n12,n13,n23,n123
    real(kind=wp), dimension(4) :: pj12,pj13,pj23,pj123
    real(kind=wp) :: pj1p,pj2p,pj3p, pj1m,pj2m,pj3m

    tau1 = 0._wp
    !> tau1 has a non-null value for V+>=2 partons
    if (npar.lt.6) return

    pV(1:4) = kin(npar)%p(1:4, npar-1) + kin(npar)%p(1:4, npar)
    yV = v1_y(pV)
    bfac = exp(-yV)

    if (npar.eq.6) then
       call boost_zhyp(bfac, kin(npar)%p(1:4,3), pj1)
       call boost_zhyp(bfac, kin(npar)%p(1:4,4), pj2)
       pj12 = pj1+pj2
       call get_dir_vec(pj12, n12)
       pj1p = pj1(4) - pj1(3)
       pj1m = pj1(4) + pj1(3)
       pj2p = pj2(4) - pj2(3)
       pj2m = pj2(4) + pj2(3)
       tau1 = min( pj1p, pj1m, pj2p, pj2m, dot(n12, pj12) )
    elseif (npar.eq.7) then
       call boost_zhyp(bfac, kin(npar)%p(1:4,3), pj1)
       call boost_zhyp(bfac, kin(npar)%p(1:4,4), pj2)
       call boost_zhyp(bfac, kin(npar)%p(1:4,5), pj3)
       pj12  = pj1+pj2
       pj13  = pj1+pj3
       pj23  = pj2+pj3
       pj123 = pj1+pj2+pj3
       call get_dir_vec(pj12,  n12)
       call get_dir_vec(pj13,  n13)
       call get_dir_vec(pj23,  n23)
       call get_dir_vec(pj123, n123)
       pj1p = pj1(4) - pj1(3)
       pj1m = pj1(4) + pj1(3)
       pj2p = pj2(4) - pj2(3)
       pj2m = pj2(4) + pj2(3)
       pj3p = pj3(4) - pj3(3)
       pj3m = pj3(4) + pj3(3)
       tau1 = min( pj1p + pj2p, pj1p + pj2m, pj1p + pj3p, pj1p + pj3m, &
            pj1m + pj2p, pj1m + pj2m, pj1m + pj3p, pj1m + pj3m, &
            pj2p + pj3p, pj2p + pj3m, pj2m + pj3p, pj2m + pj3m, &
            pj1p + dot(n23, pj23), pj2p + dot(n13, pj13), pj3p + dot(n12, pj12), &
            pj1m + dot(n23, pj23), pj2m + dot(n13, pj13), pj3m + dot(n12, pj12), &
            dot(n123, pj123) )
    else
       print *, "1-jettiness implemented only for npar <=7, npar: ", npar
       stop
    endif

  end function eval_VJ_tau1

  !>
  !> 1-jettiness for DIS, in the lab frame and excluding beam A clusterings
  !>
  function eval_DIS_tau1(npar) result(tau1)
    real(kind=wp) :: tau1
    integer, intent(in) :: npar
    real(kind=wp), dimension(4) :: pj1,pj2,pjsum
    real(kind=wp) :: d45,d24,d25

    tau1 = 0._wp
    !> tau1 has a non-null value for >=2 partons
    if (npar.lt.5) return

    if (npar.gt.5) then
       print *, "DIS 1-jettiness implemented only for npar = 5, npar: ", npar
       stop
    end if

    pj1 = kin(npar)%p(1:4,npar-1)
    pj2 = kin(npar)%p(1:4,npar)

    !> check implementation with a couple of PS points
    !pj1 = (/ -22.624d0, 128.633d0, -631.65d0, 645.012d0 /)
    !pj2 = (/ 26.6222d0, -117.608d0, -157.496d0, 198.357d0 /)
    !INFO from Analyzer: Tau1 = 13.3616
    !pj1 = (/ 20.9366d0, 31.6699d0, -7.26534d0,  38.6537d0 /)
    !pj2 = (/ 54.0892d0, -28.8005d0, -614.814d0, 617.86d0 /)
    !INFO from Analyzer: Tau1 = 3.04631

    !>------------------------------------------------------------
    pjsum = pj1+pj2

    ! final state clustering
    d45 = trinorm(pj1) + trinorm(pj2) - trinorm(pjsum)
    ! clustering of the 4th particle (always parton in disorder) with beamB
    d24 = trinorm(pj1) + pj1(3)
    ! clustering of the 5th particle (always parton in disorder) with beamB
    d25 = trinorm(pj2) + pj2(3)

    ! now check which is the minimum distance measure
    ! final-state clustering is the minimum
    if(d45 < d24 .and. d45 < d25) then
       tau1 = d45
       ! 4th particle is clustered to beamB
    elseif(d24 < d45 .and. d24 < d25) then
       tau1 = pj1(4) + pj1(3)
       ! 5th particle is clustered to beamB
    elseif(d25 < d45 .and. d25 < d24) then
       tau1 = pj2(4) + pj2(3)
    endif
    !>------------------------------------------------------------

  end function eval_DIS_tau1

  !> fvout is a 4-vector, we return norm of its 3-vector
  function trinorm(fvin) result(norm)
    real(kind=wp) :: norm
    real(kind=wp), dimension(4), intent(in) :: fvin
    real(kind=wp), dimension(3) :: fv3
    fv3 = fvin(1:3)
    norm = sqrt(dot_eucl(fv3, fv3))
  end function trinorm

  !> fvout is a light-like 4-vector with direction fvin
  subroutine get_dir_vec(fvin, fvout)
    real(kind=wp), dimension(4), intent(in) :: fvin
    real(kind=wp), dimension(4), intent(out) :: fvout
    real(kind=wp), dimension(3) :: fv3

    fv3 = fvin(1:3)
    fvout(1:3) = fv3/sqrt(dot_eucl(fv3, fv3))
    fvout(4) = 1._wp

  end subroutine get_dir_vec

  !> get the longitudinal boost to the frame with yV = 0
  subroutine boost_zhyp(la, fvin, fvout)
    real(kind=wp), intent(in) :: la
    real(kind=wp), dimension(4), intent(in) :: fvin
    real(kind=wp), dimension(4), intent(out) :: fvout
    real(kind=wp) :: lainv, minus, plus

    lainv = 1._wp/la
    minus = ( fvin(4) + fvin(3) ) * la
    plus  = ( fvin(4) - fvin(3) ) * lainv

    fvout(4) = 0.5_wp * ( minus + plus )
    fvout(1) = fvin(1)
    fvout(2) = fvin(2)
    fvout(3) = 0.5_wp * ( minus - plus )

  end subroutine boost_zhyp


  ! Independent W prediction study, 2026-11-13. Only new measurements.
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

  ! W predictivity functions: direct kinematics, no memoized dependencies.
  real(kind=wp) function eval_wpred(npar,code) result(res)
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
    integer, intent(in) :: npar,code
    integer :: ilep,key
    real(kind=wp), dimension(4) :: pl,pn
    real(kind=wp) :: ptl,ptnu,mtlep
    res=-1.0_wp
    if(npar<2) return
    ilep=abs(code)/1000; key=mod(abs(code),1000)
    pl=kin(npar)%p(:,npar-2+ilep); pn=kin(npar)%p(:,npar+1-ilep)
    ptl=v1_pt(pl); ptnu=v1_pt(pn)
    mtlep=sqrt(max(2.0_wp*(ptl*ptnu-pl(1)*pn(1)-pl(2)*pn(2)),0.0_wp))
    if(key==1.and.ptl<=28.0_wp) return
    if(key==2.and.ptl<=32.0_wp) return
    if(key==3.and.mtlep<=60.0_wp) return
    res=abs(v1_y(pl))
  end function eval_wpred_eta

end module EvalFuncs_mod
