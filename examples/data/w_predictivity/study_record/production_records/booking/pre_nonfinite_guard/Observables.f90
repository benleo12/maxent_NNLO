!-------------------------------------------------------------------------------
! NNLOJET
!-------------------------------------------------------------------------------
!
! MODULE: Observables [external members: `_obs`]
!
! DESCRIPTION:
!> Module to facilitate the evaluation of observables and their management.
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
module Observables_mod
!-------------------------------------------------------------------------------
  use EvalObs_mod
  use EvalFuncs_mod
  use ObsHelper_mod
  use KinData_mod
  implicit none
  private

  public :: init_obs, destroy_obs
  public :: digestActive_obs
  public :: initBuffer_obs, destroyBuffer_obs
  public :: nUsed_obs, isUsed_obs, isInteger_obs
  public :: isManual_obs, unsetManual_obs, setManual_obs
  public :: isComposite_obs, needsFlav_obs
  public :: getNameFromId_obs, getIdFromName_obs
  public :: evalAll_obs, getValue_obs, getValueComposite_obs
  public :: evalFlav_obs, saveFlav_obs, loadFlav_obs
  public :: printAll_obs
  public :: npar_current

  interface unsetManual_obs
    module procedure unsetManual_single_, unsetManual_all_
  end interface

  integer, parameter, public :: max_name_length_obs = 32
  integer, parameter, public :: max_desc_length_obs = 224  ! 256-32 (cache alignment)
  type ObsInfo_t
    character(len=max_name_length_obs) :: name
    character(len=max_desc_length_obs) :: desc
    type(EvalObs_t) :: eval
    integer :: iused = 0
    !> factor by which min(sqrt(shat)) would scale, e.g.
    !>  mll > mll_cut => min_shat > (mll_cut)^2  => ifac = 1
    !>  pt  > pt_cut  => min_shat > (2*pt_cut)^2 => ifac = 2
    integer :: ifac = 0
  end type ObsInfo_t

  integer :: n_obs_ = 0, nUsed_obs = 0
  type(ObsInfo_t), allocatable, dimension(:) :: list_obs_
  type(EvalObs_t), allocatable, dimension(:) :: used_obs_
!$OMP THREADPRIVATE(used_obs_)
  !> flavour-tagging implementaiton needs threadlocal arrays for
  !> writeObs_flav & readObs_flav
  real(kind=wp), allocatable, dimension(:) :: flav_used_value_
  logical, allocatable, dimension(:) :: flav_used_valid_
!$OMP THREADPRIVATE(flav_used_value_, flav_used_valid_)


  !> store some information on the kinematics:
  !>  * npar_current: currently active npar, can be different for the threads!
  !>    (i.e. associated with the cached obs)
  integer :: npar_current
!$OMP THREADPRIVATE(npar_current)

contains

  !-----------------------------------------------------------------------------
  !> @brief
  !> activate a new observable for the process by binding a name to
  !> an evaluation fuction (+properties)
  !
  !> @param[in] name unique observable name
  !> @param[in] desc description of the observable for the process
  !> @param[in] eval EvalObs_t containing all information on how to evaluate
  !-----------------------------------------------------------------------------
  subroutine bind_obs(name, desc, eval, ifac)
    character(len=*), intent(in) :: name, desc
    type(EvalObs_t), intent(in) :: eval
    type(ObsInfo_t), allocatable, dimension(:) :: tmp
    integer, intent(in), optional :: ifac
    integer :: ifac_

    !> minimum size of 10
    if (.not. allocated(list_obs_)) allocate (list_obs_(10))
    !> dynamically grow by factor of 2
    if (size(list_obs_) == n_obs_) then
      allocate (tmp(n_obs_))
      tmp(1:n_obs_) = list_obs_(1:n_obs_)
      deallocate (list_obs_)
      allocate (list_obs_(2*n_obs_))
      list_obs_(1:n_obs_) = tmp(1:n_obs_)
      deallocate (tmp)
    end if
    !> add the new entry
    n_obs_ = n_obs_ + 1
    ifac_ = 0
    if (present(ifac)) ifac_ = ifac
    list_obs_(n_obs_) = ObsInfo_t(name=name, desc=desc, eval=eval, ifac=ifac_)

  end subroutine bind_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Initialization routine for Observables. Register all allowed Observables
  !> with their names and associate it to the corresponding evaluation
  !> function. Position in list_obs_ corresponds to the unique identifyer of the
  !> observable.
  !-----------------------------------------------------------------------------
  subroutine init_obs()
    use Process_mod, only: name_proc
    use Spikes_mod, only: initialiseSpike
    use Flavour_mod, only: isActive_flav
    use EvalDIS_mod
#ifdef USEFASTJET
    use FastJet_mod
#endif
    !> note: depending on the process we can select the subset of allowed
    !> observables and change their respective string identifyer
    !> random number init
    integer :: i, rnd_n
    integer, dimension(:), allocatable :: rnd_seed
#ifdef USEFASTJET
    integer :: ialgo, idx
    character(:), allocatable :: abbr
    logical :: wflav = .false.
#endif

    !> set the intrinsic Fortran random number generator
    call random_seed(size=rnd_n)
    allocate(rnd_seed(rnd_n))
    rnd_seed = 42 * (/ (i - 1, i = 1, rnd_n) /)
    call random_seed(put=rnd_seed)
    deallocate(rnd_seed)

    !> init to invalid states
    npar_current = -1

    !------------------------
    ! set boson observables !
    !------------------------
    !> `npar-1` : anti-lepton
    !> `npar`   : lepton
    !> @todo check that the lepton assignments are correct

    select case (name_proc)
    case ("Z", "ZJ", "ZJ_H", "ZJJ", "ZJJJ", "ZQT", "Z_EW")
      ! leptons from the Z -> lp lm decay
      call bind_obs("ptlm", "transverse momentum of l-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("Elm", "energy of l-", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("ylm", "(pseudo-)rapidity of l-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ylm", "|ylm|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptlp", "transverse momentum of l+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("Elp", "energy of l+", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("ylp", "(pseudo-)rapidity of l+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ylp", "|ylp|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      ! leading & sub-leading lepton
      call bind_obs("ptl1", "transverse momentum of leading lepton", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=1))
      call bind_obs("yl1", "(pseudo-)rapidity of leading lepton", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=2))
      call bind_obs("abs_yl1", "|yl1|", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=2, decorator=abs_deco))
      call bind_obs("ptl2", "transverse momentum of sub-leading lepton", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=3))
      call bind_obs("yl2", "(pseudo-)rapidity of sub-leading lepton", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=4))
      call bind_obs("abs_yl2", "|yl2|", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=4, decorator=abs_deco))
      call bind_obs("dphi_l1l2", "Azimuthal angle difference of di-lepton", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=6))
      call bind_obs("ptavg_l1l2", "(ptl1+ptl2)/2", &
                    EvalObs_t(eval=eval_ptavg_l1l2))
      call bind_obs("ptavg_geom_l1l2", "sqrt(ptl1*ptl2)", &
                    EvalObs_t(eval=eval_ptavg_geom_l1l2))
      ! Z-boson = lp + lm
      call bind_obs("ptz", "transverse momentum of Z", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("mll", "invariant mass of Z", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      ! Chebyshev moments T_n(u(m_ll)), linear map on [66,116] GeV.
      ! Book each as a single-bin PROFILE histogram (profile = T) to get the
      ! literal event-level moment Sum_j w_j T_n; divide by rate for <T_n>.
      ! imem is NEGATIVE (order = abs(imem)): the eval is non-memoized, so we
      ! must avoid the imem>0 cache-reset call eval_mem(0,0) in unset().
      call bind_obs("chebT_ctl_1", "T_1 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-1))
      call bind_obs("chebT_ctl_2", "T_2 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-2))
      call bind_obs("chebT_ctl_3", "T_3 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-3))
      call bind_obs("chebT_ctl_4", "T_4 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-4))
      call bind_obs("chebT_ctl_5", "T_5 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-5))
      call bind_obs("chebT_ctl_6", "T_6 |cos th*_CS| (ll)", EvalObs_t(eval_mem=eval_chebT_ctl, imem=-6))
      call bind_obs("chebTT_mllct_11", "T_1(m_ll) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-11))
      call bind_obs("chebTT_mllct_12", "T_1(m_ll) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-12))
      call bind_obs("chebTT_mllct_13", "T_1(m_ll) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-13))
      call bind_obs("chebTT_mllct_14", "T_1(m_ll) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-14))
      call bind_obs("chebTT_mllct_21", "T_2(m_ll) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-21))
      call bind_obs("chebTT_mllct_22", "T_2(m_ll) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-22))
      call bind_obs("chebTT_mllct_23", "T_2(m_ll) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-23))
      call bind_obs("chebTT_mllct_24", "T_2(m_ll) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-24))
      call bind_obs("chebTT_mllct_31", "T_3(m_ll) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-31))
      call bind_obs("chebTT_mllct_32", "T_3(m_ll) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-32))
      call bind_obs("chebTT_mllct_33", "T_3(m_ll) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-33))
      call bind_obs("chebTT_mllct_34", "T_3(m_ll) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-34))
      call bind_obs("chebTT_mllct_41", "T_4(m_ll) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-41))
      call bind_obs("chebTT_mllct_42", "T_4(m_ll) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-42))
      call bind_obs("chebTT_mllct_43", "T_4(m_ll) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-43))
      call bind_obs("chebTT_mllct_44", "T_4(m_ll) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_mllct, imem=-44))
      call bind_obs("chebT_mll_1", "T_1 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-1), ifac=1)
      call bind_obs("chebT_mll_2", "T_2 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-2), ifac=1)
      call bind_obs("chebT_mll_3", "T_3 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-3), ifac=1)
      call bind_obs("chebT_mll_4", "T_4 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-4), ifac=1)
      call bind_obs("chebT_mll_5", "T_5 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-5), ifac=1)
      call bind_obs("chebT_mll_6", "T_6 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-6), ifac=1)
      call bind_obs("chebT_mll_7", "T_7 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-7), ifac=1)
      call bind_obs("chebT_mll_8", "T_8 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-8), ifac=1)
      call bind_obs("chebT_mll_9", "T_9 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-9), ifac=1)
      call bind_obs("chebT_mll_10", "T_10 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-10), ifac=1)
      call bind_obs("chebT_mll_11", "T_11 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-11), ifac=1)
      call bind_obs("chebT_mll_12", "T_12 of m_ll (lin 66-116)", &
                    EvalObs_t(eval_mem=eval_chebT_mll, imem=-12), ifac=1)
      ! Chebyshev moments T_n(u(|y_ll|)), linear map on [0,2.4].
      call bind_obs("chebT_absyz_1", "T_1 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-1))
      call bind_obs("chebT_absyz_2", "T_2 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-2))
      call bind_obs("chebT_absyz_3", "T_3 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-3))
      call bind_obs("chebT_absyz_4", "T_4 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-4))
      call bind_obs("chebT_absyz_5", "T_5 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-5))
      call bind_obs("chebT_absyz_6", "T_6 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-6))
      call bind_obs("chebT_absyz_7", "T_7 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-7))
      call bind_obs("chebT_absyz_8", "T_8 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-8))
      call bind_obs("chebT_absyz_9", "T_9 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-9))
      call bind_obs("chebT_absyz_10", "T_10 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-10))
      call bind_obs("chebT_absyz_11", "T_11 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-11))
      call bind_obs("chebT_absyz_12", "T_12 of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebT_absyz, imem=-12))
      ! WEIGHTED Chebyshev T_n(u)/sqrt(1-u^2), event-level, no regulator (basis test).
      call bind_obs("chebW_mll_0", "T_0/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-10), ifac=1)
      call bind_obs("chebW_mll_1", "T_1/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-1), ifac=1)
      call bind_obs("chebW_mll_2", "T_2/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-2), ifac=1)
      call bind_obs("chebW_mll_3", "T_3/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-3), ifac=1)
      call bind_obs("chebW_mll_4", "T_4/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-4), ifac=1)
      call bind_obs("chebW_mll_5", "T_5/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-5), ifac=1)
      call bind_obs("chebW_mll_6", "T_6/sqrt(1-u^2) of m_ll (BW 66-116)", &
                    EvalObs_t(eval_mem=eval_chebW_mll, imem=-6), ifac=1)
      call bind_obs("chebW_absyz_0", "T_0/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-10))
      call bind_obs("chebW_absyz_1", "T_1/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-1))
      call bind_obs("chebW_absyz_2", "T_2/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-2))
      call bind_obs("chebW_absyz_3", "T_3/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-3))
      call bind_obs("chebW_absyz_4", "T_4/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-4))
      call bind_obs("chebW_absyz_5", "T_5/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-5))
      call bind_obs("chebW_absyz_6", "T_6/sqrt(1-u^2) of |y_ll| (lin 0-2.4)", &
                    EvalObs_t(eval_mem=eval_chebW_absyz, imem=-6))
      ! Chebyshev moments T_n(u(pT_ll)), LOG map [30,2500] (orders 1-20; 13-20 added 2026-09-18).
      ! Book with per-histogram selector: select ptz min=30 max=500.
      call bind_obs("chebT_ptz_1", "T_1 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-1), ifac=2)
      call bind_obs("chebT_ptz_2", "T_2 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-2), ifac=2)
      call bind_obs("chebT_ptz_3", "T_3 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-3), ifac=2)
      call bind_obs("chebT_ptz_4", "T_4 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-4), ifac=2)
      call bind_obs("chebT_ptz_5", "T_5 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-5), ifac=2)
      call bind_obs("chebT_ptz_6", "T_6 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-6), ifac=2)
      call bind_obs("chebT_ptz_7", "T_7 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-7), ifac=2)
      call bind_obs("chebT_ptz_8", "T_8 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-8), ifac=2)
      call bind_obs("chebT_ptz_9", "T_9 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-9), ifac=2)
      call bind_obs("chebT_ptz_10", "T_10 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-10), ifac=2)
      call bind_obs("chebT_ptz_11", "T_11 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-11), ifac=2)
      call bind_obs("chebT_ptz_12", "T_12 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-12), ifac=2)
      call bind_obs("chebT_ptz_13", "T_13 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-13), ifac=2)
      call bind_obs("chebT_ptz_14", "T_14 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-14), ifac=2)
      call bind_obs("chebT_ptz_15", "T_15 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-15), ifac=2)
      call bind_obs("chebT_ptz_16", "T_16 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-16), ifac=2)
      call bind_obs("chebT_ptz_17", "T_17 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-17), ifac=2)
      call bind_obs("chebT_ptz_18", "T_18 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-18), ifac=2)
      call bind_obs("chebT_ptz_19", "T_19 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-19), ifac=2)
      call bind_obs("chebT_ptz_20", "T_20 of pT_ll (log 30-2500)", &
                    EvalObs_t(eval_mem=eval_chebT_ptz, imem=-20), ifac=2)
      ! SMOOTH-profile recoil moments: w(pT)*T_n, profile [30,60]->500, log map
      ! [30,2500].  Single-bin PROFILE histograms, NO selector (w does the window).
      ! <T_n>_FO(w)=prof(chebTw_ptz_n)/prof(w_ptz); R=prof(w_ptz)/sigma_fid.
      call bind_obs("w_ptz", "recoil profile w(pT_ll) [30,60,500]", &
                    EvalObs_t(eval=eval_w_ptz), ifac=2)
      call bind_obs("chebTw_ptz_1", "w*T_1 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-1), ifac=2)
      call bind_obs("chebTw_ptz_2", "w*T_2 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-2), ifac=2)
      call bind_obs("chebTw_ptz_3", "w*T_3 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-3), ifac=2)
      call bind_obs("chebTw_ptz_4", "w*T_4 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-4), ifac=2)
      call bind_obs("chebTw_ptz_5", "w*T_5 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-5), ifac=2)
      call bind_obs("chebTw_ptz_6", "w*T_6 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-6), ifac=2)
      call bind_obs("chebTw_ptz_7", "w*T_7 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-7), ifac=2)
      call bind_obs("chebTw_ptz_8", "w*T_8 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-8), ifac=2)
      call bind_obs("chebTw_ptz_9", "w*T_9 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-9), ifac=2)
      call bind_obs("chebTw_ptz_10", "w*T_10 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-10), ifac=2)
      call bind_obs("chebTw_ptz_11", "w*T_11 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-11), ifac=2)
      call bind_obs("chebTw_ptz_12", "w*T_12 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-12), ifac=2)
      call bind_obs("chebTw_ptz_13", "w*T_13 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-13), ifac=2)
      call bind_obs("chebTw_ptz_14", "w*T_14 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-14), ifac=2)
      call bind_obs("chebTw_ptz_15", "w*T_15 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-15), ifac=2)
      call bind_obs("chebTw_ptz_16", "w*T_16 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-16), ifac=2)
      call bind_obs("chebTw_ptz_17", "w*T_17 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-17), ifac=2)
      call bind_obs("chebTw_ptz_18", "w*T_18 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-18), ifac=2)
      call bind_obs("chebTw_ptz_19", "w*T_19 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-19), ifac=2)
      call bind_obs("chebTw_ptz_20", "w*T_20 pT_ll (log 30-2500)", EvalObs_t(eval_mem=eval_chebTw_ptz, imem=-20), ifac=2)
      ! ---- DY+jet (ZJ) smooth-profile moments: pT_j1, pT_j2 (above cut), dphi(l1,l2) (away from pi)
      call bind_obs("w_ptj1", "profile w(pT_j1)", EvalObs_t(eval=eval_w_ptj1, min_njets=1), ifac=2)
      call bind_obs("chebTw_ptj1_1", "w*T_1 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-1), ifac=2)
      call bind_obs("chebTw_ptj1_2", "w*T_2 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-2), ifac=2)
      call bind_obs("chebTw_ptj1_3", "w*T_3 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-3), ifac=2)
      call bind_obs("chebTw_ptj1_4", "w*T_4 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-4), ifac=2)
      call bind_obs("chebTw_ptj1_5", "w*T_5 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-5), ifac=2)
      call bind_obs("chebTw_ptj1_6", "w*T_6 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-6), ifac=2)
      call bind_obs("chebTw_ptj1_7", "w*T_7 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-7), ifac=2)
      call bind_obs("chebTw_ptj1_8", "w*T_8 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-8), ifac=2)
      call bind_obs("chebTw_ptj1_9", "w*T_9 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-9), ifac=2)
      call bind_obs("chebTw_ptj1_10", "w*T_10 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-10), ifac=2)
      call bind_obs("chebTw_ptj1_11", "w*T_11 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-11), ifac=2)
      call bind_obs("chebTw_ptj1_12", "w*T_12 pT_j1", EvalObs_t(min_njets=1, eval_mem=eval_chebTw_ptj1, imem=-12), ifac=2)
      call bind_obs("w_ptj2", "profile w(pT_j2)", EvalObs_t(eval=eval_w_ptj2, min_njets=2), ifac=2)
      call bind_obs("w_ptj12", "w(pTj1)w(pTj2)", EvalObs_t(min_njets=2, eval=eval_w_ptj12))
      call bind_obs("chebTT_ptj12_11", "w T_1(pTj1) T_1(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-11))
      call bind_obs("chebTT_ptj12_12", "w T_1(pTj1) T_2(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-12))
      call bind_obs("chebTT_ptj12_13", "w T_1(pTj1) T_3(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-13))
      call bind_obs("chebTT_ptj12_14", "w T_1(pTj1) T_4(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-14))
      call bind_obs("chebTT_ptj12_21", "w T_2(pTj1) T_1(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-21))
      call bind_obs("chebTT_ptj12_22", "w T_2(pTj1) T_2(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-22))
      call bind_obs("chebTT_ptj12_23", "w T_2(pTj1) T_3(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-23))
      call bind_obs("chebTT_ptj12_24", "w T_2(pTj1) T_4(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-24))
      call bind_obs("chebTT_ptj12_31", "w T_3(pTj1) T_1(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-31))
      call bind_obs("chebTT_ptj12_32", "w T_3(pTj1) T_2(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-32))
      call bind_obs("chebTT_ptj12_33", "w T_3(pTj1) T_3(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-33))
      call bind_obs("chebTT_ptj12_34", "w T_3(pTj1) T_4(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-34))
      call bind_obs("chebTT_ptj12_41", "w T_4(pTj1) T_1(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-41))
      call bind_obs("chebTT_ptj12_42", "w T_4(pTj1) T_2(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-42))
      call bind_obs("chebTT_ptj12_43", "w T_4(pTj1) T_3(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-43))
      call bind_obs("chebTT_ptj12_44", "w T_4(pTj1) T_4(pTj2)", EvalObs_t(min_njets=2, eval_mem=eval_chebTT_ptj12, imem=-44))
      call bind_obs("chebTw_ptj2_1", "w*T_1 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-1), ifac=2)
      call bind_obs("chebTw_ptj2_2", "w*T_2 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-2), ifac=2)
      call bind_obs("chebTw_ptj2_3", "w*T_3 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-3), ifac=2)
      call bind_obs("chebTw_ptj2_4", "w*T_4 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-4), ifac=2)
      call bind_obs("chebTw_ptj2_5", "w*T_5 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-5), ifac=2)
      call bind_obs("chebTw_ptj2_6", "w*T_6 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-6), ifac=2)
      call bind_obs("chebTw_ptj2_7", "w*T_7 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-7), ifac=2)
      call bind_obs("chebTw_ptj2_8", "w*T_8 pT_j2", EvalObs_t(min_njets=2, eval_mem=eval_chebTw_ptj2, imem=-8), ifac=2)
      call bind_obs("w_dphill", "profile w(pi-dphi_ll)", EvalObs_t(eval=eval_w_dphill))
      call bind_obs("chebTw_dphill_1", "w*T_1 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-1))
      call bind_obs("chebTw_dphill_2", "w*T_2 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-2))
      call bind_obs("chebTw_dphill_3", "w*T_3 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-3))
      call bind_obs("chebTw_dphill_4", "w*T_4 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-4))
      call bind_obs("chebTw_dphill_5", "w*T_5 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-5))
      call bind_obs("chebTw_dphill_6", "w*T_6 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-6))
      call bind_obs("chebTw_dphill_7", "w*T_7 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-7))
      call bind_obs("chebTw_dphill_8", "w*T_8 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-8))
      call bind_obs("chebTw_dphill_9", "w*T_9 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-9))
      call bind_obs("chebTw_dphill_10", "w*T_10 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-10))
      call bind_obs("chebTw_dphill_11", "w*T_11 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-11))
      call bind_obs("chebTw_dphill_12", "w*T_12 (pi-dphi_ll)", EvalObs_t(eval_mem=eval_chebTw_dphill, imem=-12))
      ! ---- DY lepton Chebyshev moments (leading/subleading by pT) ----
      call bind_obs("chebT_ptl1_1", "T_1 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-1), ifac=2)
      call bind_obs("chebT_ptl1_2", "T_2 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-2), ifac=2)
      call bind_obs("chebT_ptl1_3", "T_3 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-3), ifac=2)
      call bind_obs("chebT_ptl1_4", "T_4 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-4), ifac=2)
      call bind_obs("chebT_ptl1_5", "T_5 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-5), ifac=2)
      call bind_obs("chebT_ptl1_6", "T_6 ptl1", EvalObs_t(eval_mem=eval_chebT_ptl1, imem=-6), ifac=2)
      call bind_obs("chebT_ptl2_1", "T_1 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-1), ifac=2)
      call bind_obs("chebT_ptl2_2", "T_2 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-2), ifac=2)
      call bind_obs("chebT_ptl2_3", "T_3 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-3), ifac=2)
      call bind_obs("chebT_ptl2_4", "T_4 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-4), ifac=2)
      call bind_obs("chebT_ptl2_5", "T_5 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-5), ifac=2)
      call bind_obs("chebT_ptl2_6", "T_6 ptl2", EvalObs_t(eval_mem=eval_chebT_ptl2, imem=-6), ifac=2)
      call bind_obs("chebT_yl1_1", "T_1 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-1))
      call bind_obs("chebT_yl1_2", "T_2 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-2))
      call bind_obs("chebT_yl1_3", "T_3 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-3))
      call bind_obs("chebT_yl1_4", "T_4 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-4))
      call bind_obs("chebT_yl1_5", "T_5 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-5))
      call bind_obs("chebT_yl1_6", "T_6 yl1", EvalObs_t(eval_mem=eval_chebT_yl1, imem=-6))
      call bind_obs("chebT_yl2_1", "T_1 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-1))
      call bind_obs("chebT_yl2_2", "T_2 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-2))
      call bind_obs("chebT_yl2_3", "T_3 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-3))
      call bind_obs("chebT_yl2_4", "T_4 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-4))
      call bind_obs("chebT_yl2_5", "T_5 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-5))
      call bind_obs("chebT_yl2_6", "T_6 yl2", EvalObs_t(eval_mem=eval_chebT_yl2, imem=-6))
      call bind_obs("etz", "transverse energy: sqrt(mll^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      call bind_obs("yz", "rapidity of Z", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yz", "|yz|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaz", "pseudo-rapidity of Z", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etaz", "|etaz|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("xFz", "xF = 2*pz/sqrt(s) Feynman-x variable of Z/gamma", &
                    EvalObs_t(eval=eval_xF_V))
      call bind_obs("Ez", "E = pV(4); Energy component of Z", &
                    EvalObs_t(eval=eval_E_V))
      call bind_obs("ptzsq", "ptz^2: squared transverse momentum of Z vector boson", &
                    EvalObs_t(eval=eval_ptsq_V))
      call bind_obs("dr_l12", "Delta R (lp, lm)", &
                    EvalObs_t(eval=eval_dR_l12))
      ! Z : two charged leptons
      call bind_obs("min_dr_lj1", "minimum of Delta R (leading jet, leptons)", &
                    EvalObs_t(eval=eval_min_dR_l12j1))
      call bind_obs("min_dr_lj", "minimum of Delta R (jets, leptons)", &
                    EvalObs_t(eval=eval_min_dR_l12j))
      call bind_obs("min_dr_zj", "minimum of Delta R (jets, Z-boson)", &
                    EvalObs_t(eval=eval_min_dR_Vj))
      call bind_obs("ptrat_zclj", "ptz / ptj[closest]", &
                    EvalObs_t(eval=eval_ptrat_Vclj, min_njets=1))

      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)

      call bind_obs("z_ht", "etz + sum pt{partons}", &
                    EvalObs_t(eval=eval_ZJ_HT))

      call bind_obs("phi_star", "\phi^*_\eta ", &
                    EvalObs_t(eval=eval_phi_star_eta))
      ! obs for Z + >=1 jet
      call bind_obs("pt_Zj", "pt of the Z + leading-jet system", &
                    EvalObs_t(eval=eval_pt_Vj, min_njets=1), ifac=2)
      call bind_obs("ptavg_Zj", " = (ptz + ptj1)/2", &
                    EvalObs_t(eval=eval_ptavg_Vj, min_njets=1))
      call bind_obs("dpt_Zj", " = ptz - ptj1", &
                    EvalObs_t(eval=eval_dpt_Vj, min_njets=1))
      call bind_obs("abs_dpt_Zj", " = |ptz - ptj1|", &
                    EvalObs_t(eval=eval_dpt_Vj, min_njets=1, decorator=abs_deco))
      call bind_obs("ystar_Zj", "ystar = |yZ-yj1|/2", &
                    EvalObs_t(eval=eval_ystar_Vj, min_njets=1))
      call bind_obs("yboost_Zj", "yboost = |yZ+yj1|/2", &
                    EvalObs_t(eval=eval_yboost_Vj, min_njets=1))
      call bind_obs("dphi_Zj1", "Azimuthal angle difference of V and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dphi_Zj2", "Azimuthal angle difference of V and 2nd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-2, min_njets=2))
      call bind_obs("dphi_Zj3", "Azimuthal angle difference of V and 3rd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-3, min_njets=3))
      call bind_obs("dphi_Zj4", "Azimuthal angle difference of V and 4th jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-4, min_njets=4))
      ! min(\sum deltaR(jet_i,lepton))
      ! call set_active(EvalObs_t(eval=eval_min_dR_l1j),  "min_dr_l1j", &
      !                & "minimum of Delta R (jets, lm)")
      ! call set_active(EvalObs_t(eval=eval_min_dR_l2j),  "min_dr_l2j", &
      !                & "minimum of Delta R (jets, lp)")
      call bind_obs("max_pt_in_gap", "highest pt of jets within y-gap of j1 & j2", &
                    EvalObs_t(eval=eval_VFH_max_pt_in_gap, min_njets=3))
      call bind_obs("deltaphi_j12_rap_ord", "signed azimuthal angle between j1 & j2 (y1 > y2)", &
                    EvalObs_t(eval=eval_deltaphi_j12_rap_ord, min_njets=2))

      ! obs comparing Z+j1 system with j1 alone
      call bind_obs("ptZj_vs_ptj1", " pt of Z+j1 system compared to pt of j1", &
                    EvalObs_t(eval_mem=eval_pt_Vj_vs_pt_j1, imem=-1), ifac=2)

      ! Collins-Soper frame (angles wrt l1   = 47
      call bind_obs("cos_theta_CS", "cos(theta) in Collins-Soper frame ", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=1))
      call bind_obs("theta_CS", "theta in Collins-Soper frame ", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=2))
      call bind_obs("phi_CS", "phi in Collins-Soper frame ", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=3))
      ! projectors for the angular coefficients A_i [1606.00689]
      call bind_obs("proj_A0", "projector for the angular coefficient A0 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=4))
      call bind_obs("proj_A1", "projector for the angular coefficient A1 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=5))
      call bind_obs("proj_A2", "projector for the angular coefficient A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=6))
      call bind_obs("proj_A3", "projector for the angular coefficient A3 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=7))
      call bind_obs("proj_A4", "projector for the angular coefficient A4 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=8))
      call bind_obs("proj_A5", "projector for the angular coefficient A5 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=9))
      call bind_obs("proj_A6", "projector for the angular coefficient A6 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=10))
      call bind_obs("proj_A7", "projector for the angular coefficient A7 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=11))
      call bind_obs("proj_A0mA2", "projector for the difference A0-A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=12))
      call bind_obs("cos_theta_CS_0PT", "cos(theta) in Collins-Soper frame, usable at PTZ=0 ", &
                    EvalObs_t(eval=eval_cos_theta_CS_0PT))
      ! dynamical scales
      call bind_obs("dscale1", "sqrt(mll^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("dscale2", "etz + ptj1", &
                    EvalObs_t(eval=eval_ZJ_dscale2, min_njets=1))
      call bind_obs("dscale3", "etz + sum pt{jets}", &
                    EvalObs_t(eval=eval_ZJ_dscale3))
      if(isActive_flav())then
        ! flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj2", "transverse momentum of subleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-2, min_flav_njets=2, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj3", "transverse momentum of subsubleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-3, min_flav_njets=3, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_etaj1", "pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_etaj1", "absolute pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yj1", "rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_yj1", "absolute rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dy_Zj1", "deltay(abs) of leading flavour-tagged jet with Z", &
                      EvalObs_t(eval_mem=eval_flav_deltay_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dphi_Zj1", "deltaphi of leading flavour-tagged jet with Z", &
                      EvalObs_t(eval_mem=eval_flav_deltaphi_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dR_Zj1", "deltaR of leading flavour-tagged jet with Z", &
                      EvalObs_t(eval_mem=eval_flav_deltaR_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yboost_Zj1", "yboost of leading flavour-tagged jet with Z", &
                      EvalObs_t(eval_mem=eval_flav_yboost_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dscale2", "etz + ptj1_flav", &
                      EvalObs_t(eval=eval_flav_VJ_dscale2, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_xF_ptj1", "xFeynman of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_xF_ptjet_i, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        ! end flavour-tagging
      endif

      call bind_obs("tau0", "0-jettiness for V+j", &
                    EvalObs_t(eval=eval_VJ_tau0))
      call bind_obs("tau1", "1-jettiness for V+j", &
                    EvalObs_t(eval=eval_VJ_tau1))


    case ("WM", "WMJ", "WMJUNSYM", "WMJ_H", "WMJJ", "WMJJJ", "WMQT")
      ! leptons from the Wm -> nu-bar lm decay
      call bind_obs("ptlm", "transverse momentum of l-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("Elm", "energy of l-", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Enu", "energy of neutrino", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("ylm", "(pseudo-)rapidity of l-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ylm", "|ylm|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("etmiss", "missing transverse energy/momentum", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ynu", "(pseudo-)rapidity of nu-bar", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ynu", "|ynu|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("deltaphi_lj1", "Azimuthal angle difference between lepton and first jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(1, 1), min_njets=1))
      call bind_obs("deltaphi_lj2", "Azimuthal angle difference between lepton and second jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(1, 2), min_njets=2))
      call bind_obs("deltaphi_lj3", "Azimuthal angle difference between lepton and third jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(1, 3), min_njets=3))
      ! Wm-boson = nu-bar + lm
      call bind_obs("ptw", "transverse momentum of W", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("mt", "transverse mass of W", &
                    EvalObs_t(eval=eval_mt_V), ifac=1)
      ! Wm : only one charged lepton at `npar`
      !      note, same name as in Z but different eval function
      call bind_obs("min_dr_lj", "minimum of Delta R (jets, l-)", &
                    EvalObs_t(eval=eval_min_dR_l1j))
      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      call bind_obs("ptjlm", "Transverse component of vector sum of jet and l-", &
                    EvalObs_t(eval=eval_pt_jl1, min_njets=1), ifac=2)
      call bind_obs("htjlm", "Sclar sum of ptj1 and ptlm", &
                    EvalObs_t(eval=eval_ht_jl1, min_njets=1))
      call bind_obs("yw", "rapidity of W", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yw", "|yw|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaw", "pseudo-rapidity of W", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("mln", "invariant mass of W", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("r_w", "ratio of ptw/mln", &
                    EvalObs_t(eval=eval_r_V))
      call bind_obs("etw", "transverse energy: sqrt(mln^2 + ptw^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      ! ---- W event-level Chebyshev moments (maxent_upgrade input): m_W BW map, |y_W| lin, pT_W profiled
      call bind_obs("chebT_mw_1", "T_1 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-1), ifac=1)
      call bind_obs("chebT_mw_2", "T_2 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-2), ifac=1)
      call bind_obs("chebT_mw_3", "T_3 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-3), ifac=1)
      call bind_obs("chebT_mw_4", "T_4 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-4), ifac=1)
      call bind_obs("chebT_mw_5", "T_5 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-5), ifac=1)
      call bind_obs("chebT_mw_6", "T_6 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-6), ifac=1)
      call bind_obs("chebT_mw_7", "T_7 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-7), ifac=1)
      call bind_obs("chebT_mw_8", "T_8 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-8), ifac=1)
      call bind_obs("chebT_mw_9", "T_9 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-9), ifac=1)
      call bind_obs("chebT_mw_10", "T_10 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-10), ifac=1)
      call bind_obs("chebT_mw_11", "T_11 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-11), ifac=1)
      call bind_obs("chebT_mw_12", "T_12 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-12), ifac=1)
      call bind_obs("chebT_absyw_1", "T_1 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-1))
      call bind_obs("chebT_absyw_2", "T_2 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-2))
      call bind_obs("chebT_absyw_3", "T_3 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-3))
      call bind_obs("chebT_absyw_4", "T_4 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-4))
      call bind_obs("chebT_absyw_5", "T_5 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-5))
      call bind_obs("chebT_absyw_6", "T_6 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-6))
      call bind_obs("chebT_absyw_7", "T_7 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-7))
      call bind_obs("chebT_absyw_8", "T_8 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-8))
      call bind_obs("chebT_absyw_9", "T_9 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-9))
      call bind_obs("chebT_absyw_10", "T_10 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-10))
      call bind_obs("chebT_absyw_11", "T_11 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-11))
      call bind_obs("chebT_absyw_12", "T_12 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-12))
      call bind_obs("w_ptw", "recoil profile w(pT_W) [25,50,500]", EvalObs_t(eval=eval_w_ptw), ifac=2)
      call bind_obs("chebTw_ptw_1", "w*T_1 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-1), ifac=2)
      call bind_obs("chebTw_ptw_2", "w*T_2 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-2), ifac=2)
      call bind_obs("chebTw_ptw_3", "w*T_3 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-3), ifac=2)
      call bind_obs("chebTw_ptw_4", "w*T_4 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-4), ifac=2)
      call bind_obs("chebTw_ptw_5", "w*T_5 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-5), ifac=2)
      call bind_obs("chebTw_ptw_6", "w*T_6 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-6), ifac=2)
      call bind_obs("chebTw_ptw_7", "w*T_7 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-7), ifac=2)
      call bind_obs("chebTw_ptw_8", "w*T_8 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-8), ifac=2)
      call bind_obs("chebTw_ptw_9", "w*T_9 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-9), ifac=2)
      call bind_obs("chebTw_ptw_10", "w*T_10 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-10), ifac=2)
      call bind_obs("chebTw_ptw_11", "w*T_11 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-11), ifac=2)
      call bind_obs("chebTw_ptw_12", "w*T_12 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-12), ifac=2)
      call bind_obs("chebTw_ptw_13", "w*T_13 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-13), ifac=2)
      call bind_obs("chebTw_ptw_14", "w*T_14 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-14), ifac=2)
      call bind_obs("chebTw_ptw_15", "w*T_15 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-15), ifac=2)
      call bind_obs("chebTw_ptw_16", "w*T_16 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-16), ifac=2)
      call bind_obs("chebTw_ptw_17", "w*T_17 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-17), ifac=2)
      call bind_obs("chebTw_ptw_18", "w*T_18 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-18), ifac=2)
      call bind_obs("chebTw_ptw_19", "w*T_19 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-19), ifac=2)
      call bind_obs("chebTw_ptw_20", "w*T_20 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-20), ifac=2)
      ! ---- signed Collins-Soper decay angle of the charged lepton, lin map [-1,1]
      call bind_obs("chebT_ctw_1", "T_1 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-1))
      call bind_obs("chebT_ctw_2", "T_2 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-2))
      call bind_obs("chebT_ctw_3", "T_3 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-3))
      call bind_obs("chebT_ctw_4", "T_4 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-4))
      call bind_obs("chebT_ctw_5", "T_5 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-5))
      call bind_obs("chebT_ctw_6", "T_6 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-6))
      call bind_obs("chebT_ctw_7", "T_7 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-7))
      call bind_obs("chebT_ctw_8", "T_8 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwm, imem=-8))
      call bind_obs("costh_w", "signed cos th*_CS of the charged lepton (W rest frame)", EvalObs_t(eval_mem=eval_costhCS_w, imem=-1))
      call bind_obs("pred_D_7", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1001))
      call bind_obs("pred_D_8", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1002))
      call bind_obs("pred_D_9", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1003))
      call bind_obs("pred_D_10", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1004))
      call bind_obs("pred_M_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1005))
      call bind_obs("pred_M_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1006))
      call bind_obs("pred_M_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1007))
      call bind_obs("pred_M_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1008))
      call bind_obs("pred_M_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1009))
      call bind_obs("pred_M_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1010))
      call bind_obs("pred_N_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1011))
      call bind_obs("pred_N_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1012))
      call bind_obs("pred_N_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1013))
      call bind_obs("pred_N_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1014))
      call bind_obs("pred_N_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1015))
      call bind_obs("pred_N_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1016))
      call bind_obs("pred_N_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1017))
      call bind_obs("pred_NJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1018))
      call bind_obs("pred_NJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1019))
      call bind_obs("pred_NJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1020))
      call bind_obs("pred_NJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1021))
      call bind_obs("pred_NJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1022))
      call bind_obs("pred_NJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1023))
      call bind_obs("pred_G_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1024))
      call bind_obs("pred_G_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1025))
      call bind_obs("pred_G_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1026))
      call bind_obs("pred_G_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1027))
      call bind_obs("pred_G_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1028))
      call bind_obs("pred_G_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1029))
      call bind_obs("pred_G_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1030))
      call bind_obs("pred_GJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1031))
      call bind_obs("pred_GJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1032))
      call bind_obs("pred_GJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1033))
      call bind_obs("pred_GJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1034))
      call bind_obs("pred_GJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1035))
      call bind_obs("pred_GJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1036))
      call bind_obs("pred_W_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1037))
      call bind_obs("pred_W_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1038))
      call bind_obs("pred_W_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1039))
      call bind_obs("pred_W_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1040))
      call bind_obs("pred_W_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1041))
      call bind_obs("pred_W_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1042))
      call bind_obs("pred_W_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1043))
      call bind_obs("pred_WJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1044))
      call bind_obs("pred_WJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1045))
      call bind_obs("pred_WJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1046))
      call bind_obs("pred_WJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1047))
      call bind_obs("pred_WJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1048))
      call bind_obs("pred_WJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1049))
      call bind_obs("pred_H_0_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1050))
      call bind_obs("pred_H_0_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1051))
      call bind_obs("pred_H_0_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1052))
      call bind_obs("pred_H_0_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1053))
      call bind_obs("pred_H_0_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1054))
      call bind_obs("pred_H_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1055))
      call bind_obs("pred_H_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1056))
      call bind_obs("pred_H_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1057))
      call bind_obs("pred_H_1_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1058))
      call bind_obs("pred_H_1_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1059))
      call bind_obs("pred_H_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1060))
      call bind_obs("pred_H_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1061))
      call bind_obs("pred_H_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1062))
      call bind_obs("pred_H_2_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1063))
      call bind_obs("pred_H_2_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1064))
      call bind_obs("pred_S_0_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1065))
      call bind_obs("pred_S_0_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1066))
      call bind_obs("pred_S_0_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1067))
      call bind_obs("pred_S_0_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1068))
      call bind_obs("pred_S_0_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1069))
      call bind_obs("pred_S_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1070))
      call bind_obs("pred_S_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1071))
      call bind_obs("pred_S_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1072))
      call bind_obs("pred_S_1_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1073))
      call bind_obs("pred_S_1_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1074))
      call bind_obs("pred_S_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1075))
      call bind_obs("pred_S_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1076))
      call bind_obs("pred_S_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1077))
      call bind_obs("pred_S_2_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1078))
      call bind_obs("pred_S_2_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1079))
      call bind_obs("expl_qc_0_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1080))
      call bind_obs("expl_qc_0_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1081))
      call bind_obs("expl_qc_0_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1082))
      call bind_obs("expl_qc_0_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1083))
      call bind_obs("expl_qc_0_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1084))
      call bind_obs("expl_qc_0_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1085))
      call bind_obs("expl_qc_0_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1086))
      call bind_obs("expl_qc_0_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1087))
      call bind_obs("expl_qc_0_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1088))
      call bind_obs("expl_qc_0_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1089))
      call bind_obs("expl_qc_0_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1090))
      call bind_obs("expl_qc_0_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1091))
      call bind_obs("expl_qc_0_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1092))
      call bind_obs("expl_qc_0_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1093))
      call bind_obs("expl_qc_0_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1094))
      call bind_obs("expl_qc_1_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1095))
      call bind_obs("expl_qc_1_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1096))
      call bind_obs("expl_qc_1_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1097))
      call bind_obs("expl_qc_1_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1098))
      call bind_obs("expl_qc_1_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1099))
      call bind_obs("expl_qc_1_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1100))
      call bind_obs("expl_qc_1_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1101))
      call bind_obs("expl_qc_1_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1102))
      call bind_obs("expl_qc_1_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1103))
      call bind_obs("expl_qc_1_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1104))
      call bind_obs("expl_qc_1_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1105))
      call bind_obs("expl_qc_1_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1106))
      call bind_obs("expl_qc_1_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1107))
      call bind_obs("expl_qc_1_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1108))
      call bind_obs("expl_qc_1_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1109))
      call bind_obs("expl_qc_2_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1110))
      call bind_obs("expl_qc_2_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1111))
      call bind_obs("expl_qc_2_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1112))
      call bind_obs("expl_qc_2_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1113))
      call bind_obs("expl_qc_2_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1114))
      call bind_obs("expl_qc_2_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1115))
      call bind_obs("expl_qc_2_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1116))
      call bind_obs("expl_qc_2_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1117))
      call bind_obs("expl_qc_2_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1118))
      call bind_obs("expl_qc_2_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1119))
      call bind_obs("expl_qc_2_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1120))
      call bind_obs("expl_qc_2_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1121))
      call bind_obs("expl_qc_2_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1122))
      call bind_obs("expl_qc_2_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1123))
      call bind_obs("expl_qc_2_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1124))
      call bind_obs("expl_qm_0_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1125))
      call bind_obs("expl_qm_0_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1126))
      call bind_obs("expl_qm_0_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1127))
      call bind_obs("expl_qm_0_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1128))
      call bind_obs("expl_qm_0_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1129))
      call bind_obs("expl_qm_0_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1130))
      call bind_obs("expl_qm_0_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1131))
      call bind_obs("expl_qm_0_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1132))
      call bind_obs("expl_qm_0_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1133))
      call bind_obs("expl_qm_1_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1134))
      call bind_obs("expl_qm_1_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1135))
      call bind_obs("expl_qm_1_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1136))
      call bind_obs("expl_qm_1_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1137))
      call bind_obs("expl_qm_1_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1138))
      call bind_obs("expl_qm_1_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1139))
      call bind_obs("expl_qm_1_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1140))
      call bind_obs("expl_qm_1_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1141))
      call bind_obs("expl_qm_1_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1142))
      call bind_obs("expl_qm_2_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1143))
      call bind_obs("expl_qm_2_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1144))
      call bind_obs("expl_qm_2_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1145))
      call bind_obs("expl_qm_2_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1146))
      call bind_obs("expl_qm_2_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1147))
      call bind_obs("expl_qm_2_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1148))
      call bind_obs("expl_qm_2_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1149))
      call bind_obs("expl_qm_2_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1150))
      call bind_obs("expl_qm_2_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1151))
      call bind_obs("hold_pt28", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1152))
      call bind_obs("hold_pt32", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1153))
      call bind_obs("hold_hard", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1154))
      call bind_obs("hold_pt32hard", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1155))
      call bind_obs("hold_mt60", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1156))
      call bind_obs("hold_combined", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-1157))
      call bind_obs("pred_eta1", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-1001))
      call bind_obs("pred_eta2", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-1002))
      call bind_obs("pred_eta3", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-1003))
      call bind_obs("wval_pt_1", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-1))
      call bind_obs("wval_pt_2", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-2))
      call bind_obs("wval_pt_3", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-3))
      call bind_obs("wval_pt_4", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-4))
      call bind_obs("wval_pt_5", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-5))
      call bind_obs("wval_pt_6", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-6))
      call bind_obs("wval_pt_7", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-7))
      call bind_obs("wval_pt_8", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-8))
      call bind_obs("wval_pt_9", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-9))
      call bind_obs("wval_pt_10", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-10))
      call bind_obs("wval_pt_11", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-11))
      call bind_obs("wval_pt_12", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-12))
      call bind_obs("wval_pt_13", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-13))
      call bind_obs("wval_pt_14", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-14))
      call bind_obs("wval_pt_15", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-15))
      call bind_obs("wval_pt_16", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-16))
      call bind_obs("wval_pt_17", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-17))
      call bind_obs("wval_pt_18", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-18))
      call bind_obs("wval_pt_19", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-19))
      call bind_obs("wval_pt_20", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-20))
      call bind_obs("wval_yc_10", "windowed W rapidity decay product 10", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1010))
      call bind_obs("wval_yc_11", "windowed W rapidity decay product 11", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1011))
      call bind_obs("wval_yc_12", "windowed W rapidity decay product 12", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1012))
      call bind_obs("wval_yc_13", "windowed W rapidity decay product 13", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1013))
      call bind_obs("wval_yc_14", "windowed W rapidity decay product 14", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1014))
      call bind_obs("wval_yc_20", "windowed W rapidity decay product 20", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1020))
      call bind_obs("wval_yc_21", "windowed W rapidity decay product 21", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1021))
      call bind_obs("wval_yc_22", "windowed W rapidity decay product 22", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1022))
      call bind_obs("wval_yc_23", "windowed W rapidity decay product 23", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1023))
      call bind_obs("wval_yc_24", "windowed W rapidity decay product 24", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1024))
      call bind_obs("wval_yc_30", "windowed W rapidity decay product 30", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1030))
      call bind_obs("wval_yc_31", "windowed W rapidity decay product 31", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1031))
      call bind_obs("wval_yc_32", "windowed W rapidity decay product 32", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1032))
      call bind_obs("wval_yc_33", "windowed W rapidity decay product 33", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1033))
      call bind_obs("wval_yc_34", "windowed W rapidity decay product 34", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1034))
      call bind_obs("wval_yc_40", "windowed W rapidity decay product 40", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1040))
      call bind_obs("wval_yc_41", "windowed W rapidity decay product 41", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1041))
      call bind_obs("wval_yc_42", "windowed W rapidity decay product 42", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1042))
      call bind_obs("wval_yc_43", "windowed W rapidity decay product 43", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1043))
      call bind_obs("wval_yc_44", "windowed W rapidity decay product 44", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-1044))
      call bind_obs("wval_eta27", "charged-lepton eta after pT>27", &
                    EvalObs_t(eval_mem=eval_wval_eta, imem=-127))
      call bind_obs("wval_eta30", "charged-lepton eta after pT>30", &
                    EvalObs_t(eval_mem=eval_wval_eta, imem=-130))
      call bind_obs("w_ctw", "decay-angle window w(cos th*) [|c|<0.5, ramp to 0.7]", EvalObs_t(eval_mem=eval_w_ctw, imem=-1))
      call bind_obs("chebTw_ctw_1", "w*T_1 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-1))
      call bind_obs("chebTw_ctw_2", "w*T_2 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-2))
      call bind_obs("chebTw_ctw_3", "w*T_3 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-3))
      call bind_obs("chebTw_ctw_4", "w*T_4 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-4))
      call bind_obs("chebTw_ctw_5", "w*T_5 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-5))
      call bind_obs("chebTw_ctw_6", "w*T_6 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwm, imem=-6))
      call bind_obs("w_ptwctw", "joint window w(pT_W) w(cos th*)", EvalObs_t(eval_mem=eval_w_ptwctw, imem=-1))
      call bind_obs("chebTT_ptwctw_11", "w T_1(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-11))
      call bind_obs("chebTT_ptwctw_12", "w T_1(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-12))
      call bind_obs("chebTT_ptwctw_13", "w T_1(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-13))
      call bind_obs("chebTT_ptwctw_14", "w T_1(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-14))
      call bind_obs("chebTT_ptwctw_21", "w T_2(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-21))
      call bind_obs("chebTT_ptwctw_22", "w T_2(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-22))
      call bind_obs("chebTT_ptwctw_23", "w T_2(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-23))
      call bind_obs("chebTT_ptwctw_24", "w T_2(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-24))
      call bind_obs("chebTT_ptwctw_31", "w T_3(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-31))
      call bind_obs("chebTT_ptwctw_32", "w T_3(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-32))
      call bind_obs("chebTT_ptwctw_33", "w T_3(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-33))
      call bind_obs("chebTT_ptwctw_34", "w T_3(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-34))
      call bind_obs("chebTT_ptwctw_41", "w T_4(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-41))
      call bind_obs("chebTT_ptwctw_42", "w T_4(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-42))
      call bind_obs("chebTT_ptwctw_43", "w T_4(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-43))
      call bind_obs("chebTT_ptwctw_44", "w T_4(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwm, imem=-44))
      call bind_obs("dscale1", "sqrt(mln^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("dscale2", "etw + ptj1", &
                    EvalObs_t(eval=eval_ZJ_dscale2, min_njets=1))
      call bind_obs("dscale3", "etw + sum pt{jets} ", &
                    EvalObs_t(eval=eval_ZJ_dscale3))
      call bind_obs("dphi_Wj1", "Azimuthal angle difference of V and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dphi_Wj2", "Azimuthal angle difference of V and 2nd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-2, min_njets=2))
      call bind_obs("dphi_Wj3", "Azimuthal angle difference of V and 3rd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-3, min_njets=3))
      call bind_obs("dphi_Wj4", "Azimuthal angle difference of V and 4th jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-4, min_njets=4))
      call bind_obs("max_pt_in_gap", "highest pt of jets within y-gap of j1 & j2", &
                    EvalObs_t(eval=eval_VFH_max_pt_in_gap, min_njets=3))
      call bind_obs("deltaphi_j12_rap_ord", "signed azimuthal angle between j1 & j2 (y1 > y2)", &
                    EvalObs_t(eval=eval_deltaphi_j12_rap_ord, min_njets=2))
      ! Helicity frame (angles wrt l1   = 47
      call bind_obs("cos_theta_HF", "cos(theta) in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=1))
      call bind_obs("theta_HF", "theta in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=2))
      call bind_obs("phi_HF", "phi in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=3))
      ! projectors for the helicity frame angular coefficients A_i [1606.00689]
      call bind_obs("proj_A0_HF", "projector for the angular coefficient A0 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=4))
      call bind_obs("proj_A1_HF", "projector for the angular coefficient A1 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=5))
      call bind_obs("proj_A2_HF", "projector for the angular coefficient A2 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=6))
      call bind_obs("proj_A3_HF", "projector for the angular coefficient A3 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=7))
      call bind_obs("proj_A4_HF", "projector for the angular coefficient A4 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=8))
      call bind_obs("proj_A5_HF", "projector for the angular coefficient A5 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=9))
      call bind_obs("proj_A6_HF", "projector for the angular coefficient A6 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=10))
      call bind_obs("proj_A7_HF", "projector for the angular coefficient A7 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=11))
      call bind_obs("proj_A0mA2_HF", "projector for the difference A0-A2 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=12))
      call bind_obs("f0", "helicity fraction f0", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=13))
      call bind_obs("fr", "helicity fraction fR", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=14))
      call bind_obs("fl", "helicity fraction fL", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=15))
      call bind_obs("lp", "Lepton projection variable using the lm", &
                    EvalObs_t(eval=eval_lp_l1))
      call bind_obs("cos_theta_2D", "2D prjection of cos theta* ", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=16))
      ! projectors for the angular coefficients A_i [1606.00689]
      call bind_obs("proj_A0", "projector for the angular coefficient A0 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=4))
      call bind_obs("proj_A1", "projector for the angular coefficient A1 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=5))
      call bind_obs("proj_A2", "projector for the angular coefficient A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=6))
      call bind_obs("proj_A3", "projector for the angular coefficient A3 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=7))
      call bind_obs("proj_A4", "projector for the angular coefficient A4 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=8))
      call bind_obs("proj_A5", "projector for the angular coefficient A5 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=9))
      call bind_obs("proj_A6", "projector for the angular coefficient A6 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=10))
      call bind_obs("proj_A7", "projector for the angular coefficient A7 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=11))
      call bind_obs("proj_A0mA2", "projector for the difference A0-A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=12))
      call bind_obs("cos_theta_CS_0PT", "cos(theta) in Collins-Soper frame, usable at PTZ=0 ", &
                    EvalObs_t(eval=eval_cos_theta_CS_0PT))
      if(isActive_flav())then
        ! flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj2", "transverse momentum of subleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-2, min_flav_njets=2, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj3", "transverse momentum of subsubleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-3, min_flav_njets=3, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_etaj1", "pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_etaj1", "absolute pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yj1", "rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_yj1", "absolute rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dy_Wj1", "deltay(abs) of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltay_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dphi_Wj1", "deltaphi of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltaphi_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dR_Wj1", "deltaR of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltaR_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yboost_Wj1", "yboost of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_yboost_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dscale2", "etw + ptj1_flav", &
                      EvalObs_t(eval=eval_flav_VJ_dscale2, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dscale3", "etw + ptj1_flav", &
                      EvalObs_t(eval=eval_flav_VJ_dscale3, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abseta_lmj1", "pseudo-rapidity of the lepton lm and lead flav-jet system", &
                      EvalObs_t(eval_mem=eval_flav_eta_lep_jet, imem=-idx_lep_jet(1, 1), decorator=abs_deco,  &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_minv_lmj1", "invariant mass of the lepton lm and lead flav-jet system", &
                      EvalObs_t(eval_mem=eval_flav_minv_lep_jet, imem=-idx_lep_jet(1, 1), min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dR_lmj1", "deltaR of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_deltaR_lep_jet, imem=-idx_lep_jet(1, 1), decorator=abs_deco,  &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_dphi_lmj1", "deltaphi of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_deltaphi_lep_jet, imem=-idx_lep_jet(1, 1), decorator=abs_deco,  &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_pt_lmj1", "pt of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_pt_lep_jet, imem=-idx_lep_jet(1, 1), min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_sumq_j1", "number of quark - number of antiquark inside leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_sumq_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        !end flavour-tagging
      endif

    case ("WP", "WPJ", "WPJUNSYM", "WPJ_H", "WPJJ", "WPJJJ", "WPQT")
      ! leptons from the Wp -> lp nu decay
      call bind_obs("etmiss", "missing transverse energy/momentum", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptlp", "transverse momentum of l+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("Elp", "energy of l+", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Enu", "energy of neutrino", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("ylp", "(pseudo-)rapidity of l+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ylp", "|ylp|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("ynu", "(pseudo-)rapidity of nu", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ynu", "|ynu|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("deltaphi_lj1", "Azimuthal angle difference between lepton and first jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(2, 1), min_njets=1))
      call bind_obs("deltaphi_lj2", "Azimuthal angle difference between lepton and second jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(2, 2), min_njets=2))
      call bind_obs("deltaphi_lj3", "Azimuthal angle difference between lepton and third jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_lepton_jet, imem=-idx_lep_jet(2, 3), min_njets=3))
      ! Wp-boson = lp + nu
      call bind_obs("ptw", "transverse momentum of W", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("mt", "transverse mass of W", &
                    EvalObs_t(eval=eval_mt_V), ifac=1)
      ! Wp : only one charged lepton at `npar-1`
      !      note, same name as in Z but different eval function
      call bind_obs("min_dr_lj", "minimum of Delta R (jets, l+)", &
                    EvalObs_t(eval=eval_min_dR_l2j))
      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      call bind_obs("ptjlp", "Transverse component of vector sum of jet and l+", &
                    EvalObs_t(eval=eval_pt_jl2, min_njets=1), ifac=2)
      call bind_obs("htjlp", "Sclar sum of ptj1 and ptlp", &
                    EvalObs_t(eval=eval_ht_jl2, min_njets=1))
      call bind_obs("deltay_zj1", "|yz - yj1|", &
                    EvalObs_t(eval=eval_deltay_Vj1, min_njets=1))
      call bind_obs("yw", "rapidity of W", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yw", "|yw|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaw", "pseudo-rapidity of W", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("mln", "invariant mass of W", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("r_w", "ratio of ptw/mln", &
                    EvalObs_t(eval=eval_r_V))
      call bind_obs("etw", "transverse energy: sqrt(mln^2 + ptw^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      ! ---- W event-level Chebyshev moments (maxent_upgrade input): m_W BW map, |y_W| lin, pT_W profiled
      call bind_obs("chebT_mw_1", "T_1 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-1), ifac=1)
      call bind_obs("chebT_mw_2", "T_2 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-2), ifac=1)
      call bind_obs("chebT_mw_3", "T_3 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-3), ifac=1)
      call bind_obs("chebT_mw_4", "T_4 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-4), ifac=1)
      call bind_obs("chebT_mw_5", "T_5 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-5), ifac=1)
      call bind_obs("chebT_mw_6", "T_6 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-6), ifac=1)
      call bind_obs("chebT_mw_7", "T_7 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-7), ifac=1)
      call bind_obs("chebT_mw_8", "T_8 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-8), ifac=1)
      call bind_obs("chebT_mw_9", "T_9 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-9), ifac=1)
      call bind_obs("chebT_mw_10", "T_10 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-10), ifac=1)
      call bind_obs("chebT_mw_11", "T_11 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-11), ifac=1)
      call bind_obs("chebT_mw_12", "T_12 of m_W (BW 40-1000)", EvalObs_t(eval_mem=eval_chebT_mw, imem=-12), ifac=1)
      call bind_obs("chebT_absyw_1", "T_1 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-1))
      call bind_obs("chebT_absyw_2", "T_2 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-2))
      call bind_obs("chebT_absyw_3", "T_3 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-3))
      call bind_obs("chebT_absyw_4", "T_4 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-4))
      call bind_obs("chebT_absyw_5", "T_5 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-5))
      call bind_obs("chebT_absyw_6", "T_6 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-6))
      call bind_obs("chebT_absyw_7", "T_7 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-7))
      call bind_obs("chebT_absyw_8", "T_8 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-8))
      call bind_obs("chebT_absyw_9", "T_9 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-9))
      call bind_obs("chebT_absyw_10", "T_10 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-10))
      call bind_obs("chebT_absyw_11", "T_11 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-11))
      call bind_obs("chebT_absyw_12", "T_12 of |y_W| (lin 0-4)", EvalObs_t(eval_mem=eval_chebT_absyw, imem=-12))
      call bind_obs("w_ptw", "recoil profile w(pT_W) [25,50,500]", EvalObs_t(eval=eval_w_ptw), ifac=2)
      call bind_obs("chebTw_ptw_1", "w*T_1 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-1), ifac=2)
      call bind_obs("chebTw_ptw_2", "w*T_2 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-2), ifac=2)
      call bind_obs("chebTw_ptw_3", "w*T_3 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-3), ifac=2)
      call bind_obs("chebTw_ptw_4", "w*T_4 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-4), ifac=2)
      call bind_obs("chebTw_ptw_5", "w*T_5 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-5), ifac=2)
      call bind_obs("chebTw_ptw_6", "w*T_6 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-6), ifac=2)
      call bind_obs("chebTw_ptw_7", "w*T_7 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-7), ifac=2)
      call bind_obs("chebTw_ptw_8", "w*T_8 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-8), ifac=2)
      call bind_obs("chebTw_ptw_9", "w*T_9 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-9), ifac=2)
      call bind_obs("chebTw_ptw_10", "w*T_10 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-10), ifac=2)
      call bind_obs("chebTw_ptw_11", "w*T_11 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-11), ifac=2)
      call bind_obs("chebTw_ptw_12", "w*T_12 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-12), ifac=2)
      call bind_obs("chebTw_ptw_13", "w*T_13 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-13), ifac=2)
      call bind_obs("chebTw_ptw_14", "w*T_14 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-14), ifac=2)
      call bind_obs("chebTw_ptw_15", "w*T_15 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-15), ifac=2)
      call bind_obs("chebTw_ptw_16", "w*T_16 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-16), ifac=2)
      call bind_obs("chebTw_ptw_17", "w*T_17 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-17), ifac=2)
      call bind_obs("chebTw_ptw_18", "w*T_18 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-18), ifac=2)
      call bind_obs("chebTw_ptw_19", "w*T_19 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-19), ifac=2)
      call bind_obs("chebTw_ptw_20", "w*T_20 pT_W (log 25-2500)", EvalObs_t(eval_mem=eval_chebTw_ptw, imem=-20), ifac=2)
      ! ---- signed Collins-Soper decay angle of the charged lepton, lin map [-1,1]
      call bind_obs("chebT_ctw_1", "T_1 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-1))
      call bind_obs("chebT_ctw_2", "T_2 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-2))
      call bind_obs("chebT_ctw_3", "T_3 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-3))
      call bind_obs("chebT_ctw_4", "T_4 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-4))
      call bind_obs("chebT_ctw_5", "T_5 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-5))
      call bind_obs("chebT_ctw_6", "T_6 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-6))
      call bind_obs("chebT_ctw_7", "T_7 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-7))
      call bind_obs("chebT_ctw_8", "T_8 of cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebT_ctwp, imem=-8))
      call bind_obs("costh_w", "signed cos th*_CS of the charged lepton (W rest frame)", EvalObs_t(eval_mem=eval_costhCS_w, imem=-2))
      call bind_obs("pred_D_7", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2001))
      call bind_obs("pred_D_8", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2002))
      call bind_obs("pred_D_9", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2003))
      call bind_obs("pred_D_10", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2004))
      call bind_obs("pred_M_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2005))
      call bind_obs("pred_M_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2006))
      call bind_obs("pred_M_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2007))
      call bind_obs("pred_M_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2008))
      call bind_obs("pred_M_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2009))
      call bind_obs("pred_M_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2010))
      call bind_obs("pred_N_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2011))
      call bind_obs("pred_N_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2012))
      call bind_obs("pred_N_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2013))
      call bind_obs("pred_N_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2014))
      call bind_obs("pred_N_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2015))
      call bind_obs("pred_N_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2016))
      call bind_obs("pred_N_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2017))
      call bind_obs("pred_NJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2018))
      call bind_obs("pred_NJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2019))
      call bind_obs("pred_NJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2020))
      call bind_obs("pred_NJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2021))
      call bind_obs("pred_NJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2022))
      call bind_obs("pred_NJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2023))
      call bind_obs("pred_G_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2024))
      call bind_obs("pred_G_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2025))
      call bind_obs("pred_G_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2026))
      call bind_obs("pred_G_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2027))
      call bind_obs("pred_G_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2028))
      call bind_obs("pred_G_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2029))
      call bind_obs("pred_G_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2030))
      call bind_obs("pred_GJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2031))
      call bind_obs("pred_GJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2032))
      call bind_obs("pred_GJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2033))
      call bind_obs("pred_GJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2034))
      call bind_obs("pred_GJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2035))
      call bind_obs("pred_GJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2036))
      call bind_obs("pred_W_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2037))
      call bind_obs("pred_W_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2038))
      call bind_obs("pred_W_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2039))
      call bind_obs("pred_W_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2040))
      call bind_obs("pred_W_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2041))
      call bind_obs("pred_W_5", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2042))
      call bind_obs("pred_W_6", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2043))
      call bind_obs("pred_WJ_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2044))
      call bind_obs("pred_WJ_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2045))
      call bind_obs("pred_WJ_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2046))
      call bind_obs("pred_WJ_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2047))
      call bind_obs("pred_WJ_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2048))
      call bind_obs("pred_WJ_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2049))
      call bind_obs("pred_H_0_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2050))
      call bind_obs("pred_H_0_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2051))
      call bind_obs("pred_H_0_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2052))
      call bind_obs("pred_H_0_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2053))
      call bind_obs("pred_H_0_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2054))
      call bind_obs("pred_H_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2055))
      call bind_obs("pred_H_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2056))
      call bind_obs("pred_H_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2057))
      call bind_obs("pred_H_1_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2058))
      call bind_obs("pred_H_1_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2059))
      call bind_obs("pred_H_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2060))
      call bind_obs("pred_H_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2061))
      call bind_obs("pred_H_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2062))
      call bind_obs("pred_H_2_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2063))
      call bind_obs("pred_H_2_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2064))
      call bind_obs("pred_S_0_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2065))
      call bind_obs("pred_S_0_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2066))
      call bind_obs("pred_S_0_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2067))
      call bind_obs("pred_S_0_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2068))
      call bind_obs("pred_S_0_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2069))
      call bind_obs("pred_S_1_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2070))
      call bind_obs("pred_S_1_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2071))
      call bind_obs("pred_S_1_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2072))
      call bind_obs("pred_S_1_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2073))
      call bind_obs("pred_S_1_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2074))
      call bind_obs("pred_S_2_0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2075))
      call bind_obs("pred_S_2_1", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2076))
      call bind_obs("pred_S_2_2", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2077))
      call bind_obs("pred_S_2_3", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2078))
      call bind_obs("pred_S_2_4", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2079))
      call bind_obs("expl_qc_0_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2080))
      call bind_obs("expl_qc_0_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2081))
      call bind_obs("expl_qc_0_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2082))
      call bind_obs("expl_qc_0_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2083))
      call bind_obs("expl_qc_0_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2084))
      call bind_obs("expl_qc_0_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2085))
      call bind_obs("expl_qc_0_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2086))
      call bind_obs("expl_qc_0_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2087))
      call bind_obs("expl_qc_0_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2088))
      call bind_obs("expl_qc_0_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2089))
      call bind_obs("expl_qc_0_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2090))
      call bind_obs("expl_qc_0_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2091))
      call bind_obs("expl_qc_0_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2092))
      call bind_obs("expl_qc_0_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2093))
      call bind_obs("expl_qc_0_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2094))
      call bind_obs("expl_qc_1_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2095))
      call bind_obs("expl_qc_1_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2096))
      call bind_obs("expl_qc_1_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2097))
      call bind_obs("expl_qc_1_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2098))
      call bind_obs("expl_qc_1_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2099))
      call bind_obs("expl_qc_1_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2100))
      call bind_obs("expl_qc_1_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2101))
      call bind_obs("expl_qc_1_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2102))
      call bind_obs("expl_qc_1_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2103))
      call bind_obs("expl_qc_1_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2104))
      call bind_obs("expl_qc_1_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2105))
      call bind_obs("expl_qc_1_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2106))
      call bind_obs("expl_qc_1_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2107))
      call bind_obs("expl_qc_1_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2108))
      call bind_obs("expl_qc_1_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2109))
      call bind_obs("expl_qc_2_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2110))
      call bind_obs("expl_qc_2_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2111))
      call bind_obs("expl_qc_2_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2112))
      call bind_obs("expl_qc_2_0_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2113))
      call bind_obs("expl_qc_2_0_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2114))
      call bind_obs("expl_qc_2_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2115))
      call bind_obs("expl_qc_2_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2116))
      call bind_obs("expl_qc_2_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2117))
      call bind_obs("expl_qc_2_1_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2118))
      call bind_obs("expl_qc_2_1_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2119))
      call bind_obs("expl_qc_2_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2120))
      call bind_obs("expl_qc_2_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2121))
      call bind_obs("expl_qc_2_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2122))
      call bind_obs("expl_qc_2_2_proj27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2123))
      call bind_obs("expl_qc_2_2_proj30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2124))
      call bind_obs("expl_qm_0_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2125))
      call bind_obs("expl_qm_0_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2126))
      call bind_obs("expl_qm_0_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2127))
      call bind_obs("expl_qm_0_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2128))
      call bind_obs("expl_qm_0_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2129))
      call bind_obs("expl_qm_0_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2130))
      call bind_obs("expl_qm_0_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2131))
      call bind_obs("expl_qm_0_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2132))
      call bind_obs("expl_qm_0_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2133))
      call bind_obs("expl_qm_1_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2134))
      call bind_obs("expl_qm_1_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2135))
      call bind_obs("expl_qm_1_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2136))
      call bind_obs("expl_qm_1_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2137))
      call bind_obs("expl_qm_1_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2138))
      call bind_obs("expl_qm_1_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2139))
      call bind_obs("expl_qm_1_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2140))
      call bind_obs("expl_qm_1_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2141))
      call bind_obs("expl_qm_1_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2142))
      call bind_obs("expl_qm_2_0_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2143))
      call bind_obs("expl_qm_2_0_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2144))
      call bind_obs("expl_qm_2_0_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2145))
      call bind_obs("expl_qm_2_1_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2146))
      call bind_obs("expl_qm_2_1_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2147))
      call bind_obs("expl_qm_2_1_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2148))
      call bind_obs("expl_qm_2_2_pt0", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2149))
      call bind_obs("expl_qm_2_2_pt27", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2150))
      call bind_obs("expl_qm_2_2_pt30", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2151))
      call bind_obs("hold_pt28", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2152))
      call bind_obs("hold_pt32", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2153))
      call bind_obs("hold_hard", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2154))
      call bind_obs("hold_pt32hard", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2155))
      call bind_obs("hold_mt60", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2156))
      call bind_obs("hold_combined", "W predictivity common-vector function", &
                    EvalObs_t(eval_mem=eval_wpred, imem=-2157))
      call bind_obs("pred_eta1", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-2001))
      call bind_obs("pred_eta2", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-2002))
      call bind_obs("pred_eta3", "W held-out lepton eta", &
                    EvalObs_t(eval_mem=eval_wpred_eta, imem=-2003))
      call bind_obs("wval_pt_1", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-1))
      call bind_obs("wval_pt_2", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-2))
      call bind_obs("wval_pt_3", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-3))
      call bind_obs("wval_pt_4", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-4))
      call bind_obs("wval_pt_5", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-5))
      call bind_obs("wval_pt_6", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-6))
      call bind_obs("wval_pt_7", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-7))
      call bind_obs("wval_pt_8", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-8))
      call bind_obs("wval_pt_9", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-9))
      call bind_obs("wval_pt_10", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-10))
      call bind_obs("wval_pt_11", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-11))
      call bind_obs("wval_pt_12", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-12))
      call bind_obs("wval_pt_13", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-13))
      call bind_obs("wval_pt_14", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-14))
      call bind_obs("wval_pt_15", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-15))
      call bind_obs("wval_pt_16", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-16))
      call bind_obs("wval_pt_17", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-17))
      call bind_obs("wval_pt_18", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-18))
      call bind_obs("wval_pt_19", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-19))
      call bind_obs("wval_pt_20", "W recoil moment on acting range 25-500", &
                    EvalObs_t(eval_mem=eval_wval_pt, imem=-20))
      call bind_obs("wval_yc_10", "windowed W rapidity decay product 10", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2010))
      call bind_obs("wval_yc_11", "windowed W rapidity decay product 11", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2011))
      call bind_obs("wval_yc_12", "windowed W rapidity decay product 12", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2012))
      call bind_obs("wval_yc_13", "windowed W rapidity decay product 13", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2013))
      call bind_obs("wval_yc_14", "windowed W rapidity decay product 14", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2014))
      call bind_obs("wval_yc_20", "windowed W rapidity decay product 20", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2020))
      call bind_obs("wval_yc_21", "windowed W rapidity decay product 21", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2021))
      call bind_obs("wval_yc_22", "windowed W rapidity decay product 22", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2022))
      call bind_obs("wval_yc_23", "windowed W rapidity decay product 23", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2023))
      call bind_obs("wval_yc_24", "windowed W rapidity decay product 24", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2024))
      call bind_obs("wval_yc_30", "windowed W rapidity decay product 30", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2030))
      call bind_obs("wval_yc_31", "windowed W rapidity decay product 31", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2031))
      call bind_obs("wval_yc_32", "windowed W rapidity decay product 32", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2032))
      call bind_obs("wval_yc_33", "windowed W rapidity decay product 33", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2033))
      call bind_obs("wval_yc_34", "windowed W rapidity decay product 34", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2034))
      call bind_obs("wval_yc_40", "windowed W rapidity decay product 40", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2040))
      call bind_obs("wval_yc_41", "windowed W rapidity decay product 41", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2041))
      call bind_obs("wval_yc_42", "windowed W rapidity decay product 42", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2042))
      call bind_obs("wval_yc_43", "windowed W rapidity decay product 43", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2043))
      call bind_obs("wval_yc_44", "windowed W rapidity decay product 44", &
                    EvalObs_t(eval_mem=eval_wval_yctw, imem=-2044))
      call bind_obs("wval_eta27", "charged-lepton eta after pT>27", &
                    EvalObs_t(eval_mem=eval_wval_eta, imem=-227))
      call bind_obs("wval_eta30", "charged-lepton eta after pT>30", &
                    EvalObs_t(eval_mem=eval_wval_eta, imem=-230))
      call bind_obs("w_ctw", "decay-angle window w(cos th*) [|c|<0.5, ramp to 0.7]", EvalObs_t(eval_mem=eval_w_ctw, imem=-2))
      call bind_obs("chebTw_ctw_1", "w*T_1 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-1))
      call bind_obs("chebTw_ctw_2", "w*T_2 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-2))
      call bind_obs("chebTw_ctw_3", "w*T_3 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-3))
      call bind_obs("chebTw_ctw_4", "w*T_4 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-4))
      call bind_obs("chebTw_ctw_5", "w*T_5 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-5))
      call bind_obs("chebTw_ctw_6", "w*T_6 cos th*_CS(W)", EvalObs_t(eval_mem=eval_chebTw_ctwp, imem=-6))
      call bind_obs("w_ptwctw", "joint window w(pT_W) w(cos th*)", EvalObs_t(eval_mem=eval_w_ptwctw, imem=-2))
      call bind_obs("chebTT_ptwctw_11", "w T_1(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-11))
      call bind_obs("chebTT_ptwctw_12", "w T_1(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-12))
      call bind_obs("chebTT_ptwctw_13", "w T_1(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-13))
      call bind_obs("chebTT_ptwctw_14", "w T_1(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-14))
      call bind_obs("chebTT_ptwctw_21", "w T_2(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-21))
      call bind_obs("chebTT_ptwctw_22", "w T_2(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-22))
      call bind_obs("chebTT_ptwctw_23", "w T_2(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-23))
      call bind_obs("chebTT_ptwctw_24", "w T_2(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-24))
      call bind_obs("chebTT_ptwctw_31", "w T_3(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-31))
      call bind_obs("chebTT_ptwctw_32", "w T_3(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-32))
      call bind_obs("chebTT_ptwctw_33", "w T_3(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-33))
      call bind_obs("chebTT_ptwctw_34", "w T_3(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-34))
      call bind_obs("chebTT_ptwctw_41", "w T_4(pT_W) T_1(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-41))
      call bind_obs("chebTT_ptwctw_42", "w T_4(pT_W) T_2(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-42))
      call bind_obs("chebTT_ptwctw_43", "w T_4(pT_W) T_3(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-43))
      call bind_obs("chebTT_ptwctw_44", "w T_4(pT_W) T_4(cos th*)", EvalObs_t(eval_mem=eval_chebTT_ptwctwp, imem=-44))
      call bind_obs("dscale1", "sqrt(mln^2 + sum_i ptj_i^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("dscale2", "etw + ptj1", &
                    EvalObs_t(eval=eval_ZJ_dscale2, min_njets=1))
      call bind_obs("dscale3", "etw + sum pt{jets}", &
                    EvalObs_t(eval=eval_ZJ_dscale3))
      call bind_obs("dphi_Wj1", "Azimuthal angle difference of V and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dphi_Wj2", "Azimuthal angle difference of V and 2nd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-2, min_njets=2))
      call bind_obs("dphi_Wj3", "Azimuthal angle difference of V and 3rd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-3, min_njets=3))
      call bind_obs("dphi_Wj4", "Azimuthal angle difference of V and 4th jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-4, min_njets=4))
      call bind_obs("max_pt_in_gap", "highest pt of jets within y-gap of j1 & j2", &
                    EvalObs_t(eval=eval_VFH_max_pt_in_gap, min_njets=3))
      call bind_obs("deltaphi_j12_rap_ord", "signed azimuthal angle between j1 & j2 (y1 > y2)", &
                    EvalObs_t(eval=eval_deltaphi_j12_rap_ord, min_njets=2))
      ! Helicity frame (angles wrt l2   = 47
      call bind_obs("cos_theta_HF", "cos(theta) in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(1 + 16)))
      call bind_obs("theta_HF", "theta in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(2 + 16)))
      call bind_obs("phi_HF", "phi in the helicity frame", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(3 + 16)))
      ! projectors for the helicity frame angular coefficients A_i [1606.00689]
      call bind_obs("proj_A0_HF", "projector for the angular coefficient A0 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(4 + 16)))
      call bind_obs("proj_A1_HF", "projector for the angular coefficient A1 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(5 + 16)))
      call bind_obs("proj_A2_HF", "projector for the angular coefficient A2 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(6 + 16)))
      call bind_obs("proj_A3_HF", "projector for the angular coefficient A3 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(7 + 16)))
      call bind_obs("proj_A4_HF", "projector for the angular coefficient A4 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(8 + 16)))
      call bind_obs("proj_A5_HF", "projector for the angular coefficient A5 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(9 + 16)))
      call bind_obs("proj_A6_HF", "projector for the angular coefficient A6 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(10 + 16)))
      call bind_obs("proj_A7_HF", "projector for the angular coefficient A7 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(11 + 16)))
      call bind_obs("proj_A0mA2_HF", "projector for the difference A0-A2 in the helicity frame [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(12 + 16)))
      call bind_obs("lp", "Lepton projection variable using the lp", &
                    EvalObs_t(eval=eval_lp_l2))
      call bind_obs("f0", "helicity fraction f0", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(13 + 16)))
      call bind_obs("fr", "helicity fraction fR", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(15 + 16)))
      call bind_obs("fl", "helicity fraction fL", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(14 + 16)))
      call bind_obs("cos_theta_2D", "2D prjection of cos theta* ", &
                    EvalObs_t(eval_mem=eval_Helicity_Frame, imem=(16 + 16)))
      ! projectors for the angular coefficients A_i [1606.00689]
      call bind_obs("proj_A0", "projector for the angular coefficient A0 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=4))
      call bind_obs("proj_A1", "projector for the angular coefficient A1 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=5))
      call bind_obs("proj_A2", "projector for the angular coefficient A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=6))
      call bind_obs("proj_A3", "projector for the angular coefficient A3 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=7))
      call bind_obs("proj_A4", "projector for the angular coefficient A4 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=8))
      call bind_obs("proj_A5", "projector for the angular coefficient A5 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=9))
      call bind_obs("proj_A6", "projector for the angular coefficient A6 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=10))
      call bind_obs("proj_A7", "projector for the angular coefficient A7 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=11))
      call bind_obs("proj_A0mA2", "projector for the difference A0-A2 [1606.00689]", &
                    EvalObs_t(eval_mem=eval_Collins_Soper, imem=12))
      call bind_obs("cos_theta_CS_0PT", "cos(theta) in Collins-Soper frame, usable at PTZ=0 ", &
                    EvalObs_t(eval=eval_cos_theta_CS_0PT))
      if(isActive_flav())then
        ! flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj2", "transverse momentum of subleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-2, min_flav_njets=2, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_ptj3", "transverse momentum of subsubleading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-3, min_flav_njets=3, &
                      needs_flav=.true.), ifac=2)
        call bind_obs("flav_etaj1", "pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_etaj1", "absolute pseudorapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_eta_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yj1", "rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abs_yj1", "absolute rapidity of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_y_ji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dy_Wj1", "deltay(abs) of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltay_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dphi_Wj1", "deltaphi of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltaphi_Vji, imem=-1, decorator=abs_deco, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dR_Wj1", "deltaR of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_deltaR_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_yboost_Wj1", "yboost of leading flavour-tagged jet with W", &
                      EvalObs_t(eval_mem=eval_flav_yboost_Vji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dscale2", "etw + ptj1_flav", &
                      EvalObs_t(eval=eval_flav_VJ_dscale2, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dscale3", "etw + ptj1_flav", &
                      EvalObs_t(eval=eval_flav_VJ_dscale3, min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_abseta_lpj1", "pseudo-rapidity of the lepton lp and lead flav-jet system", &
                      EvalObs_t(eval_mem=eval_flav_eta_lep_jet, imem=-idx_lep_jet(2, 1), decorator=abs_deco, &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_minv_lpj1", "invariant mass of the lepton lp and lead flav-jet system", &
                      EvalObs_t(eval_mem=eval_flav_minv_lep_jet, imem=-idx_lep_jet(2, 1), min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_dR_lpj1", "deltaR of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_deltaR_lep_jet, imem=-idx_lep_jet(2, 1), decorator=abs_deco, &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_dphi_lpj1", "deltaphi of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_deltaphi_lep_jet, imem=-idx_lep_jet(2, 1), decorator=abs_deco, &
                      min_flav_njets=1, needs_flav=.true.))
        call bind_obs("flav_pt_lpj1", "pt of leading flavour-tagged jet with lm", &
                      EvalObs_t(eval_mem=eval_flav_pt_lep_jet, imem=-idx_lep_jet(2, 1), min_flav_njets=1, &
                      needs_flav=.true.))
        call bind_obs("flav_sumq_j1", "number of quark - number of antiquark inside leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_sumq_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.))
        !end flavour-tagging
      endif

    case ("WPWMJJ")
      ! The particle list contains {.., nu, l+, l-, nub}
      ! -> 1st lepton [npar-1]   l-   | W-
      !    2nd lepton [npar]     nub  |
      !    3rd lepton [npar-3]   nu     | W+
      !    4th lepton [npar-2]   l+     |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlm", "transverse momentum of l-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptnub", "transverse momentum of nub", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptnu", "transverse momentum of nu", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptlp", "transverse momentum of l+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("Elm", "energy of l-", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Enub", "energy of nub", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Enu", "energy of nu", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Elp", "energy of l+", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("etalm", "(pseudo-)rapidity of l-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("etanub", "(pseudo-)rapidity of nub", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etanu", "(pseudo-)rapidity of nu", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etalp", "(pseudo-)rapidity of l+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_etalm", "|etalm|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("abs_etanub", "|etanub|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etanu", "|etanu|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etalp", "|etalp|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptwm", "transverse momentum of Wm", &
      !               id_pt_l12)
      ! call bind_obs("ptwp", "transverse momentum of Wp", &
      !               id_pt_l34)
      ! call bind_obs("ywm", "rapidity of Wm", &
      !               id_y_l12)
      ! call bind_obs("ywp", "rapidity of Wp", &
      !               id_y_l34)
      ! call bind_obs("abs_ywm", "|ywm|", &
      !               id_absy_l12)
      ! call bind_obs("abs_ywp", "|ywp|", &
      !               id_absy_l34)
      ! call bind_obs("mwm", "invariant mass of Wm", &
      !               id_m_l12)
      ! call bind_obs("mwp", "invariant mass of Wp", &
      !               id_m_l34)
      ! ! VBS system = Wm + Wp
      ! call bind_obs("ptww", "transverse momentum of WpWm system", &
      !               id_pt_l1234)
      ! call bind_obs("mww", "invariant mass of WpWm system", &
      !               id_m_l1234)
      ! call bind_obs("yww", "rapidity of WpWm system", &
      !               id_y_l1234)
      ! call bind_obs("abs_yww", "|yww|", &
      !               id_absy_l1234)
      ! ! charged leptons (important, because detectable in experiment)
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l14)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l14)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l14)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l14)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l14)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l14)
      ! call bind_obs("mll", "invariant mass of dilepton system", &
      !               id_m_l14)
      ! call bind_obs("ptll", "transverse momentum of dilepton system", &
      !               id_pt_l14)
      ! call bind_obs("ptmiss", "missing transverse momentum", &
      !               id_pt_l23)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l14j)
    case ("WPWPJJ")
      ! The particle list contains {.., nu, l+, l+, nu}
      ! -> 1st lepton [npar-1]   l+(lp2)   | W+1
      !    2nd lepton [npar]     nu(nu2)   |
      !    3rd lepton [npar-3]   nu(nu1)     | W+2
      !    4th lepton [npar-2]   l+(lp1)    |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlp1", "transverse momentum of l+1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("ptnu1", "transverse momentum of nu1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptnu2", "transverse momentum of nu2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptlp2", "transverse momentum of l+2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("Elp1", "energy of l+1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("Enu1", "energy of nu1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Enu2", "energy of nu2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Elp2", "energy of l+2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("etalp1", "(pseudo-)rapidity of l+1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("etanu1", "(pseudo-)rapidity of nu1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etanu2", "(pseudo-)rapidity of nu2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etalp2", "(pseudo-)rapidity of l+2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_etalp1", "|etalp1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      call bind_obs("abs_etanu1", "|etanu1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etanu2", "|etanu2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etalp2", "|etalp2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptw1", "transverse momentum of Wp1", &
      !               id_pt_l34)
      ! call bind_obs("ptw2", "transverse momentum of Wp2", &
      !               id_pt_l12)
      ! call bind_obs("yw1", "rapidity of Wp1", &
      !               id_y_l34)
      ! call bind_obs("yw2", "rapidity of Wp2", &
      !               id_y_l12)
      ! call bind_obs("abs_yw1", "|ywp1|", &
      !               id_absy_l34)
      ! call bind_obs("abs_yw2", "|ywp2|", &
      !               id_absy_l12)
      ! call bind_obs("mw1", "invariant mass of Wp1", &
      !               id_m_l34)
      ! call bind_obs("mw2", "invariant mass of Wp2", &
      !               id_m_l12)
      ! ! VBS system = Wp1 + Wp2
      ! call bind_obs("ptww", "transverse momentum of WpWp system", &
      !               id_pt_l1234)
      ! call bind_obs("mww", "invariant mass of WpWp system", &
      !               id_m_l1234)
      ! call bind_obs("yww", "rapidity of WpWp system", &
      !               id_y_l1234)
      ! call bind_obs("abs_yww", "|yww|", &
      !               id_absy_l1234)
      ! ! charged leptons (important, because detectable in experiment)
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l14)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l14)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l14)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l14)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l14)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l14)
      ! call bind_obs("mll", "invariant mass of dilepton system", &
      !               id_m_l14)
      ! call bind_obs("ptll", "transverse momentum of dilepton system", &
      !               id_pt_l14)
      ! call bind_obs("ptmiss", "missing transverse momentum", &
      !               id_pt_l23)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l14j)

    case ("WMWMJJ")
      ! The particle list contains {.., nub, l-, l-, nub}
      ! -> 1st lepton [npar-1]   l-   | W-1
      !    2nd lepton [npar]     nub  |
      !    3rd lepton [npar-3]   nub    | W-2
      !    4th lepton [npar-2]   l-     |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlm1", "transverse momentum of l-1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptnub1", "transverse momentum of nub1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptnub2", "transverse momentum of nub2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptlm2", "transverse momentum of l-2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("Elm1", "energy of l-1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Enub1", "energy of nub1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Enub2", "energy of nub2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Elm2", "energy of l-2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("etalm1", "(pseudo-)rapidity of l-1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("etanub1", "(pseudo-)rapidity of nub1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etanub2", "(pseudo-)rapidity of nub2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etalm2", "(pseudo-)rapidity of l-2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_etalm1", "|etalm1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("abs_etanub1", "|etanub1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etanub2", "|etanub2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etalm2", "|etalm2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptw1", "transverse momentum of Wm1", &
      !               id_pt_l12)
      ! call bind_obs("ptw2", "transverse momentum of Wm2", &
      !               id_pt_l34)
      ! call bind_obs("yw1", "rapidity of Wm1", &
      !               id_y_l12)
      ! call bind_obs("yw2", "rapidity of Wm2", &
      !               id_y_l34)
      ! call bind_obs("abs_yw1", "|ywm1|", &
      !               id_absy_l12)
      ! call bind_obs("abs_yw2", "|ywm2|", &
      !               id_absy_l34)
      ! call bind_obs("mw1", "invariant mass of Wm1", &
      !               id_m_l12)
      ! call bind_obs("mw2", "invariant mass of Wm2", &
      !               id_m_l34)
      ! ! VBS system = Wm1 + Wm2
      ! call bind_obs("ptww", "transverse momentum of WmWm system", &
      !               id_pt_l1234)
      ! call bind_obs("mww", "invariant mass of WmWm system", &
      !               id_m_l1234)
      ! call bind_obs("yww", "rapidity of WmWm system", &
      !               id_y_l1234)
      ! call bind_obs("abs_yww", "|yww|", &
      !               id_absy_l1234)
      ! ! charged leptons (important, because detectable in experiment)
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l14)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l14)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l14)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l14)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l14)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l14)
      ! call bind_obs("mll", "invariant mass of dilepton system", &
      !               id_m_l14)
      ! call bind_obs("ptll", "transverse momentum of dilepton system", &
      !               id_pt_l14)
      ! call bind_obs("ptmiss", "missing transverse momentum", &
      !               id_pt_l23)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l14j)

    case ("WPZJJ")
      ! The particle list contains {.., nu, l+, l-, l+}
      ! -> 1st lepton [npar-1]   l-   | Z
      !    2nd lepton [npar]     l+   |
      !    3rd lepton [npar-3]   nu     | W+
      !    4th lepton [npar-2]   l+     |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlm1", "transverse momentum of lm1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptlp1", "transverse momentum of lp1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptnu2", "transverse momentum of nu2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptlp2", "transverse momentum of lp2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("Elm1", "energy of lm1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Elp1", "energy of lp1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Enu2", "energy of nu2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Elp2", "energy of lp2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("etalm1", "(pseudo-)rapidity of lm1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("etalp1", "(pseudo-)rapidity of lp1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etanu2", "(pseudo-)rapidity of nu2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etalp2", "(pseudo-)rapidity of lp2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_etalm1", "|etalm1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("abs_etalp1", "|etalp1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etanu2", "|etanu2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etalp2", "|etalp2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptw", "transverse momentum of Wp", &
      !               id_pt_l34)
      ! call bind_obs("ptz", "transverse momentum of Z", &
      !               id_pt_l12)
      ! call bind_obs("yw", "rapidity of Wp", &
      !               id_y_l34)
      ! call bind_obs("yz", "rapidity of Z", &
      !               id_y_l12)
      ! call bind_obs("abs_yw", "|yw|", &
      !               id_absy_l34)
      ! call bind_obs("abs_yz", "|yz|", &
      !               id_absy_l12)
      ! call bind_obs("mw", "invariant mass of Wp", &
      !               id_m_l34)
      ! call bind_obs("mz", "invariant mass of Z", &
      !               id_m_l12)
      ! ! VBS system = Wp + Z
      ! call bind_obs("ptwz", "transverse momentum of WpZ system", &
      !               id_pt_l1234)
      ! call bind_obs("mwz", "invariant mass of WpZ system", &
      !               id_m_l1234)
      ! call bind_obs("ywz", "rapidity of WpZ system", &
      !               id_y_l1234)
      ! call bind_obs("abs_ywz", "|yww|", &
      !               id_absy_l1234)
      ! ! charged leptons (important, because detectable in experiment)
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l124)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l124)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l124)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l124)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l124)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l124)
      ! call bind_obs("mlll", "invariant mass of trilepton system", &
      !               id_m_l124)
      ! call bind_obs("ptlll", "transverse momentum of trilepton system", &
      !               id_pt_l124)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l124j)

    case ("WMZJJ")
      ! The particle list contains {.., l-, l+, l-, nub}
      ! -> 1st lepton [npar-1]   l-   | W-
      !    2nd lepton [npar]     nub  |
      !    3rd lepton [npar-3]   l-     | Z
      !    4th lepton [npar-2]   l+     |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlm1", "transverse momentum of lm1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptnub1", "transverse momentum of nub1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptlm2", "transverse momentum of lm2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptlp2", "transverse momentum of lp2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("Elm1", "energy of lm1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Enub1", "energy of nub1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Elm2", "energy of lm2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Elp2", "energy of lp2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("etalm1", "(pseudo-)rapidity of lm1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("etanub1", "(pseudo-)rapidity of nub1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etalm2", "(pseudo-)rapidity of lm2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etalp2", "(pseudo-)rapidity of lp2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_etalm1", "|etalm1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("abs_etanub1", "|etanub1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etalm2", "|etalm2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etalp2", "|etalp2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptw", "transverse momentum of Wm", &
      !               id_pt_l12)
      ! call bind_obs("ptz", "transverse momentum of Z", &
      !               id_pt_l34)
      ! call bind_obs("yw", "rapidity of Wm", &
      !               id_y_l12)
      ! call bind_obs("yz", "rapidity of Z", &
      !               id_y_l34)
      ! call bind_obs("abs_yw", "|yw|", &
      !               id_absy_l12)
      ! call bind_obs("abs_yz", "|yz|", &
      !               id_absy_l34)
      ! call bind_obs("mw", "invariant mass of Wm", &
      !               id_m_l12)
      ! call bind_obs("mz", "invariant mass of Z", &
      !               id_m_l34)
      ! ! VBS system = Wp + Z
      ! call bind_obs("ptwz", "transverse momentum of WmZ system", &
      !               id_pt_l1234)
      ! call bind_obs("mwz", "invariant mass of WmZ system", &
      !               id_m_l1234)
      ! call bind_obs("ywz", "rapidity of WmZ system", &
      !               id_y_l1234)
      ! call bind_obs("abs_ywz", "|yww|", &
      !               id_absy_l1234)
      ! ! charged leptons (important, because detectable in experiment)
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l134)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l134)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l134)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l134)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l134)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l134)
      ! call bind_obs("mlll", "invariant mass of trilepton system", &
      !               id_m_l134)
      ! call bind_obs("ptlll", "transverse momentum of trilepton system", &
      !               id_pt_l134)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l134j)

    case ("ZZJJ")
      ! The particle list contains {.., l-, l+, l-, l+}
      ! -> 1st lepton [npar-1]   l-   | Z1
      !    2nd lepton [npar]     l+   |
      !    3rd lepton [npar-3]   l-     | Z2
      !    4th lepton [npar-2]   l+     |
      ! (see eval_pt_lep subroutine)
      ! single leptons from the gauge boson decay
      call bind_obs("ptlm1", "transverse momentum of l-1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ptlp1", "transverse momentum of l+1", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ptlm2", "transverse momentum of l-2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("ptlp2", "transverse momentum of l+2", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("Elm1", "energy of l-1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-1))
      call bind_obs("Elp1", "energy of l+1", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-2))
      call bind_obs("Elm2", "energy of l-2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-3))
      call bind_obs("Elp2", "energy of l+2", &
                    EvalObs_t(eval_mem=eval_E_lep, imem=-4))
      call bind_obs("etalm1", "(pseudo-)rapidity of l-1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("etalp1", "(pseudo-)rapidity of l+1", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("etalm2", "(pseudo-)rapidity of l-2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("etalp2", "(pseudo-)rapidity of l+2", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_etalm1", "|etalm1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("abs_etalp1", "|etalp1|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("abs_etalm2", "|etalm2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("abs_etalp2", "|etalp2|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ! gauge bosons
      ! call bind_obs("ptz1", "transverse momentum of Z1", &
      !               id_pt_l12)
      ! call bind_obs("ptz2", "transverse momentum of Z2", &
      !               id_pt_l34)
      ! call bind_obs("yz1", "rapidity of Z1", &
      !               id_y_l12)
      ! call bind_obs("yz2", "rapidity of Z2", &
      !               id_y_l34)
      ! call bind_obs("abs_yz1", "|yz1|", &
      !               id_absy_l12)
      ! call bind_obs("abs_yz2", "|yz2|", &
      !               id_absy_l34)
      ! call bind_obs("mz1", "invariant mass of Z1", &
      !               id_m_l12)
      ! call bind_obs("mz2", "invariant mass of Z2", &
      !               id_m_l34)
      ! ! VBS system = Z1 + Z2
      ! call bind_obs("ptzz", "transverse momentum of ZZ system", &
      !               id_pt_l1234)
      ! call bind_obs("mzz", "invariant mass of ZZ system", &
      !               id_m_l1234)
      ! call bind_obs("yzz", "rapidity of ZZ system", &
      !               id_y_l1234)
      ! call bind_obs("abs_yzz", "|yzz|", &
      !               id_absy_l1234)
      ! ! all leptons are charged and thus detectable
      ! call bind_obs("ptl1", "transverse momentum of leading lepton", &
      !               id_pt_leading_l1234)
      ! call bind_obs("etal1", "(pseudo-)rapidity of leading lepton", &
      !               id_y_leading_l1234)
      ! call bind_obs("abs_etal1", "|etal1|", &
      !               id_absy_leading_l1234)
      ! call bind_obs("ptl2", "transverse momentum of leading sub-lepton", &
      !               id_pt_subleading_l1234)
      ! call bind_obs("etal2", "(pseudo-)rapidity of leading sub-lepton", &
      !               id_y_subleading_l1234)
      ! call bind_obs("abs_etal2", "|etal2|", &
      !               id_absy_subleading_l1234)
      ! call bind_obs("deltaR_lj", "minimum deltaR between any charged lepton and jet", &
      !               id_deltaR_l1234j)

    case ("H", "HJ", "HJJ", "HJJJ",  &
         & "HM", "HJMREW", "HJJMREW", "HJM", "HJJM", "HJJJM",  &
         & "HINC", "HQT")
      ! ---- gg->H event-level Chebyshev moments
      call bind_obs("chebT_yh_1", "T_1 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-1))
      call bind_obs("chebT_yh_2", "T_2 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-2))
      call bind_obs("chebT_yh_3", "T_3 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-3))
      call bind_obs("chebT_yh_4", "T_4 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-4))
      call bind_obs("chebT_yh_5", "T_5 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-5))
      call bind_obs("chebT_yh_6", "T_6 |y_H|", EvalObs_t(eval_mem=eval_chebT_yh, imem=-6))
      call bind_obs("w_pth", "profile w(pT_H)", EvalObs_t(eval=eval_w_pth), ifac=2)
      call bind_obs("chebTw_pth_1", "w*T_1 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-1), ifac=2)
      call bind_obs("chebTw_pth_2", "w*T_2 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-2), ifac=2)
      call bind_obs("chebTw_pth_3", "w*T_3 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-3), ifac=2)
      call bind_obs("chebTw_pth_4", "w*T_4 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-4), ifac=2)
      call bind_obs("chebTw_pth_5", "w*T_5 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-5), ifac=2)
      call bind_obs("chebTw_pth_6", "w*T_6 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-6), ifac=2)
      call bind_obs("chebTw_pth_7", "w*T_7 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-7), ifac=2)
      call bind_obs("chebTw_pth_8", "w*T_8 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-8), ifac=2)
      call bind_obs("chebTw_pth_9", "w*T_9 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-9), ifac=2)
      call bind_obs("chebTw_pth_10", "w*T_10 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-10), ifac=2)
      call bind_obs("chebTw_pth_11", "w*T_11 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-11), ifac=2)
      call bind_obs("chebTw_pth_12", "w*T_12 pT_H", EvalObs_t(eval_mem=eval_chebTw_pth, imem=-12), ifac=2)
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)

      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      call bind_obs("pthjj", "transverse momentum of H plus two leading jets", &
                    EvalObs_t(eval=eval_pt_Vjj, min_njets=2), ifac=2)
      call bind_obs("pthj", "transverse momentum of H plus the leading jet", &
                    EvalObs_t(eval=eval_pt_Vj, min_njets=1), ifac=2)
      call bind_obs("dpt_hj", " = pth - ptj1", &
                    EvalObs_t(eval=eval_dpt_Vj, min_njets=1))
      call bind_obs("abs_dpt_hj", " = |pth - ptj1|", &
                    EvalObs_t(eval=eval_dpt_Vj, min_njets=1, decorator=abs_deco))
      call bind_obs("mhj", "invariant mass of H plus the leading jet", &
                    EvalObs_t(eval=eval_minv_Vj, min_njets=1), ifac=1)
      call bind_obs("deltay_hj1", "|yh - yj1|", &
                    EvalObs_t(eval=eval_deltay_Vj1, min_njets=1))
      call bind_obs("deltaR_hj1", "cone size between V and j1", &
                    EvalObs_t(eval=eval_dR_Vj1, min_njets=1))
      call bind_obs("deltaR_hj2", "cone size between V and j2", &
                    EvalObs_t(eval=eval_dR_Vj2, min_njets=2))
      call bind_obs("dphi_hj1", "Azimuthal angle difference of H and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dphi_hj2", "Azimuthal angle difference of H and 2nd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-2, min_njets=2))
      call bind_obs("eth", "sqrt(m[h]^2 + pt{h}^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      call bind_obs("dscale1", "sqrt(mgg(H)^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("dscale2", "eth + ptj1", &
                    EvalObs_t(eval=eval_ZJ_dscale2, min_njets=1))
      call bind_obs("dscale3", "eth + sum pt{jets}", &
                    EvalObs_t(eval=eval_ZJ_dscale3))
      call bind_obs("dscale4", "sqrt(mH/2 * sqrt(mH^2/4 + ptH^2))", &
                    EvalObs_t(eval=eval_VFH_dscale)) !eq.(2) of 1506.02660
      call bind_obs("phi_star", "\phi^*_\eta ", &
                    EvalObs_t(eval=eval_phi_star_eta))
      call bind_obs("H_HT", "et_Higgs + sum pt{partons} [post Les Houches 2017]", &
                    EvalObs_t(eval=eval_ZJ_HT))
      call bind_obs("htpt_parton", "ptz + sum pt{partons}", &
                    EvalObs_t(eval=eval_ZJ_HTpT))
      call bind_obs("dphi_hj12", "Azimuthal angle between the H and the di-jet system", &
                    EvalObs_t(eval=eval_deltaphi_Vj12, min_njets=2))
      call bind_obs("deltaphi_j12_rap_ord", "signed azimuthal angle between j1 & j2 (y1 > y2)", &
                    EvalObs_t(eval=eval_deltaphi_j12_rap_ord, min_njets=2))
      call bind_obs("h_zep", "|\eta_H - 0.5 * (\eta_j1 + \eta_j2)|", &
                    EvalObs_t(eval=eval_h_zep))
      call bind_obs("h_zstar", "h_zep/deltaeta_j12", &
                    EvalObs_t(eval=eval_h_zstar))

    case ("HTO2PM", "HTO2PJM", "HTO2P", "HTO2PJ", "HTO2PJJ", "HTO2PJJJ", "HTO2PINC", "HTO2PQT")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      ! photon from the H -> gamma gamma decay
      call bind_obs("nphotons", "number of (isolated) photon", &
                    EvalObs_t(eval=eval_nphotons, is_integer=.true.))
      call bind_obs("ptg1", "transverse momentum of leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, min_nphotons=1), ifac=2)
      call bind_obs("yg1", "(pseudo-)rapidity of leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, min_nphotons=1))
      call bind_obs("abs_yg1", "|yg1|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, decorator=abs_deco, min_nphotons=1))
      call bind_obs("ptg2", "transverse momentum of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-2, min_nphotons=2), ifac=2)
      call bind_obs("yg2", "(pseudo-)rapidity of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, min_nphotons=2))
      call bind_obs("abs_yg2", "|yg2|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, decorator=abs_deco, min_nphotons=2))
      call bind_obs("min_dr_gj", "minimum of Delta R (jets, photons)", &
                    EvalObs_t(eval=eval_min_dR_gamj, min_njets=1, min_nphotons=1))
      call bind_obs("dphi_g1g2", "Azimuthal angle difference of di-photon", &
                    EvalObs_t(eval=eval_deltaphi_gam12, min_nphotons=2))
      call bind_obs("dy_g1g2", "(pseudo-)rapidity difference of di-photon", &
                    EvalObs_t(eval=eval_deltay_gam12, min_nphotons=2))
      call bind_obs("abs_dy_g1g2", "|dy_g1g2|", &
                    EvalObs_t(eval=eval_abs_deltay_gam12, min_nphotons=2))
      call bind_obs("dr_g1g2", "Delta R (g1, g2)", &
                    EvalObs_t(eval=eval_deltaR_gam12, min_nphotons=2))
      call bind_obs("abs_cos_theta", "photon decay angles in the lab frame", &
                    EvalObs_t(eval=eval_costhst_gam12, min_nphotons=2))
      call bind_obs("cos_thetaCS", "photon decay angles in the Collins-Soper frame ", &
                    EvalObs_t(eval=eval_costhstCS_gam12, min_nphotons=2))
      call bind_obs("abs_cos_thetaCS", "photon decay angles in the Collins-Soper frame ", &
                    EvalObs_t(eval=eval_abscosthstCS_gam12, min_nphotons=2))
      call bind_obs("phi_star", "\phi^*_\eta", &
                    EvalObs_t(eval=eval_phi_star_gam12, min_nphotons=2))
      call bind_obs("pTt_g1g2", "pTgg component transverse to thrust axis: standard impl.", &
      EvalObs_t(eval=eval_absptt_gam12, min_nphotons=2))
      call bind_obs("ptt_g1g2", "pTgg component transverse to thrust axis: Projection impl.", &
      EvalObs_t(eval=eval_ptt_gam12, min_nphotons=2))
      call bind_obs("at_g1g2", "pTt if dphi_gg > pi/2, or pTgg if dphi_gg <= pi/2", &
      EvalObs_t(eval=eval_absat_gam12, min_nphotons=2))
      call bind_obs("Mg1g2", "invariant mass of diphoton system", &
                    EvalObs_t(eval=eval_minv_gam12, min_nphotons=2), ifac=1)
      call bind_obs("pt_g1g2", "pt of diphoton system", &
                    EvalObs_t(eval=eval_pt_gam12, min_nphotons=2), ifac=2)
      call bind_obs("ptavg_g1g2", "(ptg1+ptg2)/2", &
                    EvalObs_t(eval=eval_ptavg_gam12, min_nphotons=2))
      call bind_obs("ptavg_geom_g1g2", "sqrt(ptg1*ptg2)", &
                    EvalObs_t(eval=eval_ptavg_geom_gam12, min_nphotons=2))
      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      call bind_obs("pthjj", "transverse momentum of H plus two leading jets", &
                    EvalObs_t(eval=eval_pt_Vjj, min_njets=2), ifac=2)
      call bind_obs("pthj", "transverse momentum of H plus the leading jet", &
                    EvalObs_t(eval=eval_pt_Vj, min_njets=1), ifac=2)
      call bind_obs("deltay_hj1", "|yh - yj1|", &
                    EvalObs_t(eval=eval_deltay_Vj1, min_njets=1))
      call bind_obs("dphi_hj1", "Azimuthal angle difference of H and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("eth", "sqrt(mgg(H)^2 + pth^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      call bind_obs("tau_j1", "tau-jet of the leading jet", &
                    EvalObs_t(eval=eval_tau_j1))
      call bind_obs("max_tau_jet", "the maximum of all tau-jet among all jets", &
                    EvalObs_t(eval=eval_max_tau_jet))
      call bind_obs("sum_tau_jets", "sum of all jet's tau-jet", &
                    EvalObs_t(eval=eval_sum_tau_jets))
      call bind_obs("dscale1", "sqrt(mgg(H)^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("dscale2", "eth + ptj1", &
                    EvalObs_t(eval=eval_ZJ_dscale2, min_njets=1))
      call bind_obs("dscale3", "eth + sum pt{jets}", &
                    EvalObs_t(eval=eval_ZJ_dscale3))
      call bind_obs("h_ht", "sqrt(m[h]^2 + pt{h}^2) + sum pt{partons}", &
                    EvalObs_t(eval=eval_ZJ_HT))

    case ("HTO2TAUM", "HTO2TAUJM", "HTO2TAU", "HTO2TAUJ", "HTO2TAUJJ", "HTO2TAUJJJ", "HTO2TAUINC", "HTO2TAUQT")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      ! tau from the H -> tau+ tau- decay
      call bind_obs("pttau1", "transverse momentum of leading tau", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=1))
      call bind_obs("ytau1", "(pseudo-)rapidity of leading tau", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=2))
      call bind_obs("abs_ytau1", "|ytau1|", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=2, decorator=abs_deco))
      call bind_obs("pttau2", "transverse momentum of sub-leading tau", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=3))
      call bind_obs("ytau2", "(pseudo-)rapidity of sub-leading tau", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=4))
      call bind_obs("abs_ytau2", "|ytau2|", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=4, decorator=abs_deco))
      call bind_obs("min_dr_tauj", "minimum of Delta R (jets, taus)", &
                    EvalObs_t(eval=eval_min_dR_l12j))
      call bind_obs("dphi_tau1tau2", "Azimuthal angle difference of tau+ and tau-", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=6))
      call bind_obs("dy_tau1tau2", "(pseudo-)rapidity difference of tau+ and tau-", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=5))
      call bind_obs("abs_dy_tau1tau2", "|dy_tau1tau2|", &
                    EvalObs_t(eval_mem=eval_2l_nth, imem=5, decorator=abs_deco))
      call bind_obs("costar_tautau", "tau decay angles in the Collins-Soper frame", &
                    EvalObs_t(eval=eval_costhetastar))
      call bind_obs("phi_star", "\phi^*_\eta", &
                    EvalObs_t(eval=eval_phi_star_eta))
      call bind_obs("ptt_tau1tau2", "pTt of tau+ and tau-", &
                    EvalObs_t(eval=eval_ptt_l1l2))
      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      call bind_obs("pthjj", "transverse momentum of H plus two leading jets", &
                    EvalObs_t(eval=eval_pt_Vjj, min_njets=2), ifac=2)
      call bind_obs("pthj", "transverse momentum of H plus the leading jet", &
                    EvalObs_t(eval=eval_pt_Vj, min_njets=1), ifac=2)
      call bind_obs("deltay_hj1", "|yh - yj1|", &
                    EvalObs_t(eval=eval_deltay_Vj1, min_njets=1))
      call bind_obs("dphi_hj1", "Azimuthal angle difference of H and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dscale1", "sqrt(mgg(H)^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_ZJ_dscale1))
      call bind_obs("eth", "sqrt(mgg(H)^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      call bind_obs("tau_j1", "tau-jet of the leading jet", &
                    EvalObs_t(eval=eval_tau_j1))
      call bind_obs("max_tau_jet", "the maximum of all tau-jet among all jets", &
                    EvalObs_t(eval=eval_max_tau_jet))
      call bind_obs("sum_tau_jets", "sum of all jet's tau-jet", &
                    EvalObs_t(eval=eval_sum_tau_jets))

    case ("HTO2L1PM", "HTO2L1PJM", "HTO2L1P", "HTO2L1PJ", "HTO2L1PJJ", "HTO2L1PJJJ", "HTO2L1PINC", "HTO2L1PQT")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V2l1p), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V2l1p))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V2l1p, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V2l1p))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V2l1p, decorator=abs_deco))
      call bind_obs("eth", "Higgs transvere energy", &
                    EvalObs_t(eval=eval_et_3l), ifac=1)
      ! Z-boson
      call bind_obs("ptll", "transverse momentum of chanrge letpon system", &
                    EvalObs_t(eval=eval_pt_2l1p_V), ifac=2)
      ! leptons and photon from the H -> lp lm gamma decay
      call bind_obs("m23", "invariant mass of di-lepton system", &
                    EvalObs_t(eval=eval_minv_V23), ifac=1)
      call bind_obs("min_dr_lp", "the minimum delta_R value between the phton and one of the leptons", &
                    EvalObs_t(eval=eval_min_dR_l12p3))
      call bind_obs("ptgamma", "transverse momentum of photon", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ygamma", "rapidity of the photon", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ygamma", "|ygamma|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("ptlm", "transverse momentum of negative charged lepton", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("ptlp", "transverse momentum of positive charged lepton", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      ! ordered leptons from the Z -> lp lm decay
      call bind_obs("ptl1st", "transverse momentum of the leading lepton in the 2l system", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=1))
      call bind_obs("yl1st", "(pseudo-)rapidity of the leading lepton in the 2l system", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=2))
      call bind_obs("abs_yl1st", "|yl1st|", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=2, decorator=abs_deco))
      call bind_obs("ptl2nd", "transverse momentum of the sub-leading lepton in the 2l system", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=3))
      call bind_obs("yl2nd", "(pseudo-)rapidity of the sub-leading lepton in the 2l system", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=4))
      call bind_obs("abs_yl2nd", "|yl2nd|", &
                    EvalObs_t(eval_mem=eval_2l1p_nth, imem=4, decorator=abs_deco))

    case ("HTO4EM", "HTO4EJM", "HTO4E", "HTO4EJ", "HTO4EJJ", "HTO4EJJJ", "HTO4EINC", "HTO4EQT", &
      "HH", "HHJ", "HHJJ", "HHJJJ")
      ! leptons from the Z -> lp lm decay
      call bind_obs("ptem1", "transverse momentum of one e-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("yem1", "(pseudo-)rapidity of one e-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_yem1", "|ye1m|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptep1", "transverse momentum of one e+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("yep1", "(pseudo-)rapidity of one e+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_yep1", "|ye1p|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("ptem2", "transverse momentum of another e-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("yem2", "(pseudo-)rapidity of another e-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("abs_yem2", "|ye2m|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("ptep2", "transverse momentum of another e+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("yep2", "(pseudo-)rapidity of another e+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_yep2", "|ye2p|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ordered leptons from the two Z -> lp lm decay
      call bind_obs("pte1st", "transverse momentum of the leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=1))
      call bind_obs("ye1st", "(pseudo-)rapidity of the leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=2))
      call bind_obs("abs_ye1st", "|ye1st|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=2, decorator=abs_deco))
      call bind_obs("pte2nd", "transverse momentum of  the sub-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=4))
      call bind_obs("ye2nd", "(pseudo-)rapidity of  the sub-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=5))
      call bind_obs("abs_ye2nd", "|ye2nd|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=5, decorator=abs_deco))
      call bind_obs("pte3rd", "transverse momentum of the thrid-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=7))
      call bind_obs("ye3rd", "(pseudo-)rapidity of  the thrid-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=8))
      call bind_obs("abs_ye3rd", "|ye3rd|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=8, decorator=abs_deco))
      call bind_obs("pte4th", "transverse momentum of the least-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=10))
      call bind_obs("ye4th", "(pseudo-)rapidity of the least-leading lepton in the 4e system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=11))
      call bind_obs("abs_ye4th", "|ye4th|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=11, decorator=abs_deco))
      ! Z-boson = one e+ + e- system
      call bind_obs("ptepem1", "transverse momentum of one Z-->e+e- ", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yepem1", "rapidity of one Z-->e+e-", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yepem1", "|yepem1|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaepem1", "pseudo-rapidity of one Z-->e+e-", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etaepem1", "|etaepem1|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("mepem1", "invariant mass of one Z-->e+e-", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("etepem1", "transverse energy of one e+e- pair: sqrt(mepem1^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      ! Z-boson = another e+ + e- system
      call bind_obs("ptepem2", "transverse momentum of another Z-->e+e- ", &
                    EvalObs_t(eval=eval_pt_V2), ifac=2)
      call bind_obs("yepem2", "rapidity of another Z-->e+e-", &
                    EvalObs_t(eval=eval_y_V2))
      call bind_obs("abs_yepem2", "|ymupmum2|", &
                    EvalObs_t(eval=eval_y_V2, decorator=abs_deco))
      call bind_obs("etaepem2", "pseudo-rapidity of another Z-->e+e-", &
                    EvalObs_t(eval=eval_eta_V2))
      call bind_obs("abs_etaepem2", "|etamupmum2|", &
                    EvalObs_t(eval=eval_eta_V2, decorator=abs_deco))
      call bind_obs("mepem2", "invariant mass of another Z-->e+e-", &
                    EvalObs_t(eval=eval_minv_V2), ifac=1)
      call bind_obs("etmupmum2", "transverse energy: sqrt(mepem2^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V2), ifac=1)
      ! angular separation
      call bind_obs("min_dr_epem", "minimum Delta R_e+e- of the two e+e- systems", &
                    EvalObs_t(eval=eval_min_dR_l_sf))
      call bind_obs("min_dr_e1e2", "minimum Delta R of ep1_ep2, ep1_em2, em1_ep2, em1_em2", &
                    EvalObs_t(eval=eval_min_dR_l_df))
      call bind_obs("min_dr_lj", "minimum of Delta R (jets, leptons)", &
                    EvalObs_t(eval=eval_min_dR_l1234j))
      call bind_obs("dr_e1st_j1", "Delta R distance between (leading jet, leading lepton)", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=3, min_njets=1))
      ! Z-boson = 4e ordering system
      call bind_obs("m2lfarestmZ", "invariant mass of SFOS e+e- pair farest to the Z-mass", &
                    EvalObs_t(eval_mem=eval_minv_mZSFOS, imem=1), ifac=1)
      call bind_obs("m2lfarmZ", "invariant mass of SFOS e+e- pair third close to the Z-mass", &
                    EvalObs_t(eval_mem=eval_minv_mZSFOS, imem=2), ifac=1)
      call bind_obs("m2lclosemZ", "invariant mass of SFOS e+e- pair second close to the Z-mass", &
                    EvalObs_t(eval_mem=eval_minv_mZSFOS, imem=3), ifac=1)
      call bind_obs("m2lZ1", "invariant mass of SFOS e+e- pair closest to Z-mass (Z1 candidate)", &
                    EvalObs_t(eval_mem=eval_minv_mZSFOS, imem=4), ifac=1)
      call bind_obs("m2lZ2", "invariant mass of SFOS e+e- pair which are not inside m2lZ1 (Z2)", &
                    EvalObs_t(eval_mem=eval_minv_mZSFOS, imem=5), ifac=1)
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("m4l", "invariant mass of H", &
                    EvalObs_t(eval=eval_minv_V4l), ifac=1)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V4l))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V4l, decorator=abs_deco))
      ! dynamical scales
      call bind_obs("dscale1", "sqrt(m4l^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale1))
      call bind_obs("dscale2", "sqrt(m4l^2 + ptz^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale2))

    case ("HTO2E2MUM", "HTO2E2MUJM", "HTO2E2MU", "HTO2E2MUJ", "HTO2E2MUJJ", "HTO2E2MUJJJ", "HTO2E2MUINC")
      ! leptons from the Z -> lp lm decay
      call bind_obs("ptmum", "transverse momentum of mu-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ymum", "(pseudo-)rapidity of mu-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ymum", "|ymum|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptmup", "transverse momentum of mu+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ymup", "(pseudo-)rapidity of mu+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ymup", "|ymup|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      call bind_obs("ptem", "transverse momentum of e-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-3), ifac=2)
      call bind_obs("yem", "(pseudo-)rapidity of e-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3))
      call bind_obs("abs_yem", "|yem|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-3, decorator=abs_deco))
      call bind_obs("ptep", "transverse momentum of e+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-4), ifac=2)
      call bind_obs("yep", "(pseudo-)rapidity of e+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4))
      call bind_obs("abs_yep", "|yep|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-4, decorator=abs_deco))
      ! ordered leptons from the two Z -> lp lm decay
      call bind_obs("ptl1st", "transverse momentum of the leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=1))
      call bind_obs("yl1st", "(pseudo-)rapidity of the leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=2))
      call bind_obs("abs_yl1st", "|yl1st|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=2, decorator=abs_deco))
      call bind_obs("ptl2nd", "transverse momentum of  the sub-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=4))
      call bind_obs("yl2nd", "(pseudo-)rapidity of  the sub-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=5))
      call bind_obs("abs_yl2nd", "|yl2nd|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=5, decorator=abs_deco))
      call bind_obs("ptl3rd", "transverse momentum of the thrid-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=7))
      call bind_obs("yl3rd", "(pseudo-)rapidity of  the thrid-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=8))
      call bind_obs("abs_yl3rd", "|yl3rd|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=8, decorator=abs_deco))
      call bind_obs("ptl4th", "transverse momentum of the least-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=10))
      call bind_obs("yl4th", "(pseudo-)rapidity of the least-leading lepton in the 2e2mu system", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=11))
      call bind_obs("abs_yl4th", "|yl4th|", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=11, decorator=abs_deco))
      ! Z-boson = mu+ + mu- system
      call bind_obs("ptmupmum", "transverse momentum of Z-->mu+mu- ", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("ymupmum", "rapidity of Z-->mu+mu-", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_ymupmum", "|ymupmum|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaepem", "pseudo-rapidity of Z-->e+e-", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etamupmum", "|etamupmum|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("mmupmum", "invariant mass of Z-->mu+mu-", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("etmupmum", "transverse energy: sqrt(mmupmum^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      call bind_obs("dr_mupmum", "Delta R (mu+, mu-)", &
                    EvalObs_t(eval=eval_dR_l12))
      ! Z-boson = e+ + e- system
      call bind_obs("ptepem", "transverse momentum of Z-->e+e- ", &
                    EvalObs_t(eval=eval_pt_V2), ifac=2)
      call bind_obs("yepem", "rapidity of Z-->e+e-", &
                    EvalObs_t(eval=eval_y_V2))
      call bind_obs("abs_yepem", "|yepem|", &
                    EvalObs_t(eval=eval_y_V2, decorator=abs_deco))
      call bind_obs("etaepem", "pseudo-rapidity of Z-->e+e-", &
                    EvalObs_t(eval=eval_eta_V2))
      call bind_obs("abs_etaepem", "|etaepem|", &
                    EvalObs_t(eval=eval_eta_V2, decorator=abs_deco))
      call bind_obs("mepem", "invariant mass of Z-->e+e-", &
                    EvalObs_t(eval=eval_minv_V2), ifac=1)
      call bind_obs("etepem", "transverse energy: sqrt(mepem^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V2), ifac=1)
      call bind_obs("dr_epem", "Delta R (e+, e-)", &
                    EvalObs_t(eval=eval_dR_l34))
      ! Z-boson alternative l+l- system
      call bind_obs("mepmum", "invariant mass of Z-->e+mu-", &
                    EvalObs_t(eval=eval_minv_V23), ifac=1)
      call bind_obs("memmup", "invariant mass of Z-->e-mu+", &
                    EvalObs_t(eval=eval_minv_V14), ifac=1)
      ! angular separation
      call bind_obs("min_dr_sf", "minimum of Delta R (same flavour leptons)", &
                    EvalObs_t(eval=eval_min_dR_l_sf))
      call bind_obs("min_dr_df", "minimum of Delta R (different flavour leptons)", &
                    EvalObs_t(eval=eval_min_dR_l_df))
      call bind_obs("min_dr_lj", "minimum of Delta R (jets, leptons)", &
                    EvalObs_t(eval=eval_min_dR_l1234j))
      call bind_obs("min_dr_muj", "minimum of Delta R (jets, muons)", &
                    EvalObs_t(eval=eval_min_dR_l12j))
      call bind_obs("min_dr_ej", "minimum of Delta R (jets, electrons)", &
                    EvalObs_t(eval=eval_min_dR_l34j))
      call bind_obs("dr_l1st_j1", "Delta R distance between (leading jet, leading lepton)", &
                    EvalObs_t(eval_mem=eval_4l_nth, imem=3, min_njets=1))
      ! Z-boson = four lepton ordering system
      call bind_obs("m2lnearmZ", "invariant mass of lepton pair closest to the Z-mass", &
                    EvalObs_t(eval=eval_minv_mZ1), ifac=1)
      call bind_obs("m2lfarmZ", "invariant mass of lepton pair fartest to the Z-mass", &
                    EvalObs_t(eval=eval_minv_mZ2), ifac=1)
      call bind_obs("m2em2muvsmZ", "identify the SFOS minv closest to mZ and label 1 (2) for m2e (m2mu)", &
                    EvalObs_t(eval=eval_minvSFOS_vs_mZ, is_integer=.true.))
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("m4l", "invariant mass of H", &
                    EvalObs_t(eval=eval_minv_V4l), ifac=1)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V4l))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V4l, decorator=abs_deco))
      call bind_obs("eth", "Higgs transvere energy", &
                    EvalObs_t(eval=eval_et_4l), ifac=1)
      ! dynamical scales
      call bind_obs("dscale1", "sqrt(m4l^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale1))
      call bind_obs("dscale2", "sqrt(m4l^2 + ptz^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale2))

    case ("HTO2L2NM", "HTO2L2NJM", "HTO2L2N", "HTO2L2NJ", "HTO2L2NJJ", "HTO2L2NJJJ", "HTO2L2NINC", "HTO2L2NQT")
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("m2l2n", "invariant mass of H", &
                    EvalObs_t(eval=eval_minv_V4l), ifac=1)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V4l))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V4l, decorator=abs_deco))
      ! dynamical scales
      call bind_obs("dscale1", "sqrt(m2l2n^2 + sum pt{jets}^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale1))
      call bind_obs("dscale2", "sqrt(m2l2n^2 + ptz^2)", &
                    EvalObs_t(eval=eval_H4lJ_dscale2))

    case ("VFH", "VFHJ", "VFHFULL")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etah", "pseudo-rapidity of H", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etah", "|etah|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("pthjj", "transverse momentum of H plus two leading jets", &
                    EvalObs_t(eval=eval_pt_Vjj, min_njets=2), ifac=2)

      ! including non-coloured final states
      call bind_obs("ht_full", "scalar pt sum over leptons and jets", &
                    EvalObs_t(eval=eval_ht_full), ifac=1)
      ! VFH cuts
      call bind_obs("deltay", "|y1 - y2|", &
                    EvalObs_t(eval=eval_VFH_deltay, min_njets=2))
      call bind_obs("y1xy2", "y1*y2", &
                    EvalObs_t(eval=eval_VFH_y1xy2, min_njets=2))
      call bind_obs("dscale", "sqrt(mH/2 * sqrt(mH^2/4 + ptH^2))", &
                    EvalObs_t(eval=eval_VFH_dscale))  ! eq.(2) of 1506.02660
      call bind_obs("z3", " z3 = (y3 - (y1+y2)/2)/(y1-y2)", & ! normalised central rapidity distribution of 3rd jet:
                    EvalObs_t(eval=eval_VFH_z3, min_njets=3))
      call bind_obs("ht_parton", "et_Higgs + sum pt{partons} [post Les Houches 2017]", &
                    EvalObs_t(eval=eval_ZJ_HT))
      call bind_obs("deltaphi_j12_rap_ord", "signed azimuthal angle between j1 & j2 (y1 > y2)", &
                    EvalObs_t(eval=eval_deltaphi_j12_rap_ord, min_njets=2))
      call bind_obs("dphi_hj1", "Azimuthal angle difference of H and 1st jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-1, min_njets=1))
      call bind_obs("dphi_hj2", "Azimuthal angle difference of H and 2nd jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_V_jet, imem=-2, min_njets=2))
      call bind_obs("h_zep", "|\eta_H - 0.5 * (\eta_j1 + \eta_j2)|", &
                    EvalObs_t(eval=eval_h_zep))
      call bind_obs("h_zstar", "h_zep/deltaeta_j12", &
                    EvalObs_t(eval=eval_h_zstar))

    case ("VFHTO2P", "VFHTO2PJ")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))

    case ("VFHTO2TAU", "VFHTO2TAUJ")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)

    case ("VFHTO2L1P", "VFHTO2L1PJ")
      ! H-boson
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V2l1p), ifac=2)
      call bind_obs("m23", "invariant mass of di-lepton system", &
                    EvalObs_t(eval=eval_minv_V23), ifac=1)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V2l1p))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V2l1p, decorator=abs_deco))

    case ("VFHTO2E2MU", "VFHTO2E2MUJ")
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))

    case ("VFHTO4E", "VFHTO4EJ")
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))

    case ("VFHTO2L2N", "VFHTO2L2NJ")
      ! H-boson = l1 + l2 +l3 +l4
      call bind_obs("pth", "transverse momentum of H", &
                    EvalObs_t(eval=eval_pt_V4l), ifac=2)
      call bind_obs("yh", "rapidity of H", &
                    EvalObs_t(eval=eval_y_V4l))
      call bind_obs("abs_yh", "|yh|", &
                    EvalObs_t(eval=eval_y_V4l, decorator=abs_deco))
      
    case ("GJ", "GJ_FRAG", "G_EW", "GJJ", "GJJJ", "GJ_H")
      call bind_obs("nphotons", "number of (isolated) photon", &
                    EvalObs_t(eval=eval_nphotons, is_integer=.true.))
      call bind_obs("pt_gam", "pt of photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, min_nphotons=1), ifac=2)
      call bind_obs("y_gam", "rapidity of photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, min_nphotons=1))
      call bind_obs("abs_y_gam", "|y_gam|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, decorator=abs_deco, min_nphotons=1))
      call bind_obs("sqrt_pt_gam", "square root of pt_gam", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, decorator=sqrt_abs_deco, min_nphotons=1), ifac=2)
      call bind_obs("dphi_gam_j1", "azimuthal separation between photon and leading jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("m_gam_j1", "invariant mass of the photon and leading jet system", &
                    EvalObs_t(eval_mem=eval_minv_gamj1, imem=-1, min_njets=1, min_nphotons=1), ifac=1)
      call bind_obs("costhetastar_gam_j1", "costhetastar between the photon and leading jet", &
                    EvalObs_t(eval_mem=eval_costhstar_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_costhetastar_gam_j1", "|costhetastar_gam_j1|", &
                    EvalObs_t(eval_mem=eval_costhstar_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("y_gam_plus_y_j1", "sum of y_gam and y_j1", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_y_gam_plus_y_j1", "|y_gam_plus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("y_gam_minus_y_j1", "difference of y_gam and y_j1", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_y_gam_minus_y_j1", "|y_gam_minus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("min_dr_gamj", "min(DetaR(photon,jet))", &
                    EvalObs_t(eval=eval_min_dR_gamj, min_njets=1, min_nphotons=1))
      call bind_obs("min_dr_exp_gam1j1", "min(DetaR_exp(photon,jet))", &
                    EvalObs_t(eval=eval_min_dR_exp_gam1j1, min_njets=1, min_nphotons=1))
      call bind_obs("imbalance_pt_gam1j1", "pt photon / pt jet", &
                    EvalObs_t(eval=eval_imbalance_pt_gam1j1, min_njets=1, min_nphotons=1))

    case ("GG", "GG_FRAG", "GGJ", "GGJFC", "GGJJ", "GGJJFC", "GGJJJFC", "GG_EW", "GGQT")
      ! ---- diphoton angular moments (fix the m-cos correlation)
      call bind_obs("chebT_cts_1", "T_1 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-1))
      call bind_obs("chebT_cts_2", "T_2 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-2))
      call bind_obs("chebT_cts_3", "T_3 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-3))
      call bind_obs("chebT_cts_4", "T_4 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-4))
      call bind_obs("chebT_cts_5", "T_5 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-5))
      call bind_obs("chebT_cts_6", "T_6 |cos th*|", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_cts, imem=-6))
      call bind_obs("w_dpa", "w(pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval=eval_w_dpa))
      call bind_obs("chebTw_dpa_1", "w T_1 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-1))
      call bind_obs("chebTw_dpa_2", "w T_2 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-2))
      call bind_obs("chebTw_dpa_3", "w T_3 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-3))
      call bind_obs("chebTw_dpa_4", "w T_4 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-4))
      call bind_obs("chebTw_dpa_5", "w T_5 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-5))
      call bind_obs("chebTw_dpa_6", "w T_6 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_dpa, imem=-6))
      call bind_obs("chebT_dpa_1", "T_1 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-1))
      call bind_obs("chebT_dpa_2", "T_2 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-2))
      call bind_obs("chebT_dpa_3", "T_3 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-3))
      call bind_obs("chebT_dpa_4", "T_4 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-4))
      call bind_obs("chebT_dpa_5", "T_5 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-5))
      call bind_obs("chebT_dpa_6", "T_6 (pi-dphi_gg)", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_dpa, imem=-6))
      ! ---- diphoton event-level Chebyshev moments
      call bind_obs("chebT_maa_1", "T_1 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-1), ifac=1)
      call bind_obs("chebT_maa_2", "T_2 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-2), ifac=1)
      call bind_obs("chebT_maa_3", "T_3 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-3), ifac=1)
      call bind_obs("chebT_maa_4", "T_4 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-4), ifac=1)
      call bind_obs("chebT_maa_5", "T_5 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-5), ifac=1)
      call bind_obs("chebT_maa_6", "T_6 m_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebT_maa, imem=-6), ifac=1)
      call bind_obs("w_ptaa", "gg recoil profile w(pt_aa)", EvalObs_t(eval=eval_w_ptaa, min_nphotons=2), ifac=2)
      call bind_obs("chebTw_ptaa_1", "w*T_1 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-1), ifac=2)
      call bind_obs("chebTw_ptaa_2", "w*T_2 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-2), ifac=2)
      call bind_obs("chebTw_ptaa_3", "w*T_3 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-3), ifac=2)
      call bind_obs("chebTw_ptaa_4", "w*T_4 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-4), ifac=2)
      call bind_obs("chebTw_ptaa_5", "w*T_5 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-5), ifac=2)
      call bind_obs("chebTw_ptaa_6", "w*T_6 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-6), ifac=2)
      call bind_obs("chebTw_ptaa_7", "w*T_7 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-7), ifac=2)
      call bind_obs("chebTw_ptaa_8", "w*T_8 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-8), ifac=2)
      call bind_obs("chebTw_ptaa_9", "w*T_9 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-9), ifac=2)
      call bind_obs("chebTw_ptaa_10", "w*T_10 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-10), ifac=2)
      call bind_obs("chebTw_ptaa_11", "w*T_11 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-11), ifac=2)
      call bind_obs("chebTw_ptaa_12", "w*T_12 pt_aa", EvalObs_t(min_nphotons=2, eval_mem=eval_chebTw_ptaa, imem=-12), ifac=2)
      call bind_obs("nphotons", "number of (isolated) photon", &
                    EvalObs_t(eval=eval_nphotons, is_integer=.true.))
      call bind_obs("ptg1", "transverse momentum of leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, min_nphotons=1), ifac=2)
      call bind_obs("sqrt_ptg1", "square root of ptg1", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, decorator=sqrt_abs_deco, min_nphotons=1), ifac=2)
      call bind_obs("yg1", "(pseudo-)rapidity of leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, min_nphotons=1))
      call bind_obs("abs_yg1", "|yg1|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, decorator=abs_deco, min_nphotons=1))
      call bind_obs("ptg2", "transverse momentum of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-2, min_nphotons=2), ifac=2)
      call bind_obs("sqrt_ptg2", "square root of ptg2", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-2, decorator=sqrt_abs_deco, min_nphotons=2), ifac=2)
      call bind_obs("yg2", "(pseudo-)rapidity of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, min_nphotons=2))
      call bind_obs("abs_yg2", "|yg2|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, decorator=abs_deco, min_nphotons=2))
      call bind_obs("min_dr_gj", "minimum of Delta R (jets, photons)", &
                    EvalObs_t(eval=eval_min_dR_gamj, min_njets=1, min_nphotons=1))
      call bind_obs("dphi_g1g2", "Azimuthal angle difference of di-photon", &
                    EvalObs_t(eval=eval_deltaphi_gam12, min_nphotons=2))
      call bind_obs("pi_dphi_g1g2", "pi - dphi_g1g2", &
                    EvalObs_t(eval=eval_pi_deltaphi_gam12, min_nphotons=2))
      call bind_obs("dy_g1g2", "(pseudo-)rapidity difference of di-photon", &
                    EvalObs_t(eval=eval_deltay_gam12, min_nphotons=2))
      call bind_obs("abs_dy_g1g2", "|dy_g1g2|", &
                    EvalObs_t(eval=eval_abs_deltay_gam12, min_nphotons=2))
      call bind_obs("dr_g1g2", "Delta R (g1, g2)", &
                    EvalObs_t(eval=eval_deltaR_gam12, min_nphotons=2))
      call bind_obs("costar_gg", "photon decay angles in the lab frame", &
                    EvalObs_t(eval=eval_costhst_gam12, min_nphotons=2))
      call bind_obs("abscosthetastareta", "ATLAS scattering angle: |cos(theta*_eta)|", &
                    EvalObs_t(eval=eval_abscosthst_gam12, min_nphotons=2))
      call bind_obs("phi_star", "\phi^*_\eta ", &
                    EvalObs_t(eval=eval_phi_star_gam12, min_nphotons=2))
      call bind_obs("pTt_g1g2", "pTgg component transverse to thrust axis: standard impl.", &
      EvalObs_t(eval=eval_absptt_gam12, min_nphotons=2))
      call bind_obs("ptt_g1g2", "pTgg component transverse to thrust axis: Projection impl.", &
      EvalObs_t(eval=eval_ptt_gam12, min_nphotons=2))
      call bind_obs("at_g1g2", "pTt if dphi_gg > pi/2, or pTgg if dphi_gg <= pi/2", &
      EvalObs_t(eval=eval_absat_gam12, min_nphotons=2))
      call bind_obs("Mg1g2", "invariant mass of diphoton system", &
                    EvalObs_t(eval=eval_minv_gam12, min_nphotons=2), ifac=1)
      call bind_obs("pt_g1g2", "pt of diphoton system", &
                    EvalObs_t(eval=eval_pt_gam12, min_nphotons=2), ifac=2)
      call bind_obs("r_g1g2", "ratio between the pt over mgg", &
                    EvalObs_t(eval=eval_r_gam12, min_nphotons=2))
      call bind_obs("ptavg_g1g2", "(ptg1+ptg2)/2", &
                    EvalObs_t(eval=eval_ptavg_gam12, min_nphotons=2))
      call bind_obs("ptavg_geom_g1g2", "sqrt(ptg1*ptg2)", &
                    EvalObs_t(eval=eval_ptavg_geom_gam12, min_nphotons=2))
      call bind_obs("yg1g2", "rapidity diphoton system", &
                    EvalObs_t(eval=eval_y_gam12, min_nphotons=2))
      call bind_obs("abs_yg1g2", "|yg1g2|", &
                    EvalObs_t(eval=eval_y_gam12, decorator=abs_deco, min_nphotons=2))
      call bind_obs("minv_pt_g1g2", "sqrt(m(g1g2)^2 + pt(g1g2)^2)/4", &
                    EvalObs_t(eval=eval_minv_pt_gam12, min_nphotons=2), ifac=1)
      call bind_obs("Et_g1g2", "sqrt(m(g1g2)^2 + pt(g1g2)^2)", &
                    EvalObs_t(eval=eval_Et_gam12, min_nphotons=2), ifac=1)
      call bind_obs("cos_thetaCS", "diphoton scattering angles in the Collins-Soper frame ", &
                    EvalObs_t(eval=eval_costhstCS_gam12, min_nphotons=2))
      call bind_obs("abs_cos_thetaCS", "diphoton scattering angles in the Collins-Soper frame ", &
                    EvalObs_t(eval=eval_abscosthstCS_gam12, min_nphotons=2))
      call bind_obs("ht_g1g2", "sqrt(m(g1g2)^2 + pt(g1g2)^2)+sum of pT(jets)", &
                    EvalObs_t(eval=eval_ht_g1g2, min_nphotons=2))
      !> gamma + jet specific observables
      call bind_obs("dphi_gam_j1", "azimuthal separation of leading photon & leading jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_costhetastar_gam_j1", "|costhetastar_gam_j1|", &
                    EvalObs_t(eval_mem=eval_costhstar_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("m_gam_j1", "invariant mass of the leading photon and leading jet", &
                    EvalObs_t(eval_mem=eval_minv_gamj1, imem=-1, min_njets=1, min_nphotons=1), ifac=1)
      call bind_obs("abs_y_gam_plus_y_j1", "|y_gam1_plus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("y_gam_plus_y_j1", "sum of y_gam1 and y_j1", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_y_gam_plus_y_j1", "|y_gam1_plus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      call bind_obs("y_gam_minus_y_j1", "difference of y_gam1 and y_j1", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-1, min_njets=1, min_nphotons=1))
      call bind_obs("abs_y_gam_minus_y_j1", "|y_gam1_minus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-1, decorator=abs_deco, min_njets=1, min_nphotons=1))
      !> gamma + jet specific observables
      call bind_obs("dphi_gam2_j1", "azimuthal separation of subleading photon & leading jet", &
                    EvalObs_t(eval_mem=eval_deltaphi_gamj1, imem=-2, min_njets=1, min_nphotons=2))
      call bind_obs("abs_costhetastar_gam2_j1", "|costhetastar_gam2_j1|", &
                    EvalObs_t(eval_mem=eval_costhstar_gamj1, imem=-2, decorator=abs_deco, min_njets=1, min_nphotons=2))
      call bind_obs("m_gam2_j1", "invariant mass of the subleading photon and leading jet", &
                    EvalObs_t(eval_mem=eval_minv_gamj1, imem=-2, min_njets=1, min_nphotons=2), ifac=1)
      call bind_obs("abs_y_gam2_plus_y_j1", "|y_gam2_plus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-2, decorator=abs_deco, min_njets=1, min_nphotons=2))
      call bind_obs("y_gam2_plus_y_j1", "sum of y_gam2 and y_j1", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-2, min_njets=1, min_nphotons=2))
      call bind_obs("abs_y_gam2_plus_y_j1", "|y_gam2_plus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yPlus_gamj1, imem=-2, decorator=abs_deco, min_njets=1, min_nphotons=2))
      call bind_obs("y_gam2_minus_y_j1", "difference of y_gam2 and y_j1", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-2, min_njets=1, min_nphotons=2))
      call bind_obs("abs_y_gam2_minus_y_j1", "|y_gam2_minus_y_j1|", &
                    EvalObs_t(eval_mem=eval_yMinus_gamj1, imem=-2, decorator=abs_deco, min_njets=1, min_nphotons=2))
!       call bind_obs("pt_iso_g1g2", "pt of diphoton system where photons could cluster with partons", &
!                     EvalObs_t(eval=eval_pt_iso_gam12, min_nphotons=2), ifac=2)

    case ("GGG")
      call bind_obs("nphotons", "number of (isolated) photon", &
                    EvalObs_t(eval=eval_nphotons, is_integer=.true.))
      call bind_obs("ptg1", "transverse momentum of leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-1, min_nphotons=1), ifac=2)
      call bind_obs("yg1", "(pseudo-)rapidity of leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, min_nphotons=1))
      call bind_obs("abs_yg1", "|yg1|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-1, decorator=abs_deco, min_nphotons=1))
      call bind_obs("ptg2", "transverse momentum of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-2, min_nphotons=2), ifac=2)
      call bind_obs("yg2", "(pseudo-)rapidity of sub-leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, min_nphotons=2))
      call bind_obs("abs_yg2", "|yg2|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, decorator=abs_deco, min_nphotons=2))
      call bind_obs("ptg3", "transverse momentum of sub-sub-leading photon", &
                    EvalObs_t(eval_mem=eval_pt_gam, imem=-3, min_nphotons=3), ifac=2)
      call bind_obs("yg3", "(pseudo-)rapidity of sub-sub-leading photon", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-3, min_nphotons=3))
      call bind_obs("abs_yg3", "|yg3|", &
                    EvalObs_t(eval_mem=eval_y_gam, imem=-2, decorator=abs_deco, min_nphotons=3))
      call bind_obs("min_dr_gj", "minimum of Delta R (jets, photons)", &
                    EvalObs_t(eval=eval_min_dR_gamj, min_njets=1, min_nphotons=1))

    case ("DIS1J", "SFDIS")
      ! DIS electron cuts
      call bind_obs("q2", "momentum transfer of the lepton", &
                    EvalObs_t(eval=eval_q2_dis))
      call bind_obs("q", "sqrt of q2", &
                    EvalObs_t(eval=eval_sqrtQ2_dis))
      call bind_obs("Ee", "energy of the final-state electron", &
                    EvalObs_t(eval=eval_Ee_dis))
      call bind_obs("thetae", "polar angle to beam line of the final-state electron", &
                    EvalObs_t(eval=eval_thetae_dis))
      call bind_obs("x", "Bjorken x", &
                    EvalObs_t(eval=eval_x_dis))
      call bind_obs("y", "fraction of electron energy transferred to proton", &
                    EvalObs_t(eval=eval_y_dis))
      call bind_obs("y01", "0 to 1 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=1))
      call bind_obs("y12", "1 to 2 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=2))
      call bind_obs("y23", "2 to 3 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("y34", "3 to 4 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("resrho", "resummed jet mass (rho)", &
                    EvalObs_t(eval=eval_resrho_dis))
      call bind_obs("thetae", "polar angle to beam line of the final-state electron", &
                    EvalObs_t(eval=eval_thetae_dis))

      call bind_obs("ptl", "transverse momentum of outgoing neutrino", &
                    EvalObs_t(eval=eval_dis_ptl))
      ! DIJET [FOR USE WITH NJET SELECTORS]
      call bind_obs("etavg_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_etavg_j12, min_njets=2))
      call bind_obs("et2_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_et2_j12, min_njets=2))
      call bind_obs("avgeta_j12", "(etaj1+etaj2)/2", &
                    EvalObs_t(eval=eval_avgeta_j12, min_njets=2))
      call bind_obs("m12", "dijet invariant mass", &
                    EvalObs_t(eval=eval_minv_j12, min_njets=2), ifac=1)

      call bind_obs("tau1", "1-jettiness for DIS", &
                    EvalObs_t(eval=eval_DIS_tau1))

    case ("DISWM1J", "DISWP1J", "SFDISWM", "SFDISWP")
      ! DIS electron cuts
      call bind_obs("q2", "momentum transfer of the lepton", &
                    EvalObs_t(eval=eval_q2_dis))
      call bind_obs("q", "sqrt of q2", &
                    EvalObs_t(eval=eval_sqrtQ2_dis))
      call bind_obs("Ee", "energy of the final-state electron", &
                    EvalObs_t(eval=eval_Ee_dis))
      call bind_obs("thetae", "polar angle to beam line of the final-state electron", &
                    EvalObs_t(eval=eval_thetae_dis))

      call bind_obs("x", "Bjorken x", &
                    EvalObs_t(eval=eval_x_dis))
      call bind_obs("y", "fraction of electron energy transferred to proton", &
                    EvalObs_t(eval=eval_y_dis))
      call bind_obs("y01", "0 to 1 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=1))
      call bind_obs("y12", "1 to 2 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=2))
      call bind_obs("y23", "2 to 3 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("y34", "3 to 4 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("ptl", "transverse momentum of outgoing neutrino", &
                    EvalObs_t(eval=eval_dis_ptl))
      ! DIJET [FOR USE WITH NJET SELECTORS]
      call bind_obs("etavg_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_etavg_j12, min_njets=2))
      call bind_obs("et2_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_et2_j12, min_njets=2))
      call bind_obs("avgeta_j12", "(etaj1+etaj2)/2", &
                    EvalObs_t(eval=eval_avgeta_j12, min_njets=2))
      call bind_obs("m12", "dijet invariant mass", &
                    EvalObs_t(eval=eval_minv_j12, min_njets=2), ifac=1)

    case ("DISG")
      call bind_obs("y_DISG", "electron energy fraction carried by photon", &
                    EvalObs_t(eval=eval_y_disg))
      call bind_obs("xgamma_DISG", "xgamma", &
                    EvalObs_t(eval=eval_disg_xgamma, min_njets=2))
      call bind_obs("xproton_DISG", "xproton", &
                    EvalObs_t(eval=eval_disg_xproton, min_njets=2))
      call bind_obs("cos_DISG", "cos", &
                    EvalObs_t(eval=eval_disg_cos, min_njets=2))

    case ("DIS")
      ! DIS electron cuts
      call bind_obs("thetae", "polar angle to beam line of the final-state electron", &
                    EvalObs_t(eval=eval_thetae_dis))
      call bind_obs("dscl", "sqrt((Q2+ptavg_j12**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl, min_njets=2))
      call bind_obs("dscl3j", "sqrt((Q2+ptavg_j123**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl3j, min_njets=3))
      call bind_obs("dsclj1", "sqrt((Q2+pt_jet1**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj1, min_njets=1))
      call bind_obs("dsclj2", "sqrt((Q2+pt_jet2**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj2, min_njets=2))
      call bind_obs("dsclj3", "sqrt((Q2+pt_jet3**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj3, min_njets=3))
      call bind_obs("dsclj4", "sqrt((Q2+pt_jet4**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj4, min_njets=4))
      call bind_obs("dsclj1l", "sqrt((Q2+pt_jet1_lab**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj1l, min_njets=1))
      call bind_obs("dsclj2l", "sqrt((Q2+pt_jet2_lab**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj2l, min_njets=2))
      call bind_obs("dsclj3l", "sqrt((Q2+pt_jet3_lab**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj3l, min_njets=3))
      call bind_obs("dsclj4l", "sqrt((Q2+pt_jet4_lab**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj4l, min_njets=4))
      call bind_obs("dsclZEUS", "sqrt(Q2+etavg_j12**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS, min_njets=2))
      call bind_obs("dsclZEUS2", "sqrt(Q2+etavg_jall**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS2, min_njets=1))
      call bind_obs("q2", "momentum transfer of the lepton", &
                    EvalObs_t(eval=eval_q2_dis))
      call bind_obs("W2", "(photon-proton energy)^2", &
                    EvalObs_t(eval=eval_W2_dis))
      call bind_obs("W", "photon-proton energy", &
                    EvalObs_t(eval=eval_W_dis))
      call bind_obs("disnu", "energy of photon in proton restframe", &
                    EvalObs_t(eval=eval_nu_dis))
      call bind_obs("visbyW", "E_visible/W", &
                    EvalObs_t(eval=eval_dis_visbyW))
      call bind_obs("cosgammah", "cos(angle) of scattered quark ", &
                    EvalObs_t(eval=eval_cosgammah_dis))
      call bind_obs("q", "sqrt of q2", &
                    EvalObs_t(eval=eval_sqrtQ2_dis))
      call bind_obs("Ee", "energy of the final-state electron", &
                    EvalObs_t(eval=eval_Ee_dis))
      call bind_obs("x", "Bjorken x", &
                    EvalObs_t(eval=eval_x_dis))
      call bind_obs("y", "fraction of electron energy transferred to proton", &
                    EvalObs_t(eval=eval_y_dis))
      call bind_obs("xi2", "xbj*(1d0+m12**2/q2)", &
                    EvalObs_t(eval=eval_dis_xi2, min_njets=2))
      call bind_obs("dis_thrust", "Thrust in DIS w.r.t. boson axis (tau=1-T)", &
                    EvalObs_t(eval=eval_dis_thrust))
      call bind_obs("dis_thrust_c", "Thrust tau w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=1))
      call bind_obs("dis_JB", "Jet Broadening w.r.t. boson axis in DIS", &
                    EvalObs_t(eval=eval_dis_JB))
      call bind_obs("dis_JB_c", "Jet Broadening w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=2))
      call bind_obs("dis_JM2", "Squared Jet Mass in DIS", &
                    EvalObs_t(eval=eval_dis_JM2))
      call bind_obs("dis_C", "C-parameter in DIS", &
                    EvalObs_t(eval=eval_dis_C))
      call bind_obs("eta1_0", "HERA eta of leading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j1_0))
      call bind_obs("eta2_0", "HERA eta of subleading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j2_0))
      call bind_obs("lgxi2", "log(xi2)", &
                    EvalObs_t(eval=eval_dis_Lgxi2, min_njets=2))
      call bind_obs("etas", "abs(eta_j1-eta_j2)/2", &
                    EvalObs_t(eval=eval_dis_etas, min_njets=2))
      call bind_obs("xi3", "xbj*(1d0+m123**2/q2)", &
                    EvalObs_t(eval=eval_dis_xi3, min_njets=3))
      call bind_obs("ptavg_12", "(ptj1+ptj2)/2", &
                    EvalObs_t(eval=eval_ptavg_j12, min_njets=2))
      call bind_obs("ptavg_123", "(ptj1+ptj2+ptj3)/3", &
                    EvalObs_t(eval=eval_ptavg_j123, min_njets=3))
      call bind_obs("ptavg_all", "(ptj1+ptj2+...)/njets", &
                    EvalObs_t(eval=eval_ptavg_jall, min_njets=1))
      call bind_obs("etavg_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_etavg_j12, min_njets=2))
      call bind_obs("et2_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_et2_j12, min_njets=2))
      call bind_obs("etavg_123", "(etj1+etj2+etj3)/3", &
                    EvalObs_t(eval=eval_etavg_j123, min_njets=3))
      call bind_obs("etavg_all", "(etj1+etj2+...)/njets", &
                    EvalObs_t(eval=eval_etavg_jall, min_njets=1))
      call bind_obs("hera_etj1", "Et of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj1, min_njets=1))
      call bind_obs("hera_etj2", "Et of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj2, min_njets=2))
      call bind_obs("hera_etj3", "Et of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj3, min_njets=3))
      call bind_obs("hera_etj4", "Et of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj4, min_njets=4))
      call bind_obs("hera_et2j1", "Et2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j1, min_njets=1))
      call bind_obs("hera_et2j2", "Et2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j2, min_njets=2))
      call bind_obs("hera_et2j3", "Et2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j3, min_njets=3))
      call bind_obs("hera_et2j4", "Et2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j4, min_njets=4))
      call bind_obs("hera_etaj1", "Eta of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj1, min_njets=1))
      call bind_obs("hera_etaj2", "Eta of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj2, min_njets=2))
      call bind_obs("hera_etaj3", "Eta of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj3, min_njets=3))
      call bind_obs("hera_etaj4", "Eta of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj4, min_njets=4))
      call bind_obs("hera_et2byQ2j1", "Et^2/Q2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j1))
      call bind_obs("hera_et2byQ2j2", "Et^2/Q2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j2))
      call bind_obs("hera_et2byQ2j3", "Et^2/Q2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j3))
      call bind_obs("hera_et2byQ2j4", "Et^2/Q2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j4))
      call bind_obs("hera_FJ_reweight", "Q2+sumET2", &
                    EvalObs_t(eval=eval_DIS_hera_FJ_reweight))
      call bind_obs("hera_deta1", "etaj1-etaj2 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta1, min_njets=3))
      call bind_obs("hera_deta2", "etafj-etaj1 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta2, min_njets=3))
      call bind_obs("hera_etaavg_12", "0.5*(etaj1+etaj2)", &
                    EvalObs_t(eval=eval_dis_hera_etaavg_j12, min_njets=2))
      call bind_obs("hera_deltaeta_12", "|etaj1-etaj2|", &
                    EvalObs_t(eval=eval_dis_hera_deltaeta_j12, min_njets=2))
      call bind_obs("gammap_eta1", "eta1 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j1, min_njets=1))
      call bind_obs("gammap_eta2", "eta2 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j2, min_njets=2))
      call bind_obs("delta_phis", "phi1-phi2 in proton gamma frame", &
                    EvalObs_t(eval=eval_dis_delta_phis, min_njets=2))
      call bind_obs("xgamma", "(E-pZ)_LabJets / (2*Ee*y)", &
                    EvalObs_t(eval=eval_dis_hera_xgamma))

      call bind_obs("y01", "0 to 1 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=1))
      call bind_obs("y12", "1 to 2 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=2))
      call bind_obs("y23", "2 to 3 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("y34", "3 to 4 transition rate (JADE) *before* jet cuts", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("ptl", "transverse momentum of lepton", &
                    EvalObs_t(eval=eval_dis_ptl))

      call bind_obs("tau1", "1-jettiness for DIS", &
                    EvalObs_t(eval=eval_DIS_tau1))

    case ("DISWM")
      ! DIS electron cuts
      call bind_obs("dscl", "sqrt((Q2+ptavg_j12**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl, min_njets=2))
      call bind_obs("dscl3j", "sqrt((Q2+ptavg_j123**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl3j, min_njets=3))
      call bind_obs("dsclj1", "sqrt((Q2+pt_jet1**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj1, min_njets=1))
      call bind_obs("dsclj2", "sqrt((Q2+pt_jet2**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj2, min_njets=2))
      call bind_obs("dsclj3", "sqrt((Q2+pt_jet3**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj3, min_njets=3))
      call bind_obs("dsclj4", "sqrt((Q2+pt_jet4**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj4, min_njets=4))
      call bind_obs("dsclZEUS", "sqrt(Q2+etavg_j12**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS, min_njets=2))
      call bind_obs("dsclZEUS2", "sqrt(Q2+etavg_jall**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS2, min_njets=1))
      call bind_obs("q2", "momentum transfer of the lepton", &
                    EvalObs_t(eval=eval_q2_dis))
      call bind_obs("W2", "(photon-proton energy)^2", &
                    EvalObs_t(eval=eval_W2_dis))
      call bind_obs("W", "photon-proton energy", &
                    EvalObs_t(eval=eval_W_dis))
      call bind_obs("disnu", "energy of photon in proton restframe", &
                    EvalObs_t(eval=eval_nu_dis))
      call bind_obs("visbyW", "E_visible/W", &
                    EvalObs_t(eval=eval_dis_visbyW))
      call bind_obs("cosgammah", "cos(angle) of scattered quark ", &
                    EvalObs_t(eval=eval_cosgammah_dis))
      call bind_obs("q", "sqrt of q2", &
                    EvalObs_t(eval=eval_sqrtQ2_dis))
      call bind_obs("Ee", "energy of the final-state electron", &
                    EvalObs_t(eval=eval_Ee_dis))
      call bind_obs("x", "Bjorken x", &
                    EvalObs_t(eval=eval_x_dis))
      call bind_obs("y", "fraction of electron energy transferred to proton", &
                    EvalObs_t(eval=eval_y_dis))
      call bind_obs("xi2", "xbj*(1d0+m12**2/q2)", &
                    EvalObs_t(eval=eval_dis_xi2, min_njets=2))
      call bind_obs("dis_thrust", "Thrust in DIS w.r.t. boson axis (tau=1-T)", &
                    EvalObs_t(eval=eval_dis_thrust))
      call bind_obs("dis_thrust_c", "Thrust tau w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=1))
      call bind_obs("dis_JB", "Jet Broadening w.r.t. boson axis in DIS", &
                    EvalObs_t(eval=eval_dis_JB))
      call bind_obs("dis_JB_c", "Jet Broadening w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=2))
      call bind_obs("dis_JM2", "Squared Jet Mass in DIS", &
                    EvalObs_t(eval=eval_dis_JM2))
      call bind_obs("dis_C", "C-parameter in DIS", &
                    EvalObs_t(eval=eval_dis_C))
      call bind_obs("eta1_0", "HERA eta of leading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j1_0))
      call bind_obs("eta2_0", "HERA eta of subleading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j2_0))
      call bind_obs("lgxi2", "log(xi2)", &
                    EvalObs_t(eval=eval_dis_Lgxi2, min_njets=2))
      call bind_obs("etas", "abs(eta_j1-eta_j2)/2", &
                    EvalObs_t(eval=eval_dis_etas, min_njets=2))
      call bind_obs("xi3", "xi3", &
                    EvalObs_t(eval=eval_dis_xi3, min_njets=3))
      call bind_obs("ptavg_12", "(ptj1+ptj2)/2", &
                    EvalObs_t(eval=eval_ptavg_j12, min_njets=2))
      call bind_obs("ptavg_123", "(ptj1+ptj2+ptj3)/3", &
                    EvalObs_t(eval=eval_ptavg_j123, min_njets=3))
      call bind_obs("ptavg_all", "(ptj1+ptj2+...)/njets", &
                    EvalObs_t(eval=eval_ptavg_jall, min_njets=1))
      call bind_obs("etavg_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_etavg_j12, min_njets=2))
      call bind_obs("et2_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_et2_j12, min_njets=2))
      call bind_obs("etavg_123", "(etj1+etj2+etj3)/3", &
                    EvalObs_t(eval=eval_etavg_j123, min_njets=3))
      call bind_obs("etavg_all", "(etj1+etj2+...)/njets", &
                    EvalObs_t(eval=eval_etavg_jall, min_njets=1))
      call bind_obs("hera_etj1", "Et of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj1, min_njets=1))
      call bind_obs("hera_etj2", "Et of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj2, min_njets=2))
      call bind_obs("hera_etj3", "Et of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj3, min_njets=3))
      call bind_obs("hera_etj4", "Et of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj4, min_njets=4))
      call bind_obs("hera_et2j1", "Et2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j1, min_njets=1))
      call bind_obs("hera_et2j2", "Et2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j2, min_njets=2))
      call bind_obs("hera_et2j3", "Et2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j3, min_njets=3))
      call bind_obs("hera_et2j4", "Et2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j4, min_njets=4))
      call bind_obs("hera_etaj1", "Eta of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj1, min_njets=1))
      call bind_obs("hera_etaj2", "Eta of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj2, min_njets=2))
      call bind_obs("hera_etaj3", "Eta of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj3, min_njets=3))
      call bind_obs("hera_etaj4", "Eta of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj4, min_njets=4))
      call bind_obs("hera_et2byQ2j1", "Et^2/Q2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j1))
      call bind_obs("hera_et2byQ2j2", "Et^2/Q2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j2))
      call bind_obs("hera_et2byQ2j3", "Et^2/Q2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j3))
      call bind_obs("hera_et2byQ2j4", "Et^2/Q2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j4))
      call bind_obs("hera_FJ_reweight", "Q2+sumET2", &
                    EvalObs_t(eval=eval_DIS_hera_FJ_reweight))
      call bind_obs("hera_deta1", "etaj1-etaj2 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta1, min_njets=3))
      call bind_obs("hera_deta2", "etafj-etaj1 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta2, min_njets=3))
      call bind_obs("hera_etaavg_12", "0.5*(etaj1+etaj2)", &
                    EvalObs_t(eval=eval_dis_hera_etaavg_j12, min_njets=2))
      call bind_obs("hera_deltaeta_12", "|etaj1-etaj2|", &
                    EvalObs_t(eval=eval_dis_hera_deltaeta_j12, min_njets=2))
      call bind_obs("gammap_eta1", "eta1 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j1, min_njets=1))
      call bind_obs("gammap_eta2", "eta2 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j2, min_njets=2))
      call bind_obs("delta_phis", "phi1-phi2 in proton gamma frame", &
                    EvalObs_t(eval=eval_dis_delta_phis, min_njets=2))
      call bind_obs("xgamma", "(E-pZ)_LabJets / (2*Ee*y)", &
                    EvalObs_t(eval=eval_dis_hera_xgamma))
      call bind_obs("y01", "0 to 1 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=1))
      call bind_obs("y12", "1 to 2 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=2))
      call bind_obs("y23", "2 to 3 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("y34", "3 to 4 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("ptl", "transverse momentum of outgoing neutrino", &
                    EvalObs_t(eval=eval_dis_ptl))

    case ("DISWP")
      ! DIS electron cuts
      call bind_obs("dscl", "sqrt((Q2+ptavg_j12**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl, min_njets=2))
      call bind_obs("dscl3j", "sqrt((Q2+ptavg_j123**2)/2)", &
                    EvalObs_t(eval=eval_dis_dscl3j, min_njets=3))
      call bind_obs("dsclj1", "sqrt((Q2+pt_jet1**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj1, min_njets=1))
      call bind_obs("dsclj2", "sqrt((Q2+pt_jet2**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj2, min_njets=2))
      call bind_obs("dsclj3", "sqrt((Q2+pt_jet3**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj3, min_njets=3))
      call bind_obs("dsclj4", "sqrt((Q2+pt_jet4**2)/2)", &
                    EvalObs_t(eval=eval_dis_dsclj4, min_njets=4))
      call bind_obs("dsclZEUS", "sqrt(Q2+etavg_j12**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS, min_njets=2))
      call bind_obs("dsclZEUS2", "sqrt(Q2+etavg_jall**2)", &
                    EvalObs_t(eval=eval_dis_dsclZEUS2, min_njets=1))
      call bind_obs("q2", "momentum transfer of the lepton", &
                    EvalObs_t(eval=eval_q2_dis))
      call bind_obs("W2", "(photon-proton energy)^2", &
                    EvalObs_t(eval=eval_W2_dis))
      call bind_obs("W", "photon-proton energy", &
                    EvalObs_t(eval=eval_W_dis))
      call bind_obs("disnu", "energy of photon in proton restframe", &
                    EvalObs_t(eval=eval_nu_dis))
      call bind_obs("visbyW", "E_visible/W", &
                    EvalObs_t(eval=eval_dis_visbyW))
      call bind_obs("cosgammah", "cos(angle) of scattered quark ", &
                    EvalObs_t(eval=eval_cosgammah_dis))
      call bind_obs("q", "sqrt of q2", &
                    EvalObs_t(eval=eval_sqrtQ2_dis))
      call bind_obs("Ee", "energy of the final-state electron", &
                    EvalObs_t(eval=eval_Ee_dis))
      call bind_obs("x", "Bjorken x", &
                    EvalObs_t(eval=eval_x_dis))
      call bind_obs("y", "fraction of electron energy transferred to proton", &
                    EvalObs_t(eval=eval_y_dis))
      call bind_obs("xi2", "xbj*(1d0+m12**2/q2)", &
                    EvalObs_t(eval=eval_dis_xi2, min_njets=2))
      call bind_obs("dis_thrust", "Thrust in DIS w.r.t. boson axis (tau=1-T)", &
                    EvalObs_t(eval=eval_dis_thrust))
      call bind_obs("dis_thrust_c", "Thrust tau w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=1))
      call bind_obs("dis_JB", "Jet Broadening w.r.t. boson axis in DIS", &
                    EvalObs_t(eval=eval_dis_JB))
      call bind_obs("dis_JB_c", "Jet Broadening w.r.t. thrust axis in DIS", &
                    EvalObs_t(eval_mem=eval_DIS_Taxis, imem=2))
      call bind_obs("dis_JM2", "Squared Jet Mass in DIS", &
                    EvalObs_t(eval=eval_dis_JM2))
      call bind_obs("dis_C", "C-parameter in DIS", &
                    EvalObs_t(eval=eval_dis_C))
      call bind_obs("eta1_0", "HERA eta of leading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j1_0))
      call bind_obs("eta2_0", "HERA eta of subleading jet in Breit frame before cuts", &
                    EvalObs_t(eval=eval_dis_hera_eta_j2_0))
      call bind_obs("lgxi2", "log(xi2)", &
                    EvalObs_t(eval=eval_dis_Lgxi2, min_njets=2))
      call bind_obs("etas", "abs(eta_j1-eta_j2)/2", &
                    EvalObs_t(eval=eval_dis_etas, min_njets=2))
      call bind_obs("xi3", "abs(eta_j1-eta_j2)/2", &
                    EvalObs_t(eval=eval_dis_xi3, min_njets=3))
      call bind_obs("ptavg_12", "(ptj1+ptj2)/2", &
                    EvalObs_t(eval=eval_ptavg_j12, min_njets=2))
      call bind_obs("ptavg_123", "(ptj1+ptj2+ptj3)/3", &
                    EvalObs_t(eval=eval_ptavg_j123, min_njets=3))
      call bind_obs("ptavg_all", "(ptj1+ptj2+...)/njets", &
                    EvalObs_t(eval=eval_ptavg_jall, min_njets=1))
      call bind_obs("etavg_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_etavg_j12, min_njets=2))
      call bind_obs("et2_12", "(etj1+etj2)/2", &
                    EvalObs_t(eval=eval_et2_j12, min_njets=2))
      call bind_obs("etavg_123", "(etj1+etj2+etj3)/3", &
                    EvalObs_t(eval=eval_etavg_j123, min_njets=3))
      call bind_obs("etavg_all", "(etj1+etj2+...)/njets", &
                    EvalObs_t(eval=eval_etavg_jall, min_njets=1))
      call bind_obs("hera_etj1", "Et of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj1, min_njets=1))
      call bind_obs("hera_etj2", "Et of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj2, min_njets=2))
      call bind_obs("hera_etj3", "Et of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj3, min_njets=3))
      call bind_obs("hera_etj4", "Et of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etj4, min_njets=4))
      call bind_obs("hera_et2j1", "Et2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j1, min_njets=1))
      call bind_obs("hera_et2j2", "Et2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j2, min_njets=2))
      call bind_obs("hera_et2j3", "Et2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j3, min_njets=3))
      call bind_obs("hera_et2j4", "Et2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2j4, min_njets=4))
      call bind_obs("hera_etaj1", "Eta of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj1, min_njets=1))
      call bind_obs("hera_etaj2", "Eta of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj2, min_njets=2))
      call bind_obs("hera_etaj3", "Eta of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj3, min_njets=3))
      call bind_obs("hera_etaj4", "Eta of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_etaj4, min_njets=4))
      call bind_obs("hera_et2byQ2j1", "Et^2/Q2 of leading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j1))
      call bind_obs("hera_et2byQ2j2", "Et^2/Q2 of subleading jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j2))
      call bind_obs("hera_et2byQ2j3", "Et^2/Q2 of 3rd jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j3))
      call bind_obs("hera_et2byQ2j4", "Et^2/Q2 of 4th jet in hera frame", &
                    EvalObs_t(eval=eval_dis_hera_et2byq2j4))
      call bind_obs("hera_FJ_reweight", "Q2+sumET2", &
                    EvalObs_t(eval=eval_DIS_hera_FJ_reweight))
      call bind_obs("hera_deta1", "etaj1-etaj2 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta1, min_njets=3))
      call bind_obs("hera_deta2", "etafj-etaj1 for +2 jets", &
                    EvalObs_t(eval=eval_dis_hera_deta2, min_njets=3))
      call bind_obs("hera_etaavg_12", "0.5*(etaj1+etaj2)", &
                    EvalObs_t(eval=eval_dis_hera_etaavg_j12, min_njets=2))
      call bind_obs("hera_deltaeta_12", "|etaj1-etaj2|", &
                    EvalObs_t(eval=eval_dis_hera_deltaeta_j12, min_njets=2))
      call bind_obs("gammap_eta1", "eta1 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j1, min_njets=1))
      call bind_obs("gammap_eta2", "eta2 in gamma proton frame", &
                    EvalObs_t(eval=eval_gammap_eta_j2, min_njets=2))
      call bind_obs("delta_phis", "phi1-phi2 in proton gamma frame", &
                    EvalObs_t(eval=eval_dis_delta_phis, min_njets=2))
      call bind_obs("xgamma", "(E-pZ)_LabJets / (2*Ee*y)", &
                    EvalObs_t(eval=eval_dis_hera_xgamma))
      call bind_obs("y01", "0 to 1 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=1))
      call bind_obs("y12", "1 to 2 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=2))
      call bind_obs("y23", "2 to 3 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("y34", "3 to 4 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("ptl", "transverse momentum of outgoing neutrino", &
                    EvalObs_t(eval=eval_dis_ptl))

      !> ep em observables
    case ("EPEM", "EPEM_H", "EPEMJJ", "EPEMJJ_H", "EPEMINC")
      call bind_obs("C", "C parameter", &
                    EvalObs_t(eval=eval_epem_C))
      call bind_obs("D", "D parameter", &
                    EvalObs_t(eval=eval_epem_D))
      call bind_obs("WJB", "Wide jet broadening", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=4))
      call bind_obs("TJB", "total jet broadening", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=5))
      call bind_obs("T", "Thrust", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=1))
      call bind_obs("1mT", "1-T", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=2))
      call bind_obs("cosT", "angle between thrust and z axis", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=3))
      call bind_obs("angP", "angular projection for thrust", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=8))
      call bind_obs("HJM", "Heavy jet mass", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=6))
      call bind_obs("SHM", "Sum hemisphere masses", &
                    EvalObs_t(eval_mem=eval_Taxis, imem=7))
      call bind_obs("EFPG3B", "Energy flow polynomial G3b (triangle, sqrt edges)", &
                    EvalObs_t(eval=eval_epem_EFPG3b, min_npar=5))
      call bind_obs("EFPK32", "Energy flow polynomial K3^(2) (double-edge triangle)", &
                    EvalObs_t(eval=eval_epem_EFPK32, min_npar=5))
      call bind_obs("EFPC5", "Energy flow polynomial C5 (5-cycle)", &
                    EvalObs_t(eval=eval_epem_EFPC5, min_npar=5))
      call bind_obs("EFPTHETA122", "Energy flow polynomial theta_{1,2,2} (theta graph)", &
                    EvalObs_t(eval=eval_epem_EFPTHETA122, min_npar=5))
      call bind_obs("y45", "5 to 4 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=5))
      call bind_obs("y34", "4 to 3 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=4))
      call bind_obs("y23", "2 to 3 transition rate (JADE)", &
                    EvalObs_t(eval_mem=eval_jet_ytrans, imem=3))
      call bind_obs("cost_ej1", "costh between leading jet and electron direction", &
                    EvalObs_t(eval=eval_epem_costh_ej1))
      call bind_obs("costh_en3j", "costh bewteen 3 jet plane and electron direction", &
                    EvalObs_t(eval=eval_epem_costh_en3j))
      call bind_obs("orchi", "angle between 3 jet and e^- - leading jet plane", &
                    EvalObs_t(eval=eval_epem_chi))
      if(isActive_flav())then
        ! flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        ! end flavour-tagging
      endif

      ! parton lambda:the three eigenvalues of sphericity tensor( the theta matrix)

      call bind_obs("lambda1", "parton lambda1", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=1))
      call bind_obs("lambda2", "parton lambda2", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=2))
      call bind_obs("lambda3", "parton lambda3", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=3))
      call bind_obs("Sph", "parton Spherity", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=4))
      call bind_obs("Al", "parton aplanarity", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=5))
      call bind_obs("P", "parton planarity", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=6))
      call bind_obs("lambdaC", "parton C parameter lambda", &
                    EvalObs_t(eval_mem=eval_partonlambda, imem=7))


      ! ridge of parton : long-pseudorapidity-range correlation between partonn1 and partonn2
      call bind_obs("dEta12", "delta pseudorapidity between parton1 & parton2 ", &
                    EvalObs_t(eval=eval_epem_dEta12, min_npar=5))
      call bind_obs("dPhi12", "azimuthal angle between parton1 & parton2", &
                    EvalObs_t(eval=eval_epem_dPhi12, min_npar=5))
      call bind_obs("dEta13", "delta pseudorapidity between parton1 & parton3", &
                    EvalObs_t(eval=eval_epem_dEta13, min_npar=5))
      call bind_obs("dPhi13", "azimuthal angle between parton1 & parton3", &
                    EvalObs_t(eval=eval_epem_dPhi13, min_npar=5))
      call bind_obs("dEta23", "delta pseudorapidity between parton2 & parton3", &
                    EvalObs_t(eval=eval_epem_dEta23, min_npar=5))
      call bind_obs("dPhi23", "azimuthal angle between parton2 & parton3", &
                    EvalObs_t(eval=eval_epem_dPhi23, min_npar=5))

      call bind_obs("dEta14", "delta pseudorapidity between parton1 & parton4", &
                    EvalObs_t(eval=eval_epem_dEta14, min_npar=6))
      call bind_obs("dPhi14", "azimuthal angle between parton1 & parton4", &
                    EvalObs_t(eval=eval_epem_dPhi14, min_npar=6))
      call bind_obs("dEta24", "delta pseudorapidity between parton2 & parton4", &
                    EvalObs_t(eval=eval_epem_dEta24, min_npar=6))
      call bind_obs("dPhi24", "azimuthal angle between parton2 & parton4", &
                    EvalObs_t(eval=eval_epem_dPhi24, min_npar=6))
      call bind_obs("dEta34", "delta pseudorapidity between parton3 & parton4", &
                    EvalObs_t(eval=eval_epem_dEta34, min_npar=6))
      call bind_obs("dPhi34", "azimuthal angle between parton3 & parton4", &
                    EvalObs_t(eval=eval_epem_dPhi34, min_npar=6))

      call bind_obs("dEta15", "delta pseudorapidity between parton1 & parton5", &
                    EvalObs_t(eval=eval_epem_dEta15, min_npar=7))
      call bind_obs("dPhi15", "azimuthal angle between parton1 & parton5", &
                    EvalObs_t(eval=eval_epem_dPhi15, min_npar=7))
      call bind_obs("dEta25", "delta pseudorapidity between parton2 & parton5", &
                    EvalObs_t(eval=eval_epem_dEta25, min_npar=7))
      call bind_obs("dPhi25", "azimuthal angle between parton2 & parton5", &
                    EvalObs_t(eval=eval_epem_dPhi25, min_npar=7))
      call bind_obs("dEta35", "delta pseudorapidity between parton3 & parton5", &
                    EvalObs_t(eval=eval_epem_dEta35,  min_npar=7))
      call bind_obs("dPhi35", "azimuthal angle between parton3 & parton5", &
                    EvalObs_t(eval=eval_epem_dPhi35, min_npar=7))
      call bind_obs("dEta45", "delta pseudorapidity between parton4 & parton5", &
                    EvalObs_t(eval=eval_epem_dEta45, min_npar=7))
      call bind_obs("dPhi45", "azimuthal angle between parton4 & parton5", &
                    EvalObs_t(eval=eval_epem_dPhi45, min_npar=7))

      call bind_obs("jC", "jet C parameter ", &
                    EvalObs_t(eval=eval_epem_jC))

      ! jet lambda:the three eigenvalues of sphericity tensor( the thetajet matrix)

      call bind_obs("jlambda1", "jet lambda1", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=1))
      call bind_obs("jlambda2", "jet lambda2", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=2))
      call bind_obs("jlambda3", "jet lambda3", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=3))
      call bind_obs("jSph", "jet Spherity", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=4))
      call bind_obs("jAl", "jet aplanarity", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=5))
      call bind_obs("jP", "jet planarity", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=6))
      call bind_obs("jlambdaC", "jet C parameter lambda", &
                    EvalObs_t(eval_mem=eval_jetlambda, imem=7))
      ! Sphericity axis for jets
      call bind_obs("jaxis1", "jet axis1", &
                    EvalObs_t(eval_mem=eval_jetaxis, imem=1))
      call bind_obs("jaxis2", "jet axis2", &
                    EvalObs_t(eval_mem=eval_jetaxis, imem=2))
      call bind_obs("jaxis3", "jet axis3", &
                    EvalObs_t(eval_mem=eval_jetaxis, imem=3))

      ! Sphericity axis for parton
      call bind_obs("axis1", "parton axis1", &
                    EvalObs_t(eval_mem=eval_partonaxis, imem=1))
      call bind_obs("axis2", "parton axis2", &
                    EvalObs_t(eval_mem=eval_partonaxis, imem=2))
      call bind_obs("axis3", "parton axis3", &
                    EvalObs_t(eval_mem=eval_partonaxis, imem=3))


      ! ridge of jets : long-pseudorapidity-range correlation between jetn1 and jetn2

      call bind_obs("jdEta12", "delta pseudorapidity between j1 & j2", &
                    EvalObs_t(eval=eval_epem_jdEta12, min_njets=3))
      call bind_obs("jdPhi12", "azimuthal angle between j1 & j2", &
                    EvalObs_t(eval=eval_epem_jdPhi12, min_njets=3))
      call bind_obs("jdEta13", "delta pseudorapidity between j1 & j3", &
                    EvalObs_t(eval=eval_epem_jdEta13, min_njets=3))
      call bind_obs("jdPhi13", "azimuthal angle between j1 & j3", &
                    EvalObs_t(eval=eval_epem_jdPhi13, min_njets=3))
      call bind_obs("jdEta23", "delta pseudorapidity between j2 & j3", &
                    EvalObs_t(eval=eval_epem_jdEta23, min_njets=3))
      call bind_obs("jdPhi23", "azimuthal angle between j2 & j3", &
                    EvalObs_t(eval=eval_epem_jdPhi23, min_njets=3))

      call bind_obs("jdEta14", "delta pseudorapidity between j1 & j4", &
                    EvalObs_t(eval=eval_epem_jdEta14, min_njets=4))
      call bind_obs("jdPhi14", "azimuthal angle between j1 & j4", &
                    EvalObs_t(eval=eval_epem_jdPhi14, min_njets=4))
      call bind_obs("jdEta24", "delta pseudorapidity between j2 & j4", &
                    EvalObs_t(eval=eval_epem_jdEta24, min_njets=4))
      call bind_obs("jdPhi24", "azimuthal angle between j2 & j4", &
                    EvalObs_t(eval=eval_epem_jdPhi24, min_njets=4))
      call bind_obs("jdEta34", "delta pseudorapidity between j3 & j4", &
                    EvalObs_t(eval=eval_epem_jdEta34, min_njets=4))
      call bind_obs("jdPhi34", "azimuthal angle between j3 & j4", &
                    EvalObs_t(eval=eval_epem_jdPhi34, min_njets=4))

      call bind_obs("jdEta15", "delta pseudorapidity between j1 & j5", &
                    EvalObs_t(eval=eval_epem_jdEta15, min_njets=5))
      call bind_obs("jdPhi15", "azimuthal angle between j1 & j5", &
                    EvalObs_t(eval=eval_epem_jdPhi15, min_njets=5))
      call bind_obs("jdEta25", "delta pseudorapidity between j2 & j5", &
                    EvalObs_t(eval=eval_epem_jdEta25, min_njets=5))
      call bind_obs("jdPhi25", "azimuthal angle between j2 & j5", &
                    EvalObs_t(eval=eval_epem_jdPhi25, min_njets=5))
      call bind_obs("jdEta35", "delta pseudorapidity between j3 & j5", &
                    EvalObs_t(eval=eval_epem_jdEta35, min_njets=5))
      call bind_obs("jdPhi35", "azimuthal angle between j3 & j5", &
                    EvalObs_t(eval=eval_epem_jdPhi35, min_njets=5))
      call bind_obs("jdEta45", "delta pseudorapidity between j4 & j5", &
                    EvalObs_t(eval=eval_epem_jdEta45, min_njets=5))
      call bind_obs("jdPhi45", "azimuthal angle between j4 & j5", &
                    EvalObs_t(eval=eval_epem_jdPhi45, min_njets=5))

    case ("1JET", "2JET", "1JETFC", "2JETFC", "3JET", "4JET", "5JET")
      !> multi jet observables
      call bind_obs("ptavg_j12", "(ptj1+ptj2)/2", &
                    EvalObs_t(eval=eval_ptavg_j12, min_njets=2))
      call bind_obs("ptavg_all", "(ptj1+ptj2+...)/njets", &
                    EvalObs_t(eval=eval_ptavg_jall, min_njets=1))
      call bind_obs("ptavg_geom_j12", "sqrt(ptj1*ptj2)", &
                    EvalObs_t(eval=eval_ptavg_geom_j12, min_njets=2))
      call bind_obs("ptavg_geom_all", "(ptj1*ptj2*...)^{1/njets}", &
                    EvalObs_t(eval=eval_ptavg_geom_jall, min_njets=1))

      call bind_obs("yb_part", "technical parton-level observable", &
                    EvalObs_t(eval=eval_yb_part))
      call bind_obs("ys_part", "technical parton-level observable", &
                    EvalObs_t(eval=eval_ys_part))

      ! call bind_obs("TEEC_phi", "transverse energy-energy correlation: phi", &
      !               id_TEEC_phi)
      ! call bind_obs("TEEC_tau", "transverse energy-energy correlation: tau", &
      !               id_TEEC_tau)
      ! call bind_obs("TEEC_cos_phi", "transverse energy-energy correlation: cos(phi)", &
      !               id_TEEC_cos_phi)
      ! call bind_obs("TEEC_rwgt", "reweighting function for TEEC as regulator", &
      !               EvalObs_t(eval=eval_TEEC_rwgt, min_njets=2))

    case ("ZH", "ZHJ", "ZHJJ", "ZHJJJ")
      ! leptons from the Z -> lp lm decay
      call bind_obs("ptlm", "transverse momentum of l-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ylm", "(pseudo-)rapidity of l-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ylm", "|ylm|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptlp", "transverse momentum of l+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ylp", "(pseudo-)rapidity of l+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ylp", "|ylp|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      ! Z-boson
      call bind_obs("ptz", "transverse momentum of Z ", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yz", "rapidity of Z", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yz", "|yz|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaz", "pseudo-rapidity of Z", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etaz", "|etaz|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("mz", "invariant mass of Z", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("etz", "transverse energy: sqrt(mz^2 + ptz^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      ! Higgs boson
      call bind_obs("pth", "transverse momentum of Higgs", &
                    EvalObs_t(eval=eval_pth_VH))
      ! ZH system
      call bind_obs("ptzh", "transverse momentum of ZH system", &
                    EvalObs_t(eval=eval_pt_VH), ifac=2)
      call bind_obs("mzh", "invariant mass of ZH system", &
                    EvalObs_t(eval=eval_minv_VH), ifac=1)

      if(isActive_flav())then
      ! general flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)

        ! flavour-tagging related
        call bind_obs("mbb", "invariant mass of two b-tagged jets closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_minv_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("ptbb", &
                      "transverse momentum of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_pt_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("dRbb", &
                      "angular separation of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_dR_jj, min_flav_njets=2, &
                      needs_flav=.true.))
      !end flavour-tagging
      endif

    case ("WPH", "WPHJ", "WPHJJ", "WPHJJJ")
      ! leptons from the W+ -> l+ nu decay
      call bind_obs("ptnu", "transverse momentum of nu", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ynu", "(pseudo-)rapidity of nu", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ynu", "|ynu|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptlp", "transverse momentum of l+", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ylp", "(pseudo-)rapidity of l+", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ylp", "|ylp|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      ! W-boson
      call bind_obs("ptw", "transverse momentum of W+", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yw", "rapidity of W+", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yw", "|yw+|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaw", "pseudo-rapidity of W+", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etaw", "|etaw+|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("mw", "invariant mass of W+", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("etw", "transverse energy: sqrt(mw^2 + ptw^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)
      ! Higgs boson
      call bind_obs("pth", "transverse momentum of Higgs", &
                    EvalObs_t(eval=eval_pth_VH))
      ! WH system
      call bind_obs("ptwh", "transverse momentum of WH system", &
                    EvalObs_t(eval=eval_pt_VH), ifac=2)
      call bind_obs("mwh", "invariant mass of WH system", &
                    EvalObs_t(eval=eval_minv_VH), ifac=1)

      if(isActive_flav())then
      ! general flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)

        ! flavour-tagging related
        call bind_obs("mbb", "invariant mass of two b-tagged jets closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_minv_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("ptbb", &
                      "transverse momentum of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_pt_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("dRbb", &
                      "angular separation of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_dR_jj, min_flav_njets=2, &
                      needs_flav=.true.))
      ! end flavour-tagging
      endif

    case ("WMH", "WMHJ", "WMHJJ", "WMHJJJ")
      ! leptons from the W- -> nub lm decay
      call bind_obs("ptlm", "transverse momentum of l-", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-1), ifac=2)
      call bind_obs("ylm", "(pseudo-)rapidity of l-", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1))
      call bind_obs("abs_ylm", "|ylm|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-1, decorator=abs_deco))
      call bind_obs("ptnu", "transverse momentum of nub", &
                    EvalObs_t(eval_mem=eval_pt_lep, imem=-2), ifac=2)
      call bind_obs("ynu", "(pseudo-)rapidity of nub", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2))
      call bind_obs("abs_ynu", "|ylnub|", &
                    EvalObs_t(eval_mem=eval_y_lep, imem=-2, decorator=abs_deco))
      ! W-boson
      call bind_obs("ptw", "transverse momentum of W-", &
                    EvalObs_t(eval=eval_pt_V), ifac=2)
      call bind_obs("yw", "rapidity of W-", &
                    EvalObs_t(eval=eval_y_V))
      call bind_obs("abs_yw", "|yw-|", &
                    EvalObs_t(eval=eval_y_V, decorator=abs_deco))
      call bind_obs("etaw", "pseudo-rapidWty of W-", &
                    EvalObs_t(eval=eval_eta_V))
      call bind_obs("abs_etaw", "|etaw-|", &
                    EvalObs_t(eval=eval_eta_V, decorator=abs_deco))
      call bind_obs("mw", "invariant mass of W-", &
                    EvalObs_t(eval=eval_minv_V), ifac=1)
      call bind_obs("etw", "transverse energy: sqrt(mw^2 + ptw^2)", &
                    EvalObs_t(eval=eval_et_V), ifac=1)

      ! WH system
      call bind_obs("ptwh", "transverse momentum of WH system", &
                    EvalObs_t(eval=eval_pt_VH), ifac=2)
      call bind_obs("mwh", "invariant mass of WH system", &
                    EvalObs_t(eval=eval_minv_VH), ifac=1)


      if(isActive_flav())then
      ! general flavour-tagging
        call bind_obs("flav_njets", "number of flavour-tagged jets", &
                      EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                      needs_flav=.true.))
        call bind_obs("flav_ptj1", "transverse momentum of leading flavour-tagged jet", &
                      EvalObs_t(eval_mem=eval_flav_pt_ji, imem=-1, min_flav_njets=1, &
                      needs_flav=.true.), ifac=2)

        ! flavour-tagging related
        call bind_obs("mbb", "invariant mass of two b-tagged jets closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_minv_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("ptbb", &
                      "transverse momentum of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_pt_jj, min_flav_njets=2, &
                      needs_flav=.true.))
        call bind_obs("dRbb", &
                      "angular separation of two b-tagged jets whose invariant mass is closest to the Higgs mass", &
                      EvalObs_t(eval=eval_flav_dR_jj, min_flav_njets=2, &
                      needs_flav=.true.))
      ! end flavour-tagging
      endif

    end select

    !---------------------!
    ! default observables !
    !---------------------!

    call bind_obs("njets", "number of reconstructed jets", &
                  EvalObs_t(eval=eval_njets, is_integer=.true.))

    call bind_obs("npartons", "number of partons", &
                  EvalObs_t(eval=eval_npartons, is_integer=.true., digest=.false.))

    if(isActive_flav())then
      call bind_obs("flav_njets", "number of flavour-tagged jets", &
                    EvalObs_t(eval=eval_flav_njets, is_integer=.true., &
                    needs_flav=.true.))
    endif

    call bind_obs("nphotons", "number of (isolated) photon", &
                  EvalObs_t(eval=eval_nphotons, is_integer=.true.))

    ! leading jet
    call bind_obs("ptj1", "transverse momentum of leading jet", &
                  EvalObs_t(eval_mem=eval_pt_jet, imem=-1, min_njets=1), ifac = 2)
    call bind_obs("ej1", "energy of leading jet", &
                  EvalObs_t(eval_mem=eval_e_jet, imem=-1, min_njets=1))
    call bind_obs("yj1", "rapidity of leading jet", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-1, min_njets=1))
    call bind_obs("abs_yj1", "|yj1|", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-1, decorator=abs_deco, min_njets=1))
    call bind_obs("etaj1", "pseudo-rapidity of leading jet", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-1, min_njets=1))
    call bind_obs("abs_etaj1", "|etaj1|", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-1, decorator=abs_deco, min_njets=1))
    ! sub-leading jet
    call bind_obs("ptj2", "transverse momentum of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_pt_jet, imem=-2, min_njets=2), ifac = 2)
    call bind_obs("ej2", "energy of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_e_jet, imem=-2, min_njets=2))
    call bind_obs("yj2", "rapidity of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-2, min_njets=2))
    call bind_obs("abs_yj2", "|yj2|", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-2, decorator=abs_deco, min_njets=2))
    call bind_obs("etaj2", "pseudo-rapidity of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-2, min_njets=2))
    call bind_obs("abs_etaj2", "|etaj2|", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-2, decorator=abs_deco, min_njets=2))
    ! 3rd jet
    call bind_obs("ptj3", "transverse momentum of 3rd jet", &
                  EvalObs_t(eval_mem=eval_pt_jet, imem=-3, min_njets=3), ifac = 2)
    call bind_obs("ej3", "energy of 3rd jet", &
                  EvalObs_t(eval_mem=eval_e_jet, imem=-3, min_njets=3))
    call bind_obs("yj3", "rapidity of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-3, min_njets=3))
    call bind_obs("abs_yj3", "|yj3|", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-3, decorator=abs_deco, min_njets=3))
    call bind_obs("etaj3", "pseudo-rapidity of 3rd jet", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-3, min_njets=3))
    call bind_obs("abs_etaj3", "|etaj3|", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-3, decorator=abs_deco, min_njets=3))
    ! 4th jet
    call bind_obs("ptj4", "transverse momentum of 4th jet", &
                  EvalObs_t(eval_mem=eval_pt_jet, imem=-4, min_njets=4), ifac = 2)
    call bind_obs("ej4", "energy of 4th jet", &
                  EvalObs_t(eval_mem=eval_e_jet, imem=-4, min_njets=4))
    call bind_obs("yj4", "rapidity of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-4, min_njets=4))
    call bind_obs("abs_yj4", "|yj4|", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-4, decorator=abs_deco, min_njets=4))
    call bind_obs("etaj4", "pseudo-rapidity of 4th jet", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-4, min_njets=4))
    call bind_obs("abs_etaj4", "|etaj4|", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-4, decorator=abs_deco, min_njets=4))
    call bind_obs("ptsum_jets", "sum of jet pt's", &
                  EvalObs_t(eval=eval_sum_pt_jets))
    ! 5th jet
    call bind_obs("ptj5", "transverse momentum of 5th jet", &
                  EvalObs_t(eval_mem=eval_pt_jet, imem=-5, min_njets=5), ifac = 2)
    call bind_obs("ej5", "energy of 5th jet", &
                  EvalObs_t(eval_mem=eval_e_jet, imem=-5, min_njets=5))
    call bind_obs("yj5", "rapidity of sub-leading jet", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-5, min_njets=5))
    call bind_obs("abs_yj5", "|yj5|", &
                  EvalObs_t(eval_mem=eval_y_jet, imem=-5, decorator=abs_deco, min_njets=5))
    call bind_obs("etaj5", "pseudo-rapidity of 5th jet", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-5, min_njets=5))
    call bind_obs("abs_etaj5", "|etaj5|", &
                  EvalObs_t(eval_mem=eval_eta_jet, imem=-5, decorator=abs_deco, min_njets=5))
    ! dijet invariant mass
    call bind_obs("m12", "dijet invariant mass", &
                  EvalObs_t(eval=eval_minv_j12, min_njets=2), ifac=1)
    call bind_obs("m45", "dijet invariant mass j4+j5", &
                  EvalObs_t(eval=eval_minv_j45, min_njets=5), ifac=1)
    ! more di-jet observables
    call bind_obs("deltapt_j12", "|ptj1-ptj2|", &
                  EvalObs_t(eval=eval_deltapt_j12, min_njets=2))
    call bind_obs("pt_j12", "pT of the dijet system", &
                  EvalObs_t(eval=eval_pt_j12, min_njets=2), ifac=2)
    call bind_obs("dR_j12", "delta R between leading and subleading jet", &
                  EvalObs_t(eval=eval_dR_j12, min_njets=2))
    call bind_obs("deltaeta_j12", "|etaj1-etaj2|", &
                  EvalObs_t(eval=eval_deltaeta_j12, min_njets=2))
    call bind_obs("avgeta_j12", "(etaj1+etaj2)/2", &
                  EvalObs_t(eval=eval_avgeta_j12, min_njets=2))
    call bind_obs("avgeta_j123", "(etaj1+etaj2+etaj3)/3", &
                  EvalObs_t(eval=eval_avgeta_j123, min_njets=3))
    call bind_obs("deltay_j12", "|yj1-yj2|", &
                  EvalObs_t(eval=eval_deltay_j12, min_njets=2))
    call bind_obs("deltay_j13", "|yj1-yj3|", &
                  EvalObs_t(eval=eval_deltay_j13, min_njets=3))
    call bind_obs("deltay_j23", "|yj2-yj3|", &
                  EvalObs_t(eval=eval_deltay_j23, min_njets=3))
    call bind_obs("deltaphi_j12", "azimuthal angle between j1 & j2", &
                  EvalObs_t(eval=eval_deltaphi_j12, min_njets=2))
    call bind_obs("chi", "chi = exp(|yj1-yj2|)", &
                  EvalObs_t(eval=eval_chi_j12, min_njets=2))
    call bind_obs("ystar", "ystar = |yj1-yj2|/2", &
                  EvalObs_t(eval=eval_ystar_j12, min_njets=2))
    call bind_obs("yboost", "yboost = |yj1+yj2|/2", &
                  EvalObs_t(eval=eval_yboost_j12, min_njets=2))
    call bind_obs("max_abs_y_j12", "max(|yj1|,|yj2|)", &
                  EvalObs_t(eval=eval_max_abs_y_j12, min_njets=2))
    call bind_obs("ptj1EXP_03ystar", "ptj1*dexp(0.3*ystar)", &
                  EvalObs_t(eval=eval_pt_j1expYstar))

    call bind_obs("deltaeta_jfb", "|etajf-etajb| (most fwd & bwd jet)", &
                  EvalObs_t(eval_mem=eval_jet_fb, imem=2, min_njets=2))
    call bind_obs("deltay_jfb", "|yjf-yjb| (most fwd & bwd jet)", &
                  EvalObs_t(eval_mem=eval_jet_fb, imem=1, min_njets=2))
    call bind_obs("deltaphi_jfb", "azimuthal angle between jf & jb (most fwd & bwd jet)", &
                  EvalObs_t(eval_mem=eval_jet_fb, imem=2, min_njets=2))

    ! trijet
    call bind_obs("m123", "trijet invariant mass", &
                  EvalObs_t(eval=eval_minv_j123), ifac=1)
    call bind_obs("ystar_j123", "ystar = |yj1-yj2|+|yj2-yj3|+|yj1-yj3|", &
                  EvalObs_t(eval=eval_ystar_j123, min_njets=3))
    call bind_obs("ymax_j123", "sgn(|max{yj1,2,3}|-|min{yj1,2,3}|)*max{|yj1|,|yj2|,|yj3|}", &
                  EvalObs_t(eval=eval_ymax_j123, min_njets=3))
    call bind_obs("abs_ymax_j123", "|ymax|", &
                  EvalObs_t(eval=eval_ymax_j123, decorator=abs_deco, min_njets=3))
    call bind_obs("deltaphi_j13", "azimuthal angle between j1 & j3", &
                  EvalObs_t(eval=eval_deltaphi_j13))
    call bind_obs("deltaphi_j14", "azimuthal angle between j1 & j4", &
                  EvalObs_t(eval=eval_deltaphi_j14))
    call bind_obs("deltaphi_j15", "azimuthal angle between j1 & j5", &
                  EvalObs_t(eval=eval_deltaphi_j15))
    call bind_obs("deltaphi_j23", "azimuthal angle between j2 & j3", &
                  EvalObs_t(eval=eval_deltaphi_j23))
    call bind_obs("deltaphi_j24", "azimuthal angle between j2 & j4", &
                  EvalObs_t(eval=eval_deltaphi_j24))
    call bind_obs("deltaphi_j25", "azimuthal angle between j2 & j5", &
                  EvalObs_t(eval=eval_deltaphi_j25))
    call bind_obs("deltaphi_j34", "azimuthal angle between j3 & j4", &
                  EvalObs_t(eval=eval_deltaphi_j34))
    call bind_obs("deltaphi_j35", "azimuthal angle between j3 & j5", &
                  EvalObs_t(eval=eval_deltaphi_j35))
    call bind_obs("deltaphi_j45", "azimuthal angle between j4 & j5", &
                  EvalObs_t(eval=eval_deltaphi_j45))

    ! HT only summing over coloured final-state particles
    call bind_obs("ht_jets", "HT of jets", &
                  EvalObs_t(eval=eval_ht_jets), ifac=1)
    call bind_obs("ht_part", "HT of partons", &
                  EvalObs_t(eval=eval_ht_part), ifac=1)

    !> PDF momentum fractions
    call bind_obs("x1_pdf", "momentum fraction entering PDF (+z direction)", &
                  EvalObs_t(eval=eval_x1_pdf, digest=.false.))
    call bind_obs("x2_pdf", "momentum fraction entering PDF (-z direction)", &
                  EvalObs_t(eval=eval_x2_pdf, digest=.false.))

    call bind_obs("smin_over_shat", "technical cut emulator", &
                  EvalObs_t(eval=eval_smin_over_shat))



#ifdef USEFASTJET

    !---------------------!
    ! FastJet observables !
    !---------------------!

    do ialgo = 1, get_n_fj()
      abbr = get_abbr_fj(ialgo)

      call bind_obs(trim(abbr)//"_njets", "[FJ] number of jets", &
           EvalObs_t(eval_mem=eval_njets_fj, imem=-ialgo, is_integer=.true.))

      ! leading jet
      idx = get_idx_fj(ialgo, 1)
      call bind_obs(trim(abbr)//"_ptj1", "[FJ] transverse momentum of leading jet", &
           EvalObs_t(eval_mem=eval_pt_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=1))
      call bind_obs(trim(abbr)//"_yj1", "[FJ] rapidity of leading jet", &
           EvalObs_t(eval_mem=eval_y_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=1))
      call bind_obs(trim(abbr)//"_etaj1", "[FJ] pseudorapidity of leading jet", &
           EvalObs_t(eval_mem=eval_eta_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=1))

      ! sub-leading jet
      idx = get_idx_fj(ialgo, 2)
      call bind_obs(trim(abbr)//"_ptj2", "[FJ] transverse momentum of sub-leading jet", &
           EvalObs_t(eval_mem=eval_pt_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=2))
      call bind_obs(trim(abbr)//"_etaj2", "[FJ] pseudorapidity of sub-leading jet", &
           EvalObs_t(eval_mem=eval_eta_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=2))

      if(isActive_flav())then
         if (.not. get_wflav_fj(ialgo)) continue

         call bind_obs(trim(abbr)//"_flav_njets", "[FJ] number of flavoured jets", &
              EvalObs_t(eval_mem=eval_flav_njets_fj, imem=-ialgo, is_integer=.true., &
              needs_flav=.true.))

         idx = get_idx_fj(ialgo, 1)
         call bind_obs(trim(abbr)//"_flav_ptj1", "[FJ] transverse momentum of leading flavoured-jet", &
              EvalObs_t(eval_mem=eval_flav_pt_fj, imem=-idx, ialgo_fj=ialgo, min_flav_njets_fj=1, &
              needs_flav=.true.), ifac = 2)
         call bind_obs(trim(abbr)//"_flav_yj1", "[FJ] rapidity of leading flavoured-jet", &
              EvalObs_t(eval_mem=eval_flav_y_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=1, &
              needs_flav=.true.), ifac = 2)
         call bind_obs(trim(abbr)//"_flav_etaj1", "[FJ] pseudorapidity of leading flavoured-jet", &
              EvalObs_t(eval_mem=eval_flav_eta_fj, imem=-idx, ialgo_fj=ialgo, min_njets_fj=1, &
              needs_flav=.true.), ifac = 2)
         call bind_obs(trim(abbr)//"_flav_abs_etaj1", "[FJ] absolute pseudorapidity of leading flavoured-jet", &
              EvalObs_t(eval_mem=eval_flav_eta_fj, imem=-idx, decorator=abs_deco, ialgo_fj=ialgo, min_njets_fj=1, &
              needs_flav=.true.), ifac = 2)
      endif

    end do  ! ialgo

#endif

    !---------------------!
    ! feature observables !
    !---------------------!

    !> identified hadrons
    select case (name_proc)
    case ("ZJ_H", "WMJ_H", "WPJ_H", "EPEM_H", "EPEMJJ_H", "GJ_H")
      call bind_obs("pt_had", "transverse momentum of the identified hadron", &
                    EvalObs_t(eval=eval_pt_had), ifac=2)
      call bind_obs("p_had", "total momentum of the identified hadron", &
                    EvalObs_t(eval=eval_p_had))
      call bind_obs("pt_had_vs_ptj1", &
                    "transverse momentum of the identified hadron with respect to the leading-jet axis [1904.08878]", &
                    EvalObs_t(eval=eval_pt_had_vs_ptj1))
      call bind_obs("pt_had_vs_ptj", &
                    "transverse momentum of the identified hadron with respect to the closest jet axis [1904.08878]", &
                    EvalObs_t(eval=eval_pt_had_vs_ptj))
      call bind_obs("pl_frac_had1", &
                    "Longitudinal momentum fraction of the identified hadron w.r.t the leading-jet axis [1904.08878]", &
                    EvalObs_t(eval=eval_pl_frac_had1))
      call bind_obs("pl_frac_had", &
                    "Longitudinal momentum fraction of the identified hadron w.r.t the closest jet axis [1904.08878]", &
                    EvalObs_t(eval=eval_pl_frac_had))
      call bind_obs("E_had", "energy of the identified hadron", &
                    EvalObs_t(eval=eval_E_had))
      call bind_obs("x_had", "energy fraction of the identified hadron", &
                    EvalObs_t(eval=eval_x_had))
      call bind_obs("y_had", "rapidity of the identified hadron", &
                    EvalObs_t(eval=eval_y_had))
      call bind_obs("costh_had", "cos(theta) of identified hadron w.r.t. e^- beam", &
                    EvalObs_t(eval=eval_costh_had))
      call bind_obs("abs_y_had", "|y_had|", &
                    EvalObs_t(eval=eval_y_had, decorator=abs_deco))
      call bind_obs("dr_hadj1", "distance between hadron and leading-jet axis", &
                    EvalObs_t(eval=eval_deltaR_hadj1))
      call bind_obs("dr_hadj", "distance between hadron and closest jet axis", &
                    EvalObs_t(eval=eval_deltaR_hadj))
      call bind_obs("dr_hadz", "distance between hadron and the Z-boson", &
                    EvalObs_t(eval=eval_deltaR_hadZ))
      call bind_obs("dphi_hadz", "azimuthal angle between hadron and the Z-boson", &
                    EvalObs_t(eval=eval_deltaphi_hadZ))
      call bind_obs("xit_hadZ", "log of the inverse of the logitudinal fraction of transverse momentum &
                    &of the hadron with respect to the Z boson [2103.04377]", &
                    EvalObs_t(eval=eval_xit_hadZ))
      call bind_obs("pt_ratio_hadj", &
                    "ratio between the transverse momentum of the identified hadron and the closest jet", &
                    EvalObs_t(eval=eval_pt_ratio_hadj))
      call bind_obs("eta_had", "pseudorapidity of the identified hadron", &
                    EvalObs_t(eval=eval_eta_had))
      call bind_obs("abs_eta_had", "absolute value of pseudorapidity of the identified hadron", &
                    EvalObs_t(eval=eval_eta_had, decorator=abs_deco))
    end select

    select case (name_proc)
    case ("GJ_H")
      call bind_obs("m_gam_had", "invariant mass of the photon and the hadron", &
                    EvalObs_t(eval=eval_minv_gamhad, min_nphotons=1), ifac=1)
    end select

    !> P2B variables
    select case (name_proc)
    case ("Z", "ZJ", "WM", "WMJ", "WP", "WPJ", "H", "HJ", "HM", "HJM", &
          "HTO2PM", "HTO2PJM", "HTO2P", "HTO2PJ", &
          "HTO2TAUM", "HTO2TAUJM", "HTO2TAU", "HTO2TAUJ", &
          "GG", "GG_FRAG", "GGJ", "GGJFC")
      call bind_obs("p2b_qt", "transverse momentum of the colour-singlet *before* P2B", &
                    EvalObs_t(is_manual=.true., digest=.false.))
    end select

    !> Born projected variables
    select case (name_proc)
    case ("Z", "ZJ", "WM", "WMJ", "WP", "WPJ", "H", "HJ", "HM", "HJM",  &
         & "HTO2PM", "HTO2PJM", "HTO2P", "HTO2PJ",  &
         & "HTO2TAUM", "HTO2TAUJM", "HTO2TAU", "HTO2TAUJ",  &
         & "GG", "GG_FRAG", "GGJ", "GGJFC")
      call bind_obs("xib1", "Born: (1-xi1)", &
                    EvalObs_t(eval_mem=eval_born_xi, imem=1))
      call bind_obs("xib2", "Born: (1-xi2)", &
                    EvalObs_t(eval_mem=eval_born_xi, imem=2))
      call bind_obs("min_xib", "Born: min(xib1,xib2)", &
                    EvalObs_t(eval_mem=eval_born_xi, imem=3))
    end select

    !--------------------!
    ! global observables !
    !--------------------!

    !> DIS observables that are mandatory to be active
    select case (name_proc)
    case ("DIS1J", "SFDIS", "DISWM1J", "DISWP1J", "SFDISWM", "SFDISWP", &
          "DIS", "DISWM", "DISWP", "DISG")

      id_Q2_dis = getIdFromName_obs("q2", no_error=.true.)
      ! id_sqrtQ2_dis    = getIdFromName_obs("q", no_error=.true.)
      ! id_Ee_dis        = getIdFromName_obs("Ee", no_error=.true.)
      ! id_thetae_dis    = getIdFromName_obs("thetae", no_error=.true.)
      ! id_x_dis         = getIdFromName_obs("x", no_error=.true.)
      ! id_y_dis         = getIdFromName_obs("y", no_error=.true.)
      ! id_W2_dis        = getIdFromName_obs("W2", no_error=.true.)
      ! id_W_dis         = getIdFromName_obs("W", no_error=.true.)
      ! id_cosgammah_dis = getIdFromName_obs("cosgammah", no_error=.true.)
      ! id_nu_dis        = getIdFromName_obs("disnu", no_error=.true.)
      ! id_resrho_dis    = getIdFromName_obs("resrho", no_error=.true.)
      ! id_y_disg        = getIdFromName_obs("y_DISG", no_error=.true.)

    end select

    !--------!
    ! spikes !
    !--------!

    call bind_obs("s13", "s13 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(1, 3), digest=.false.))
    call bind_obs("s14", "s14 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(1, 4), digest=.false.))
    call bind_obs("s15", "s15 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(1, 5), digest=.false.))
    call bind_obs("s16", "s16 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(1, 6), digest=.false.))
    call bind_obs("s23", "s23 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(2, 3), digest=.false.))
    call bind_obs("s24", "s24 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(2, 4), digest=.false.))
    call bind_obs("s25", "s25 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(2, 5), digest=.false.))
    call bind_obs("s26", "s26 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(2, 6), digest=.false.))
    call bind_obs("s34", "s34 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(3, 4), digest=.false.))
    call bind_obs("s35", "s35 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(3, 5), digest=.false.))
    call bind_obs("s36", "s36 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(3, 6), digest=.false.))
    call bind_obs("s45", "s45 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(4, 5), digest=.false.))
    call bind_obs("s46", "s46 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(4, 6), digest=.false.))
    call bind_obs("s56", "s56 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(5, 6), digest=.false.))

    call bind_obs("s17", "s17 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(1, 7), digest=.false.))
    call bind_obs("s27", "s27 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(2, 7), digest=.false.))
    call bind_obs("s37", "s37 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(3, 7), digest=.false.))
    call bind_obs("s47", "s47 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(4, 7), digest=.false.))
    call bind_obs("s57", "s57 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(5, 7), digest=.false.))
    call bind_obs("s67", "s67 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sij, imem=-idx_sij(6, 7), digest=.false.))

    call bind_obs("s346", "s346 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(3,4,6), digest=.false.))
    call bind_obs("s356", "s356 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(3,5,6), digest=.false.))
    call bind_obs("s456", "s456 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(4,5,6), digest=.false.))
    call bind_obs("s347", "s347 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(3,4,7), digest=.false.))
    call bind_obs("s357", "s357 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(3,5,7), digest=.false.))
    call bind_obs("s457", "s457 invariant / shat", &
                  EvalObs_t(eval_mem=eval_sijk, imem=-idx_sijk(4,5,7), digest=.false.))

  end subroutine init_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> routine to deallocate global members
  !-----------------------------------------------------------------------------
  subroutine destroy_obs()
    if (allocated(list_obs_)) deallocate (list_obs_)
    if (allocated(used_obs_)) deallocate (used_obs_)
  end subroutine destroy_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> determine if the observable is an integer type
  !
  !> @param[in] id unique identifier for the observabes
  !-----------------------------------------------------------------------------
  pure logical function isInteger_obs(id)
    integer, intent(in) :: id
    integer :: i
    if (allocated(used_obs_)) then
      isInteger_obs = used_obs_(id)%is_integer
    else
      do i = 1, n_obs_
        if (list_obs_(i)%iused /= id) cycle
        isInteger_obs = list_obs_(i)%eval%is_integer
        return
      end do
    end if
    !> technically, error but we want to keep this pure?
  end function isInteger_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> determine if the associated observable is a manual one
  !> manual variables have to be set/unset by the user
  !> and it is their responsibility to manage their state
  !
  !> @param[in] id unique identifier for the observabes
  !> @return bool to indicate it's manual
  !-----------------------------------------------------------------------------
  pure logical function isManual_obs(id)
    integer, intent(in) :: id
    integer :: i
    if (allocated(used_obs_)) then
      isManual_obs = used_obs_(id)%is_manual
    else
      do i = 1, n_obs_
        if (list_obs_(i)%iused /= id) cycle
        isManual_obs = list_obs_(i)%eval%is_manual
        return
      end do
    end if
    !> technically, error but we want to keep this pure?
  end function isManual_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> determine if the associated observable is a composite one
  !> composite variables allow for multiple binnings in one event
  !
  !> @param[in] id unique identifier for the observabes
  !-----------------------------------------------------------------------------
  pure logical function isComposite_obs(id)
    integer, intent(in) :: id
    integer :: i
    !> my lazy fix to make 'writeBuffer_hist' work
    isComposite_obs = .false.
    if ((id < 1) .or. (id > nUsed_obs)) return
    if (allocated(used_obs_)) then
      isComposite_obs = used_obs_(id)%is_composite
    else
      do i = 1, n_obs_
        if (list_obs_(i)%iused /= id) cycle
        isComposite_obs = list_obs_(i)%eval%is_composite
        return
      end do
    end if
    !> technically, error but we want to keep this pure?
  end function isComposite_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> determine if the associated observable is flavour-sensitive
  !
  !> @param[in] id unique identifier for the observabes
  !-----------------------------------------------------------------------------
  pure logical function needsFlav_obs(id)
    integer, intent(in) :: id
    integer :: i
    if (allocated(used_obs_)) then
      needsFlav_obs = used_obs_(id)%needs_flav
    else
      do i = 1, n_obs_
        if (list_obs_(i)%iused /= id) cycle
        needsFlav_obs = list_obs_(i)%eval%needs_flav
        return
      end do
    end if
    !> technically, error but we want to keep this pure?
  end function needsFlav_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> inqure if an observable is used in a run
  !> if it's used, return the id of that observable, zero otherwise
  !
  !> @param[in] name string containing the name of an observable
  !> @return unique id if used, 0. otherwise
  !-----------------------------------------------------------------------------
  pure function isUsed_obs(name) result(id)
    use StrHelper_mod, only: to_upper_str
    integer :: id
    character(len=*), intent(in) :: name
    integer :: i

    id = 0

    do i = 1, n_obs_
      if (trim(to_upper_str(list_obs_(i)%name)) /= &
         & trim(to_upper_str(name))) cycle
      id = list_obs_(i)%iused
      return
    end do

  end function isUsed_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> print out a list of active observables
  !-----------------------------------------------------------------------------
  subroutine digestActive_obs(unit)
    integer, optional, intent(in) :: unit
    integer :: i
    integer :: iout
    character(len=*), parameter :: fmt_str = '(1x,"*",3x,A,T103,"*")'
    character(len=*), parameter :: fmt_act = '(1x,"*",3x,A,T40," --- ",A,T103,"*")'

    if (present(unit)) then
      iout = unit
    else
      iout = 6 ! output to screen
    end if

    write (iout, fmt_str) ""
    write (iout, fmt_str) " > process-specific observables <  "
    write (iout, fmt_str) ""
    do i = 1, n_obs_
      if (.not. list_obs_(i)%eval%digest) cycle
      write (iout, fmt_act) list_obs_(i)%name, trim(list_obs_(i)%desc)
    end do

  end subroutine digestActive_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> routine to initialize the (threadprivate) observables cache
  !
  !> @param[in] maxnpartons largest # partons
  !-----------------------------------------------------------------------------
  subroutine initBuffer_obs(maxnpartons)
    integer, intent(in) :: maxnpartons
    integer :: i, j
    logical :: flav_used_

    allocate (used_obs_(nUsed_obs))
    flav_used_ = .false.

    do i = 1, n_obs_
      j = list_obs_(i)%iused
      if (j == 0) cycle
      used_obs_(j) = list_obs_(i)%eval
      if (list_obs_(i)%eval%needs_flav) flav_used_ = .true.
    end do

    npar_current = -1

    call clearCache_obs()
    call unsetManual_obs()

    if (flav_used_) then
      allocate (flav_used_value_(nUsed_obs))
      allocate (flav_used_valid_(nUsed_obs))
    end if

  end subroutine initBuffer_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> routine to deallocate the (threadprivate) observables cache
  !-----------------------------------------------------------------------------
  subroutine destroyBuffer_obs()
    deallocate (used_obs_)
    if (allocated(flav_used_value_)) deallocate(flav_used_value_)
    if (allocated(flav_used_valid_)) deallocate(flav_used_valid_)
  end subroutine destroyBuffer_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Convert the observable identifier to its corresponding name string
  !
  !> @param[in] id unique observable identifier
  !> @param[out] desc optional string of the description
  !> @param[out] ifac optional integer indicating min(sqrt(shat)) scaling
  !> @return name string
  !-----------------------------------------------------------------------------
  function getNameFromId_obs(id, desc, ifac) result(res)
    character(len=max_name_length_obs) :: res
    integer, intent(in) :: id
    character(len=max_desc_length_obs), intent(out), optional :: desc
    integer, intent(out), optional :: ifac
    integer :: i

    if (id < 1 .or. id > nUsed_obs)  &
      &  stop "getNameFromId_obs: invalid observable id"

    res = ""
    if (present(desc)) desc = ""
    if (present(ifac)) ifac = 0

    do i = 1, n_obs_
      if (list_obs_(i)%iused /= id) cycle
      res = list_obs_(i)%name
      if (present(desc)) desc = list_obs_(i)%desc
      if (present(ifac)) ifac = list_obs_(i)%ifac
      return
    end do

    stop "getNameFromId_obs: couldn't match observable id"

  end function getNameFromId_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Convert a name string to the observable identifier
  !
  !> @param[in] name string containing the name of an observable
  !> @param[in] no_error bool to skip unmatched names (returns 0)
  !> @return observable id (0 if unsuccessful)
  !-----------------------------------------------------------------------------
  function getIdFromName_obs(name, no_error) result(id)
    use StrHelper_mod, only: to_upper_str
    use Flavour_mod, only: setActive_flav
    integer :: id
    character(len=*), intent(in) :: name
    logical, intent(in), optional :: no_error
    integer :: i

    id = 0

    do i = 1, n_obs_
      if (trim(to_upper_str(list_obs_(i)%name)) /= &
          trim(to_upper_str(name))) cycle

      if (list_obs_(i)%iused == 0) then
        nUsed_obs = nUsed_obs + 1
        list_obs_(i)%iused = nUsed_obs
        !> turn on flavour-tagging if flavour-sensitive observable is switched on
        if (list_obs_(i)%eval%needs_flav) call setActive_flav(.true.)
      end if

      id = list_obs_(i)%iused
      return

    end do

    if (present(no_error)) then
      if (no_error) return
    end if

    print *, "getIdFromName_obs: unmatched observable: ", name
    stop
  end function getIdFromName_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> set the value for a manual observable
  !
  !> @param[in] id unique identifier for the observabe
  !> @param[in] value value of the manual observable
  !-----------------------------------------------------------------------------
  subroutine setManual_obs(id, value)
    integer, intent(in) :: id
    double precision, intent(in) :: value

#ifndef NDEBUG
    if (.not. used_obs_(id)%is_manual) then
      print *, "setManual_obs: tried to set a non-manual observable: ",  &
        &  getNameFromId_obs(id)
      stop
    end if
#endif

    call used_obs_(id)%set(value)

  end subroutine setManual_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> unset a manual observable (<-> make invalid)
  !
  !> @param[in] id unique identifier for the observabe
  !-----------------------------------------------------------------------------
  subroutine unsetManual_single_(id)
    integer, intent(in) :: id

#ifndef NDEBUG
    if (.not. used_obs_(id)%is_manual) then
      print *, "unsetManual_single_: tried to unset a non-manual observable: ",  &
        &  getNameFromId_obs(id)
      stop
    end if
#endif

    call used_obs_(id)%unset()

  end subroutine unsetManual_single_

  !-----------------------------------------------------------------------------
  !> @brief
  !> unset all manual observables (<-> make invalid)
  !-----------------------------------------------------------------------------
  subroutine unsetManual_all_()
    integer :: i

    do i = 1, nUsed_obs
      if (.not. used_obs_(i)%is_manual) cycle
      call used_obs_(i)%unset()
    end do

  end subroutine unsetManual_all_

  !-----------------------------------------------------------------------------
  !> @brief
  !> reset the cache (do it before calling the eval_obs routines)
  !
  !> @param[in] onlyFlav optional flag to only clear flavour-sensitive obs
  !-----------------------------------------------------------------------------
  subroutine clearCache_obs(onlyFlav)
    logical, intent(in), optional :: onlyFlav
    integer :: i
    logical :: onlyFlav_

    !> manual observables are skipped as user should manage them

    onlyFlav_ = .false.
    if (present(onlyFlav)) onlyFlav_ = onlyFlav

    do i = 1, nUsed_obs
      if (used_obs_(i)%is_manual) cycle
      if (onlyFlav_ .and. (.not. used_obs_(i)%needs_flav)) cycle
      call used_obs_(i)%unset()
    end do

  end subroutine clearCache_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Evaluate all registered observables (non flavour-sensitive)
  !
  !> @param[in] npar kinematics specified by the number of external particles
  !> @param[in] skipJetAlgo optional flag to skip jet clustering
  !> @param[in] skipCache optional flag to skip reading & writing to the cache
  !-----------------------------------------------------------------------------
  subroutine evalAll_obs(npar, skipJetAlgo, skipCache)
    use KinData_mod, only: kin
    use Jets_mod, only: recombine_jet
#ifdef USEFASTJET
    use FastJet_mod, only: recombine_fj
#endif
    integer, intent(in) :: npar
    logical, intent(in), optional :: skipJetAlgo, skipCache
    logical :: skipJetAlgo_, skipCache_
    integer :: i
    real(kind=wp) :: value
    logical :: valid

    !> save the current npar
    npar_current = npar

    !> clean up the local cache before calling `eval_obs`
    call clearCache_obs()

    skipCache_ = .false.
    if (present(skipCache)) skipCache_ = skipCache

    !> read in cached values
    if (kin(npar)%qcachedObs .and. .not. skipCache_) then
      do i = 1, nUsed_obs
        call used_obs_(i)%set(kin(npar)%cacheObs(i))
      end do
      return
    end if

    !> need to actually compute the observables
    skipJetAlgo_ = .false.
    if (present(skipJetAlgo)) skipJetAlgo_ = skipJetAlgo
    if (.not. skipJetAlgo_) then
      call recombine_jet(npar)
#ifdef USEFASTJET
      call recombine_fj(npar)
#endif
    end if

    !> set manual observables here...
    ! call evalAll_manual(npar)

    !> compute all used observables
    do i = 1, nUsed_obs
      !> flavour-sensitive observables *cannot* be cached!
      !> default init to invalid state so cannot be cut on
      if (used_obs_(i)%needs_flav) cycle
      value = used_obs_(i)%get(npar, valid)
      if (valid .and. .not. skipCache_) kin(npar)%cacheObs(i) = value
    end do

  end subroutine evalAll_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Evaluate flavour-sensitive registered observables
  !
  !> @param[in] npar kinematics specified by the number of external partlicles
  !-----------------------------------------------------------------------------
  subroutine evalFlav_obs(npar)
    !use Flavour_mod, only: writeObs_flav
    integer, intent(in) :: npar
    integer :: i
    real(kind=wp) :: value
    logical :: valid

    !> clean up the local cache before calling `eval_obs`
    call clearCache_obs(onlyFlav=.true.)

    !> compute all flavour-sensitive observables
    do i = 1, nUsed_obs
      if (.not. used_obs_(i)%needs_flav) cycle
      value = used_obs_(i)%get(npar, valid)
    end do

  end subroutine evalFlav_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Save observable values & flags to the flavour-buffer
  !> we need a simple call-back routine to avoid a cyclic dependence
  !-----------------------------------------------------------------------------
  subroutine saveFlav_obs()
    use Flavour_mod, only: writeObs_flav
    integer :: id
    do id = 1, nUsed_obs
      flav_used_value_(id) = used_obs_(id)%get(-1, flav_used_valid_(id))
    end do
    call writeObs_flav(flav_used_value_, flav_used_valid_)
  end subroutine saveFlav_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Load observable values & flags from the flavour-buffer
  !>  we need a simple call-back routine to avoid a cyclic dependence
  !-----------------------------------------------------------------------------
  subroutine loadFlav_obs()
    use Flavour_mod, only: readObs_flav
    integer :: id
    call readObs_flav(flav_used_value_, flav_used_valid_)
    do id = 1, nUsed_obs
      call used_obs_(id)%unset()
      call used_obs_(id)%set(flav_used_value_(id), flav_used_valid_(id))
      !if (flav_used_valid_(id)) call used_obs_(id)%set(flav_used_value_(id))
    end do
  end subroutine loadFlav_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> Print all registered observables
  !
  !> @param[in] npar kinematics specified by the number of external partlicles
  !> @param[in] skipCache optional flag to skip reading from the cache
  !> @param[in] unit optional logical unit connected for formatted output (file, screen)
  !-----------------------------------------------------------------------------
  subroutine printAll_obs(npar, skipCache, unit)
    use KinData_mod
    integer, intent(in) :: npar
    logical, intent(in), optional :: skipCache
    integer, intent(in), optional :: unit
    logical :: skipCache_
    integer :: iout, i
    real(kind=wp) :: value
    real(kind=wp), dimension(nUsed_obs) :: valueUsed_npar
    logical, dimension(nUsed_obs) :: validUsed_npar
    character(len=*), parameter :: fmt_int = '(1x,"*",3x,A,T30,I10,T51,"*")'
    character(len=*), parameter :: fmt_dble = '(1x,"*",3x,A,T30,1e18.12,T51,"*")'
    character(len=*), parameter :: fmt_invalid = '(1x,"*",3x,A,T30,"<invalid>",T51,"*")'

    skipCache_ = .false.
    if (present(skipCache)) skipCache_ = skipCache

    iout = 6  ! output to screen
    if (present(unit)) iout = unit

    validUsed_npar(:) = .false.

    !> get the cached values or recompute and store in local arrays
    if (kin(npar)%qcachedObs .and. .not. skipCache_) then
      do i = 1, nUsed_obs
        value = used_obs_(i)%get(npar, validUsed_npar(i))
        valueUsed_npar(i) = kin(npar)%cacheObs(i)
        if (abs((value - valueUsed_npar(i))/(value + valueUsed_npar(i))) > 1e-6_wp) then
          print *, "printAll_obs: mismatch for ", getNameFromId_obs(i)
          print *, " > cache  = ", validUsed_npar(i)
          print *, " > recomp = ", value
        end if
      end do
    else
      write (iout, *) ">>> re-evaluate observables <<<"
      call clearCache_obs()
      do i = 1, nUsed_obs
        valueUsed_npar(i) = used_obs_(i)%get(npar, validUsed_npar(i))
      end do
    end if

    !> pretty printout
    write (iout, '(/,1x,50("*"),T15," printAll_obs(",I1,") ")') npar
    write (iout, fmt_int) "cacheID", kin(npar)%cacheID
    write (iout, fmt_int) "npar_from", kin(npar)%npar_from
    do i = 1, nUsed_obs
      if (validUsed_npar(i)) then
        if (used_obs_(i)%is_integer) then
          write (iout, fmt_int) trim(getNameFromId_obs(i)), INT(valueUsed_npar(i))
        else
          write (iout, fmt_dble) trim(getNameFromId_obs(i)), valueUsed_npar(i)
        end if
      else
        write (iout, fmt_invalid) trim(getNameFromId_obs(i))
      end if
    end do
    write (iout, '(1x,50("*"))')

  end subroutine printAll_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> retrieve the value for the observable (id)
  !
  !> @param[in] id unique observable identifier
  !> @param[out] valid flag
  !> @return value of the observable
  !-----------------------------------------------------------------------------
  function getValue_obs(id, valid) result(res)
    real(kind=wp) :: res
    integer, intent(in) :: id
    logical, intent(out) :: valid

#ifndef NDEBUG
    if (id < 1 .or. id > nUsed_obs)  &
      &  stop "getValue_obs: invalid observable id"
#endif

    !> get cached values (invalid npar -> will error if not cached)
    res = used_obs_(id)%get(-1, valid)

  end function getValue_obs

  !-----------------------------------------------------------------------------
  !> @brief
  !> retrieve the value for the observable (id)
  !
  !> @param[in] id unique observable identifier
  !> @param[out] valid flag
  !> @param[in] isub optional index for composite observables
  !> @param[inout] wgt optional weight for the binning
  !> @return value of the observable
  !-----------------------------------------------------------------------------
  function getValueComposite_obs(id, valid, isub, wgt) result(res)
    use Constants_mod, only: pi
    real(kind=wp) :: res
    integer, intent(in) :: id
    logical, intent(out) :: valid
    integer, optional, intent(in) :: isub
    double precision, optional, intent(inout) :: wgt
    res = 0._wp
    stop "getValueComposite_obs: to implement"
  end function getValueComposite_obs

end module Observables_mod
