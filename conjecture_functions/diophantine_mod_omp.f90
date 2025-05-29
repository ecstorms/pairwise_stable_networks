module diophantine_mod
  use omp_lib
  implicit none
contains

!====================
!  Binomial coefficient
!====================
pure integer(kind=8) function comb(n, k) result(res)
  integer(kind=8), intent(in) :: n, k
  integer(kind=8) :: i
  if (k < 0_8 .or. k > n) then
     res = 0_8
  else
     res = 1_8
     do i = 1_8, k
        res = res * (n - i + 1_8) / i
     end do
  end if
end function comb

!====================
!  Necessary inequality checks
!====================
pure logical function ok_tuple(cand, n)
  integer(kind=8), intent(in) :: cand(0:n), n
  integer(kind=8) :: total, first, i, k, lhs, rhs

  total = sum(cand)
  if (total /= 2_8**n) then
     ok_tuple = .false.; return
  end if

  first = sum([(i*cand(i), i = 0, n)])
  if (first /= n * 2_8**(n - 1)) then
     ok_tuple = .false.; return
  end if

  do k = 2_8, n
     lhs = sum([(comb(i, k) * cand(i), i = k, n)])
     rhs = comb(n, k) * 2_8**(n - k)
     if (lhs < rhs) then
        ok_tuple = .false.; return
     end if
  end do
  ok_tuple = .true.
end function ok_tuple

!====================================
!  Depth‑first enumeration (may recurse deeply)
!====================================
recursive subroutine search(level, cand, n, upper, sols, nsol, max_sols)
  integer(kind=8), intent(in)    :: level, n, upper, max_sols
  integer(kind=8), intent(inout) :: cand(0:n)
  integer(kind=8), intent(inout) :: sols(max_sols, n + 1)
  integer(kind=8), intent(inout) :: nsol
  integer(kind=8) :: val

  if (level > n) then
     if (ok_tuple(cand, n) .and. ok_tuple(cand(n:0:-1), n)) then
        ! Append solution in a threadsafe manner
        !$omp critical (solution_append)
        if (nsol < max_sols) then
           nsol = nsol + 1_8
           sols(nsol, :) = cand
        end if
        !$omp end critical (solution_append)
     end if
  else
     do val = 0_8, upper
        cand(level) = val
        call search(level + 1_8, cand, n, upper, sols, nsol, max_sols)
     end do
  end if
end subroutine search

!====================================
!  Public wrapper with OpenMP parallelism
!====================================
subroutine diophantine_solutions(n, out, nsol, max_sols)
  use omp_lib
  integer(kind=8), intent(in)  :: n, max_sols
  integer(kind=8), intent(out) :: out(max_sols, n + 1)
  integer(kind=8), intent(out) :: nsol

  integer(kind=8) :: sols(max_sols, n + 1)
  integer(kind=8) :: upper, val
  integer(kind=8) :: cand(0:n)

  if (n < 2_8) stop "n must be >= 2"

  nsol  = 0_8
  upper = 2_8**n

  ! Parallelise the very first branching level – each thread explores a distinct subtree.
  !$omp parallel do schedule(dynamic,1) default(shared) private(cand) &
  !$omp             firstprivate(n, upper, max_sols)
  do val = 0_8, upper
     cand = 0_8
     cand(0) = val
     call search(1_8, cand, n, upper, sols, nsol, max_sols)
  end do
  !$omp end parallel do

  ! Copy results back
  if (nsol > 0) out(1:nsol, :) = sols(1:nsol, :)
end subroutine diophantine_solutions

end module diophantine_mod
